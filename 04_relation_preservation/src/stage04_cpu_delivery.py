"""CPU-only audit, tables and figures for stage04. Does not initialize CUDA."""
import csv,json,subprocess
from pathlib import Path
import torch,numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import stage04_repeat as run
s1,s2=run.s1,run.s2
ROOT=run.ROOT;OUT=ROOT/'04_relation_preservation/outputs/stage04'
read=run.read


def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))


def classification(a):
    if not a['objects_alive']:return 'object_missing'
    if a['identity_or_instances_uncertain'] or a['invalid_truncated'] or a['uncertain']:return 'multiple_instances_or_uncertain'
    if a['new_relation']!='clear_success':return 'new_target_not_repaired'
    if a['old_relation']=='clear_success':return 'repaired_old_maintained'
    if a['old_relation']=='clear_failure':return 'repaired_old_clear_failure_degradation_candidate'
    return 'multiple_instances_or_uncertain'


def draw_image(folder,a):
    im=Image.open(folder/'final.png').convert('RGB');dr=ImageDraw.Draw(im)
    for key,color in [('A','red'),('B','cyan')]:
        box=a['objects'][key]['bbox_pixels']
        if box:dr.rectangle(box,outline=color,width=3);dr.text((box[0],max(0,box[1]-14)),key,fill=color)
    for extra in a.get('extra_objects',[]):
        if extra.get('bbox_pixels'):dr.rectangle(extra['bbox_pixels'],outline='yellow',width=3)
    im.save(folder/'final_annotated.png')


def main():
    assert not torch.cuda.is_initialized()
    cfg=read(run.CONFIG);hashes,cpu=run.fingerprints(cfg)
    lock=read(OUT/'protocol_lock.json');assert lock['fingerprints']==hashes
    assert s2.sha(OUT/'AGENTS_at_lock.md')==lock['agents_sha256_before_gpu']
    selected=run.selection(OUT)['selected_seeds'];screening=run.records(OUT)
    assert [r['seed'] for r in screening]==cfg['seed_candidates'][:len(screening)]
    runs=[];annotations={};sample_cases=[];table=[];restore_rows=[];inner=[];steps=[];heat=[];initials=[]
    prior_cp=torch.load(ROOT/cfg['checkpoint'],map_location='cpu',weights_only=False)
    for record in screening:
        seed=record['seed'];case=OUT/f'seed{seed}';m=read(case/'baseline_run_metrics.json');assert m['completed'];runs.append(m)
        a=read(case/'U/annotation.json');assert a['image_sha256']==s2.sha(case/'U/final.png')
        cp=torch.load(case/'U/checkpoint.pt',map_location='cpu',weights_only=False)
        assert cp['fingerprints']==hashes and cp['switches']==cfg['active_switches'] and cp['next_index']==5 and cp['boundary']=='after scheduler.step'
        assert len(cp['scheduler_state']['derivatives'])==4 and cp['effective_config']['seed']==seed
        initial=cp['initial_latent']['__tensor__'];assert torch.isfinite(initial).all() and initial.dtype==torch.float16
        assert s1.tensor_hash(initial)==record['initial_latent_sha256']==cp['effective_config']['initial_latent_sha256']
        assert s1.nested_equal(cp['conditioning'],prior_cp['conditioning'])
        initials.append({'seed':seed,'tensor_sha256':record['initial_latent_sha256'],'checkpoint_sha256':s2.sha(case/'U/checkpoint.pt')})
        annotations[str(seed)]={'U':a};draw_image(case/'U',a)
        ref=torch.load(case/'U/reference_latents.pt',map_location='cpu',weights_only=False);assert all(torch.isfinite(t).all() for t in ref.values())
    assert len({r['tensor_sha256'] for r in initials})==len(initials)
    assert s1.tensor_hash(prior_cp['initial_latent']['__tensor__']) not in {r['tensor_sha256'] for r in initials}
    for seed in selected:
        case=OUT/f'seed{seed}';path=case/'intervene_run_metrics.json'
        if not path.exists() or not read(path)['completed']:continue
        m=read(path);runs.append(m);gates=read(case/'restoration_metrics.json');assert gates['passed']
        ref=torch.load(case/'U/reference_latents.pt',map_location='cpu',weights_only=False);restored=torch.load(case/'R/latent.pt',map_location='cpu',weights_only=False)
        assert torch.equal(ref['first'],restored['first']) and torch.equal(ref['final'],restored['final'])
        assert np.array_equal(np.asarray(Image.open(case/'U/final.png')),np.asarray(Image.open(case/'R/final.png')))
        for key in ['first_step','final_latent']:assert gates[key]['torch_equal'] and gates[key]['max_abs_error']==0
        for key in ['first_step','final_latent']:restore_rows.append({'seed':seed,'item':key,**gates[key]})
        restore_rows.append({'seed':seed,'item':'image','torch_equal':gates['image']['pixel_exact'],'max_abs_error':gates['image']['max_abs_error']})
        a=read(case/'G20/annotation.json');assert a['image_sha256']==s2.sha(case/'G20/final.png')
        checked=s2.evaluate_annotation(read(case/'G20/annotation_input.json'),cfg);assert checked==a
        annotations[str(seed)]['G20']=a;draw_image(case/'G20',a)
        u=annotations[str(seed)]['U'];cls=classification(a)
        result={'seed':seed,'classification':cls,'baseline':u,'guided':a,'delta_dx':None if a['dx'] is None else a['dx']-u['dx'],'delta_dy':None if a['dy'] is None else a['dy']-u['dy'],'center_joint_success':a['joint_success'],'scene_content_preserved':a['scene_content_preserved'],'restoration':gates}
        sample_cases.append(result)
        for kind,t in [('U',u),('G20',a)]:
            table.append({'seed':seed,'branch':kind,'A_bbox_pixels':json.dumps(t['objects']['A']['bbox_pixels']),'B_bbox_pixels':json.dumps(t['objects']['B']['bbox_pixels']),'dx':t['dx'],'dy':t['dy'],'old_relation':t['old_relation'],'new_relation':t['new_relation'],'joint_success':t['joint_success'],'objects_alive':t['objects_alive'],'uncertain':t['uncertain'] or t['identity_or_instances_uncertain'], 'relation_uncertain':t['uncertain'],'identity_or_instances_uncertain':t['identity_or_instances_uncertain'],'red_apple_count':t['red_apple_count'],'major_blue_cup_count':t['major_blue_cup_count'],'all_cup_count':t['all_cup_count'],'all_cup_count_range':json.dumps(t.get('all_cup_count_range',[t['all_cup_count'],t['all_cup_count']])), 'cup_count_uncertain':t.get('all_cup_count_uncertain',False),'extra_objects':json.dumps(t.get('extra_objects',[]),ensure_ascii=False),'new_objects_relative_to_baseline':json.dumps(t.get('new_objects_relative_to_baseline',[]),ensure_ascii=False),'scene_content_preserved':True if kind=='U' else t['scene_content_preserved'],'classification':cls if kind=='G20' else 'qualified_start'})
        for kind in ['inner','step','heat']:
            data=rows(case/f'G20/{kind}_metrics.csv');target={'inner':inner,'step':steps,'heat':heat}[kind];target.extend({'seed':seed,**r} for r in data)
            if kind=='inner':assert all(r['finite']=='True' and r['nonzero_gradient']=='True' and r['update_executed']=='True' and r['model_parameter_grad_present']=='False' and float(r['relative_update'])<=.1 for r in data)
        for annotated in [False,True]:
            canvas=Image.new('RGB',(1024,552),'white');dr=ImageDraw.Draw(canvas)
            for col,kind in enumerate(['U','G20']):
                t=annotations[str(seed)][kind];dr.text((col*512+5,5),f'seed {seed} {kind} dx={t["dx"]} dy={t["dy"]}',fill='black')
                canvas.paste(Image.open(case/kind/('final_annotated.png' if annotated else 'final.png')).resize((512,512)),(col*512,40))
            canvas.save(case/('U_G20_annotated.png' if annotated else 'U_G20_original.png'))
    # All recorded GPU attempts count even if partial; do not sum successful-only costs.
    all_runs=[read(p) for p in OUT.glob('seed*/*_run_metrics.json')]
    assert all(m['switches_at_start']==cfg['active_switches'] for m in all_runs)
    gpu=sum(m['gpu_activity_wall_seconds'] for m in all_runs);updates=sum(m['inner_updates_total'] for m in all_runs)
    started=sum(m.get('trajectories_started',0) for m in all_runs);completed=sum(m.get('trajectories_completed',0) for m in all_runs)
    assert gpu<=600 and updates<=300 and started<=12
    ordering=[read(OUT/f'seed{r["seed"]}/baseline_run_metrics.json') for r in screening]+[read(OUT/f'seed{s}/intervene_run_metrics.json') for s in selected if (OUT/f'seed{s}/intervene_run_metrics.json').exists()]
    elapsed=0.;count=0
    for m in ordering:assert abs(m['previous_gpu_seconds']-elapsed)<1e-8 and m['previous_inner_updates']==count;elapsed+=m['gpu_activity_wall_seconds'];count+=m['inner_updates_total']
    assert count==updates
    s1.csv_rows(OUT/'screening.csv',screening);s1.dump_json(OUT/'initial_latent_identities.json',initials)
    s1.dump_json(OUT/'relation_results.json',sample_cases);s1.csv_rows(OUT/'relation_objects.csv',table);s1.csv_rows(OUT/'restoration_metrics.csv',restore_rows)
    for name,data in [('inner',inner),('step',steps),('heat',heat)]:s1.csv_rows(OUT/f'{name}_metrics.csv',data)
    s1.dump_json(OUT/'all_annotations.json',annotations)
    canvas=Image.new('RGB',(1536,2*552),'white');dr=ImageDraw.Draw(canvas)
    for j,r in enumerate(screening):
        x=(j%3)*512;y=(j//3)*552
        dr.text((x+6,y+6),f'seed {r["seed"]}: '+('QUALIFIED' if r['qualified'] else 'REJECTED'),fill='black')
        dr.text((x+6,y+20),f'dx={r["dx"]}, dy={r["dy"]}',fill='black')
        canvas.paste(Image.open(OUT/f'seed{r["seed"]}/U/final_annotated.png').resize((512,512)),(x,y+40))
    canvas.save(OUT/'screening_annotated.png')
    for index in [5,9]:
        files=[OUT/f'seed{s}/G20/heat_{index:02d}_{phase}.npz' for s in selected for phase in ['before','after'] if (OUT/f'seed{s}/G20/heat_{index:02d}_{phase}.npz').exists()]
        if not files:continue
        raw=[np.load(p) for p in files];assert all(np.isfinite(z[k]).all() for z in raw for k in z.files)
        fig,axes=plt.subplots(2,len(raw),figsize=(3.3*len(raw),6),squeeze=False,layout='constrained');scale=[]
        for row,layer in enumerate(['mid','up']):
            vmax=max(float(z[layer].max()) for z in raw);scale.append({'layer':layer,'vmin':0,'vmax':vmax})
            for col,(z,p) in enumerate(zip(raw,files)):
                assert set(z.files)=={'mid','up','mask'} and z['mask'].sum()==288
                ax=axes[row,col];im=ax.imshow(z[layer],vmin=0,vmax=vmax,cmap='magma',extent=(0,1,1,0));ax.axhline(3/32,color='cyan',lw=.8);ax.axhline(12/32,color='cyan',lw=.8);ax.set_title(p.parent.parent.name+' '+p.stem.split('_')[-1]+' / '+layer)
            fig.colorbar(im,ax=list(axes[row]),fraction=.022,pad=.02)
        fig.suptitle(f'Index {index}: apple / same per-layer scale across seeds and phases');fig.savefig(OUT/f'heat_index{index}.png',dpi=125);plt.close(fig)
        s1.dump_json(OUT/f'heat_index{index}_scales.json',scale)
    classes={name:sum(r['classification']==name for r in sample_cases) for name in ['new_target_not_repaired','repaired_old_maintained','repaired_old_clear_failure_degradation_candidate','object_missing','multiple_instances_or_uncertain']}
    repaired=classes['repaired_old_maintained']+classes['repaired_old_clear_failure_degradation_candidate']
    done=len(sample_cases)==len(selected) and not (OUT/'STOP.json').exists()
    fallback=bool(done and selected and repaired==0 and classes['new_target_not_repaired']>0 and classes['new_target_not_repaired']==sum(r['classification'] in ['new_target_not_repaired','repaired_old_maintained','repaired_old_clear_failure_degradation_candidate'] for r in sample_cases))
    summary={'task':'stage04','status':'complete' if done else 'incomplete','completed':done,'base_commit':cfg['base_commit'],'study_type':'qualified-start small-sample exploration; no population success-rate estimate','new_baselines_generated':len(screening),'new_qualified_starts':len(selected),'restorations_completed':sum((OUT/f'seed{s}/restoration_metrics.json').exists() and read(OUT/f'seed{s}/restoration_metrics.json').get('passed',False) for s in selected),'new_guided_completed_and_annotated':len(sample_cases),'selected_seeds':selected,'selection_frozen_before_guidance':True,'classifications':classes,'new_repaired_count':repaired,'geometrically_evaluable_new_cases':sum(r['classification'] in ['new_target_not_repaired','repaired_old_maintained','repaired_old_clear_failure_degradation_candidate'] for r in sample_cases),'scene_content_failures':sum(not r['scene_content_preserved'] for r in sample_cases),'multiple_blue_cup_or_target_identity_uncertain_cases':sum(r['guided']['identity_or_instances_uncertain'] for r in sample_cases),'cup_count_uncertain_cases':sum(r['guided'].get('all_cup_count_uncertain',False) for r in sample_cases),'geometry_results':sample_cases,'overall_success_rate':None,'relation_degradation_evidence':classes['repaired_old_clear_failure_degradation_candidate']>0,'historical_seed19':{'included_in_new_counts':False,'new_run':False,'new_relation':'clear_success','old_relation':'clear_success','scene_content_preserved':False,'source':'04_relation_preservation/outputs/stage03c/stage03c_summary.json'},'fallback_to_FLUX_triggered':fallback,'conclusion':'finite new cases support further review only' if repaired else ('all valid new cases failed; stop SDXL tuning and prepare FLUX single case' if fallback else 'inconclusive: no qualified starts, incomplete execution, or uncertain objects'),'protection_development_authorized':False,'FLUX_GPU_work_this_turn':0,'FLUX_preparation_config':'04_relation_preservation/configs/stage04_flux_single_preparation.json' if fallback else None,'GPU_work_seconds_all_attempts':gpu,'inner_updates_all_attempts':updates,'trajectories_started':started,'trajectories_completed':completed,'peak_allocated_mib':max(m['peak_allocated_mib'] for m in all_runs),'peak_reserved_mib':max(m['peak_reserved_mib'] for m in all_runs),'config':cfg,'environment':all_runs[0]['environment'],'active_switches':all_runs[0]['switches_at_start'],'all_model_weight_hashes_exact':all(m['model_content_hashes']==cfg['model_weight_hashes'] for m in all_runs),'commands':[m['argv'] for m in ordering],'sampling_source_sha256':s2.sha(Path(run.__file__)),'CPU_delivery_source_sha256':s2.sha(Path(__file__))}
    s1.dump_json(OUT/'stage04_summary.json',summary)
    pts=list(OUT.rglob('*.pt'));ignored=subprocess.check_output(['git','check-ignore','--stdin'],input='\n'.join(str(p.relative_to(ROOT)) for p in pts)+'\n',text=True,cwd=ROOT).splitlines();assert len(ignored)==len(pts)
    s1.dump_json(OUT/'artifact_audit.json',{'historical_and_current_locks_unchanged':True,'selection_lock_unchanged':True,'all_initial_latents_distinct_once_saved':True,'no_new_initial_latent_matches_historical_seed19':True,'all_conditioning_exact_to_03C':True,'selected_R_tensor_and_pixel_exact':True,'all_guidance_finite_parameters_grad_absent':True,'cumulative_budget_not_reset':True,'all_pt_git_ignored':True,'local_pt_count':len(pts),'local_pt_bytes':sum(p.stat().st_size for p in pts),'CPU_cuda_initialized':torch.cuda.is_initialized(),'CPU_delivery_sha256':s2.sha(Path(__file__))})
    print(json.dumps({k:v for k,v in summary.items() if k not in ['config','geometry_results','environment','active_switches','commands']},indent=2,ensure_ascii=False))


if __name__=='__main__':main()
