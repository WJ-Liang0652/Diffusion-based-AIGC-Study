"""CPU-only delivery/audit; never runs a model or draws new random noise."""
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import stage01_sdxl as s1
import stage02_relations as s2

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'04_relation_preservation/outputs/stage03c'
DATA=OUT/'repair01'


def read(p): return json.loads(p.read_text())


def rows(p):
    with p.open(newline='') as f: return list(csv.DictReader(f))


def main():
    assert not torch.cuda.is_initialized()
    locks=[OUT/'protocol_lock.json',DATA/'protocol_lock.json']
    for p in locks:
        lock=read(p);assert all(s2.sha(ROOT/n)==h for n,h in lock['fingerprints'].items())
        assert s2.sha(p.parent/'AGENTS_at_lock.md')==lock['agents_sha256_before_gpu']
    run0=read(OUT/'prepare_run_metrics.json');prep=read(DATA/'prepare_run_metrics.json');guide=read(DATA/'guide_run_metrics.json')
    assert not run0['completed'] and run0['inner_updates_total']==0 and 'last_completed' not in run0
    assert prep['completed'] and guide['completed'] and guide['G5']['completed'] and guide['G20']['completed']
    assert prep['switches_at_start']==guide['switches_at_start']==prep['config']['active_switches']
    cfg=read(ROOT/'04_relation_preservation/configs/stage03c_repair01.json')
    r=read(DATA/'restoration_metrics.json');assert r['passed'] and r['image']['pixel_exact']
    assert all(r[k]['torch_equal'] and r[k]['max_abs_error']==0 for k in ['first_step','final_latent'])
    ref=torch.load(DATA/'U/reference_latents.pt',map_location='cpu',weights_only=False)
    rr=torch.load(DATA/'R/latent.pt',map_location='cpu',weights_only=False)
    assert torch.equal(ref['final'],rr['final'])
    cp=torch.load(DATA/'U/checkpoint.pt',map_location='cpu',weights_only=False)
    old=torch.load(ROOT/cfg['checkpoint'],map_location='cpu',weights_only=False)
    assert s1.tensor_hash(cp['initial_latent']['__tensor__'])==old['effective_config']['initial_latent_sha256']
    assert cp['conditioning'].keys()==old['conditioning'].keys() and s1.nested_equal(cp['conditioning'],old['conditioning'])
    assert cp['next_index']==5 and len(cp['scheduler_state']['derivatives'])==4
    annotations={name:read(DATA/name/'annotation.json') for name in ['U','G5','G20']}
    assert annotations['U']['baseline_qualified']
    assert annotations['G20']['joint_success'] and annotations['G20']['cup_category_total_count']==2
    for name,a in annotations.items():
        assert a['image_sha256']==s2.sha(DATA/name/'final.png')
        checked=s2.evaluate_annotation(read(DATA/name/'annotation_input.json'),cfg)
        assert checked['dx']==a['dx'] and checked['dy']==a['dy'] and checked['joint_success']==a['joint_success']
        image=Image.open(DATA/name/'final.png').convert('RGB');draw=ImageDraw.Draw(image)
        for key,color in [('A','red'),('B','cyan')]:
            box=a['objects'][key]['bbox_pixels'];draw.rectangle(box,outline=color,width=4)
            draw.text((box[0],max(0,box[1]-15)),key,fill=color)
        for extra in a.get('extra_objects',[]):
            if 'bbox_pixels' in extra:
                box=extra['bbox_pixels'];draw.rectangle(box,outline='yellow',width=3)
                draw.text((box[0],box[1]-15),'extra mug',fill='yellow')
        image.save(DATA/name/'final_annotated.png')
    for typ in ['original','annotated']:
        canvas=Image.new('RGB',(1536,556),'white');draw=ImageDraw.Draw(canvas)
        for column,name in enumerate(['U','G5','G20']):
            a=annotations[name]
            draw.text((column*512+6,6),f'{name}: dx={a["dx"]:.6f}, dy={a["dy"]:.6f}',fill='black')
            label='old PASS / new PASS + extra mug' if name=='G20' else 'old PASS / new FAIL'
            draw.text((column*512+6,22),label,fill='black')
            image=Image.open(DATA/name/('final.png' if typ=='original' else 'final_annotated.png'))
            canvas.paste(image.resize((512,512)),(column*512,44))
        canvas.save(OUT/f'U_G5_G20_{typ}.png')
    s1.dump_json(OUT/'relation_results.json',annotations)
    s1.csv_rows(OUT/'relations.csv',[{'branch':name,'dx':a['dx'],'dy':a['dy'],'old_relation':a['old_relation'],
        'new_relation':a['new_relation'],'joint_success':a['joint_success'],'objects_alive':a['objects_alive'],
        'uncertain':a['uncertain'],'extra_objects':json.dumps(a.get('extra_objects',[]),ensure_ascii=False)} for name,a in annotations.items()])
    s1.dump_json(OUT/'restoration_metrics.json',r)
    s1.csv_rows(OUT/'restoration_metrics.csv',[{'item':key,'max_abs_error':r[key]['max_abs_error'],'exact':r[key]['torch_equal']} for key in ['first_step','final_latent']]+[{'item':'image','max_abs_error':r['image']['max_abs_error'],'exact':r['image']['pixel_exact']}])
    all_inner=[];all_steps=[];all_heat=[];guidance_summary={}
    for name in ['G5','G20']:
        ir=rows(DATA/name/'inner_metrics.csv');sr=rows(DATA/name/'step_metrics.csv');hr=rows(DATA/name/'heat_metrics.csv')
        assert len(ir)==(25 if name=='G5' else 100)
        assert all(x['finite']=='True' and x['model_parameter_grad_present']=='False' and float(x['relative_update'])<=.1 for x in ir)
        assert all(int(x['executed_inner_iterations'])==(5 if name=='G5' else 20) for x in sr)
        all_inner.extend({'branch':name,**x} for x in ir);all_steps.extend({'branch':name,**x} for x in sr);all_heat.extend({'branch':name,**x} for x in hr)
        guidance_summary[name]={'inner_updates':len(ir),'all_finite':True,'parameter_grad_present':False,
            'gradient_norm_range':[min(float(x['gradient_norm']) for x in ir),max(float(x['gradient_norm']) for x in ir)],
            'update_norm_range':[min(float(x['update_norm']) for x in ir),max(float(x['update_norm']) for x in ir)],
            'max_relative_update':max(float(x['relative_update']) for x in ir),'steps':sr}
    assert len(all_inner)==125 and guide['inner_updates_total']==125
    a=rows(DATA/'G5/inner_metrics.csv')[:5];b=rows(DATA/'G20/inner_metrics.csv')[:5]
    assert a==b, 'Same starting snapshot and first5 updates should reproduce at index5'
    s1.csv_rows(OUT/'inner_metrics.csv',all_inner);s1.csv_rows(OUT/'step_metrics.csv',all_steps);s1.csv_rows(OUT/'heat_metrics.csv',all_heat)
    s1.dump_json(OUT/'guidance_summary.json',guidance_summary)
    panels=[]
    for index in [5,9]:
        files=[DATA/name/f'heat_{index:02d}_{phase}.npz' for name in ['G5','G20'] for phase in ['before','after']]
        raw=[np.load(p) for p in files]
        for z in raw: assert set(z.files)=={'mid','up','mask'} and all(np.isfinite(z[k]).all() for k in z.files) and z['mask'].sum()==288
        fig,axes=plt.subplots(2,4,figsize=(13.5,7))
        scales=[]
        for row,layer in enumerate(['mid','up']):
            vmax=max(float(z[layer].max()) for z in raw)
            for col,(z,label) in enumerate(zip(raw,['G5 before','G5 after','G20 before','G20 after'])):
                ax=axes[row,col];im=ax.imshow(z[layer],vmin=0,vmax=vmax,cmap='magma',extent=(0,1,1,0))
                ax.axhline(3/32,color='cyan',lw=.8);ax.axhline(12/32,color='cyan',lw=.8)
                ax.set_title(f'{label} / {layer}');ax.set_xlabel('normalized x');ax.set_ylabel('normalized y')
            fig.colorbar(im,ax=list(axes[row]),fraction=.018,pad=.02)
            scales.append({'layer':layer,'vmin':0.,'vmax':vmax,'unit':'mean attention probability'})
        fig.suptitle(f'Index {index}: same per-layer scale across branches and phases / apple only')
        fig.subplots_adjust(top=.88,right=.88,hspace=.35,wspace=.4)
        file=OUT/f'heat_index{index}.png';fig.savefig(file,dpi=130);plt.close(fig)
        panels.append({'file':file.name,'scales':scales,'inputs':[str(p.relative_to(ROOT)) for p in files]})
    s1.dump_json(OUT/'heat_panel_scales.json',panels)
    runs=[run0,prep,guide];gpu=sum(x['gpu_activity_wall_seconds'] for x in runs)
    summary={'task':'stage03c','status':'complete','completed':True,'base_commit':cfg['base_commit'],
        'G20_new_relation_repaired':True,'G5_new_relation_repaired':False,'G20_old_relation_maintained':True,
        'joint_success_by_prespecified_center_rules':True,'pure_relation_degradation_candidate':False,
        'interpretation':'single_case_joint_center_relation_success_with_extra_object_and_scene_fidelity_failure',
        'quality_limits':['Additional red-white mug/gray-blue handle appears at prior apple position','Leafy supporting branch generated; partial left-border crop','Apple size/position and scene content change; temporal instance identity not tracked'],
        'scene_content_preserved':False,'protection_effectiveness_tested':False,'fallback_to_FLUX_triggered':False,
        'decision':'SDXL stop-loss trigger (both branches fail) not met. Stop this bounded experiment and return for review; no expansion.',
        'config':cfg,'restoration':r,'relations':annotations,'guidance':guidance_summary,
        'cost':{'gpu_activity_seconds_including_failure':gpu,'normal_trajectories_completed':4,'guidance_updates':125,
            'local_repairs_used':1,'peak_allocated_mib':max(x['peak_allocated_mib'] for x in runs),
            'peak_reserved_mib':max(x['peak_reserved_mib'] for x in runs)},
        'attempts':[{'path':str((OUT if i==0 else DATA).relative_to(ROOT)), 'mode':x['mode'],'completed':x['completed'],
            'gpu_seconds':x['gpu_activity_wall_seconds'],'argv':x['argv']} for i,x in enumerate(runs)],
        'same_first5_G5_G20_index5_scalar_metrics_exact':True,'no_extra_FLUX_GPU_run':True}
    assert gpu<600 and summary['cost']['guidance_updates']==125
    s1.dump_json(OUT/'stage03c_summary.json',summary)
    pts=sorted(OUT.rglob('*.pt'))
    ignored=subprocess.check_output(['git','check-ignore','--stdin'],input='\n'.join(str(p.relative_to(ROOT)) for p in pts)+'\n',text=True,cwd=ROOT).splitlines()
    assert len(ignored)==len(pts)
    audit={'all_original_and_repair_lock_fingerprints_unchanged':True,'old_sources_inputs_preserved':True,
        'repair_history_preserved':True,'restoration_exact_from_disk':True,'initial_latent_exact_to_old_saved':True,
        'conditioning_exact_to_old_saved':True,'R_final_latent_audited_again_on_CPU':True,
        'G5_G20_first5_index5_scalar_metrics_exact':True,'all_guidance_finite_parameter_grad_absent':True,
        'local_pt_count':len(pts),'local_pt_bytes':sum(p.stat().st_size for p in pts),'all_pt_git_ignored':True,
        'heat_raw_count':len(list(DATA.glob('G*/heat_*.npz'))),'post_sampling_cpu_cuda_initialized':torch.cuda.is_initialized(),
        'cpu_delivery_source_sha256':s2.sha(Path(__file__).resolve())}
    s1.dump_json(OUT/'artifact_audit.json',audit)
    print(json.dumps({'G20_dx':annotations['G20']['dx'],'G20_dy':annotations['G20']['dy'],'cost':summary['cost'],
        'fallback_to_FLUX_triggered':False,'quality_limits':summary['quality_limits'],'local_tensor_bytes':audit['local_pt_bytes']},indent=2))


if __name__=='__main__': main()
