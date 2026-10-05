"""CPU-only audit and render of an incomplete stage03 attempt; never samples."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import stage02_relations as s2
from stage01_sdxl import dump_json, csv_rows

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'04_relation_preservation/outputs/stage03'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    cfg = read(ROOT/'04_relation_preservation/configs/stage03.json')
    run = read(OUT/'run_metrics.json')
    lock = read(OUT/'protocol_lock.json')
    assert all(s2.sha(ROOT/name)==digest for name,digest in lock['fingerprints'].items())
    assert run['status']=='incomplete' and not run['completed']
    assert run['gates']['U_image']['pixel_exact']
    assert not run['gates']['instrumentation_guidance']['torch_equal']
    assert not (OUT/'G20').exists() and not (OUT/'G5/final.png').exists()
    olddir = ROOT/'04_relation_preservation/outputs/stage02b/P1'
    old = rows(olddir/'intervention/guided_inner_iteration_metrics.csv')[:5]
    new = rows(OUT/'G5/inner_metrics.csv')
    scalar_differences = []
    for left,right in zip(old,new):
        different = {key:{'stage02b':left[key],'stage03':right[key]} for key in left if left[key]!=right[key]}
        scalar_differences.append({'inner_index':int(right['inner_index']),'different_fields':different})
    before_exact = {}
    for layer in ['mid','up']:
        x = np.load(OUT/'U/heat_05_apple_before.npz')[layer]
        y = np.load(OUT/'G5/heat_05_apple_before.npz')[layer]
        before_exact[layer] = {'array_exact':bool(np.array_equal(x,y)), 'max_abs_error':float(np.abs(x-y).max())}
        assert before_exact[layer]['array_exact']
    raw_audit = []
    for path in sorted(OUT.glob('*/heat_*.npz')):
        with np.load(path) as z:
            assert set(z.files)=={'mid','up','mask'}
            assert all(z[k].shape==(32,32) and np.isfinite(z[k]).all() for k in z.files)
            assert z['mask'].sum()==288 and z['mask'].mean()==.28125
            raw_audit.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'finite':True,
                             'mask_area_ratio':.28125,'grid':[32,32]})
    diagnostic = {'locked_sources_and_inputs_unchanged':True,'no_G20_started':True,'G5_final_absent':True,
        'U_vs_G5_initial_heatmaps':before_exact,'G5_vs_stage02b_scalar_differences':scalar_differences,
        'first_observed_historical_divergence':'inner0 gradient_norm_fp32; pre-update E/mid/up ratios exactly match',
        'first_gradient_norm_fp32_delta':float(new[0]['gradient_norm_fp32'])-float(old[0]['gradient_norm_fp32']),
        'probe_vs_instrumented_guidance':run['gates']['instrumentation_guidance'],
        'numerical_root_cause':'unresolved; no additional GPU attempt permitted by the failed gate',
        'original_probe_scalar_arrays_persisted':False,'raw_array_audit':raw_audit,
        'cpu_delivery_source_sha256':s2.sha(Path(__file__).resolve())}
    dump_json(OUT/'failure_diagnostic.json',diagnostic)

    # Pixel equality permits reuse of reviewed old boxes, visually rechecked on U.
    annotation = copy.deepcopy(read(olddir/'seed19/baseline_annotation_input.json'))
    annotation.update(branch='U',image_sha256=s2.sha(OUT/'U/final.png'),
        method='manual visual recheck of U, with reviewed 02B bbox reused after exact pixel equality',
        extra_objects=[],review_status='ready_for_user_review',
        notes='Visible apple stem and cup handle included; shadows excluded. Identical pixels to reviewed 02B baseline.')
    dump_json(OUT/'U/annotation_input.json',annotation)
    result = s2.evaluate_annotation(annotation,cfg)
    dump_json(OUT/'U/annotation.json',result)
    boxed = Image.open(OUT/'U/final.png').convert('RGB')
    draw = ImageDraw.Draw(boxed)
    for key,color in [('A','red'),('B','cyan')]:
        box = result['objects'][key]['bbox_pixels']
        draw.rectangle(box,outline=color,width=4)
        draw.text((box[0],box[1]-16),key,fill=color)
    boxed.save(OUT/'U/final_annotated.png')
    canvas = Image.new('RGB',(1536,544),'white')
    draw = ImageDraw.Draw(canvas)
    canvas.paste(boxed.resize((512,512)),(0,32))
    draw.text((8,8),'U: dx=.377930, dy=-.033203; old PASS, new FAIL',fill='black')
    draw.text((520,8),'G5: incomplete (failed index5 equality gate)',fill='black')
    draw.text((1032,8),'G20: not run (blocked by G5 gate)',fill='black')
    draw.text((570,220),'No final image / bbox result',fill='black')
    draw.text((1082,220),'No final image / bbox result',fill='black')
    canvas.save(OUT/'branch_status_bbox_comparison.png')

    relation_summary = {'U':result,'G5':{'status':'incomplete','evaluated':False,'objects_alive':None,'dx':None,'dy':None,
        'old_relation':'not_evaluated','new_relation':'not_evaluated','joint_success':None,'reason':'guidance equality failed before index5 scheduler.step'},
        'G20':{'status':'not_run','evaluated':False,'objects_alive':None,'dx':None,'dy':None,
        'old_relation':'not_evaluated','new_relation':'not_evaluated','joint_success':None,'reason':'blocked by G5 gate'}}
    dump_json(OUT/'relation_results.json',relation_summary)
    csv_rows(OUT/'relations.csv',[{'branch':name,'status':'complete' if name=='U' else r['status'],
        'dx':r['dx'],'dy':r['dy'],'objects_alive':r['objects_alive'],'old_relation':r['old_relation'],
        'new_relation':r['new_relation'],'joint_success':r['joint_success']} for name,r in relation_summary.items()])
    heat_rows = []
    for name in ['U','G5']:
        heat_rows += [{'branch':name,**r} for r in rows(OUT/name/'heat_metrics.csv')]
    csv_rows(OUT/'heat_metrics.csv',heat_rows)
    inner_rows = [{'branch':'G5',**r} for r in new]
    csv_rows(OUT/'inner_metrics.csv',inner_rows)

    def panel(filename, groups, titles, note):
        # Each column group shares its per-layer vmin/vmax across all cells.
        fig,axes=plt.subplots(2,len(groups),figsize=(4.1*len(groups),7),squeeze=False)
        color_metadata = []
        for row,layer in enumerate(['mid','up']):
            vmax = max(float(np.load(p)[layer].max()) for p in groups)
            for col,path in enumerate(groups):
                z=np.load(path)
                ax=axes[row,col]
                shown=ax.imshow(z[layer],vmin=0,vmax=vmax,cmap='magma',extent=(0,1,1,0))
                ax.contour(np.linspace(.5/32,1-.5/32,32),np.linspace(.5/32,1-.5/32,32),z['mask'],levels=[.5],colors=['cyan'],linewidths=.8)
                ax.set_title(f'{titles[col]} / {layer}')
                ax.set_xlabel('normalized x');ax.set_ylabel('normalized y')
            fig.colorbar(shown,ax=list(axes[row]),fraction=.02,pad=.02)
            color_metadata.append({'layer':layer,'vmin':0.,'vmax':vmax,'raw_unit':'mean attention probability'})
        fig.suptitle(note)
        fig.subplots_adjust(top=.88,wspace=.32,hspace=.35,right=.88)
        fig.savefig(OUT/filename,dpi=130);plt.close(fig)
        return {'file':filename,'scales':color_metadata,'inputs':[str(p.relative_to(ROOT)) for p in groups]}
    panels=[]
    panels.append(panel('heat_index5.png',[OUT/'U/heat_05_apple_before.npz',OUT/'G5/heat_05_apple_before.npz',OUT/'G5/heat_05_apple_after.npz'],
        ['U before','G5 before','G5 after (FAILED GATE)'],'Index5 apple / matched scales; G5 is invalid for branch outcome comparison'))
    panels.append(panel('heat_U_window.png',[OUT/f'U/heat_{i:02d}_apple_before.npz' for i in range(5,10)],
        [f'U index{i}' for i in range(5,10)],'U apple window / same per-layer scale across indices (diagnostic only)'))
    panels.append(panel('heat_U_late.png',[OUT/'U/heat_50_apple_late.npz',OUT/'U/heat_50_cup_late.npz'],
        ['U index50 apple','U index50 cup (read-only)'],'Last step BEFORE update / cup has no guidance loss; cyan mask is apple target'))
    dump_json(OUT/'heat_panel_scales.json',panels)
    summary={'task':'stage03','status':'incomplete','experiment_completed':False,
        'answer_G20_repaired_new_relation':None,'answer':'Cannot assess: G5 instrumentation exact-equality gate failed; G20 not started.',
        'interpretation':'invalid_comparability_not_a_G20_negative_result',
        'base_commit':cfg['base_commit'],'config':cfg,'gates':run['gates'],
        'branches':{'U':{'status':'complete','relation_result':result},'G5':relation_summary['G5'],'G20':relation_summary['G20']},
        'cost':{'full_continuations_completed':1,'inner_updates_including_probe':run['total_inner_updates_including_probe'],
            'gpu_work_seconds':run['gpu_work_seconds'],'gpu_budget_charge_conservative_seconds':run['total_process_seconds'],
            'total_run_process_seconds':run['total_process_seconds'],
            'peak_allocated_mib':run['peak_allocated_mib'],'peak_reserved_mib':run['peak_reserved_mib']},
        'diagnostic':diagnostic,'artifacts':{'run_metrics':'run_metrics.json','failure_diagnostic':'failure_diagnostic.json',
            'protocol_lock':'protocol_lock.json','bbox_comparison':'branch_status_bbox_comparison.png',
            'heat_panels':['heat_index5.png','heat_U_window.png','heat_U_late.png']},
        'stopped_after_failed_gate':True,'no_threshold_relaxation':True}
    dump_json(OUT/'stage03_summary.json',summary)
    print(json.dumps({'status':summary['status'],'cost':summary['cost'],'U_dx':result['dx'],'U_dy':result['dy'],
        'initial_maps_exact':before_exact,'first_gradient_norm_fp32_delta':diagnostic['first_gradient_norm_fp32_delta']},indent=2))


if __name__=='__main__':
    main()
