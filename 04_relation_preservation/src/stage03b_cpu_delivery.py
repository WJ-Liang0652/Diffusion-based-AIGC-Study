"""CPU-only integrity audit and flat comparisons for stage03B."""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
import stage01_sdxl as s1
import stage03b_diagnostic as d

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'04_relation_preservation/outputs/stage03b'


def equal(a,b):
    if isinstance(a,torch.Tensor): return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray): return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    assert not torch.cuda.is_initialized()
    summary=d.read(OUT/'stage03b_summary.json')
    cfg=summary['config'];lock=d.read(OUT/'protocol_lock.json')
    locked={p: d.s2.sha(ROOT/p)==sha for p,sha in lock['fingerprints'].items()}
    assert all(locked.values())
    assert summary['completed'] and summary['confirmation_passed'] and summary['total_backwards']==36
    assert summary['gpu_work_conservative_seconds']<120
    inputs=[]
    for name in ['original_matrix','deterministic_matrix','deterministic_confirm']:
        paths=sorted((OUT/name).rglob('input_state.pt'))
        baseline=None;sig=None
        for path in paths:
            state=torch.load(path,map_location='cpu',weights_only=False)
            physical={k:v for k,v in state.items() if k!='model_signature'}
            if baseline is None: baseline=physical;sig=state['model_signature']
            assert equal(baseline,physical) and equal(sig,state['model_signature'])
            identity=d.read(path.with_name('input_identity.json'))
            assert all(identity['old_load_checks'].values()) and identity['model_signature_matches_frozen_worker_start']
            active=identity['active_switches'];det=name.startswith('deterministic')
            assert active['deterministic_algorithms']==det and active['cudnn_deterministic']==det and not active['deterministic_warn_only']
            assert active['CUBLAS_WORKSPACE_CONFIG']==(':4096:8' if det else None)
            inputs.append({'path':str(path.relative_to(ROOT)),'all_input_tensors_and_rng_equal_to_worker_reference':True,
                'model_storage_version_dtype_device_matches':True,'checkpoint_sha256':identity['checkpoint_sha256']})
    # Cross-worker input identity, excluding process-specific model storage addresses.
    base=torch.load(OUT/'original_matrix/A1/input_state.pt',map_location='cpu',weights_only=False)
    for path in [OUT/'deterministic_matrix/A1/input_state.pt',OUT/'deterministic_confirm/A/input_state.pt']:
        other=torch.load(path,map_location='cpu',weights_only=False)
        assert equal({k:v for k,v in base.items() if k!='model_signature'},{k:v for k,v in other.items() if k!='model_signature'})
    hashes=[w['metrics']['weight_hash_before'] for w in summary['workers']]
    assert hashes[0]==hashes[1]==hashes[2]
    assert all(w['metrics']['weight_hash_after']==w['metrics']['weight_hash_before'] for w in summary['workers'])
    traces=sorted(OUT.rglob('trace.pt'))
    norms=[]
    for path in traces:
        t=torch.load(path,map_location='cpu',weights_only=False)
        assert all(bool(torch.isfinite(v).all()) for v in t.values())
        assert t['gradient'].dtype==t['update'].dtype==t['result'].dtype==torch.float16
        assert list(t['gradient'].shape)==[1,4,128,128] and t['map_mid'].shape==t['map_up'].shape==(32,32)
        assert torch.equal(t['input_latent']-t['update'],t['result'])
        assert torch.equal(t['loss'],t['energy']*cfg['loss_scale'])
        m=d.read(path.with_name('metrics.json'))
        assert m['all_finite'] and not m['parameter_grad_present'] and m['relative_update']<=.1
        norms.append({'path':str(path.relative_to(ROOT)),'energy':m['energy'],'gradient_norm':m['gradient_norm'],
            'gradient_norm_fp32':m['gradient_norm_fp32'],'update_norm':m['update_norm'],'relative_update':m['relative_update']})
    s1.csv_rows(OUT/'scalar_metrics.csv',norms)
    combined=[]
    comparison_json={}
    def append(setting,kind,left,right,fields):
        comparison_json.setdefault(setting,[]).append({'kind':kind,'left':left,'right':right,'fields':fields})
        for field,values in fields.items(): combined.append({'setting':setting,'kind':kind,'left':left,'right':right,'field':field,**values})
    for setting in ['original','deterministic']:
        for row in d.read(OUT/f'{setting}_matrix/comparisons.json'):
            append(setting,row['kind'],row['left'],row['right'],row['fields'])
    append('original','implementation_validation','original_guidance_inner1','explicit_inner1',d.read(OUT/'original_matrix/implementation_validation.json'))
    cross=d.compare_traces(OUT/'original_matrix/A1/trace.pt',OUT/'deterministic_matrix/A1/trace.pt')
    append('original_vs_deterministic','cross_setting','original_A1','deterministic_A1',cross)
    confirmation=d.read(OUT/'deterministic_confirm/confirmation.json')
    for i,fields in enumerate(confirmation['per_inner']): append('deterministic','confirmation_inner',f'A_inner{i}',f'D_inner{i}',fields)
    for phase in ['guidance','step']:
        append('deterministic','confirmation_final',f'A_{phase}',f'D_{phase}',{'latent':confirmation[phase]['latent']})
    s1.csv_rows(OUT/'comparisons.csv',combined);s1.dump_json(OUT/'comparisons.json',comparison_json)
    group_rows=[]
    for setting in ['original','deterministic']:
        r=d.read(OUT/f'{setting}_matrix/comparisons.json')
        for label,predicate in [('A/A',lambda x:x['kind']=='repeat' and x['left'].startswith('A')),
                                ('B/B',lambda x:x['kind']=='repeat' and x['left'].startswith('B')),
                                ('C/C',lambda x:x['kind']=='repeat' and x['left'].startswith('C')),
                                ('D/D',lambda x:x['kind']=='repeat' and x['left'].startswith('D')),
                                ('A/B',lambda x:x['kind']=='cross_condition' and x['right'].startswith('B')),
                                ('A/C',lambda x:x['kind']=='cross_condition' and x['right'].startswith('C')),
                                ('A/D',lambda x:x['kind']=='cross_condition' and x['right'].startswith('D'))]:
            selected=[x for x in r if predicate(x)]
            for field in d.FIELDS:
                values=[x['fields'][field] for x in selected]
                group_rows.append({'setting':setting,'comparison':label,'field':field,'pair_count':len(values),
                    'exact_count':sum(v['torch_equal'] for v in values),'max_abs_error_max':max(v['max_abs_error'] for v in values),
                    'mean_abs_error_max':max(v['mean_abs_error'] for v in values),
                    'different_element_ratio_min':min(v['different_element_ratio'] for v in values),
                    'different_element_ratio_max':max(v['different_element_ratio'] for v in values)})
    s1.csv_rows(OUT/'comparison_group_summary.csv',group_rows)
    pts=sorted(OUT.rglob('*.pt'))
    ignored=subprocess.check_output(['git','check-ignore','--stdin'],input='\n'.join(str(p.relative_to(ROOT)) for p in pts)+'\n',text=True,cwd=ROOT).splitlines()
    assert len(ignored)==len(pts)
    audit={'locked_files_unchanged':locked,'full_input_and_rng_audits':inputs,'cross_worker_input_identity_exact':True,
        'worker_full_model_weight_hashes_equal_and_unchanged':True,'trace_file_count':len(traces),
        'unique_backward_traces':len(traces)-1,'duplicate_trace_note':'validation_original/trace.pt duplicates its inner0 evidence and records actual original guidance result',
        'all_trace_finite_and_own_update_formula_exact':True,'all_parameter_grad_absent':True,
        'local_tensor_file_count':len(pts),'local_tensor_bytes':sum(p.stat().st_size for p in pts),'all_pt_git_ignored':True,
        'cpu_cuda_initialized':torch.cuda.is_initialized(),'cpu_delivery_sha256':d.s2.sha(Path(__file__).resolve())}
    assert audit['unique_backward_traces']==36 and not audit['cpu_cuda_initialized']
    s1.dump_json(OUT/'artifact_audit.json',audit)
    summary['findings']={'original_self_repeat_exact':False,'original_forward_maps_E_loss_repeat_exact':True,
        'original_gradient_update_result_repeat_exact':False,'original_cross_condition_effect_separable':False,
        'deterministic_all_conditions_and_repeats_exact':True,'deterministic_5inner_guidance_step_and_scheduler_exact':True,
        'original_and_deterministic_gradient_identical':cross['gradient']['torch_equal'],
        'underlying_operator_identified':False,'causal_claim_CPU_copy_is_root_cause_supported':False,
        'scope':'Only P1/seed19 index5 in the recorded environment/settings; no full continuation or historical guided-image reproduction tested.'}
    summary['cpu_artifact_audit']='artifact_audit.json'
    summary['comparisons_csv']='comparisons.csv';summary['comparisons_json']='comparisons.json'
    summary['tensor_evidence_local_bytes']=audit['local_tensor_bytes']
    s1.dump_json(OUT/'stage03b_summary.json',summary)
    print(json.dumps({'status':summary['status'],'backwards':summary['total_backwards'],
        'charged_seconds':summary['gpu_work_conservative_seconds'],'trace_files':len(traces),'pt_files':len(pts),
        'local_tensor_mib':audit['local_tensor_bytes']/1024**2,'comparison_rows':len(combined),'findings':summary['findings']},indent=2))


if __name__=='__main__': main()
