"""Index5-only stage03B diagnostics; parent never initializes CUDA."""
import argparse
import contextlib
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
import stage01_sdxl as s1
import stage02_relations as s2
from stage03_inner_budget import DiagnosticSampler

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'04_relation_preservation/configs/stage03b.json'
PROTOCOL = ROOT/'04_relation_preservation/notes/stage03b_protocol.md'
FIELDS = ['map_mid','map_up','energy','loss','gradient','update','result']


def read(path):
    return json.loads(path.read_text())


def preflight(cfg):
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==cfg['base_commit']
    old = read(ROOT/cfg['old_lock'])['fingerprints']
    for p,sha in old.items(): assert s2.sha(ROOT/p)==sha, f'02B fingerprint mismatch: {p}'
    a = read(ROOT/'04_relation_preservation/outputs/stage03/protocol_lock.json')
    for p,sha in a['fingerprints'].items():
        if p!='AGENTS.md': assert s2.sha(ROOT/p)==sha, f'03A fingerprint mismatch: {p}'
    cp = torch.load(ROOT/cfg['checkpoint'],map_location='cpu',weights_only=False)
    assert cp['fingerprints']==old and cp['next_index']==5 and cp['boundary']=='after scheduler.step'
    ec = cp['effective_config']
    for k in ['model_revision','prompt','seed','height','width','steps','cfg_scale','target_word','target_box','loss_scale','loss_threshold','safety_max_relative_update']:
        assert ec[k]==cfg[k], k
    assert ec['lms_order']==4 and ec['dtype']=='float16' and len(cp['scheduler_state']['derivatives'])==4
    paths = [CONFIG,PROTOCOL,Path(__file__).resolve(),ROOT/'AGENTS.md']+[ROOT/p for p in a['fingerprints'] if p!='AGENTS.md']
    paths += [ROOT/'04_relation_preservation/outputs/stage03/protocol_lock.json', ROOT/'04_relation_preservation/notes/stage03_report.md',
              ROOT/'04_relation_preservation/outputs/stage03/run_metrics.json', ROOT/'04_relation_preservation/outputs/stage03/failure_diagnostic.json']
    hashes = {str(p.relative_to(ROOT)):s2.sha(p) for p in paths}
    assert cfg['conditions']==['A','B','C','D'] and cfg['repetitions']==3 and 2+2*12+10==cfg['maximum_backwards']==36
    return old,hashes,{'passed':True,'old_checkpoint_identity_exact':True,'old_LMS_history_count':4,
        'boundary':cp['boundary'],'next_index':5,'stage03a_runtime_sources_and_inputs_exact':True,
        'historical_AGENTS_hash':a['fingerprints']['AGENTS.md'],'current_AGENTS_hash':hashes['AGENTS.md'],
        'AGENTS_difference_reason':'user-authorized current-stage update; historical lock unchanged',
        'estimated_tensor_disk_upper_mib':100}


def switches():
    return {'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),
        'allow_tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'allow_tf32_cudnn':torch.backends.cudnn.allow_tf32,
        'allow_fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'flash_sdp_enabled':torch.backends.cuda.flash_sdp_enabled(),'mem_efficient_sdp_enabled':torch.backends.cuda.mem_efficient_sdp_enabled(),
        'math_sdp_enabled':torch.backends.cuda.math_sdp_enabled(),'grad_enabled':torch.is_grad_enabled(),
        'autocast_enabled':torch.is_autocast_enabled(),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def set_mode(mode,baseline):
    torch.backends.cudnn.benchmark = baseline['cudnn_benchmark']
    torch.backends.cudnn.deterministic = baseline['cudnn_deterministic'] if mode=='original' else True
    torch.use_deterministic_algorithms(baseline['deterministic_algorithms'] if mode=='original' else True,warn_only=False)


def model_signature(pipe):
    sig = {}
    for modelname in ['unet','text_encoder','text_encoder_2','vae']:
        model = getattr(pipe,modelname)
        for kind,items in [('parameter',model.named_parameters()),('buffer',model.named_buffers())]:
            for name,t in items:
                sig[f'{modelname}/{kind}/{name}'] = [str(t.dtype),str(t.device),list(t.shape),t._version,t.data_ptr(),t.requires_grad]
    return sig


def weight_hash(pipe,budget):
    result = {}
    for name in ['unet','text_encoder','text_encoder_2','vae']:
        h=hashlib.sha256()
        model=getattr(pipe,name)
        for j,(key,value) in enumerate(model.state_dict().items()):
            if j%20==0: budget.check()
            h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(list(value.shape)).encode())
            array=value.detach().cpu().contiguous().numpy()
            h.update(memoryview(array).cast('B'))
            del array
        result[name]=h.hexdigest()
    return result


def compare(x,y):
    assert isinstance(x,torch.Tensor) and isinstance(y,torch.Tensor)
    if x.shape!=y.shape: return {'torch_equal':False,'shape_mismatch':True}
    delta=(x.double()-y.double()).abs()
    return {'torch_equal':torch.equal(x,y),'max_abs_error':float(delta.max()),'mean_abs_error':float(delta.mean()),
        'different_element_ratio':float((x!=y).double().mean()),'dtype_equal':x.dtype==y.dtype,
        'finite':bool(torch.isfinite(x).all() and torch.isfinite(y).all())}


def compare_traces(left,right):
    x=torch.load(left,map_location='cpu',weights_only=False)
    y=torch.load(right,map_location='cpu',weights_only=False)
    return {k:compare(x[k],y[k]) for k in FIELDS}


class Budget:
    def __init__(self,cfg,out,previous_seconds,previous_backwards):
        self.cfg,self.out=cfg,out
        self.previous_seconds,self.previous_backwards=previous_seconds,previous_backwards
        self.start=time.perf_counter();self.backwards=0;self.inner_updates=0
        # Adapter for unchanged 03A DiagnosticSampler.
        self.cfg={**cfg,'maximum_inner_updates':cfg['maximum_backwards']}

    def check(self):
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-self.start
        assert self.previous_seconds+elapsed<self.cfg['gpu_work_limit_seconds'], '120s GPU work budget exhausted'

    def before_backward(self):
        self.check()
        assert self.previous_backwards+self.backwards<36, '36 backwards exhausted'
        self.backwards+=1
        self.persist()

    def persist(self):
        s1.dump_json(self.out/'budget_progress.json',{'backwards':self.backwards,'worker_seconds':time.perf_counter()-self.start,
            'previous_charged_seconds':self.previous_seconds,'previous_backwards':self.previous_backwards})


class DeferredSampler(DiagnosticSampler):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.pending=None

    def conditional(self,latent,timestep,scheduler):
        self.budget.check()
        if latent.requires_grad:
            assert self.budget.inner_updates<self.budget.cfg['maximum_inner_updates']
            self.budget.inner_updates+=1
        maps,ratios,energy=s1.Sampler.conditional(self,latent,timestep,scheduler)
        if self.context is not None:
            self.pending={k:v.detach() for k,v in maps.items()}
        return maps,ratios,energy

    def flush(self):
        if self.pending is not None:
            arrays={k:v.cpu().numpy().copy() for k,v in self.pending.items()}
            index,word=self.context
            self.stats(arrays,index,'post_backward',word,self.call_index)
            self.call_index+=1
            self.pending=None
            del arrays


def cpu_trace(maps,energy,loss,leaf,gradient,sigma,result=None):
    update=gradient.detach()*sigma.square()
    calculated=leaf.detach()-update
    record={f'map_{k}':v.detach().cpu().clone() for k,v in maps.items()}
    record.update(energy=energy.detach().cpu().clone(),loss=loss.detach().cpu().clone(),
        gradient=gradient.detach().cpu().clone(),update=update.cpu().clone(),
        input_latent=leaf.detach().cpu().clone(),result=(calculated if result is None else result).detach().cpu().clone())
    return record


def scalar_trace(trace,sampler,gpu_scalars=None):
    g,u,z=trace['gradient'],trace['update'],trace['input_latent']
    row={'energy':float(trace['energy']),'loss':float(trace['loss']),
        'gradient_norm':float(g.norm()),'gradient_norm_fp32':float(g.float().norm()),
        'update_norm':float(u.norm()),'latent_norm':float(z.norm()),'relative_update':float(u.norm()/(z.norm()+s1.EPS)),
        'all_finite':all(bool(torch.isfinite(v).all()) for v in trace.values()),
        'nonzero_gradient':bool(torch.count_nonzero(g)), 'parameter_grad_present':sampler.parameters_have_grad(),
        'processor_restored':sampler.processors_restored(),'switches':switches()}
    if gpu_scalars is not None: row.update(gpu_scalars)
    return row


def gpu_norms(leaf,gradient,update):
    return {'gradient_norm':gradient.norm().item(),'gradient_norm_fp32':gradient.float().norm().item(),
        'update_norm':update.norm().item(),'latent_norm':leaf.detach().norm().item(),
        'relative_update':(update.norm()/(leaf.detach().norm()+s1.EPS)).item(),'norm_backend':'original CUDA FP16 arithmetic (FP32 extra diagnostic)'}


def save_trace(folder,trace,sampler,gpu_scalars=None):
    folder.mkdir(parents=True,exist_ok=True)
    torch.save(trace,folder/'trace.pt')
    scalar=scalar_trace(trace,sampler,gpu_scalars)
    s1.dump_json(folder/'metrics.json',scalar)
    # Persist evidence before these safety/identity comparisons.
    assert scalar['all_finite'] and not scalar['parameter_grad_present'] and scalar['processor_restored'], scalar
    assert scalar['relative_update']<=sampler.args.safety_max_relative_update, scalar
    expected=trace['input_latent']-trace['update']
    assert torch.equal(trace['result'],expected), 'Single update formula differs from original FP16 formula'
    return scalar


def make_sampler(pipe,cfg,condition,budget,folder):
    args=argparse.Namespace(**cfg,inner_iters=1)
    if condition=='A': return s1.Sampler(pipe,args,{},[3])
    cls=DeferredSampler if condition=='D' else DiagnosticSampler
    obj=cls(pipe,args,{},[3],budget=budget,folder=folder)
    obj.context=(5,'apple') if condition in ('C','D') else None
    return obj


def load_case(sampler,cfg,oldhashes,mode,baseline,signature,content_hash,folder,budget):
    folder.mkdir(parents=True,exist_ok=False)
    # Retain original load_branch assertions. Deterministic settings apply only after old acceptance.
    set_mode('original',baseline)
    latent,scheduler,cp,checks=s2.load_branch(ROOT/cfg['checkpoint'],sampler,oldhashes)
    old_environment=s2.environment()
    set_mode(mode,baseline)
    actual_sig=model_signature(sampler.pipe)
    state=s1.pack(s2.scheduler_state(scheduler))
    identity={'checkpoint_sha256':s2.sha(ROOT/cfg['checkpoint']), 'next_index':cp['next_index'],
        'latent_sha256':s1.tensor_hash(latent),'old_environment_accepted':old_environment,
        'old_load_checks':checks,'active_switches':switches(), 'model_signature_sha256':hashlib.sha256(json.dumps(actual_sig,sort_keys=True).encode()).hexdigest(),
        'model_weight_content_hash_at_worker_start':content_hash,
        'model_signature_matches_frozen_worker_start':actual_sig==signature,
        'model_weight_check_method':'full content hash at start/end; every restore validates all parameter/buffer storage/version/dtype/device/shape and requires_grad',
        'conditioning_signature':{k:s1.tensor_hash(v) for k,v in sampler.conditioning.items() if isinstance(v,torch.Tensor)}}
    s1.dump_json(folder/'input_identity.json',identity)
    torch.save({'latent':latent.detach().cpu(),'conditioning':s1.pack(sampler.conditioning),
        'scheduler_state':state,'rng':cp['rng'],'model_signature':actual_sig},folder/'input_state.pt')
    assert all(checks.values()) and actual_sig==signature and cp['next_index']==5
    assert torch.equal(latent,s1.unpack(cp['latent'])) and not sampler.parameters_have_grad()
    assert all(not p.requires_grad for model in sampler.models for p in model.parameters())
    budget.check()
    return latent,scheduler


def explicit_single(sampler,latent,scheduler,folder,budget):
    leaf=latent.detach().requires_grad_(True)
    sigma=scheduler.sigmas[5].float()
    budget.check()
    maps,ratios,energy=sampler.conditional(leaf,scheduler.timesteps[5],scheduler)
    loss=energy*sampler.args.loss_scale
    budget.before_backward()
    gradient=torch.autograd.grad(loss,leaf)[0]
    if isinstance(sampler,DeferredSampler): sampler.flush()
    update=gradient.detach()*sigma.square()
    relative=update.norm()/(leaf.detach().norm()+s1.EPS)
    finite=bool(torch.isfinite(energy) and torch.isfinite(gradient).all() and torch.isfinite(update).all())
    safe=finite and relative.item()<=sampler.args.safety_max_relative_update and not sampler.parameters_have_grad()
    # Original calculation, including FP16 norm/update. Save before safety comparison.
    result=leaf.detach()-update
    trace=cpu_trace(maps,energy,loss,leaf,gradient,sigma,result)
    scalar=save_trace(folder,trace,sampler,gpu_norms(leaf,gradient,update))
    assert safe, scalar
    del leaf,maps,ratios,energy,loss,gradient,update,relative
    if isinstance(sampler,DiagnosticSampler):
        sampler.first_maps=sampler.last_maps=None
        s1.csv_rows(folder/'heat_stats.csv',sampler.heat_rows)
    return trace,result.detach()


@contextlib.contextmanager
def capture_original_grad(sampler,scheduler,folder,budget):
    """Observe only AFTER actual backward; original guidance body stays intact."""
    original=torch.autograd.grad
    count=0
    def wrapped(*args,**kwargs):
        nonlocal count
        budget.before_backward()
        gradient=original(*args,**kwargs)
        caller=sys._getframe(1).f_locals
        # No map CPU copy or scalar statistic before original backward.
        trace=cpu_trace(caller['maps'],caller['energy'],caller['loss'],caller['leaf'],gradient[0],scheduler.sigmas[5].float())
        update=gradient[0].detach()*scheduler.sigmas[5].float().square()
        save_trace(folder/f'inner{count}',trace,sampler,gpu_norms(caller['leaf'],gradient[0],update))
        count+=1
        return gradient
    torch.autograd.grad=wrapped
    try: yield
    finally: torch.autograd.grad=original


def matrix_comparisons(folder):
    entries=[]
    for condition in ['A','B','C','D']:
        for i,j in itertools.combinations(range(3),2): entries.append((f'{condition}{i+1}',f'{condition}{j+1}','repeat'))
    for i in range(1,4):
        for c in ['B','C','D']: entries.append((f'A{i}',f'{c}{i}','cross_condition'))
    results=[]
    for left,right,kind in entries:
        fields=compare_traces(folder/left/'trace.pt',folder/right/'trace.pt')
        results.append({'left':left,'right':right,'kind':kind,'all_fields_exact':all(v['torch_equal'] and v['dtype_equal'] for v in fields.values()),'fields':fields})
    s1.dump_json(folder/'comparisons.json',results)
    s1.csv_rows(folder/'comparisons.csv',[{'left':r['left'],'right':r['right'],'kind':r['kind'],'field':k,**v} for r in results for k,v in r['fields'].items()])
    repeated=all(r['all_fields_exact'] for r in results if r['kind']=='repeat' and r['left'].startswith('A'))
    ad=all(r['all_fields_exact'] for r in results if r['kind']=='cross_condition' and r['right'].startswith('D'))
    return {'any_difference':any(not r['all_fields_exact'] for r in results),'A_repeat_exact':repeated,'AD_exact':ad,
            'confirmation_eligible':repeated and ad,'condition_repeat_exact':{c:all(r['all_fields_exact'] for r in results if r['kind']=='repeat' and r['left'].startswith(c)) for c in ['A','B','C','D']},
            'cross_condition_exact':{c:all(r['all_fields_exact'] for r in results if r['kind']=='cross_condition' and r['right'].startswith(c)) for c in ['B','C','D']}}


def worker(cfg,args,oldhashes,hashes):
    out=ROOT/cfg['output_dir']/f'{args.setting}_{args.operation}'
    assert not out.exists(), 'Worker already attempted; never overwrite'
    out.mkdir()
    budget=Budget(cfg,out,args.previous_seconds,args.previous_backwards)
    result={'setting':args.setting,'operation':args.operation,'completed':False,'argv':sys.argv}
    sampler=None
    try:
        assert read(ROOT/cfg['output_dir']/'protocol_lock.json')['fingerprints']==hashes
        baseline=switches()
        result['baseline_switches']=baseline
        assert baseline['cudnn_benchmark']==False and baseline['deterministic_algorithms']==False
        assert baseline['allow_tf32_matmul']==False and baseline['cudnn_deterministic']==False
        if args.setting=='deterministic': assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        torch.cuda.reset_peak_memory_stats()
        pipe=s2.load_pipe(cfg)
        budget.check()
        result['environment_before_new_settings']=s2.environment()
        cp=torch.load(ROOT/cfg['checkpoint'],map_location='cpu',weights_only=False)
        assert result['environment_before_new_settings']==cp['environment']
        diag,indices=s2.tokenizer_diagnostic(pipe,cfg,out/'tokenizer_diagnostic.json');assert indices==[3]
        signature=model_signature(pipe)
        assert all(str(p.dtype)=='torch.float16' and str(p.device)=='cuda:0' for model in [pipe.unet,pipe.text_encoder,pipe.text_encoder_2,pipe.vae] for p in model.parameters())
        content=weight_hash(pipe,budget)
        result['weight_hash_before']=content
        result['model_dtype_device_counts']={model:{dtype_device:sum(1 for k,v in signature.items() if k.startswith(model+'/') and str(v[:2])==dtype_device)
            for dtype_device in sorted({str(v[:2]) for k,v in signature.items() if k.startswith(model+'/')})} for model in ['unet','text_encoder','text_encoder_2','vae']}
        s1.dump_json(out/'worker_metrics.json',result)
        if args.operation=='matrix':
            if args.setting=='original':
                # Maximum two backwards: original one-inner vs explicit one-inner.
                folder=out/'validation_original'
                sampler=make_sampler(pipe,cfg,'A',budget,folder)
                z,sch=load_case(sampler,cfg,oldhashes,args.setting,baseline,signature,content,folder,budget)
                ir,sr=[],[]
                with capture_original_grad(sampler,sch,folder,budget):
                    original_result=sampler.guidance(z,sch,5,ir,sr)
                original_trace=torch.load(folder/'inner0/trace.pt',map_location='cpu',weights_only=False)
                original_trace['result']=original_result.detach().cpu().clone()
                save_trace(folder,original_trace,sampler)
                s1.csv_rows(folder/'original_inner_metrics.csv',ir);s1.csv_rows(folder/'original_step_metrics.csv',sr)
                folder2=out/'validation_explicit'
                sampler=make_sampler(pipe,cfg,'A',budget,folder2)
                z,sch=load_case(sampler,cfg,oldhashes,args.setting,baseline,signature,content,folder2,budget)
                explicit_single(sampler,z,sch,folder2,budget)
                result['implementation_validation']=compare_traces(folder/'trace.pt',folder2/'trace.pt')
                result['each_run_own_gradient_update_formula_exact']=True
                s1.dump_json(out/'implementation_validation.json',result['implementation_validation'])
            for repeat in range(1,4):
                for c in cfg['conditions']:
                    folder=out/f'{c}{repeat}'
                    sampler=make_sampler(pipe,cfg,c,budget,folder)
                    z,sch=load_case(sampler,cfg,oldhashes,args.setting,baseline,signature,content,folder,budget)
                    explicit_single(sampler,z,sch,folder,budget)
                    print(args.setting,c,repeat,'evidence persisted',flush=True)
                    del z,sch,sampler
                    sampler=None
            result['matrix_summary']=matrix_comparisons(out)
        else:
            source=read(ROOT/cfg['output_dir']/f'{args.setting}_matrix/worker_metrics.json')
            assert source['matrix_summary']['confirmation_eligible']
            assert content==source['weight_hash_before'], 'Model content differs from selected matrix'
            for c in ['A','D']:
                folder=out/c
                sampler=make_sampler(pipe,cfg,c,budget,folder)
                sampler.args.inner_iters=5
                z,sch=load_case(sampler,cfg,oldhashes,args.setting,baseline,signature,content,folder,budget)
                ir,sr=[],[]
                if c=='A':
                    with capture_original_grad(sampler,sch,folder,budget):
                        guided=sampler.guidance(z,sch,5,ir,sr)
                else:
                    prior_energy,inner_index=float('inf'),0
                    guided=z
                    while prior_energy>sampler.args.loss_threshold and inner_index<5:
                        trace,guided=explicit_single(sampler,guided,sch,folder/f'inner{inner_index}',budget)
                        prior_energy=float(trace['energy'])
                        inner_index+=1
                    # Same final conditional forward as original guidance; no backward.
                    sampler.context=None
                    with torch.no_grad():
                        maps,ratios,energy=sampler.conditional(guided,sch.timesteps[5],sch)
                    s1.dump_json(folder/'final_energy.json',{'energy':energy.item(),'mid_ratio':ratios['mid'].item(),'up_ratio':ratios['up'].item()})
                    del maps,ratios,energy
                torch.save({'latent':guided.detach().cpu(),'scheduler_state':s1.pack(s2.scheduler_state(sch))},folder/'guidance.pt')
                if c=='A':
                    s1.csv_rows(folder/'original_inner_metrics.csv',ir);s1.csv_rows(folder/'original_step_metrics.csv',sr)
                budget.check()
                stepped=sampler.step(guided,sch,5)
                torch.save({'latent':stepped.detach().cpu(),'scheduler_state':s1.pack(s2.scheduler_state(sch))},folder/'step.pt')
                print(args.setting,'confirmation',c,'guidance and step persisted',flush=True)
            confirmation={}
            for phase in ['guidance','step']:
                x=torch.load(out/'A'/f'{phase}.pt',map_location='cpu',weights_only=False)
                y=torch.load(out/'D'/f'{phase}.pt',map_location='cpu',weights_only=False)
                confirmation[phase]={'latent':compare(x['latent'],y['latent']), 'scheduler_packed_exact':s1.nested_equal(x['scheduler_state'],y['scheduler_state'])}
            confirmation['per_inner']=[compare_traces(out/'A'/f'inner{i}/trace.pt',out/'D'/f'inner{i}/trace.pt') for i in range(5)]
            confirmation['passed']=all(confirmation[p]['latent']['torch_equal'] and confirmation[p]['scheduler_packed_exact'] for p in ['guidance','step'])
            s1.dump_json(out/'confirmation.json',confirmation)
            result['confirmation']=confirmation
            # All evidence already persisted; failure is reported without threshold relaxation.
            assert confirmation['passed'], '5-inner confirmation failed exact comparison'
        result['weight_hash_after']=weight_hash(pipe,budget)
        assert result['weight_hash_after']==content and model_signature(pipe)==signature, 'Model weights changed'
        result['frozen_weights_verified']=True
        result['active_switches']=switches()
        result['completed']=True
    except Exception:
        result['error']=traceback.format_exc()
        print(result['error'],flush=True)
    finally:
        if sampler is not None: sampler.restore_processors()
        budget.persist()
        result['backwards']=budget.backwards
        result['worker_gpu_activity_wall_seconds']=time.perf_counter()-budget.start
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            result['peak_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
            result['peak_reserved_mib']=torch.cuda.max_memory_reserved()/1024**2
        s1.dump_json(out/'worker_metrics.json',result)
    return result


def orchestrate(cfg,hashes,cpu):
    out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
    lock=out/'protocol_lock.json'
    assert not lock.exists() and not (out/'run_started.json').exists(), 'Diagnostic already attempted'
    s1.dump_json(out/'cpu_checks.json',cpu)
    s1.dump_json(lock,{'fingerprints':hashes,'git_head':cfg['base_commit'],
        'locked_at_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
        'git_status_before_gpu':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'backward_limit':36,'conservative_gpu_work_limit_seconds':120})
    s1.dump_json(out/'run_started.json',{'argv':sys.argv,'parent_cuda_initialized':torch.cuda.is_initialized()})
    assert not torch.cuda.is_initialized()
    total=0.;backwards=0;workers=[]
    summary={'task':'stage03b','status':'incomplete','completed':False,'config':cfg,'workers':workers,'commands':[]}
    def launch(setting,operation):
        nonlocal total,backwards
        remaining=120-total
        assert remaining>0
        env=os.environ.copy()
        if setting=='deterministic': env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--setting',setting,'--operation',operation,
            '--previous-seconds',str(total),'--previous-backwards',str(backwards)]
        summary['commands'].append({'argv':cmd,'CUBLAS_WORKSPACE_CONFIG':env.get('CUBLAS_WORKSPACE_CONFIG')})
        start=time.perf_counter()
        timed_out=False
        logfile=out/f'{setting}_{operation}.log'
        try:
            with logfile.open('w') as f:
                proc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
            returncode=proc.returncode
        except subprocess.TimeoutExpired:
            timed_out=True;returncode=-9
        total+=time.perf_counter()-start
        wd=out/f'{setting}_{operation}'
        metrics=read(wd/'worker_metrics.json') if (wd/'worker_metrics.json').exists() else {'completed':False}
        progress=read(wd/'budget_progress.json') if (wd/'budget_progress.json').exists() else {'backwards':0}
        backwards+=progress['backwards']
        entry={'setting':setting,'operation':operation,'returncode':returncode,'timed_out':timed_out,
            'subprocess_wall_charged_seconds':time.perf_counter()-start,'metrics':metrics}
        workers.append(entry)
        summary.update(gpu_work_conservative_seconds=total,total_backwards=backwards)
        s1.dump_json(out/'stage03b_summary.json',summary)
        print(json.dumps({'setting':setting,'operation':operation,'completed':metrics.get('completed'),
            'backwards':progress['backwards'],'charged_seconds':total,'matrix':metrics.get('matrix_summary'),
            'confirmation_passed':metrics.get('confirmation',{}).get('passed')},indent=2),flush=True)
        return metrics
    try:
        orig=launch('original','matrix')
        assert orig.get('completed'), 'Original worker failed; inspect its saved traceback'
        candidates=['original'] if orig['matrix_summary']['confirmation_eligible'] else []
        if orig['matrix_summary']['any_difference']:
            det=launch('deterministic','matrix')
            if det.get('completed') and det['matrix_summary']['confirmation_eligible']: candidates.append('deterministic')
            summary['deterministic_matrix_completed']=det.get('completed',False)
        if candidates:
            summary['selected_confirmation_setting']=candidates[0]
            confirmation=launch(candidates[0],'confirm')
            summary['confirmation_passed']=confirmation.get('confirmation',{}).get('passed',False)
        else:
            summary['selected_confirmation_setting']=None
            summary['confirmation_passed']=None
        assert backwards<=36 and total<=120
        summary['completed']=all(w['metrics'].get('completed',False) for w in workers)
        summary['status']='complete' if summary['completed'] else 'incomplete'
    except Exception:
        summary['orchestrator_error']=traceback.format_exc()
        print(summary['orchestrator_error'],flush=True)
    finally:
        summary.update(gpu_work_conservative_seconds=total,total_backwards=backwards,
            peak_allocated_mib=max((w['metrics'].get('peak_allocated_mib',0) for w in workers),default=0),
            peak_reserved_mib=max((w['metrics'].get('peak_reserved_mib',0) for w in workers),default=0),
            no_full_continuation=True,no_G20=True,all_tensor_evidence_local_pt_ignored=True)
        s1.dump_json(out/'stage03b_summary.json',summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['preflight','run','worker'])
    p.add_argument('--setting',choices=['original','deterministic'])
    p.add_argument('--operation',choices=['matrix','confirm'])
    p.add_argument('--previous-seconds',type=float,default=0.)
    p.add_argument('--previous-backwards',type=int,default=0)
    args=p.parse_args();cfg=read(CONFIG)
    oldhashes,hashes,cpu=preflight(cfg)
    if args.mode=='preflight': print(json.dumps(cpu,indent=2));return
    if args.mode=='run': orchestrate(cfg,hashes,cpu)
    else: worker(cfg,args,oldhashes,hashes)


if __name__=='__main__': main()
