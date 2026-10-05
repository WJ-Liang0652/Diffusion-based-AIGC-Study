"""Bounded deterministic end-to-end SDXL validation; old guidance math unchanged."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from PIL import Image
import stage01_sdxl as s1
import stage02_relations as s2
import stage03b_diagnostic as b

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'04_relation_preservation/configs/stage03c_repair01.json'
PROTOCOL=ROOT/'04_relation_preservation/notes/stage03c_repair01_protocol.md'


def read(path): return json.loads(path.read_text())


def preflight(cfg):
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==cfg['base_commit']
    prior=ROOT/'04_relation_preservation/outputs/stage03b'
    oldhashes=read(ROOT/cfg['old_lock'])['fingerprints']
    for name,h in oldhashes.items(): assert s2.sha(ROOT/name)==h, name
    prior_lock=read(prior/'protocol_lock.json')
    for name,h in prior_lock['fingerprints'].items():
        if name!='AGENTS.md': assert s2.sha(ROOT/name)==h, name
    summary=read(prior/'stage03b_summary.json');assert summary['completed'] and summary['confirmation_passed']
    matrix=read(prior/'deterministic_matrix/worker_metrics.json')
    assert matrix['completed'] and matrix['matrix_summary']['A_repeat_exact'] and matrix['matrix_summary']['cross_condition_exact']['B']
    assert cfg['active_switches']==matrix['active_switches'] and cfg['model_weight_hashes']==matrix['weight_hash_before']
    comparisons=read(prior/'deterministic_matrix/comparisons.json')
    assert all(v['torch_equal'] and v['dtype_equal'] and v['max_abs_error']==0 for row in comparisons for v in row['fields'].values())
    raw_paths=[]
    for left,right in [('A1','A2'),('A1','A3'),('A1','B1'),('A2','B2'),('A3','B3')]:
        x,y=prior/'deterministic_matrix'/left/'trace.pt',prior/'deterministic_matrix'/right/'trace.pt'
        assert all(m['torch_equal'] for m in b.compare_traces(x,y).values())
        raw_paths += [x,y]
    for phase in ['guidance','step']:
        x,y=prior/'deterministic_confirm/A'/f'{phase}.pt',prior/'deterministic_confirm/D'/f'{phase}.pt'
        v,w=torch.load(x,map_location='cpu',weights_only=False),torch.load(y,map_location='cpu',weights_only=False)
        assert torch.equal(v['latent'],w['latent']) and s1.nested_equal(v['scheduler_state'],w['scheduler_state'])
        raw_paths += [x,y]
    cp=torch.load(ROOT/cfg['checkpoint'],map_location='cpu',weights_only=False)
    assert cp['fingerprints']==oldhashes and cp['next_index']==5 and cp['boundary']=='after scheduler.step'
    for k in ['prompt','seed','model_revision','height','width','steps','cfg_scale','target_word','target_box','loss_scale','loss_threshold','safety_max_relative_update']:
        assert cfg[k]==cp['effective_config'][k], k
    initial=cp['initial_latent']['__tensor__']
    assert s1.tensor_hash(initial)==cp['effective_config']['initial_latent_sha256']
    assert initial.dtype==torch.float16 and list(initial.shape)==[1,4,128,128]
    paths=[CONFIG,PROTOCOL,Path(__file__).resolve(),ROOT/'04_relation_preservation/configs/stage03c.json',
        ROOT/'04_relation_preservation/notes/stage03c_protocol.md',ROOT/'04_relation_preservation/src/stage03c_sdxl.py',
        ROOT/'04_relation_preservation/outputs/stage03c/protocol_lock.json',ROOT/'04_relation_preservation/outputs/stage03c/prepare_run_metrics.json']+[ROOT/p for p in prior_lock['fingerprints'] if p!='AGENTS.md']
    paths += [prior/'protocol_lock.json',prior/'stage03b_summary.json',prior/'deterministic_matrix/worker_metrics.json',
        prior/'deterministic_matrix/comparisons.json',prior/'deterministic_confirm/worker_metrics.json',prior/'deterministic_confirm/confirmation.json']+raw_paths
    hashes={str(p.relative_to(ROOT)):s2.sha(p) for p in paths}
    return hashes,{'passed':True,'03b_deterministic_AA_AB_verified_from_raw_tensors':True,
        '03b_guidance_step_latent_scheduler_verified_from_raw_tensors':True,'old_checkpoint_fingerprints_exact':True,
        'shared_initial_latent_sha256':s1.tensor_hash(initial),'initial_latent_shape':list(initial.shape),
        'negative_prompt_source':'old fingerprinted encode explicitly None; saved conditioning unchanged',
        'old_AGENTS_difference':'authorized stage03C update; historical locks unchanged'}


def canonical_scheduler_config(value):
    result=dict(value)
    if '_use_default_values' in result:
        result['_use_default_values']=sorted(result['_use_default_values'])
    return result


def configure(cfg):
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8', 'CUBLAS must be set before process start'
    expected=cfg['active_switches']
    torch.backends.cudnn.benchmark=expected['cudnn_benchmark']
    torch.backends.cudnn.deterministic=expected['cudnn_deterministic']
    torch.use_deterministic_algorithms(expected['deterministic_algorithms'],warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=expected['allow_tf32_matmul']
    torch.backends.cudnn.allow_tf32=expected['allow_tf32_cudnn']
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=expected['allow_fp16_reduced_precision_reduction']
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=expected['allow_bf16_reduced_precision_reduction']
    assert b.switches()==expected, 'Runtime flags differ from validated 03B flags'


class Budget:
    def __init__(self,cfg,metrics,previous):
        self.cfg,self.metrics,self.previous=cfg,metrics,previous
        self.start=time.perf_counter();self.inner_updates=0

    def check(self,new_branch=False):
        torch.cuda.synchronize()
        limit=self.cfg['gpu_no_new_branch_seconds'] if new_branch else self.cfg['gpu_work_limit_seconds']
        assert self.previous+time.perf_counter()-self.start<limit, f'GPU activity budget {limit}s reached'

    def timed(self,name,fn):
        self.check();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        try: return fn()
        finally:
            torch.cuda.synchronize()
            self.metrics['timings'][name]={'seconds':time.perf_counter()-start,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
            print(name,self.metrics['timings'][name],flush=True)


class PostBackwardSampler(b.DeferredSampler):
    """Validated D capture timing; original Sampler.guidance owns every update."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.first_arrays=None;self.final_arrays=None

    def flush(self):
        if self.pending is not None:
            arrays={k:v.cpu().numpy().copy() for k,v in self.pending.items()}
            if self.first_arrays is None: self.first_arrays=arrays
            self.pending=None

    def conditional(self,latent,timestep,scheduler):
        maps,ratios,energy=super().conditional(latent,timestep,scheduler)
        if self.context is not None and not latent.requires_grad:
            self.final_arrays={k:v.detach().cpu().numpy().copy() for k,v in maps.items()}
            self.pending=None
        return maps,ratios,energy

    def guidance(self,latent,scheduler,index,rows,summaries):
        self.context=(index,'apple') if index in self.args.heat_indices else None
        self.first_arrays=self.final_arrays=None
        original=torch.autograd.grad
        def grad_then_flush(*args,**kwargs):
            result=original(*args,**kwargs)
            self.flush()
            return result
        torch.autograd.grad=grad_then_flush
        try:
            result=s1.Sampler.guidance(self,latent,scheduler,index,rows,summaries)
            if self.context is not None:
                assert self.first_arrays is not None and self.final_arrays is not None
                for phase,arrays in [('before',self.first_arrays),('after',self.final_arrays)]:
                    self.stats(arrays,index,phase,'apple',0)
                    np.savez_compressed(self.folder/f'heat_{index:02d}_{phase}.npz',**arrays,mask=self.mask)
            return result
        finally:
            torch.autograd.grad=original
            self.pending=None;self.context=None
            self.first_arrays=self.final_arrays=None
            self.restore_processors()

    def step(self,latent,scheduler,index):
        self.budget.check()
        return s1.Sampler.step(self,latent,scheduler,index)


def initial_pipe(cfg,budget,metrics):
    pipe=budget.timed('model_load',lambda:s2.load_pipe(cfg))
    content=budget.timed('model_content_hash',lambda:b.weight_hash(pipe,budget))
    assert content==cfg['model_weight_hashes'], 'Model content differs from verified03B'
    metrics['model_content_hashes']=content
    old=torch.load(ROOT/cfg['checkpoint'],map_location='cpu',weights_only=False)
    env=s2.environment();expected_old={**old['environment'],'deterministic_algorithms':True}
    assert env==expected_old, 'Unexpected old-input environment difference'
    metrics['environment']=env;metrics['switches']=b.switches()
    diag,indices=s2.tokenizer_diagnostic(pipe,cfg,budget.metrics_path.parent/'tokenizer_diagnostic.json')
    assert indices==old['effective_config']['token_indices']==[3]
    return pipe,old,indices


def devices(pipe):
    for model in [pipe.unet,pipe.text_encoder,pipe.text_encoder_2]: model.to('cuda')
    pipe.vae.to(dtype=torch.float16)


def decode(pipe,z,path,budget,label):
    def work():
        s2.prepare_decode(pipe)
        return s2.decode(pipe,z,path)
    return budget.timed(label,work)


def safety(sampler,rows):
    assert sampler.processors_restored() and not sampler.parameters_have_grad()
    assert all(not p.requires_grad for model in sampler.models for p in model.parameters())
    assert all(r['finite'] and not r['model_parameter_grad_present'] and r['relative_update']<=.1 for r in rows)


def prepare(cfg,out,hashes,budget,metrics):
    assert not (out/'U').exists() and not (out/'R').exists(), 'Do not rerun a normal branch'
    (out/'U').mkdir();(out/'R').mkdir()
    pipe,old,indices=initial_pipe(cfg,budget,metrics)
    conditioning=s1.unpack(old['conditioning'])
    args=argparse.Namespace(**cfg,inner_iters=1)
    sampler=PostBackwardSampler(pipe,args,conditioning,indices,budget=budget,folder=out/'U')
    scheduler=sampler.new_scheduler();pipe.scheduler=scheduler
    actual_config=dict(scheduler.config)
    metrics['scheduler_config_check']={'actual_raw':actual_config,'old_raw':old['scheduler_config'],
        'effective_and_metadata_members_exact':canonical_scheduler_config(actual_config)==canonical_scheduler_config(old['scheduler_config'])}
    assert metrics['scheduler_config_check']['effective_and_metadata_members_exact']
    assert scheduler.timesteps.cpu().tolist()==old['effective_config']['timesteps']
    assert scheduler.sigmas.cpu().tolist()==old['effective_config']['sigmas']
    initial=s1.unpack(old['initial_latent'])
    assert s1.tensor_hash(initial)==old['effective_config']['initial_latent_sha256']
    generator=s1.restore_rng(old['rng'])
    metrics['initial_latent_sha256']=s1.tensor_hash(initial)
    metrics['conditioning_exact_to_old_saved']=s1.nested_equal(conditioning,s1.unpack(old['conditioning']))
    initial_cfg={**cfg,'layers':s1.LAYERS,'token_indices':indices,'scheduler_config':dict(scheduler.config),
        'timesteps':scheduler.timesteps.cpu().tolist(),'sigmas':scheduler.sigmas.cpu().tolist(),
        'initial_latent_sha256':metrics['initial_latent_sha256'],'lms_order':4,'conditioning_source':cfg['checkpoint']}
    s1.dump_json(out/'effective_config.json',initial_cfg)
    first=None;z=initial.clone()
    def U_run():
        nonlocal z,first
        budget.check(new_branch=True)
        for index in range(cfg['steps']):
            z=sampler.step(z,scheduler,index)
            metrics['last_completed']={'branch':'U','next_index':index+1}
            if index+1==5:
                cp={'format_version':1,'boundary':'after scheduler.step','next_index':5,
                    'latent':s1.pack(z),'initial_latent':s1.pack(initial),'scheduler_config':dict(scheduler.config),
                    'scheduler_state':s1.pack(s2.scheduler_state(scheduler)),'conditioning':s1.pack(conditioning),
                    'rng':s1.rng_state(generator),'processor_types':sampler.processors,'effective_config':initial_cfg,
                    'environment':s2.environment(),'switches':b.switches(),'fingerprints':hashes,'source_checkpoint_sha256':s2.sha(ROOT/cfg['checkpoint'])}
                assert len(scheduler.derivatives)==4 and scheduler.step_index==5
                torch.save(cp,out/'U/checkpoint.pt')
            if index==5: first=z.detach().cpu().clone()
        torch.save({'first':first,'final':z.detach().cpu().clone()},out/'U/reference_latents.pt')
        safety(sampler,[])
        return z
    baseline=budget.timed('U_51_steps',U_run)
    decode(pipe,baseline,out/'U/final.png',budget,'U_decode')
    budget.timed('U_restore_devices',lambda:devices(pipe))
    budget.check(new_branch=True)
    latent,scheduler2,cp,checks=s2.load_branch(out/'U/checkpoint.pt',sampler,hashes)
    assert cp['switches']==b.switches()==cfg['active_switches']
    assert scheduler2 is not scheduler and scheduler2.derivatives is not scheduler.derivatives
    assert all(x.data_ptr()!=y.data_ptr() for x in scheduler2.derivatives for y in scheduler.derivatives)
    reference=torch.load(out/'U/reference_latents.pt',map_location='cpu',weights_only=False)
    z=latent;first2=None
    def R_run():
        nonlocal z,first2
        for index in range(5,cfg['steps']):
            z=sampler.step(z,scheduler2,index)
            metrics['last_completed']={'branch':'R','next_index':index+1}
            if index==5: first2=z.detach().cpu().clone()
        return z
    restored=budget.timed('R_46_steps',R_run)
    gates={**checks,'first_step':s1.diff_metrics(reference['first'],first2),'final_latent':s1.diff_metrics(reference['final'],restored.cpu())}
    s1.dump_json(out/'restoration_metrics.json',gates)
    assert gates['first_step']['torch_equal'] and gates['final_latent']['torch_equal']
    decode(pipe,restored,out/'R/final.png',budget,'R_decode')
    x,y=np.asarray(Image.open(out/'U/final.png')),np.asarray(Image.open(out/'R/final.png'))
    gates['image']={'pixel_exact':bool(np.array_equal(x,y)),'max_abs_error':int(np.abs(x.astype(np.int16)-y.astype(np.int16)).max())}
    gates['passed']=gates['image']['pixel_exact']
    s1.dump_json(out/'restoration_metrics.json',gates);assert gates['passed']
    metrics['restoration']=gates
    torch.save({'final':restored.detach().cpu()},out/'R/latent.pt')
    safety(sampler,[])
    metrics['completed']=True


def guide(cfg,out,hashes,budget,metrics):
    assert read(out/'restoration_metrics.json')['passed']
    annotation=read(out/'U/annotation.json')
    assert annotation['baseline_qualified'] and annotation['image_sha256']==s2.sha(out/'U/final.png')
    assert not (out/'G5').exists() and not (out/'G20').exists(), 'No branch retry'
    pipe,old,indices=initial_pipe(cfg,budget,metrics)
    cp_path=out/'U/checkpoint.pt'
    previous=None
    for name,inner in cfg['branches'].items():
        budget.check(new_branch=True)
        folder=out/name;folder.mkdir()
        args=argparse.Namespace(**cfg,inner_iters=inner)
        sampler=PostBackwardSampler(pipe,args,{},indices,budget=budget,folder=folder)
        z,sch,cp,checks=s2.load_branch(cp_path,sampler,hashes)
        assert cp['switches']==b.switches()==cfg['active_switches']
        assert s1.nested_equal(z,s1.unpack(cp['latent']))
        if previous is not None:
            pz,ps=previous
            assert sch is not ps and sch.derivatives is not ps.derivatives and z.data_ptr()!=pz.data_ptr()
            assert all(x.data_ptr()!=y.data_ptr() for x in sch.derivatives for y in ps.derivatives)
        previous=(z,sch)
        rows,summaries=[],[]
        metrics[name]={'load_checks':checks,'independent_snapshot':True,'inner_cap':inner,'completed':False}
        def trajectory():
            nonlocal z
            for index in range(5,cfg['steps']):
                if 5<=index<10:
                    z=sampler.guidance(z,sch,index,rows,summaries)
                    print(name,summaries[-1],flush=True)
                z=sampler.step(z,sch,index)
                metrics['last_completed']={'branch':name,'next_index':index+1}
            return z
        try: final=budget.timed(name+'_46_steps_and_guidance',trajectory)
        finally:
            sampler.restore_processors();sampler.pending=None
            s1.csv_rows(folder/'inner_metrics.csv',rows);s1.csv_rows(folder/'step_metrics.csv',summaries)
            s1.csv_rows(folder/'heat_metrics.csv',sampler.heat_rows)
            metrics[name].update(inner_updates=len(rows),all_finite=all(r['finite'] for r in rows),
                parameter_grad_present=sampler.parameters_have_grad(),processors_restored=sampler.processors_restored())
        safety(sampler,rows)
        torch.save({'final':final.detach().cpu()},folder/'latent.pt')
        decode(pipe,final,folder/'final.png',budget,name+'_decode')
        metrics[name]['completed']=True
        s1.dump_json(folder/'effective_config.json',{**cp['effective_config'],'inner_iters':inner,'checkpoint_sha256':s2.sha(cp_path)})
        if name!='G20': budget.timed(name+'_restore_devices',lambda:devices(pipe))
    assert budget.inner_updates<=125
    metrics['completed']=True


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['preflight','prepare','guide'])
    a=p.parse_args();cfg=read(CONFIG);hashes,cpu=preflight(cfg)
    out=ROOT/cfg['output_dir'];out.mkdir(parents=True,exist_ok=True)
    s1.dump_json(out/'cpu_checks.json',cpu)
    if a.mode=='preflight': print(json.dumps(cpu,indent=2));return
    lock=out/'protocol_lock.json'
    if lock.exists(): assert read(lock)['fingerprints']==hashes, 'Stage03C execution identity changed'
    else:
        (out/'AGENTS_at_lock.md').write_bytes((ROOT/'AGENTS.md').read_bytes())
        s1.dump_json(lock,{'fingerprints':hashes,'agents_sha256_before_gpu':s2.sha(ROOT/'AGENTS.md'),
            'locked_at_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),'git_head':cfg['base_commit'],
            'git_status_before_gpu':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)})
    path=out/f'{a.mode}_run_metrics.json';assert not path.exists(), 'Attempt already recorded; no automatic retry'
    budget_root=ROOT/cfg['gpu_budget_root']
    previous=sum(read(q).get('gpu_activity_wall_seconds',0.) for q in budget_root.rglob('*_run_metrics.json'))
    assert previous<540
    configure(cfg)  # All formal GPU activity follows validated flags from its first operation.
    metrics={'mode':a.mode,'completed':False,'argv':sys.argv,'timings':{},'previous_gpu_seconds':previous,
        'config':cfg,'fingerprints':hashes,'switches_at_start':b.switches(),'repairs_used':cfg['repair_attempt']}
    budget=Budget(cfg,metrics,previous);budget.metrics_path=path
    s1.dump_json(path,metrics)
    try:
        if a.mode=='prepare': prepare(cfg,out,hashes,budget,metrics)
        else: guide(cfg,out,hashes,budget,metrics)
    except Exception:
        metrics['error']=traceback.format_exc();print(metrics['error'],flush=True)
        raise
    finally:
        torch.cuda.synchronize()
        metrics['gpu_activity_wall_seconds']=time.perf_counter()-budget.start
        metrics['inner_updates_total']=budget.inner_updates
        metrics['peak_allocated_mib']=max((x['peak_allocated_mib'] for x in metrics['timings'].values()),default=0.)
        metrics['peak_reserved_mib']=max((x['peak_reserved_mib'] for x in metrics['timings'].values()),default=0.)
        s1.dump_json(path,metrics)


if __name__=='__main__': main()
