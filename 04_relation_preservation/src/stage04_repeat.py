"""Fixed independent-start SDXL G20 validation. Historical samplers are imported unchanged."""
import argparse,json,sys,time,subprocess,traceback
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import stage01_sdxl as s1
import stage02_relations as s2
import stage03c_sdxl_repair01 as c

ROOT=c.ROOT
CONFIG=ROOT/'04_relation_preservation/configs/stage04.json'
PROTOCOL=ROOT/'04_relation_preservation/notes/stage04_protocol.md'
read=c.read


def fingerprints(cfg):
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==cfg['base_commit']
    prior=ROOT/'04_relation_preservation/outputs/stage03c/repair01'
    oldlock=read(prior/'protocol_lock.json')
    for n,h in oldlock['fingerprints'].items(): assert s2.sha(ROOT/n)==h,n
    historical=read(ROOT/'04_relation_preservation/configs/stage03c_repair01.json')
    for k in ['model','model_revision','height','width','dtype','scheduler','steps','cfg_scale','batch_size','cfg_forward_batch_size','target_word','target_box','checkpoint_after','guided_index_start','guided_index_stop_exclusive','loss_scale','loss_threshold','safety_max_relative_update','relation_thresholds','baseline_requires','prompt','negative_prompt','active_switches','model_weight_hashes','heat_indices']:
        assert cfg[k]==historical[k],k
    assert cfg['seed_candidates']==[101,314,777,1001,2048,4096] and cfg['inner_iters']==20
    assert cfg['maximum_inner_updates']==300 and cfg['maximum_full_continuations']==12
    assert cfg['target_qualified']==3 and cfg['maximum_baselines']==6
    summary=read(ROOT/'04_relation_preservation/outputs/stage03c/stage03c_summary.json')
    assert summary['completed'] and summary['G20_new_relation_repaired'] and summary['restoration']['passed']
    cp=torch.load(ROOT/cfg['checkpoint'],map_location='cpu',weights_only=False)
    assert cp['fingerprints']==oldlock['fingerprints'] and cp['switches']==cfg['active_switches']
    assert cp['next_index']==5 and cp['boundary']=='after scheduler.step'
    assert len(cp['scheduler_state']['derivatives'])==4
    paths=[CONFIG,PROTOCOL,Path(__file__).resolve(),Path(c.__file__).resolve(),prior/'protocol_lock.json',ROOT/cfg['checkpoint'],ROOT/'04_relation_preservation/notes/stage03c_report.md',ROOT/'04_relation_preservation/outputs/stage03c/stage03c_summary.json']+[ROOT/n for n in oldlock['fingerprints']]
    hashes={str(p.relative_to(ROOT)):s2.sha(p) for p in paths}
    return hashes,{'passed':True,'frozen_numeric_config_exact_to_03C_G20':True,'historical_lock_unchanged':True,'no_GPU_initialized':not torch.cuda.is_initialized()}


def records(out):return read(out/'screening.json') if (out/'screening.json').exists() else []


def next_seed(cfg,out):
    rows=records(out)
    assert [r['seed'] for r in rows]==cfg['seed_candidates'][:len(rows)]
    if sum(r['qualified'] for r in rows)>=3 or len(rows)==6:return None
    return cfg['seed_candidates'][len(rows)]


def freeze(cfg,out,hashes):
    rows=records(out);assert next_seed(cfg,out) is None,'Screening must finish before freeze'
    assert not list(out.glob('seed*/G20')) and not (out/'selection_lock.json').exists()
    assert rows and all(read(out/f'seed{r["seed"]}/U/annotation.json')['baseline_qualified']==r['qualified'] for r in rows)
    paths=[out/'screening.json']
    for r in rows:
        folder=out/f'seed{r["seed"]}/U'
        paths += [folder/n for n in ['final.png','annotation_input.json','annotation.json','effective_config.json','checkpoint.pt','reference_latents.pt']]
    data={'selected_seeds':[r['seed'] for r in rows if r['qualified']],'screening_count':len(rows),'qualified_count':sum(r['qualified'] for r in rows),'frozen_before_any_guided_image':True,'time_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),'protocol_lock_sha256':s2.sha(out/'protocol_lock.json'),'fingerprints':{str(p.relative_to(ROOT)):s2.sha(p) for p in paths}}
    s1.dump_json(out/'selection_lock.json',data);print(json.dumps(data,indent=2))


def selection(out):
    d=read(out/'selection_lock.json');assert d['frozen_before_any_guided_image']
    assert d['protocol_lock_sha256']==s2.sha(out/'protocol_lock.json')
    for p,h in d['fingerprints'].items():assert s2.sha(ROOT/p)==h,p
    return d


def annotate(cfg,out,seed,kind,path):
    folder=out/f'seed{seed}'/kind;assert not (folder/'annotation.json').exists()
    if kind=='U':assert next_seed(cfg,out)==seed and not (out/'selection_lock.json').exists()
    else:assert seed in selection(out)['selected_seeds'] and read(out/f'seed{seed}/intervene_run_metrics.json')['completed']
    data=read(path);assert data['image_sha256']==s2.sha(folder/'final.png')
    assert data['red_apple_count'] is not None and 'major_blue_cup_count' in data and 'all_cup_count' in data
    result=s2.evaluate_annotation(data,cfg);s1.dump_json(folder/'annotation_input.json',data);s1.dump_json(folder/'annotation.json',result)
    if kind=='U':
        rows=records(out);rows.append({'seed':seed,'candidate_index':len(rows),'qualified':result['baseline_qualified'],'dx':result['dx'],'dy':result['dy'],'uncertain':result['uncertain'],'reason':data['screening_reason'],'annotation':str((folder/'annotation.json').relative_to(ROOT)),'image_sha256':data['image_sha256'],'initial_latent_sha256':read(folder/'effective_config.json')['initial_latent_sha256']})
        s1.dump_json(out/'screening.json',rows)
    print(json.dumps(result,indent=2,ensure_ascii=False))


class Budget(c.Budget):
    def __init__(self,cfg,metrics,previous,previous_inner):
        super().__init__(cfg,metrics,previous);self.inner_updates=previous_inner;self.prior_inner=previous_inner


def baseline(cfg,out,seed,hashes,budget,metrics):
    assert next_seed(cfg,out)==seed and not (out/'selection_lock.json').exists()
    folder=out/f'seed{seed}'/'U';assert not folder.exists();folder.mkdir(parents=True)
    budget.check(new_branch=True)
    pipe,old,indices=c.initial_pipe(cfg,budget,metrics)
    condition=s1.unpack(old['conditioning']);assert old['effective_config']['prompt']==cfg['prompt']
    args=argparse.Namespace(**cfg);sampler=c.PostBackwardSampler(pipe,args,condition,indices,budget=budget,folder=folder)
    sch=sampler.new_scheduler();pipe.scheduler=sch
    assert c.canonical_scheduler_config(sch.config)==c.canonical_scheduler_config(old['scheduler_config'])
    assert sch.timesteps.cpu().tolist()==old['effective_config']['timesteps'] and sch.sigmas.cpu().tolist()==old['effective_config']['sigmas']
    generator=torch.Generator(device='cuda').manual_seed(seed)
    with torch.no_grad():initial=pipe.prepare_latents(1,pipe.unet.config.in_channels,cfg['height'],cfg['width'],condition['positive'].dtype,torch.device('cuda'),generator)
    assert initial.dtype==torch.float16 and list(initial.shape)==[1,4,128,128] and torch.isfinite(initial).all()
    ec={**cfg,'seed':seed,'layers':s1.LAYERS,'token_indices':indices,'scheduler_config':dict(sch.config),'timesteps':sch.timesteps.cpu().tolist(),'sigmas':sch.sigmas.cpu().tolist(),'initial_latent_sha256':s1.tensor_hash(initial),'boundary':'after scheduler.step'}
    s1.dump_json(folder/'effective_config.json',ec);metrics['initial_latent_sha256']=ec['initial_latent_sha256']
    z=initial.clone();first=None
    def trajectory():
        nonlocal z,first
        budget.check(new_branch=True);metrics['trajectories_started']=1
        for index in range(cfg['steps']):
            z=sampler.step(z,sch,index);metrics['last_completed']={'branch':'U','next_index':index+1}
            if index==4:
                assert len(sch.derivatives)==4 and sch.step_index==5
                cp={'format_version':1,'boundary':'after scheduler.step','next_index':5,'latent':s1.pack(z),'initial_latent':s1.pack(initial),'scheduler_config':dict(sch.config),'scheduler_state':s1.pack(s2.scheduler_state(sch)),'conditioning':s1.pack(condition),'rng':s1.rng_state(generator),'processor_types':sampler.processors,'effective_config':ec,'environment':s2.environment(),'switches':c.b.switches(),'fingerprints':hashes}
                torch.save(cp,folder/'checkpoint.pt')
            if index==5:first=z.detach().cpu().clone()
        torch.save({'first':first,'final':z.detach().cpu().clone()},folder/'reference_latents.pt');c.safety(sampler,[])
        return z
    final=budget.timed('U_51_steps',trajectory);c.decode(pipe,final,folder/'final.png',budget,'U_decode')
    metrics['completed']=True;metrics['trajectories_completed']=1


def intervene(cfg,out,seed,hashes,budget,metrics):
    selected=selection(out)['selected_seeds'];assert seed in selected
    done=[s for s in selected if (out/f'seed{s}/intervene_run_metrics.json').exists() and read(out/f'seed{s}/intervene_run_metrics.json')['completed']]
    assert done==selected[:len(done)] and seed==selected[len(done)]
    case=out/f'seed{seed}';assert not (case/'R').exists() and not (case/'G20').exists()
    (case/'R').mkdir();(case/'G20').mkdir()
    pipe,old,indices=c.initial_pipe(cfg,budget,metrics)
    sampler=c.PostBackwardSampler(pipe,argparse.Namespace(**cfg),{},indices,budget=budget,folder=case/'G20')
    cp_path=case/'U/checkpoint.pt';z,sch,cp,checks=s2.load_branch(cp_path,sampler,hashes)
    assert cp['switches']==cfg['active_switches'] and cp['effective_config']['seed']==seed
    assert s1.tensor_hash(cp['initial_latent']['__tensor__'])==cp['effective_config']['initial_latent_sha256']
    reference=torch.load(case/'U/reference_latents.pt',map_location='cpu',weights_only=False)
    first=None
    def R():
        nonlocal z,first
        budget.check(new_branch=True);metrics['trajectories_started']=1
        for index in range(5,cfg['steps']):
            z=sampler.step(z,sch,index);metrics['last_completed']={'branch':'R','next_index':index+1}
            if index==5:first=z.detach().cpu().clone()
        return z
    final=budget.timed('R_46_steps',R)
    gates={**checks,'first_step':s1.diff_metrics(reference['first'],first),'final_latent':s1.diff_metrics(reference['final'],final.cpu())}
    torch.save({'first':first,'final':final.detach().cpu()},case/'R/latent.pt');s1.dump_json(case/'restoration_metrics.json',gates)
    assert gates['first_step']['torch_equal'] and gates['final_latent']['torch_equal']
    c.decode(pipe,final,case/'R/final.png',budget,'R_decode');c.safety(sampler,[])
    x,y=np.asarray(Image.open(case/'U/final.png')),np.asarray(Image.open(case/'R/final.png'))
    gates['image']={'pixel_exact':bool(np.array_equal(x,y)),'max_abs_error':int(np.abs(x.astype(np.int16)-y.astype(np.int16)).max())};gates['passed']=gates['image']['pixel_exact']
    s1.dump_json(case/'restoration_metrics.json',gates);assert gates['passed'];metrics['restoration']=gates;metrics['trajectories_completed']=1
    c.devices(pipe);budget.check(new_branch=True)
    old_sch=sch;old_z=z
    z,sch,cp,checks=s2.load_branch(cp_path,sampler,hashes)
    assert cp['switches']==cfg['active_switches'] and s1.nested_equal(z,s1.unpack(cp['latent']))
    assert sch is not old_sch and sch.derivatives is not old_sch.derivatives and z.data_ptr()!=old_z.data_ptr()
    assert all(x.data_ptr()!=y.data_ptr() for x in sch.derivatives for y in old_sch.derivatives)
    metrics['G20_load_checks']=checks;metrics['independent_snapshot']=True
    rows,summaries=[],[]
    def G():
        nonlocal z
        budget.check(new_branch=True);metrics['trajectories_started']=2
        for index in range(5,cfg['steps']):
            if 5<=index<10:
                z=sampler.guidance(z,sch,index,rows,summaries);print(seed,summaries[-1],flush=True)
            z=sampler.step(z,sch,index);metrics['last_completed']={'branch':'G20','next_index':index+1}
        return z
    try:final=budget.timed('G20_46_steps',G)
    finally:
        sampler.restore_processors();sampler.pending=None
        for name,data in [('inner_metrics.csv',rows),('step_metrics.csv',summaries),('heat_metrics.csv',sampler.heat_rows)]:s1.csv_rows(case/'G20'/name,data)
        metrics['all_finite']=all(r['finite'] for r in rows);metrics['parameter_grad_present']=sampler.parameters_have_grad()
    c.safety(sampler,rows);torch.save({'final':final.detach().cpu()},case/'G20/latent.pt')
    c.decode(pipe,final,case/'G20/final.png',budget,'G20_decode')
    s1.dump_json(case/'G20/effective_config.json',{**cp['effective_config'],'checkpoint_sha256':s2.sha(cp_path),'selection_lock_sha256':s2.sha(out/'selection_lock.json')})
    metrics['completed']=True;metrics['trajectories_completed']=2


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['preflight','baseline','annotate','freeze','intervene']);p.add_argument('--seed',type=int);p.add_argument('--kind',choices=['U','G20']);p.add_argument('--annotation',type=Path);a=p.parse_args()
    cfg=read(CONFIG);hashes,cpu=fingerprints(cfg);out=ROOT/cfg['output_dir'];out.mkdir(exist_ok=True,parents=True)
    s1.dump_json(out/'cpu_checks.json',cpu)
    lock=out/'protocol_lock.json'
    if lock.exists():assert read(lock)['fingerprints']==hashes
    else:
        (out/'AGENTS_at_lock.md').write_bytes((ROOT/'AGENTS.md').read_bytes())
        s1.dump_json(lock,{'fingerprints':hashes,'git_head':cfg['base_commit'],'git_status_before_GPU':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),'agents_sha256_before_gpu':s2.sha(ROOT/'AGENTS.md'),'locked_at_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())})
    if a.mode=='preflight':print(json.dumps(cpu));return
    if a.mode=='annotate':annotate(cfg,out,a.seed,a.kind,a.annotation);return
    if a.mode=='freeze':freeze(cfg,out,hashes);return
    assert not (out/'STOP.json').exists(),'Prior identity/restoration/safety/budget stop; no retry'
    assert a.seed in cfg['seed_candidates']
    prev=[read(p) for p in out.glob('seed*/*_run_metrics.json')]
    previous=sum(x.get('gpu_activity_wall_seconds',0) for x in prev);previous_inner=sum(x.get('inner_updates_total',0) for x in prev)
    assert previous<540 and previous_inner<=300
    assert sum(x.get('trajectories_started',0) for x in prev)+(1 if a.mode=='baseline' else 2)<=12
    case=out/f'seed{a.seed}';case.mkdir(exist_ok=True);path=case/f'{a.mode}_run_metrics.json';assert not path.exists(),'No attempt rerun'
    c.configure(cfg)
    metrics={'mode':a.mode,'seed':a.seed,'argv':sys.argv,'completed':False,'timings':{},'previous_gpu_seconds':previous,'previous_inner_updates':previous_inner,'config':cfg,'fingerprints':hashes,'switches_at_start':c.b.switches()}
    budget=Budget(cfg,metrics,previous,previous_inner);budget.metrics_path=path;s1.dump_json(path,metrics)
    try:
        if a.mode=='baseline':baseline(cfg,out,a.seed,hashes,budget,metrics)
        else:intervene(cfg,out,a.seed,hashes,budget,metrics)
    except Exception:
        metrics['error']=traceback.format_exc();s1.dump_json(out/'STOP.json',{'error':metrics['error'],'seed':a.seed,'mode':a.mode,'incomplete':True});print(metrics['error'],flush=True);raise
    finally:
        if torch.cuda.is_initialized():torch.cuda.synchronize()
        metrics['gpu_activity_wall_seconds']=time.perf_counter()-budget.start;metrics['inner_updates_total']=budget.inner_updates-previous_inner
        metrics['peak_allocated_mib']=max((t['peak_allocated_mib'] for t in metrics['timings'].values()),default=0);metrics['peak_reserved_mib']=max((t['peak_reserved_mib'] for t in metrics['timings'].values()),default=0)
        s1.dump_json(path,metrics)


if __name__=='__main__':main()
