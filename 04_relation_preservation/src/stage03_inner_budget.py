"""Frozen stage03 wrapper: original guidance, detached diagnostics, exact gates."""
import argparse
import copy
import csv
import json
import math
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

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'04_relation_preservation/configs/stage03.json'
PROTOCOL = ROOT/'04_relation_preservation/notes/stage03_protocol.md'


def read(path):
    return json.loads(path.read_text())


def cpu_preflight(cfg):
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() == cfg['base_commit']
    old = read(ROOT/cfg['old_lock'])
    for name, digest in old['fingerprints'].items():
        assert s2.sha(ROOT/name) == digest, f'Old fingerprint mismatch: {name}'
    paths = [CONFIG, PROTOCOL, Path(__file__).resolve(), ROOT/'AGENTS.md'] + [ROOT/x for x in old['fingerprints']]
    artifacts = [ROOT/cfg['checkpoint'], ROOT/cfg['reference'], ROOT/cfg['old_lock']]
    olddir = ROOT/'04_relation_preservation/outputs/stage02b'
    artifacts += [olddir/'P1/seed19/baseline.png', olddir/'P1/intervention/guided.png',
                  olddir/'P1/intervention/guided_inner_iteration_metrics.csv', olddir/'P1/intervention/guided_step_metrics.csv']
    hashes = {str(p.relative_to(ROOT)):s2.sha(p) for p in paths+artifacts}
    cp = torch.load(ROOT/cfg['checkpoint'],map_location='cpu',weights_only=False)
    assert cp['fingerprints'] == old['fingerprints']
    assert cp['boundary'] == 'after scheduler.step' and cp['next_index'] == 5
    ec = cp['effective_config']
    for k in ['model_revision','seed','prompt','height','width','steps','cfg_scale','target_word','target_box',
              'checkpoint_after','guided_index_start','guided_index_stop_exclusive','loss_scale','loss_threshold','safety_max_relative_update']:
        assert ec[k] == cfg[k], (k,ec[k],cfg[k])
    assert ec['inner_iters'] == 5 and ec['lms_order'] == 4
    # Old effective_config omits negative_prompt; hashed original encode explicitly uses None.
    assert 'negative_prompt' not in ec and cfg['negative_prompt'] is None
    assert ec['dtype'] == 'float16' and ec['batch_size'] == 1 and ec['layers'] == s1.LAYERS
    assert len(cp['scheduler_state']['derivatives']) == 4
    # CPU-only packed checkpoint inspection; never move checkpoint to GPU here.
    assert cp['latent']['__tensor__'].dtype == torch.float16
    assert list(cp['latent']['__tensor__'].shape) == [1,4,128,128]
    ref = torch.load(ROOT/cfg['reference'],map_location='cpu',weights_only=False)
    assert set(ref) == {'first','final'} and all(t.dtype == torch.float16 and torch.isfinite(t).all() for t in ref.values())
    assert cfg['branches'] == {'U':0,'G5':5,'G20':20}
    assert sum(5*n for n in cfg['branches'].values())+5 == cfg['maximum_inner_updates'] == 130
    return old['fingerprints'], hashes, {'passed':True,'checkpoint_bytes':(ROOT/cfg['checkpoint']).stat().st_size,
            'boundary':cp['boundary'],'next_index':5,'old_checkpoint_fingerprints_exact':True,
            'packed_latent_dtype':'float16','packed_latent_shape':[1,4,128,128],
            'derivative_count':4,'reference_finite':True,'maximum_inner_updates':130}


class WorkBudget:
    def __init__(self, cfg, metrics):
        self.cfg, self.metrics = cfg, metrics
        self.seconds = 0.
        self.active = None
        self.inner_updates = 0

    def elapsed(self):
        return self.seconds + (time.perf_counter()-self.active if self.active is not None else 0.)

    def check(self, new_branch=False):
        torch.cuda.synchronize()
        limit = self.cfg['gpu_no_new_branch_seconds'] if new_branch else self.cfg['gpu_work_limit_seconds']
        assert self.elapsed() < limit, f'GPU work budget reached {self.elapsed():.3f}/{limit}s'

    def run(self, name, fn):
        assert self.active is None
        self.check()
        torch.cuda.reset_peak_memory_stats()
        self.active = time.perf_counter()
        try:
            return fn()
        finally:
            torch.cuda.synchronize()
            duration = time.perf_counter()-self.active
            self.seconds += duration
            self.active = None
            self.metrics['gpu_work_seconds'] = self.seconds
            self.metrics['timings'][name] = {'seconds':duration,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
            print(name,self.metrics['timings'][name],flush=True)


class DiagnosticSampler(s1.Sampler):
    """Only intercept conditional outputs; guidance math remains in original Sampler."""
    def __init__(self, *args, budget, folder, **kwargs):
        super().__init__(*args, **kwargs)
        self.budget, self.folder = budget, folder
        self.context = None
        self.call_index = 0
        self.first_maps = None
        self.last_maps = None
        self.heat_rows = []
        h,w = self.args.height//32,self.args.width//32
        x0,y0,x1,y1 = self.args.target_box
        self.mask = np.zeros((h,w),dtype=np.uint8)
        self.mask[math.floor(y0*h):math.ceil(y1*h),math.floor(x0*w):math.ceil(x1*w)] = 1

    def stats(self, arrays, index, phase, word, iteration):
        for layer, raw in arrays.items():
            raw = raw.astype(np.float64)
            h,w = raw.shape
            p = raw / (raw.sum()+s1.EPS)
            yy,xx = np.indices(raw.shape)
            peak = np.unravel_index(raw.argmax(),raw.shape)
            self.heat_rows.append({'denoising_index':index,'phase':phase,'word':word,'layer':layer,
                'conditional_call':iteration,'peak_value':float(raw.max()),'peak_x':(peak[1]+.5)/w,
                'peak_y':(peak[0]+.5)/h,'centroid_x':float((p*(xx+.5)/w).sum()),
                'centroid_y':float((p*(yy+.5)/h).sum()),'spatial_entropy_nats':float(-(p*np.log(p+s1.EPS)).sum()),
                'mask_area_ratio':float(self.mask.mean()),'in_box_ratio':float((raw*self.mask).sum()/(raw.sum()+s1.EPS)),
                'finite':bool(np.isfinite(raw).all()),'model_parameter_grad_present':self.parameters_have_grad()})

    def conditional(self, latent, timestep, scheduler):
        self.budget.check()
        if latent.requires_grad:
            assert self.budget.inner_updates < self.budget.cfg['maximum_inner_updates'], 'Inner budget exceeded'
            self.budget.inner_updates += 1
        maps, ratios, energy = super().conditional(latent,timestep,scheduler)
        if self.context is not None:
            arrays = {key:value.detach().cpu().numpy().copy() for key,value in maps.items()}
            index,word = self.context
            phase = 'inner_pre_update' if latent.requires_grad else 'after'
            self.stats(arrays,index,phase,word,self.call_index)
            if self.first_maps is None:
                self.first_maps = arrays
            self.last_maps = arrays
            self.call_index += 1
        return maps,ratios,energy

    def save_maps(self, arrays, index, phase, word='apple'):
        np.savez_compressed(self.folder/f'heat_{index:02d}_{word}_{phase}.npz',**arrays,mask=self.mask)

    def guidance(self, latent, scheduler, index, rows, summaries):
        self.context, self.call_index = (index,'apple'),0
        self.first_maps = self.last_maps = None
        try:
            result = super().guidance(latent,scheduler,index,rows,summaries)
            self.save_maps(self.first_maps,index,'before')
            self.save_maps(self.last_maps,index,'after')
            return result
        finally:
            self.context = None
            self.first_maps = self.last_maps = None

    @torch.no_grad()
    def observe(self, latent, scheduler, index, word, indices, phase):
        # scale_model_input can mutate scheduler flags. Restore all runtime and RNG.
        state = s1.pack(s2.scheduler_state(scheduler))
        gen = torch.Generator(device='cuda')
        rng = s1.rng_state(gen)
        saved_latent = latent.clone()
        old_tokens = self.token_indices
        try:
            self.token_indices = indices
            maps,ratios,energy = self.conditional(latent,scheduler.timesteps[index],scheduler)
            scalar = {"energy":energy.item(),"mid_ratio":ratios["mid"].item(),"up_ratio":ratios["up"].item()}
            arrays = {key:value.detach().cpu().numpy().copy() for key,value in maps.items()}
            self.stats(arrays,index,phase,word,0)
            self.save_maps(arrays,index,phase,word)
            del maps
        finally:
            self.token_indices = old_tokens
            scheduler.__dict__.update(s1.unpack(state))
            restored_gen = s1.restore_rng(rng)
        assert torch.equal(latent,saved_latent)
        assert s1.nested_equal(s2.scheduler_state(scheduler),s1.unpack(state))
        current_rng = s1.rng_state(restored_gen)
        assert torch.equal(current_rng['cpu'],rng['cpu'])
        assert all(torch.equal(x,y) for x,y in zip(current_rng['cuda'],rng['cuda']))
        assert torch.equal(current_rng['generator'],rng['generator'])
        assert current_rng['python'] == rng['python']
        assert current_rng['numpy'][0] == rng['numpy'][0]
        assert np.array_equal(current_rng['numpy'][1],rng['numpy'][1])
        assert current_rng['numpy'][2:] == rng['numpy'][2:]
        assert self.processors_restored() and not self.parameters_have_grad()
        return scalar

    def step(self, latent, scheduler, index):
        self.budget.check()
        return super().step(latent,scheduler,index)


def exact_csv(path, rows):
    with path.open(newline='') as f:
        old = list(csv.DictReader(f))
    new = [{k:str(v) for k,v in row.items()} for row in rows]
    assert old == new, f'G5 key metrics differ: {path}'
    return {'rows':len(rows),'all_fields_exact':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','run'])
    a = parser.parse_args()
    cfg = read(CONFIG)
    oldhashes,hashes,cpu = cpu_preflight(cfg)
    out = ROOT/cfg['output_dir']
    out.mkdir(parents=True,exist_ok=True)
    s1.dump_json(out/'cpu_checks.json',cpu)
    if a.mode == 'preflight':
        print(json.dumps(cpu,indent=2));return
    lock = out/'protocol_lock.json'
    if lock.exists():
        assert read(lock)['fingerprints'] == hashes, 'Stage03 identity changed'
    else:
        s1.dump_json(lock,{'fingerprints':hashes,'locked_at_utc':time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
            'git_head':cfg['base_commit'],'git_status_before_gpu':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            'old_checkpoint_identity':'validated against complete stage02b fingerprint dictionary'})
    assert not (out/'run_started.json').exists(), 'Stage03 already attempted; no automatic retry'
    s1.dump_json(out/'run_started.json',{'argv':sys.argv,'locked_before_first_gpu':True})
    metrics = {'completed':False,'status':'incomplete','timings':{},'gpu_work_seconds':0.,'branches':{},'gates':{},
               'command_argv':sys.argv,'config':cfg,'fingerprints':hashes,'cpu_checks':cpu}
    budget = WorkBudget(cfg,metrics)
    progress, sampler = {}, None
    total_start = time.perf_counter()
    try:
        metrics['environment'] = s2.environment()
        pipe = budget.run('model_load',lambda:s2.load_pipe(cfg))
        cp = torch.load(ROOT/cfg['checkpoint'],map_location='cpu',weights_only=False)
        diag,apple = s2.tokenizer_diagnostic(pipe,cfg,out/'tokenizer_diagnostic.json')
        cup1 = s1.token_indices(pipe.tokenizer,cfg['prompt'],'cup')
        cup2 = s1.token_indices(pipe.tokenizer_2,cfg['prompt'],'cup')
        assert cup1 == cup2 and cup1 and apple == cp['effective_config']['token_indices']
        metrics['cup_token_indices'] = cup1
        reference = torch.load(ROOT/cfg['reference'],map_location='cpu',weights_only=False)
        olddir = ROOT/'04_relation_preservation/outputs/stage02b/P1'
        prior_start = None
        for name, inner in cfg['branches'].items():
            budget.check(new_branch=True)
            assert read(lock)['fingerprints'] == hashes
            assert all(s2.sha(ROOT/p)==digest for p,digest in hashes.items())
            folder = out/name
            assert not folder.exists()
            folder.mkdir()
            args = argparse.Namespace(**cfg,inner_iters=max(inner,1))
            sampler = DiagnosticSampler(pipe,args,s1.unpack(cp['conditioning']),apple,budget=budget,folder=folder)
            rows,summaries = [],[]
            progress = {'branch':name,'next_index':5}
            bm = {'inner_upper_limit':inner,'completed':False}
            metrics['branches'][name] = bm
            # The original assertions validate OLD checkpoint identity, never new fingerprints.
            latent,scheduler,loaded,checks = budget.run(name+'_load_state',lambda:s2.load_branch(ROOT/cfg['checkpoint'],sampler,oldhashes))
            assert loaded['next_index'] == 5 and torch.equal(latent,s1.unpack(cp['latent']))
            if prior_start is not None:
                prior_latent,prior_sched = prior_start
                assert latent.data_ptr()!=prior_latent.data_ptr() and scheduler is not prior_sched
                assert scheduler.derivatives is not prior_sched.derivatives
                assert all(x.data_ptr()!=y.data_ptr() for x in scheduler.derivatives for y in prior_sched.derivatives)
            prior_start = (latent,scheduler)
            bm['load_checks'] = checks
            bm['independent_state'] = True
            bm['effective_config'] = {**cfg,'inner_iters':inner,'layers':s1.LAYERS,'apple_indices':apple,'cup_indices':cup1,
                'scheduler_config':dict(scheduler.config),'timesteps':scheduler.timesteps.cpu().tolist(),
                'sigmas':scheduler.sigmas.cpu().tolist(),'initial_latent_sha256':s1.tensor_hash(s1.unpack(cp['initial_latent']))}
            s1.dump_json(folder/'effective_config.json',bm['effective_config'])
            probe = None
            if name == 'G5':
                def original_probe():
                    original = s1.Sampler(pipe,args,s1.unpack(cp['conditioning']),apple)
                    z,sch,_,_ = s2.load_branch(ROOT/cfg['checkpoint'],original,oldhashes)
                    ir,sr = [],[]
                    z = original.guidance(z,sch,5,ir,sr)
                    budget.inner_updates += len(ir)
                    assert len(ir)<=5 and budget.inner_updates<=130
                    guided = z.detach().cpu().clone()
                    stepped = original.step(z,sch,5).cpu().clone()
                    return guided,stepped,ir,sr
                probe = budget.run('original_sampler_index5_probe',original_probe)
                latent,scheduler,_,checks = budget.run('G5_reload_after_probe',lambda:s2.load_branch(ROOT/cfg['checkpoint'],sampler,oldhashes))
            first = None
            def trajectory():
                nonlocal latent,first
                for index in range(5,cfg['steps']):
                    budget.check()
                    if 5<=index<10:
                        if inner:
                            latent = sampler.guidance(latent,scheduler,index,rows,summaries)
                            if name=='G5' and index==5:
                                metrics['gates']['instrumentation_guidance'] = s1.diff_metrics(probe[0],latent.cpu())
                                assert metrics['gates']['instrumentation_guidance']['torch_equal']
                                assert rows==probe[2] and summaries==probe[3]
                        else:
                            scalar = sampler.observe(latent,scheduler,index,'apple',apple,'before')
                            # Same scalar schema, no gradient or update for U.
                            recent = sampler.heat_rows[-2:]
                            summaries.append({'denoising_index':index,**scalar,'executed_inner_iterations':0,
                                'gradient_norm':0.,'update_norm':0.,'finite':all(r['finite'] for r in recent),
                                'model_parameter_grad_present':False})
                    if index==50:
                        for word,indices in [('apple',apple),('cup',cup1)]:
                            sampler.observe(latent,scheduler,index,word,indices,'late')
                    latent = sampler.step(latent,scheduler,index)
                    progress.update(next_index=index+1,latent=latent.detach())
                    if index==5:
                        first = latent.detach().cpu().clone()
                        if name=='G5':
                            metrics['gates']['instrumentation_step'] = s1.diff_metrics(probe[1],first)
                            assert metrics['gates']['instrumentation_step']['torch_equal']
                    if index%10==0 or index<10:
                        print(name,index, summaries[-1] if index<10 else '',flush=True)
                return latent
            try:
                final = budget.run(name+'_sampling_and_diagnostics',trajectory)
            finally:
                sampler.restore_processors()
                s1.csv_rows(folder/'inner_metrics.csv',rows)
                s1.csv_rows(folder/'step_metrics.csv',summaries)
                s1.csv_rows(folder/'heat_metrics.csv',sampler.heat_rows)
                bm['inner_updates'] = len(rows)
                bm['model_parameter_grad_present'] = sampler.parameters_have_grad()
                bm['all_parameters_frozen'] = all(not p.requires_grad for model in sampler.models for p in model.parameters())
                bm['all_finite'] = all(r['finite'] for r in rows+sampler.heat_rows)
                bm['processors_restored'] = sampler.processors_restored()
                bm['last_completed_next_index'] = progress['next_index']
                bm['max_relative_update'] = max((r['relative_update'] for r in rows),default=0.)
            assert not bm['model_parameter_grad_present'] and bm['all_parameters_frozen'] and bm['all_finite']
            if name=='U':
                metrics['gates']['U_first_step'] = s1.diff_metrics(reference['first'],first)
                metrics['gates']['U_final_latent'] = s1.diff_metrics(reference['final'],final.cpu())
                assert metrics['gates']['U_first_step']['torch_equal'] and metrics['gates']['U_final_latent']['torch_equal']
            if name=='G5':
                metrics['gates']['G5_inner_metrics'] = exact_csv(olddir/'intervention/guided_inner_iteration_metrics.csv',rows)
                metrics['gates']['G5_step_metrics'] = exact_csv(olddir/'intervention/guided_step_metrics.csv',summaries)
            def decode_branch():
                s2.prepare_decode(pipe)
                return s2.decode(pipe,final,folder/'final.png')
            image = budget.run(name+'_decode',decode_branch)
            if name in ('U','G5'):
                reference_image = olddir/('seed19/baseline.png' if name=='U' else 'intervention/guided.png')
                oldpix,newpix = np.asarray(Image.open(reference_image)),np.asarray(image)
                gate = {'pixel_exact':bool(np.array_equal(oldpix,newpix)),
                        'max_abs_error':int(np.abs(oldpix.astype(np.int16)-newpix.astype(np.int16)).max())}
                metrics['gates'][name+'_image'] = gate
                assert gate['pixel_exact'], f'{name} reference image mismatch'
            bm['completed'] = True
            torch.save({'final':final.detach().cpu(),'first':first},folder/'latents.pt')
            if name!='G20':
                def inference_devices():
                    for model in [pipe.unet,pipe.text_encoder,pipe.text_encoder_2]: model.to('cuda')
                    pipe.vae.to(dtype=torch.float16)
                budget.run(name+'_restore_devices',inference_devices)
            s1.dump_json(out/'run_metrics.json',metrics)
        assert budget.inner_updates<=130
        metrics['completed'] = True
        metrics['status'] = 'gpu_complete_pending_annotation'
    except Exception:
        metrics['error'] = traceback.format_exc()
        print(metrics['error'],flush=True)
        if sampler is not None: sampler.restore_processors()
        if 'latent' in progress:
            torch.save({'latent':progress['latent'].detach().cpu(),'next_index':progress['next_index']},out/progress['branch']/'incomplete_state.pt')
        raise
    finally:
        metrics['total_inner_updates_including_probe'] = budget.inner_updates
        metrics['total_process_seconds'] = time.perf_counter()-total_start
        metrics['peak_allocated_mib'] = max((v['peak_allocated_mib'] for v in metrics['timings'].values()),default=0.)
        metrics['peak_reserved_mib'] = max((v['peak_reserved_mib'] for v in metrics['timings'].values()),default=0.)
        s1.dump_json(out/'run_metrics.json',metrics)


if __name__ == '__main__':
    main()
