"""Frozen task02 protocol: baseline screening, independent disk continuations, manual bbox evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import accelerate
import diffusers
import numpy as np
from PIL import Image, ImageDraw
import torch
import transformers
from diffusers import LMSDiscreteScheduler, StableDiffusionXLPipeline

from stage01_sdxl import (Sampler, MODEL, LAYERS, pack, unpack, nested_equal, rng_state,
                         restore_rng, diff_metrics, tensor_hash, token_indices, csv_rows, dump_json)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "04_relation_preservation/configs/stage02.json"
PROTOCOL = ROOT / "04_relation_preservation/notes/stage02_protocol.md"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprints(config_path):
    paths = [config_path, PROTOCOL, Path(__file__).resolve(),
             Path(__file__).with_name("stage01_sdxl.py"), Path(__file__).with_name("attention_processor.py")]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def environment():
    return {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
            "diffusers": diffusers.__version__, "transformers": transformers.__version__,
            "accelerate": accelerate.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
            "hf_home": os.environ.get("HF_HOME"), "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32}


def lock_protocol(out, hashes):
    out.mkdir(parents=True, exist_ok=True)
    lock = out / "protocol_lock.json"
    if lock.exists():
        assert json.loads(lock.read_text())["fingerprints"] == hashes, "Frozen code/config/protocol changed"
    else:
        dump_json(lock, {"fingerprints": hashes, "locked_at_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        "git_status_before_sampling": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)})


def consumed_sampling(out):
    return sum(json.loads(p.read_text()).get("sampling_seconds", 0.0) for p in out.glob("*/run_metrics.json"))


def timed(name, function, metrics):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        return function()
    finally:
        torch.cuda.synchronize()
        timing = {"seconds": time.perf_counter()-start,
                  "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
                  "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2}
        metrics["timings"][name] = timing
        if name.endswith("sampling"):
            metrics["sampling_seconds"] += timing["seconds"]
        print(name, timing, flush=True)


class SamplingBudget:
    def __init__(self, cfg, metrics, already):
        self.cfg, self.metrics, self.already, self.stage_start = cfg, metrics, already, None

    def begin(self):
        self.stage_start = time.perf_counter()

    def check(self):
        torch.cuda.synchronize()
        total = self.already + self.metrics["sampling_seconds"] + time.perf_counter()-self.stage_start
        if total >= self.cfg["gpu_sampling_stop_seconds"]:
            raise RuntimeError(f"Sampling budget stop at {total:.2f} seconds")


def load_pipe(cfg):
    assert os.environ.get("HF_HOME") == "/root/autodl-tmp/huggingface"
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL, revision=cfg["model_revision"],
                                                   torch_dtype=torch.float16, local_files_only=True).to("cuda")
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe._guidance_scale, pipe._guidance_rescale = cfg["cfg_scale"], 0.0
    for model in [pipe.unet, pipe.text_encoder, pipe.text_encoder_2, pipe.vae]:
        model.eval().requires_grad_(False)
        model.zero_grad(set_to_none=True)
    return pipe


def tokenizer_diagnostic(pipe, cfg, path):
    data = {}
    for name, tok in [("tokenizer", pipe.tokenizer), ("tokenizer_2", pipe.tokenizer_2)]:
        ids = tok(cfg["prompt"], padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids[0].tolist()
        indices = token_indices(tok, cfg["prompt"], cfg["target_word"])
        spans = []
        for index in indices:
            if spans and spans[-1][-1]+1 == index:
                spans[-1].append(index)
            else:
                spans.append([index])
        data[name] = {"ids": ids, "tokens": tok.convert_ids_to_tokens(ids), "apple_indices": indices,
                      "apple_spans": spans, "target_word_ids": tok(cfg["target_word"], add_special_tokens=False).input_ids,
                      "special_token_ids": tok.all_special_ids}
    dump_json(path, data)
    for item in data.values():
        assert item["apple_indices"] and all(item["ids"][i] not in item["special_token_ids"] for i in item["apple_indices"]), "Invalid apple span"
    assert data["tokenizer"]["apple_indices"] == data["tokenizer_2"]["apple_indices"], "Apple token mapping blocked; union forbidden"
    # Both CLIP hidden states are concatenated in feature dimension at the same sequence position.
    return data, data["tokenizer"]["apple_indices"]


@torch.no_grad()
def encode(pipe, cfg):
    positive, negative, pooled, negative_pooled = pipe.encode_prompt(
        prompt=cfg["prompt"], device="cuda", num_images_per_prompt=1,
        do_classifier_free_guidance=True, negative_prompt=None)
    ids = pipe._get_add_time_ids((cfg["height"], cfg["width"]), (0, 0), (cfg["height"], cfg["width"]),
                                 positive.dtype, pipe.text_encoder_2.config.projection_dim).to("cuda")
    return {"positive": positive, "conditional_added": {"text_embeds": pooled, "time_ids": ids},
            "cfg_prompt_embeds": torch.cat([negative, positive]),
            "cfg_added": {"text_embeds": torch.cat([negative_pooled, pooled]), "time_ids": torch.cat([ids]*2)}}


def scheduler_state(scheduler):
    return {k: v for k, v in scheduler.__dict__.items() if k != "_internal_dict"}


@torch.no_grad()
def decode(pipe, latent, path):
    if pipe.vae.config.force_upcast:
        pipe.upcast_vae()
    dtype = next(iter(pipe.vae.post_quant_conv.parameters())).dtype
    decoded = pipe.vae.decode((latent/pipe.vae.config.scaling_factor).to(dtype=dtype), return_dict=False)[0]
    assert torch.isfinite(decoded).all(), "Non-finite decoded image"
    image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    image.save(path)
    return image


def prepare_decode(pipe):
    pipe.unet.to("cpu")
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()


def baseline(cfg, out, seed, hashes, metrics, budget):
    screening = json.loads((out/"screening.json").read_text()) if (out/"screening.json").exists() else []
    assert not any(r["qualified"] for r in screening), "A qualified seed already exists"
    assert seed == cfg["seed_candidates"][len(screening)], "Candidate order violated"
    folder = out / f"seed{seed}"
    assert not folder.exists(), "Candidate output already exists"
    folder.mkdir()
    metrics["artifact_dir"] = str(folder)
    pipe = timed("model_load", lambda: load_pipe(cfg), metrics)
    diag, indices = tokenizer_diagnostic(pipe, cfg, folder/"tokenizer_diagnostic.json")
    conditioning = encode(pipe, cfg)
    sampler = Sampler(pipe, argparse.Namespace(**cfg), conditioning, indices)
    scheduler = sampler.new_scheduler()
    pipe.scheduler = scheduler
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.no_grad():
        initial = pipe.prepare_latents(1, pipe.unet.config.in_channels, cfg["height"], cfg["width"],
                                      conditioning["positive"].dtype, torch.device("cuda"), generator)
    effective = {**cfg, "seed": seed, "scheduler_class": type(scheduler).__name__,
                 "scheduler_config": dict(scheduler.config), "timesteps": scheduler.timesteps.cpu().tolist(),
                 "sigmas": scheduler.sigmas.tolist(), "lms_order": 4, "layers": LAYERS,
                 "token_indices": indices, "initial_latent_sha256": tensor_hash(initial),
                 "initial_noise_sigma": float(scheduler.init_noise_sigma), "boundary": "after scheduler.step"}
    dump_json(folder/"effective_config.json", effective)
    latent, first_reference = initial.clone(), None

    def run():
        nonlocal latent, first_reference
        budget.begin()
        for index in range(cfg["steps"]):
            budget.check()
            latent = sampler.step(latent, scheduler, index)
            if index+1 == cfg["checkpoint_after"]:
                assert sampler.processors_restored()
                snapshot = {"format_version": 1, "boundary": "after scheduler.step", "next_index": index+1,
                            "latent": pack(latent), "initial_latent": pack(initial),
                            "scheduler_config": dict(scheduler.config), "scheduler_state": pack(scheduler_state(scheduler)),
                            "conditioning": pack(conditioning), "rng": rng_state(generator),
                            "processor_types": sampler.processors, "effective_config": effective,
                            "environment": environment(), "fingerprints": hashes}
                estimate = sum(t.numel()*t.element_size() for t in [latent, initial]+scheduler.derivatives)
                print(f"checkpoint estimate latent/history={estimate} bytes; derivatives={len(scheduler.derivatives)}", flush=True)
                torch.save(snapshot, folder/"checkpoint.pt")
            if index == cfg["checkpoint_after"]:
                first_reference = latent.detach().cpu().clone()
            if index % 10 == 0:
                print(f"baseline seed={seed} index={index}", flush=True)
        torch.save({"first": first_reference, "final": latent.detach().cpu().clone()}, folder/"reference_latents.pt")

    timed("baseline_sampling", run, metrics)
    metrics["processors_restored"] = sampler.processors_restored()
    metrics["model_parameter_grad_present"] = sampler.parameters_have_grad()
    prepare_decode(pipe)
    timed("baseline_decode", lambda: decode(pipe, latent, folder/"baseline.png"), metrics)
    metrics["decoded_finite"] = True
    metrics["checkpoint_bytes"] = (folder/"checkpoint.pt").stat().st_size
    return folder


def load_branch(path, sampler, hashes):
    cp = torch.load(path, map_location="cpu", weights_only=False)
    assert cp["format_version"] == 1 and cp["boundary"] == "after scheduler.step"
    assert cp["fingerprints"] == hashes and cp["environment"] == environment()
    assert cp["processor_types"] == sampler.processors
    scheduler = LMSDiscreteScheduler.from_config(cp["scheduler_config"])
    scheduler.__dict__.update(unpack(cp["scheduler_state"]))  # No set_timesteps reset or skipped history.
    assert scheduler.step_index == cp["next_index"] and len(scheduler.derivatives) == 4
    sampler.conditioning = unpack(cp["conditioning"])
    sampler.restore_processors()
    generator = restore_rng(cp["rng"])
    checks = {"scheduler_exact": nested_equal(scheduler_state(scheduler), unpack(cp["scheduler_state"])),
              "conditioning_exact": nested_equal(sampler.conditioning, unpack(cp["conditioning"])),
              "processors_restored": sampler.processors_restored(),
              "cpu_rng_exact": torch.equal(torch.get_rng_state(), cp["rng"]["cpu"]),
              "cuda_rng_exact": all(torch.equal(x,y) for x,y in zip(torch.cuda.get_rng_state_all(),cp["rng"]["cuda"])),
              "generator_state_exact": torch.equal(generator.get_state(), cp["rng"]["generator"])}
    assert all(checks.values())
    return unpack(cp["latent"]), scheduler, cp, checks


def continue_trajectory(sampler, latent, scheduler, next_index, guided_range, rows, summaries, budget, progress=None):
    """Continue from the loaded next_index; optionally guide a specified half-open index interval."""
    assert scheduler.step_index == next_index
    first = None
    budget.begin()
    if progress is not None:
        progress.update(latent=latent.detach(), next_index=next_index)
    for index in range(next_index, sampler.args.steps):
        budget.check()
        if guided_range is not None and guided_range[0] <= index < guided_range[1]:
            latent = sampler.guidance(latent, scheduler, index, rows, summaries)
            print(f"guided index={index} {summaries[-1]}", flush=True)
        latent = sampler.step(latent, scheduler, index)
        if progress is not None:
            progress.update(latent=latent.detach(), next_index=index+1)
        if index == next_index:
            first = latent.detach().cpu().clone()
    return latent, first


def intervene(cfg, out, hashes, metrics, budget):
    screening = json.loads((out/"screening.json").read_text())
    selected = next(r for r in screening if r["qualified"])
    seed_folder = out/f"seed{selected['seed']}"
    folder = out/"intervention"
    assert not folder.exists(), "Intervention already exists; no retry/sweep allowed"
    folder.mkdir()
    metrics["artifact_dir"] = str(folder)
    dump_json(folder/"baseline_selection_before_intervention.json", selected)
    cp_path = seed_folder/"checkpoint.pt"
    raw_cp = torch.load(cp_path, map_location="cpu", weights_only=False)
    assert raw_cp["effective_config"]["seed"] == selected["seed"]
    assert raw_cp["next_index"] == cfg["checkpoint_after"]
    pipe = timed("model_load", lambda: load_pipe(cfg), metrics)
    diag, indices = tokenizer_diagnostic(pipe, cfg, folder/"tokenizer_diagnostic.json")
    assert indices == raw_cp["effective_config"]["token_indices"]
    sampler = Sampler(pipe, argparse.Namespace(**cfg), unpack(raw_cp["conditioning"]), indices)
    reference = torch.load(seed_folder/"reference_latents.pt", map_location="cpu", weights_only=False)
    latent, scheduler, cp, checks = load_branch(cp_path, sampler, hashes)
    snapshot_before = pack(scheduler_state(scheduler))
    restored, first = timed("restored_sampling", lambda: continue_trajectory(
        sampler, latent, scheduler, cp["next_index"], None, [], [], budget), metrics)
    metrics["restoration"] = {**checks, "first_step": diff_metrics(reference["first"], first),
                               "final_latent": diff_metrics(reference["final"], restored.cpu()),
                               "next_index": cp["next_index"], "checkpoint_bytes": cp_path.stat().st_size}
    assert metrics["restoration"]["first_step"]["torch_equal"] and metrics["restoration"]["final_latent"]["torch_equal"], "Restoration failed"
    # Decode before permitting intervention so image equality is also an actual acceptance gate.
    prepare_decode(pipe)
    timed("restored_decode", lambda: decode(pipe, restored, folder/"restored.png"), metrics)
    old = np.asarray(Image.open(seed_folder/"baseline.png"))
    new = np.asarray(Image.open(folder/"restored.png"))
    metrics["restoration"]["image_pixel_exact"] = bool(np.array_equal(old,new))
    metrics["restoration"]["image_max_abs_error"] = int(np.abs(old.astype(np.int16)-new.astype(np.int16)).max())
    assert metrics["restoration"]["image_pixel_exact"], "Restored image differs"
    metrics["restoration"]["passed"] = True
    print("Restoration passed:", metrics["restoration"], flush=True)
    def restore_inference_devices():
        for model in [pipe.unet, pipe.text_encoder, pipe.text_encoder_2]:
            model.to("cuda")
        pipe.vae.to(dtype=torch.float16)
    timed("restore_inference_devices", restore_inference_devices, metrics)
    latent2, scheduler2, cp2, checks2 = load_branch(cp_path, sampler, hashes)
    independent = (scheduler2 is not scheduler and scheduler2.derivatives is not scheduler.derivatives
                   and latent2.data_ptr() != latent.data_ptr()
                   and all(x.data_ptr()!=y.data_ptr() for x in scheduler2.derivatives for y in scheduler.derivatives))
    assert independent and nested_equal(scheduler_state(scheduler2), unpack(snapshot_before))
    assert nested_equal(latent2, unpack(raw_cp["latent"]))
    metrics["guided_start"] = {**checks2, "independent_state": independent,
                              "same_checkpoint_sha256": sha(cp_path), "same_unintervened_latent": True}
    rows, summaries, progress = [], [], {}
    try:
        guided, _ = timed("guided_sampling", lambda: continue_trajectory(
            sampler, latent2, scheduler2, cp2["next_index"],
            (cfg["guided_index_start"],cfg["guided_index_stop_exclusive"]), rows, summaries, budget, progress), metrics)
    except Exception:
        # Preserve the last completed diffusion update, never label it a final sample.
        sampler.restore_processors()
        metrics["guided_completed"] = False
        metrics["guided_last_completed_next_index"] = progress.get("next_index",cp2["next_index"])
        prepare_decode(pipe)
        timed("incomplete_guided_decode", lambda: decode(pipe, progress.get("latent",latent2), folder/"guided_incomplete.png"), metrics)
        raise
    finally:
        sampler.restore_processors()
        csv_rows(folder/"guided_inner_iteration_metrics.csv",rows)
        csv_rows(folder/"guided_step_metrics.csv",summaries)
        metrics["guidance"] = {"iterations": len(rows), "actual_indices": sorted({r["denoising_index"] for r in rows}),
                               "all_finite": all(r["finite"] for r in rows),
                               "all_nonzero_gradients": all(r["nonzero_gradient"] for r in rows),
                               "model_parameter_grad_present": sampler.parameters_have_grad(),
                               "processors_restored": sampler.processors_restored(),
                               "gradient_norm_range": [min(r["gradient_norm"] for r in rows),max(r["gradient_norm"] for r in rows)] if rows else None,
                               "update_norm_range": [min(r["update_norm"] for r in rows),max(r["update_norm"] for r in rows)] if rows else None,
                               "max_relative_update": max((r["relative_update"] for r in rows),default=None)}
    metrics["guided_completed"] = True
    prepare_decode(pipe)
    timed("guided_decode",lambda:decode(pipe,guided,folder/"guided.png"),metrics)
    metrics["decoded_finite"] = True
    dump_json(folder/"effective_config.json", cp2["effective_config"])
    return folder


def relation(value, cfg):
    thresholds = cfg["relation_thresholds"]
    if value is None:
        return "uncertain"
    if value <= thresholds["clear_failure_max"]:
        return "clear_failure"
    if value >= thresholds["clear_success_min"]:
        return "clear_success"
    return "uncertain"


def evaluate_annotation(annotation, cfg):
    objects = annotation["objects"]
    for obj in objects.values():
        box = obj.get("bbox_pixels")
        if box is not None:
            x0,y0,x1,y1 = box
            assert 0<=x0<x1<=cfg["width"] and 0<=y0<y1<=cfg["height"]
            obj["bbox_normalized"] = [x0/cfg["width"],y0/cfg["height"],x1/cfg["width"],y1/cfg["height"]]
            obj["center_normalized"] = [(x0+x1)/(2*cfg["width"]),(y0+y1)/(2*cfg["height"])]
        else:
            obj["bbox_normalized"], obj["center_normalized"] = None,None
    a,b = objects["A"],objects["B"]
    dx = b["center_normalized"][0]-a["center_normalized"][0] if a["center_normalized"] and b["center_normalized"] else None
    dy = b["center_normalized"][1]-a["center_normalized"][1] if a["center_normalized"] and b["center_normalized"] else None
    missing = any(not obj["present"] for obj in [a,b])
    ambiguous = any(not obj["unique"] or not obj["identity_clear"] or obj["bbox_pixels"] is None for obj in [a,b])
    truncated = any(obj["truncated"] for obj in [a,b])
    old,new = relation(dx,cfg),relation(dy,cfg)
    if missing:
        old,new = "failure_object_missing","failure_object_missing"
    elif ambiguous or truncated:
        old,new = "uncertain","uncertain"
    qualified = not (missing or ambiguous or truncated) and dx>=cfg["baseline_requires"]["dx_min"] and dy<=cfg["baseline_requires"]["dy_max"]
    return {**annotation,"dx":dx,"dy":dy,"old_relation":old,"new_relation":new,
            "old_nominal_success": dx>=0.05 if dx is not None else None,
            "new_nominal_success": dy>=0.05 if dy is not None else None,
            "objects_alive":not missing,"identity_or_instances_uncertain":ambiguous,"invalid_truncated":truncated,
            "uncertain":old=="uncertain" or new=="uncertain","baseline_qualified":qualified,
            "joint_success":old==new=="clear_success"}


def annotate(cfg,out,annotation_path,kind,seed):
    annotation=json.loads(annotation_path.read_text())
    image_path=out/f"seed{seed}/baseline.png" if kind=="baseline" else out/"intervention/guided.png"
    assert annotation["image_sha256"]==sha(image_path),"Annotation image mismatch"
    evaluated=evaluate_annotation(annotation,cfg)
    folder=image_path.parent
    assert not (folder/f"{kind}_annotation.json").exists(),"Annotation already finalized"
    image=Image.open(image_path).convert("RGB")
    draw=ImageDraw.Draw(image)
    for key,color in [("A","red"),("B","blue")]:
        box=evaluated["objects"][key]["bbox_pixels"]
        if box is not None:
            draw.rectangle(box,outline=color,width=4)
            draw.text((box[0]+6,box[1]+6),key,fill=color,stroke_width=2,stroke_fill="white")
    image.save(folder/f"{kind}_annotated.png")
    dump_json(folder/f"{kind}_annotation.json",evaluated)
    if kind=="baseline":
        records=json.loads((out/"screening.json").read_text()) if (out/"screening.json").exists() else []
        assert seed==cfg["seed_candidates"][len(records)]
        records.append({"seed":seed,"qualified":evaluated["baseline_qualified"],"dx":evaluated["dx"],"dy":evaluated["dy"],
                        "reason":annotation["notes"],"annotation":str((folder/f"{kind}_annotation.json").relative_to(ROOT))})
        dump_json(out/"screening.json",records)
    else:
        baseline_record=json.loads((out/"intervention/baseline_selection_before_intervention.json").read_text())
        result={"seed":baseline_record["seed"],"baseline":json.loads((ROOT/baseline_record["annotation"]).read_text()),"guided":evaluated}
        result["degradation_candidate"]=(evaluated["new_relation"]=="clear_success" and evaluated["old_relation"]=="clear_failure")
        result["classification"]=("uncertain_or_invalid" if evaluated["uncertain"] or evaluated["invalid_truncated"] else
                                  "new_target_repair_failed" if evaluated["new_relation"]!="clear_success" else
                                  "joint_success" if evaluated["joint_success"] else "degradation_candidate")
        dump_json(out/"relation_results.json",result)
    print(json.dumps(evaluated,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG)
    sub=parser.add_subparsers(dest="mode",required=True)
    bp=sub.add_parser("baseline"); bp.add_argument("--seed",type=int,required=True)
    sub.add_parser("intervene")
    ap=sub.add_parser("annotate"); ap.add_argument("--annotation",type=Path,required=True)
    ap.add_argument("--kind",choices=["baseline","guided"],required=True); ap.add_argument("--seed",type=int,required=True)
    args=parser.parse_args(); cfg=json.loads(args.config.read_text()); out=ROOT/cfg["output_dir"]
    hashes=fingerprints(args.config.resolve()); lock_protocol(out,hashes)
    if args.mode=="annotate":
        annotate(cfg,out,args.annotation,args.kind,args.seed)
        return
    start=time.perf_counter(); already=consumed_sampling(out)
    assert already<cfg["gpu_sampling_stop_seconds"],"Cumulative sampling budget exhausted"
    metrics={"mode":args.mode,"timings":{},"sampling_seconds":0.0,"sampling_seconds_before":already,
             "completed":False,"environment":environment(),"fingerprints":hashes,"command_argv":sys.argv}
    budget=SamplingBudget(cfg,metrics,already)
    try:
        if args.mode=="baseline":
            baseline(cfg,out,args.seed,hashes,metrics,budget)
        else:
            intervene(cfg,out,hashes,metrics,budget)
        metrics["completed"]=True
    except Exception:
        metrics["traceback"]=traceback.format_exc()
        raise
    finally:
        metrics["total_seconds"]=time.perf_counter()-start
        metrics["peak_allocated_mib"]=max((t["peak_allocated_mib"] for t in metrics["timings"].values()),default=0)
        metrics["peak_reserved_mib"]=max((t["peak_reserved_mib"] for t in metrics["timings"].values()),default=0)
        folder=Path(metrics.get("artifact_dir",out))
        dump_json(folder/"run_metrics.json",metrics)


if __name__=="__main__":
    main()
