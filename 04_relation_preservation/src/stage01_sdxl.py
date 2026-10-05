"""Single historical SDXL/LMS case and a real disk-roundtrip continuation check."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import accelerate
import diffusers
import numpy as np
from PIL import Image, ImageDraw
import torch
import transformers
from diffusers import LMSDiscreteScheduler, StableDiffusionXLPipeline

from attention_processor import DifferentiableMapProcessor

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LAYERS = {
    "mid": "mid_block.attentions.0.transformer_blocks.0.attn2",
    "up": "up_blocks.0.attentions.0.transformer_blocks.0.attn2",
}
EPS = 1e-8


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default="A small red cabin beside a calm mountain lake at sunrise, realistic photograph")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--scheduler", choices=["lms"], default="lms")
    p.add_argument("--steps", type=int, default=51)
    p.add_argument("--cfg-scale", type=float, default=7.0)
    p.add_argument("--target-word", default="cabin")
    p.add_argument("--target-box", type=float, nargs=4, default=[0.10, 0.50, 0.40, 0.82])
    p.add_argument("--guided-steps", type=int, default=10)
    p.add_argument("--inner-iters", type=int, default=5)
    p.add_argument("--loss-scale", type=float, default=30.0)
    p.add_argument("--loss-threshold", type=float, default=0.2)
    p.add_argument("--safety-max-relative-update", type=float, default=0.1)
    p.add_argument("--checkpoint-after", type=int, default=25, help="Number of completed scheduler updates")
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.height % 32 or a.width % 32 or min(a.height, a.width) < 32:
        p.error("Resolution must be a positive multiple of 32 (selected layer grid is H/32 by W/32)")
    x0, y0, x1, y1 = a.target_box
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        p.error("Invalid normalized target box")
    if not (1 <= a.checkpoint_after < a.steps and 1 <= a.guided_steps <= a.steps and a.inner_iters >= 1):
        p.error("Invalid step counts")
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error("Output directory must be empty; historical results must not be overwritten")
    return a


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def pack(value):
    """CPU serialization copies, with original device recorded per tensor."""
    if isinstance(value, torch.Tensor):
        return {"__tensor__": value.detach().cpu().clone(), "device": str(value.device)}
    if isinstance(value, dict):
        return {key: pack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [pack(item) for item in value]
    if isinstance(value, tuple):
        return tuple(pack(item) for item in value)
    return copy.deepcopy(value)


def unpack(value):
    if isinstance(value, dict) and "__tensor__" in value:
        return value["__tensor__"].to(value["device"]).clone()
    if isinstance(value, dict):
        return {key: unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unpack(item) for item in value]
    if isinstance(value, tuple):
        return tuple(unpack(item) for item in value)
    return copy.deepcopy(value)


def nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.device == b.device and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(nested_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(nested_equal(x, y) for x, y in zip(a, b))
    return a == b


def rng_state(generator):
    return {"cpu": torch.get_rng_state().clone(), "cuda": torch.cuda.get_rng_state_all(),
            "generator": generator.get_state().clone(), "generator_device": str(generator.device),
            "python": random.getstate(), "numpy": np.random.get_state()}


def restore_rng(state):
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    generator = torch.Generator(device=state["generator_device"])
    generator.set_state(state["generator"])
    return generator


def diff_metrics(a, b):
    delta = (a.float() - b.float()).abs()
    return {"max_abs_error": delta.max().item(), "mean_abs_error": delta.mean().item(),
            "torch_equal": torch.equal(a, b), "dtype_equal": a.dtype == b.dtype,
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}


class Sampler:
    def __init__(self, pipe, args, conditioning, token_indices):
        self.pipe, self.args = pipe, args
        self.conditioning, self.token_indices = conditioning, token_indices
        self.modules = {k: pipe.unet.get_submodule(v) for k, v in LAYERS.items()}
        assert all(module.is_cross_attention for module in self.modules.values())
        self.original = {k: module.processor for k, module in self.modules.items()}
        self.processors = {k: type(module.processor).__module__ + "." + type(module.processor).__name__
                           for k, module in pipe.unet.named_modules() if hasattr(module, "processor")}
        self.models = [pipe.unet, pipe.text_encoder, pipe.text_encoder_2, pipe.vae]

    def restore_processors(self):
        for key, module in self.modules.items():
            module.set_processor(self.original[key])

    def processors_restored(self):
        return all(module.processor is self.original[k] for k, module in self.modules.items())

    def parameters_have_grad(self):
        return any(p.grad is not None for model in self.models for p in model.parameters())

    def new_scheduler(self):
        scheduler = LMSDiscreteScheduler.from_config(self.pipe.scheduler.config)
        scheduler.set_timesteps(self.args.steps, device="cuda")
        return scheduler

    def conditional(self, latent, timestep, scheduler):
        captured = {}
        try:
            for key, module in self.modules.items():
                module.set_processor(DifferentiableMapProcessor(key, captured))
            self.pipe.unet(scheduler.scale_model_input(latent, timestep), timestep,
                           encoder_hidden_states=self.conditioning["positive"],
                           added_cond_kwargs=self.conditioning["conditional_added"], return_dict=False)
            assert set(captured) == set(self.modules)
            maps, ratios = {}, {}
            h, w = self.args.height // 32, self.args.width // 32
            x0, y0, x1, y1 = self.args.target_box
            xs, xe, ys, ye = math.floor(x0*w), math.ceil(x1*w), math.floor(y0*h), math.ceil(y1*h)
            for key, probs in captured.items():
                vector = probs[0, :, :, self.token_indices].mean(dim=0).mean(dim=-1)
                if vector.numel() != h*w:
                    raise RuntimeError(f"Layer {key} has {vector.numel()} query tokens; expected {h}x{w}")
                maps[key] = vector.reshape(h, w)
                ratios[key] = maps[key][ys:ye, xs:xe].sum() / (maps[key].sum() + EPS)
            energy = sum((1-ratio)**2 for ratio in ratios.values()) / len(ratios)
            return maps, ratios, energy
        finally:
            self.restore_processors()
            captured.clear()  # Processor references must not retain an old autograd graph.

    @torch.no_grad()
    def step(self, latent, scheduler, index):
        assert self.processors_restored()
        assert scheduler.step_index in (None, index)
        timestep = scheduler.timesteps[index]
        model_input = scheduler.scale_model_input(torch.cat([latent]*2), timestep)
        prediction = self.pipe.unet(model_input, timestep,
                                    encoder_hidden_states=self.conditioning["cfg_prompt_embeds"],
                                    added_cond_kwargs=self.conditioning["cfg_added"], return_dict=False)[0]
        uncond, text = prediction.chunk(2)
        noise = uncond + self.args.cfg_scale * (text-uncond)
        result = scheduler.step(noise, timestep, latent, order=4, return_dict=False)[0]
        if not torch.isfinite(result).all():
            raise RuntimeError(f"Non-finite latent at index {index}")
        assert scheduler.step_index == index+1
        return result

    def guidance(self, latent, scheduler, index, inner_rows, summaries):
        a, first, last = self.args, None, None
        sigma = scheduler.sigmas[index].float()
        prior_energy, inner_index = float("inf"), 0
        while prior_energy > a.loss_threshold and inner_index < a.inner_iters:
            leaf = latent.detach().requires_grad_(True)
            maps, ratios, energy = self.conditional(leaf, scheduler.timesteps[index], scheduler)
            loss = energy*a.loss_scale
            gradient = torch.autograd.grad(loss, leaf)[0]
            update = gradient.detach()*sigma.square()
            # Preserve historical FP16 norm/update computation; extra float32 norm is diagnostic only.
            relative = update.norm() / (leaf.detach().norm()+EPS)
            finite = bool(torch.isfinite(energy) and torch.isfinite(gradient).all() and torch.isfinite(update).all())
            row = {"denoising_index": index, "timestep": float(scheduler.timesteps[index]),
                   "sigma": sigma.item(), "inner_index": inner_index,
                   "mid_ratio": ratios["mid"].item(), "up_ratio": ratios["up"].item(),
                   "aggregate_energy": energy.item(), "loss_scaled": loss.item(),
                   "gradient_norm": gradient.norm().item(), "gradient_norm_fp32": gradient.float().norm().item(),
                   "latent_norm": leaf.detach().norm().item(), "update_norm": update.norm().item(),
                   "relative_update": relative.item(), "finite": finite,
                   "nonzero_gradient": bool(torch.count_nonzero(gradient)),
                   "model_parameter_grad_present": self.parameters_have_grad(), "update_executed": False}
            inner_rows.append(row)
            if not finite or relative.item() > a.safety_max_relative_update or row["model_parameter_grad_present"]:
                raise RuntimeError(f"Unsafe guidance: {row}")
            if first is None:
                first = row.copy()
            latent = leaf.detach()-update
            row["update_executed"] = True
            last = row.copy()
            prior_energy = energy.item()  # Historical early-stop uses pre-update energy.
            del leaf, maps, ratios, energy, loss, gradient, update, relative
            inner_index += 1
        with torch.no_grad():
            maps, ratios, energy = self.conditional(latent, scheduler.timesteps[index], scheduler)
            summaries.append({"denoising_index": index, "timestep": float(scheduler.timesteps[index]),
                              "sigma": sigma.item(), "energy_first": first["aggregate_energy"],
                              "energy_last_pre_update": last["aggregate_energy"], "energy_final": energy.item(),
                              "mid_ratio_first": first["mid_ratio"], "mid_ratio_last": ratios["mid"].item(),
                              "up_ratio_first": first["up_ratio"], "up_ratio_last": ratios["up"].item(),
                              "executed_inner_iterations": inner_index})
        return latent.detach()


def token_indices(tokenizer, prompt, word):
    ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids[0].tolist()
    word_ids = tokenizer(word, add_special_tokens=False).input_ids
    return sorted({i for start in range(len(ids)-len(word_ids)+1)
                   if ids[start:start+len(word_ids)] == word_ids for i in range(start, start+len(word_ids))})


def csv_rows(path, rows):
    if rows:
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def timed_stage(name, function, timings):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    timings[name] = {"seconds": time.perf_counter()-start,
                     "peak_allocated_mib": torch.cuda.max_memory_allocated()/1024**2,
                     "peak_reserved_mib": torch.cuda.max_memory_reserved()/1024**2}
    print(f"{name}: {timings[name]}", flush=True)
    return result


def main():
    a = arguments()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    out = a.output_dir
    total_start = time.perf_counter()
    timings = {}
    hf_home = os.environ.get("HF_HOME")
    if not hf_home or not (Path(hf_home)/"hub"/"models--stabilityai--stable-diffusion-xl-base-1.0").is_dir():
        raise RuntimeError("Verified HF_HOME local SDXL cache required")
    environment = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
                   "diffusers": diffusers.__version__, "transformers": transformers.__version__,
                   "accelerate": accelerate.__version__, "cuda_runtime": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(0), "gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
                   "hf_home": hf_home, "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32}
    root = Path(__file__).resolve().parents[2]
    provenance = {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  "git_status_before": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
                  "command_argv": sys.argv, "code_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [Path(__file__).resolve(), Path(__file__).with_name("attention_processor.py").resolve(),
                            root/"02_layout_control/src/18_paper_backward_guidance_full_generation.py",
                            root/"02_layout_control/src/19_lms_scheduler_alignment_full_generation.py"]}}
    dump_json(out/"environment.json", environment)
    dump_json(out/"provenance.json", provenance)
    pipe = timed_stage("model_load", lambda: StableDiffusionXLPipeline.from_pretrained(
        MODEL, torch_dtype=torch.float16, local_files_only=True).to("cuda"), timings)
    pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe._guidance_scale, pipe._guidance_rescale = a.cfg_scale, 0.0
    for model in [pipe.unet, pipe.text_encoder, pipe.text_encoder_2, pipe.vae]:
        model.eval()
        model.requires_grad_(False)
        model.zero_grad(set_to_none=True)
    with torch.no_grad():
        positive, negative, pooled, negative_pooled = pipe.encode_prompt(
            prompt=a.prompt, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=None)
        time_ids = pipe._get_add_time_ids((a.height, a.width), (0, 0), (a.height, a.width),
                                        positive.dtype, pipe.text_encoder_2.config.projection_dim).to("cuda")
        conditioning = {"positive": positive, "conditional_added": {"text_embeds": pooled, "time_ids": time_ids},
                        "cfg_prompt_embeds": torch.cat([negative, positive]),
                        "cfg_added": {"text_embeds": torch.cat([negative_pooled, pooled]), "time_ids": torch.cat([time_ids]*2)}}
    indices1 = token_indices(pipe.tokenizer, a.prompt, a.target_word)
    indices2 = token_indices(pipe.tokenizer_2, a.prompt, a.target_word)
    indices = sorted(set(indices1)|set(indices2))
    if not indices:
        raise RuntimeError("Target word absent from both tokenizers")
    sampler = Sampler(pipe, a, conditioning, indices)
    baseline_scheduler = sampler.new_scheduler()
    pipe.scheduler = baseline_scheduler  # prepare_latents must use this 51-step schedule's init_noise_sigma.
    generator = torch.Generator(device="cuda").manual_seed(a.seed)
    with torch.no_grad():
        initial = pipe.prepare_latents(1, pipe.unet.config.in_channels, a.height, a.width,
                                      positive.dtype, torch.device("cuda"), generator)  # Explicit single shared x_T.
    config = {**vars(a), "output_dir": str(out.resolve()), "model": MODEL, "dtype": "float16",
              "batch_size": 1, "cfg_forward_batch_size": 2, "layers": LAYERS, "token_indices_1": indices1,
              "token_indices_2": indices2, "token_indices": indices, "scheduler_class": type(baseline_scheduler).__name__,
              "scheduler_config": dict(baseline_scheduler.config), "lms_step_order": 4,
              "timesteps": baseline_scheduler.timesteps.cpu().tolist(), "sigmas": baseline_scheduler.sigmas.tolist(),
              "initial_latent_shape": list(initial.shape), "initial_latent_sha256": tensor_hash(initial),
              "initial_noise_sigma": float(baseline_scheduler.init_noise_sigma),
              "attention_grid": [a.height//32, a.width//32], "negative_prompt": None,
              "energy": "mean((1-mid_ratio)^2,(1-up_ratio)^2)", "latent_update": "z - sigma^2 * grad(30*E), scale parameterized",
              "checkpoint_boundary": "after scheduler.step; next_index=completed updates",
              "processor_types": sampler.processors}
    dump_json(out/"config.json", config)
    print(f"Effective config: {json.dumps(config)}", flush=True)
    checkpoint_path = out/"baseline_checkpoint.pt"
    reference_first, checkpoint_in_memory = None, None

    def baseline_run():
        nonlocal reference_first, checkpoint_in_memory
        latent = initial.clone()
        for index in range(a.steps):
            latent = sampler.step(latent, baseline_scheduler, index)
            if index+1 == a.checkpoint_after:
                assert sampler.processors_restored()
                state = {k: v for k, v in baseline_scheduler.__dict__.items() if k != "_internal_dict"}
                checkpoint_in_memory = {"format_version": 1, "boundary": "after scheduler.step",
                    "next_index": index+1, "latent": pack(latent), "initial_latent": pack(initial),
                    "scheduler_config": dict(baseline_scheduler.config), "scheduler_state": pack(state),
                    "conditioning": pack(conditioning), "config": config, "environment": environment,
                    "processor_types": sampler.processors, "rng": rng_state(generator)}
                estimated = sum(v.numel()*v.element_size() for v in [latent, initial]+baseline_scheduler.derivatives)
                print(f"Checkpoint latent/history estimate: {estimated} bytes; next index {index+1}; history {len(baseline_scheduler.derivatives)}", flush=True)
                torch.save(checkpoint_in_memory, checkpoint_path)
                print(f"Checkpoint on disk: {checkpoint_path.stat().st_size} bytes", flush=True)
            if index == a.checkpoint_after:
                reference_first = latent.detach().clone()
            if index % 10 == 0 or index == a.steps-1:
                print(f"baseline index {index}/{a.steps-1}", flush=True)
        return latent

    baseline = timed_stage("baseline_51_steps_including_save", baseline_run, timings)
    inner_rows, summaries = [], []

    def guided_run():
        scheduler = sampler.new_scheduler()
        assert scheduler is not baseline_scheduler and scheduler.derivatives is not baseline_scheduler.derivatives
        latent = initial.clone()
        try:
            for index in range(a.steps):
                if index < a.guided_steps:
                    latent = sampler.guidance(latent, scheduler, index, inner_rows, summaries)
                    print(f"guided {summaries[-1]}", flush=True)
                latent = sampler.step(latent, scheduler, index)
                if index % 10 == 0 or index == a.steps-1:
                    print(f"guided index {index}/{a.steps-1}", flush=True)
        finally:
            sampler.restore_processors()
            csv_rows(out/"guided_inner_iteration_metrics.csv", inner_rows)
            csv_rows(out/"guided_step_metrics.csv", summaries)
        return latent

    guided = timed_stage("guided_51_steps", guided_run, timings)
    restoration = {}

    def resumed_run():
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert loaded["format_version"] == 1 and loaded["boundary"] == "after scheduler.step"
        assert loaded["environment"] == environment and loaded["config"] == config
        restored_scheduler = LMSDiscreteScheduler.from_config(loaded["scheduler_config"])
        # Do not call set_timesteps here: restore every live runtime field, including history.
        restored_scheduler.__dict__.update(unpack(loaded["scheduler_state"]))
        next_index = loaded["next_index"]
        assert restored_scheduler.step_index == next_index
        assert len(restored_scheduler.derivatives) == min(next_index, 4)
        restoration["scheduler_fields"] = sorted(loaded["scheduler_state"])
        restoration["scheduler_snapshot_exact"] = nested_equal(
            {k: v for k, v in restored_scheduler.__dict__.items() if k != "_internal_dict"},
            unpack(checkpoint_in_memory["scheduler_state"]))
        restored_conditioning = unpack(loaded["conditioning"])
        restoration["conditioning_exact"] = nested_equal(conditioning, restored_conditioning)
        latent = unpack(loaded["latent"])
        restoration["checkpoint_latent_exact"] = nested_equal(latent, unpack(checkpoint_in_memory["latent"]))
        restoration["initial_latent_exact"] = nested_equal(initial, unpack(loaded["initial_latent"]))
        restoration["independent_derivatives"] = all(
            x.data_ptr() != y.data_ptr() for x in restored_scheduler.derivatives for y in baseline_scheduler.derivatives)
        sampler.restore_processors()
        restoration["processor_types_exact"] = loaded["processor_types"] == sampler.processors
        restoration["processors_restored"] = sampler.processors_restored()
        restored_generator = restore_rng(loaded["rng"])
        restoration["generator_state_exact"] = torch.equal(restored_generator.get_state(), loaded["rng"]["generator"])
        restoration["cpu_rng_exact"] = torch.equal(torch.get_rng_state(), loaded["rng"]["cpu"])
        restoration["cuda_rng_exact"] = all(torch.equal(x, y) for x, y in zip(torch.cuda.get_rng_state_all(), loaded["rng"]["cuda"]))
        sampler.conditioning = restored_conditioning
        assert all(v for v in restoration.values() if isinstance(v, bool))
        for index in range(next_index, a.steps):
            latent = sampler.step(latent, restored_scheduler, index)
            if index == next_index:
                restoration["first_step"] = diff_metrics(reference_first, latent)
                print(f"Resumed first step: {restoration['first_step']}", flush=True)
            if index % 10 == 0 or index == a.steps-1:
                print(f"resumed index {index}/{a.steps-1}", flush=True)
        restoration["final_latent"] = diff_metrics(baseline, latent)
        restoration["next_index"] = next_index
        restoration["resumed_updates"] = a.steps-next_index
        restoration["derivative_count_at_checkpoint"] = len(loaded["scheduler_state"]["derivatives"])
        restoration["checkpoint_bytes"] = checkpoint_path.stat().st_size
        return latent

    resumed = timed_stage("disk_load_and_resumed_26_steps", resumed_run, timings)
    parameter_grad_present = sampler.parameters_have_grad()
    all_parameters_frozen = all(not p.requires_grad for model in sampler.models for p in model.parameters())
    pipe.unet.to("cpu")
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    torch.cuda.empty_cache()

    @torch.no_grad()
    def decode_all():
        if pipe.vae.config.force_upcast:
            pipe.upcast_vae()
        dtype = next(iter(pipe.vae.post_quant_conv.parameters())).dtype
        images = {}
        for name, latent in [("baseline", baseline), ("guided", guided), ("resumed", resumed)]:
            decoded = pipe.vae.decode((latent/pipe.vae.config.scaling_factor).to(dtype=dtype), return_dict=False)[0]
            image = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
            image.save(out/f"{name}.png")
            images[name] = image
            if name != "resumed":
                boxed = image.convert("RGB").copy()
                x0, y0, x1, y1 = a.target_box
                ImageDraw.Draw(boxed).rectangle((x0*a.width, y0*a.height, x1*a.width, y1*a.height), outline="cyan", width=4)
                boxed.save(out/f"{name}_with_target_box.png")
        for filename, names in [("baseline_guided_comparison.png", ["baseline", "guided"]),
                                ("continuous_resumed_comparison.png", ["baseline", "resumed"])]:
            canvas = Image.new("RGB", (a.width*2, a.height+32), "white")
            draw = ImageDraw.Draw(canvas)
            for i, name in enumerate(names):
                draw.text((i*a.width+10, 10), name, fill="black")
                canvas.paste(images[name], (i*a.width, 32))
            canvas.save(out/filename)
        return images

    images = timed_stage("decode_and_image_output", decode_all, timings)
    base_pixels = np.asarray(images["baseline"], dtype=np.int16)
    guided_pixels = np.asarray(images["guided"], dtype=np.int16)
    restored_pixels = np.asarray(images["resumed"], dtype=np.int16)
    restoration["final_image"] = {"pixel_exact": bool(np.array_equal(base_pixels, restored_pixels)),
                                  "max_abs_error": int(np.abs(base_pixels-restored_pixels).max())}
    restoration["passed"] = (restoration["first_step"]["torch_equal"] and restoration["final_latent"]["torch_equal"]
                               and restoration["final_image"]["pixel_exact"])
    old_dir = root/"02_layout_control/outputs/lms_scheduler_alignment_full_generation"
    historical_images = {}
    for name in ["baseline", "guided"]:
        old_pixels = np.asarray(Image.open(old_dir/f"{name}.png"), dtype=np.int16)
        new_pixels = np.asarray(images[name], dtype=np.int16)
        historical_images[name] = {"pixel_exact": bool(np.array_equal(old_pixels, new_pixels)),
                                  "mean_abs_error": float(np.abs(old_pixels-new_pixels).mean()),
                                  "max_abs_error": int(np.abs(old_pixels-new_pixels).max())} if old_pixels.shape == new_pixels.shape else {"shape_mismatch": True}
    metrics = {"restoration": restoration,
               "guidance": {"iterations": len(inner_rows), "all_finite": all(r["finite"] for r in inner_rows),
                            "all_gradients_nonzero": all(r["nonzero_gradient"] for r in inner_rows),
                            "gradient_norm_range": [min(r["gradient_norm"] for r in inner_rows), max(r["gradient_norm"] for r in inner_rows)],
                            "update_norm_range": [min(r["update_norm"] for r in inner_rows), max(r["update_norm"] for r in inner_rows)],
                            "max_relative_update": max(r["relative_update"] for r in inner_rows),
                            "model_parameter_grad_present": parameter_grad_present, "all_parameters_frozen": all_parameters_frozen,
                            "processors_restored": sampler.processors_restored(), "final_latent_vs_baseline": diff_metrics(baseline, guided),
                            "image_mean_abs_delta": float(np.abs(base_pixels-guided_pixels).mean()),
                            "image_max_abs_delta": int(np.abs(base_pixels-guided_pixels).max()),
                            "different_pixel_ratio": float(np.any(base_pixels != guided_pixels, axis=-1).mean())},
               "historical_image_comparison": historical_images, "timings": timings,
               "total_seconds": time.perf_counter()-total_start,
               "peak_allocated_mib": max(t["peak_allocated_mib"] for t in timings.values()),
               "peak_reserved_mib": max(t["peak_reserved_mib"] for t in timings.values())}
    dump_json(out/"metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    if not restoration["passed"]:
        raise RuntimeError("Continuation is not exactly equal; see recorded errors")


if __name__ == "__main__":
    main()
