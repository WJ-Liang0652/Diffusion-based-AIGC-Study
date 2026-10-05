"""Strength sweep and trajectory logging for the frozen Stage-6E configuration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import diffusers
import matplotlib
import torch
from diffusers import FluxPipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
STAGE6E_PATH = Path(__file__).with_name("19_flux_dev_tight_bbox_guidance.py")
EXPLICIT_METRICS_PATH = (
    PROJECT_DIR / "outputs" / "explicit_sampling" / "flux_dev_explicit_sampling_metrics.json"
)
CANONICAL_METRICS_PATH = (
    PROJECT_DIR
    / "outputs"
    / "tight_bbox_guidance"
    / "flux_dev_tight_bbox_block-18_eta-scale-0p25_metrics.json"
)
OUTPUT_ROOT = PROJECT_DIR / "outputs" / "strength_trajectory"
PLANNED_ETA_SCALES = (0.0, 0.125, 0.25, 0.5)
TRAJECTORY_EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta-scale", type=float, required=True, choices=PLANNED_ETA_SCALES)
    return parser.parse_args()


def scale_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_stage6e():
    spec = importlib.util.spec_from_file_location("flux_dev_tight_bbox_guidance", STAGE6E_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Stage-6E helpers from {STAGE6E_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu_tensor.view(torch.uint16).numpy().tobytes()).hexdigest()


def read_trajectory(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        return [
            {
                "step_index": int(row["step_index"]),
                "relative_deviation": float(row["relative_deviation"]),
            }
            for row in csv.DictReader(handle)
        ]


def maybe_write_summary() -> None:
    runs = []
    for scale in PLANNED_ETA_SCALES:
        run_dir = OUTPUT_ROOT / f"eta_scale_{scale_label(scale)}"
        metrics_path = run_dir / "metrics.json"
        trajectory_path = run_dir / "trajectory.csv"
        if not metrics_path.exists() or not trajectory_path.exists():
            return
        metrics = json.loads(metrics_path.read_text())
        trajectory = read_trajectory(trajectory_path)
        deviations = [item["relative_deviation"] for item in trajectory]
        runs.append(
            {
                "eta_scale": scale,
                "eta": float(metrics["eta"]),
                "max_D_t": max(deviations),
                "mean_D_t": sum(deviations) / len(deviations),
                "final_D_t": deviations[-1],
                "runtime": float(metrics["guided_runtime_seconds"]),
                "total_runtime_including_load": float(metrics["total_runtime_seconds_including_load"]),
                "trajectory": trajectory,
            }
        )

    summary_rows = [{key: value for key, value in run.items() if key != "trajectory"} for run in runs]
    summary_csv = OUTPUT_ROOT / "summary.csv"
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (OUTPUT_ROOT / "summary.json").write_text(json.dumps(summary_rows, indent=2) + "\n")

    figure, axis = plt.subplots(figsize=(8, 5))
    for run in runs:
        axis.plot(
            [item["step_index"] for item in run["trajectory"]],
            [item["relative_deviation"] for item in run["trajectory"]],
            label=f"eta_scale={run['eta_scale']:g}",
        )
    axis.set_xlabel("Denoising step index")
    axis.set_ylabel("Trajectory deviation $D_t$")
    axis.set_title("Trajectory deviation by denoising step")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "trajectory_deviation_by_step.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(
        [run["eta_scale"] for run in runs],
        [run["final_D_t"] for run in runs],
        marker="o",
    )
    axis.set_xlabel("Eta scale")
    axis.set_ylabel("Final trajectory deviation $D_t$")
    axis.set_title("Guidance strength vs final trajectory deviation")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_ROOT / "strength_vs_deviation.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    eta_scale = float(args.eta_scale)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the strength trajectory sweep")
    for required in (EXPLICIT_METRICS_PATH, CANONICAL_METRICS_PATH):
        if not required.exists():
            raise FileNotFoundError(f"Required reference missing: {required}")

    stage6e = load_stage6e()
    stage6d = stage6e.load_stage6d()
    stage5 = stage6d.load_stage5()
    stage4 = stage5.load_stage4()
    stage3 = stage4.load_stage3()
    helpers = stage3.load_helpers()
    stage6e.configure_tight_layout(stage3)

    total_start = time.perf_counter()
    baseline_state, baseline_metrics = helpers.load_baseline_state()
    initial_hash = tensor_sha256(baseline_state["initial_latents"])
    if initial_hash != baseline_metrics["initial_latents_sha256"]:
        raise RuntimeError("Initial latent hash does not match the saved baseline hash")
    if not torch.equal(
        baseline_state["pipeline_final_latents"], baseline_state["pipeline_step_latents"][-1]
    ):
        raise RuntimeError("Saved baseline final latent does not match its final trajectory state")

    explicit_metrics = json.loads(EXPLICIT_METRICS_PATH.read_text())
    expected_timesteps = torch.tensor(explicit_metrics["timesteps"], dtype=torch.float32)
    if len(expected_timesteps) != helpers.NUM_INFERENCE_STEPS:
        raise RuntimeError("Saved timestep reference has the wrong length")
    canonical_metrics = json.loads(CANONICAL_METRICS_PATH.read_text())
    base_eta = float(canonical_metrics["base_eta"])
    eta = base_eta * eta_scale

    pipe = FluxPipeline.from_pretrained(
        helpers.MODEL_ID, torch_dtype=helpers.DTYPE, local_files_only=True
    )
    target_spans, _ = helpers.locate_target_spans(pipe.tokenizer_2, helpers.PROMPT)
    for component_name in ("text_encoder", "text_encoder_2", "transformer", "vae"):
        component = getattr(pipe, component_name, None)
        if component is not None:
            component.requires_grad_(False)
    pipe.transformer.enable_gradient_checkpointing()
    pipe.enable_sequential_cpu_offload()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    state = stage4.prepare_probe_state(pipe, helpers, baseline_state)

    trajectory = []
    final_latents, image, inner_records, step_summaries, guided_runtime = stage6d.run_guided(
        pipe,
        helpers,
        stage3,
        stage4,
        stage5,
        state,
        baseline_state,
        stage6e.BLOCKS,
        target_spans,
        eta=eta,
        trajectory_reference=baseline_state["pipeline_step_latents"],
        expected_timesteps=expected_timesteps,
        trajectory_records=trajectory,
        trajectory_eps=TRAJECTORY_EPS,
        require_exact_trajectory=eta_scale == 0.0,
    )
    if eta_scale == 0.0 and (inner_records or step_summaries):
        raise RuntimeError("eta_scale=0 unexpectedly executed backward guidance")
    if len(trajectory) != helpers.NUM_INFERENCE_STEPS:
        raise RuntimeError(f"Expected 50 trajectory records, got {len(trajectory)}")

    final_reference = baseline_state["pipeline_final_latents"].to(final_latents.device)
    final_error = (final_latents.float() - final_reference.float()).abs()
    final_exact = bool(torch.equal(final_latents.detach().cpu(), final_reference.detach().cpu()))
    if eta_scale == 0.0 and not final_exact:
        raise RuntimeError("Final eta_scale=0 latent does not exactly match the baseline")

    output_dir = OUTPUT_ROOT / f"eta_scale_{scale_label(eta_scale)}"
    image_path = output_dir / "final.png"
    trajectory_path = output_dir / "trajectory.csv"
    metrics_path = output_dir / "metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    with trajectory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(trajectory[0]))
        writer.writeheader()
        writer.writerows(trajectory)

    deviations = [item["relative_deviation"] for item in trajectory]
    record = {
        "stage": "Stage 2 - guidance strength trajectory sweep",
        "model_id": helpers.MODEL_ID,
        "diffusers_version": diffusers.__version__,
        "torch_version": torch.__version__,
        "prompt": helpers.PROMPT,
        "seed": helpers.SEED,
        "width": helpers.WIDTH,
        "height": helpers.HEIGHT,
        "num_inference_steps": helpers.NUM_INFERENCE_STEPS,
        "guidance_scale": helpers.GUIDANCE_SCALE,
        "scheduler_class": type(pipe.scheduler).__name__,
        "blocks": list(stage6e.BLOCKS),
        "guided_indices": list(stage6d.GUIDED_INDICES),
        "planned_eta_scales": list(PLANNED_ETA_SCALES),
        "eta_scale": eta_scale,
        "base_eta": base_eta,
        "eta": eta,
        "trajectory_eps": TRAJECTORY_EPS,
        "initial_latents_sha256": initial_hash,
        "expected_initial_latents_sha256": baseline_metrics["initial_latents_sha256"],
        "timestep_tensor_exact_match": True,
        "all_post_step_latents_exact_match": all(item["exact_match"] for item in trajectory),
        "final_latent_exact_match": final_exact,
        "max_relative_deviation": max(deviations),
        "mean_relative_deviation": sum(deviations) / len(deviations),
        "final_relative_deviation": deviations[-1],
        "final_latent_max_abs_error": float(final_error.max().item()),
        "final_latent_mean_abs_error": float(final_error.mean().item()),
        "backward_guidance_update_count": len(inner_records),
        "trajectory_path": str(trajectory_path),
        "image_path": str(image_path),
        "guided_runtime_seconds": guided_runtime,
        "total_runtime_seconds_including_load": time.perf_counter() - total_start,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    metrics_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))

    pipe.maybe_free_model_hooks()
    torch.cuda.empty_cache()
    maybe_write_summary()


if __name__ == "__main__":
    main()
