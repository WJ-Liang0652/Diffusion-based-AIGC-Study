"""Summarize existing Stage 1-7 metrics for the group meeting.

This script is CPU/read-only with respect to experiment artifacts. It does not
run inference or alter any historical metric file.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_DIR / "outputs"
EVAL_DIR = OUTPUTS / "evaluation"
NOTES_DIR = PROJECT_DIR / "notes"


def load(relative: str):
    return json.loads((OUTPUTS / relative).read_text())


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def quantitative_rows():
    baseline42 = load("baseline/flux_dev_baseline_metrics.json")
    tight42 = load("tight_bbox_guidance/flux_dev_tight_bbox_block-18_eta-scale-0p25_metrics.json")
    stage7 = load("repeatability/stage7_repeatability_summary.json")
    ratio42 = tight42["final_target_box_check"]["box_inside_attention_ratio"]
    rows = [
        {
            "case": "apple_cup_seed42",
            "prompt": tight42["prompt"],
            "seed": tight42["seed"],
            "object_1": "apple",
            "object_1_box_inside_ratio": ratio42["apple"],
            "object_2": "cup",
            "object_2_box_inside_ratio": ratio42["cup"],
            "coarse_relocation_success": True,
            "baseline_runtime_seconds": baseline42["runtime_seconds"],
            "guided_runtime_seconds": tight42["guided_runtime_seconds"],
        }
    ]
    for case in stage7["cases"]:
        case_dir = Path(case["comparison_path"]).parent
        baseline = json.loads((case_dir / "baseline_metrics.json").read_text())
        objects = list(case["final_box_inside_ratios"])
        rows.append(
            {
                "case": case["case"],
                "prompt": case["prompt"],
                "seed": case["seed"],
                "object_1": objects[0],
                "object_1_box_inside_ratio": case["final_box_inside_ratios"][objects[0]],
                "object_2": objects[1],
                "object_2_box_inside_ratio": case["final_box_inside_ratios"][objects[1]],
                "coarse_relocation_success": bool(case["layout_success"]),
                "baseline_runtime_seconds": baseline["runtime_seconds"],
                "guided_runtime_seconds": case["guided_runtime_seconds"],
            }
        )
    for row in rows:
        row["runtime_multiplier"] = (
            row["guided_runtime_seconds"] / row["baseline_runtime_seconds"]
        )
    return rows


def write_quantitative(rows):
    csv_path = EVAL_DIR / "tight_box_quantitative_summary.csv"
    fields = list(rows[0])
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    ratios = [
        row[key]
        for row in rows
        for key in ("object_1_box_inside_ratio", "object_2_box_inside_ratio")
    ]
    stage7 = rows[1:]
    mean_overhead_seconds = sum(
        row["guided_runtime_seconds"] - row["baseline_runtime_seconds"] for row in stage7
    ) / len(stage7)
    mean_multiplier = sum(row["runtime_multiplier"] for row in stage7) / len(stage7)
    mean_overhead_percent = (mean_multiplier - 1.0) * 100.0
    relocation_count = sum(row["coarse_relocation_success"] for row in rows)

    header = (
        "| Case | Prompt | Seed | Object 1 ratio | Object 2 ratio | Coarse relocation | "
        "Baseline (s) | Guided (s) | Multiplier |\n"
        "|---|---|---:|---:|---:|---|---:|---:|---:|\n"
    )
    body = "".join(
        f"| {r['case']} | {r['prompt']} | {r['seed']} | "
        f"{r['object_1']} {fmt(r['object_1_box_inside_ratio'], 4)} | "
        f"{r['object_2']} {fmt(r['object_2_box_inside_ratio'], 4)} | "
        f"{'yes' if r['coarse_relocation_success'] else 'no'} | "
        f"{fmt(r['baseline_runtime_seconds'], 2)} | {fmt(r['guided_runtime_seconds'], 2)} | "
        f"{fmt(r['runtime_multiplier'], 3)}x |\n"
        for r in rows
    )
    markdown = f"""# Tight-box quantitative summary

All values are from existing Stage 1–7 artifacts. `box-inside ratio` means final block-18 target-box attention-mass ratio; it is not instance-mask IoU or a detector metric. `Coarse relocation` is the recorded visual judgement that both object centers/main bodies moved into the intended swapped regions.

{header}{body}
## Aggregate

- Eight-object mean box-inside ratio: **{fmt(sum(ratios) / len(ratios), 4)}**.
- Eight-object minimum box-inside ratio: **{fmt(min(ratios), 4)}** (bottle, seed 31415).
- Preliminary tested-case relocation count: **{relocation_count}/{len(rows)}**. This is not a formal benchmark success rate.
- Stage 7 three-case mean runtime overhead: **+{fmt(mean_overhead_seconds, 2)} s**, mean multiplier **{fmt(mean_multiplier, 3)}x**, or **+{fmt(mean_overhead_percent, 2)}%** versus each matching exact baseline.
- Four-case mean runtime multiplier (descriptive only): **{fmt(sum(r['runtime_multiplier'] for r in rows) / len(rows), 3)}x**.
"""
    md_path = EVAL_DIR / "tight_box_quantitative_summary.md"
    md_path.write_text(markdown)
    return {
        "csv_path": csv_path,
        "md_path": md_path,
        "mean_ratio": sum(ratios) / len(ratios),
        "min_ratio": min(ratios),
        "relocation_count": relocation_count,
        "mean_overhead_seconds": mean_overhead_seconds,
        "mean_multiplier": mean_multiplier,
        "mean_overhead_percent": mean_overhead_percent,
    }


def write_ablation():
    stage5 = load("single_step_guidance/flux_dev_single_step_guidance_rel-0p005_metrics.json")
    stage6a = load("inner_loop_guidance/flux_dev_inner_loop_k-1-3-5_rel-0p005_metrics.json")
    stage6b = load("multi_timestep_guidance/flux_dev_multi_timestep_indices-20-28-35_rel-0p005_metrics.json")
    stage6b_alt = load("multi_timestep_guidance/flux_dev_multi_timestep_indices-12-24-35_rel-0p005_metrics.json")
    stage6c = load("paper_aligned_guidance/flux_dev_paper_aligned_blocks-18_metrics.json")
    strength = load("paper_aligned_guidance/flux_dev_stage6d_strength_summary.json")
    half = load("paper_aligned_guidance/flux_dev_paper_aligned_blocks-18_eta-scale-0p25_metrics.json")
    tight = load("tight_bbox_guidance/flux_dev_tight_bbox_block-18_eta-scale-0p25_metrics.json")
    objective = load("layout_objective/flux_dev_layout_objective_metrics.json")
    spatial = load("spatial_validation/flux_dev_spatial_signal_validation_metrics.json")
    gradient = load("gradient_probe/flux_dev_gradient_probe_metrics.json")

    k_results = {r["inner_loop_count"]: r for r in stage6a["guided_results"]}
    s6b_obj = " / ".join(
        f"{r['objective_before']:.3f}->{r['objective_after']:.3f}" for r in stage6b["interventions"]
    )
    s6b_alt_obj = " / ".join(
        f"{r['objective_before']:.3f}->{r['objective_after']:.3f}" for r in stage6b_alt["interventions"]
    )
    first6c = stage6c["inner_updates"][0]

    timing_rows = [
        (
            "Stage 5: late single-step",
            "index 35, t=499.842; one normalized 0.005 update",
            f"E {stage5['layout_objective']['before']['total_energy']:.4f}->{stage5['layout_objective']['after']['total_energy']:.4f}; COM-x apple {stage5['attention_movement']['objects']['apple']['center_of_mass_delta_xy'][0]:+.3f}, cup {stage5['attention_movement']['objects']['cup']['center_of_mass_delta_xy'][0]:+.3f} tokens",
            "No visible relocation; only local pixel/texture change.",
        ),
        (
            "Stage 6A: same-timestep inner loop",
            "index 35; K=1/3/5; normalized 0.005 each update",
            "Final E " + "/".join(f"{k_results[k]['layout_after_final_update']['total_energy']:.4f}" for k in (1, 3, 5)) + "; K=5 COM-x apple +2.783, cup -2.305 tokens",
            "Monotonic objective/attention movement, but no visible relocation at K<=5.",
        ),
        (
            "Stage 6B: sparse multi-timestep",
            "K=1 at indices 20/28/35; alternate 12/24/35",
            f"Main E {s6b_obj}; alternate E {s6b_alt_obj}",
            "All interventions stable and directional; neither schedule visibly relocated objects.",
        ),
        (
            "Stage 6C: early sigma^2 guidance",
            "indices 0-9; x <- x - eta*sigma^2*grad(E); eta=96.6173",
            f"Index 0 E {first6c['objective_before']:.4f}->{first6c['objective_after']:.4f}; COM-x apple {first6c['blocks']['18']['objects']['apple']['attention_centroid_delta_xy'][0]:+.3f}, cup {first6c['blocks']['18']['objects']['cup']['attention_centroid_delta_xy'][0]:+.3f} tokens",
            "First clear visible apple/cup swap; strongest fidelity damage.",
        ),
    ]

    strength_rows = []
    for scale in ("1.0", "0.5", "0.25"):
        v = strength["variants"][scale]
        strength_rows.append(
            (
                scale + "x",
                f"{v['eta']:.4f}",
                str(len(v["per_update_relative_norms"])),
                f"{v['pixel_mae_vs_baseline']:.2f} / {v['pixel_rmse_vs_baseline']:.2f}",
                f"{v['guided_runtime_seconds']:.2f}",
                v["manual_visual_assessment"],
            )
        )

    half_diff = half["guided"]["pixel_difference_vs_baseline"]
    tight_diff = tight["guided"]["pixel_difference_vs_baseline"]
    tight_ratios = tight["final_target_box_check"]["box_inside_attention_ratio"]
    region_rows = [
        (
            "Half regions, eta 0.25x",
            "apple right half; cup left half",
            f"MAE/RMSE {half_diff['mean_absolute_pixel_difference']:.2f}/{half_diff['rmse']:.2f}; final COM-x {half['step_summaries'][-1]['final_attention_centroids']['18']['apple'][0]:.2f}/{half['step_summaries'][-1]['final_attention_centroids']['18']['cup'][0]:.2f}",
            "Clear swap, but scale inflation and major background/identity drift.",
        ),
        (
            "Tight boxes, eta 0.25x",
            "apple [27,23,46,44); cup [4,20,27,44)",
            f"MAE/RMSE {tight_diff['mean_absolute_pixel_difference']:.2f}/{tight_diff['rmse']:.2f}; box ratios {tight_ratios['apple']:.4f}/{tight_ratios['cup']:.4f}",
            "Clear swap with improved extent/scale; cup identity and background still drift.",
        ),
    ]

    spatial35 = {
        r["block_index"]: r
        for r in spatial["configuration_summaries"]
        if r["denoising_index"] == 35 and r["block_index"] in (9, 18)
    }
    block_rows = []
    for block in (9, 18):
        obj = objective["blocks"][str(block)]
        sp = spatial35[block]["joint_statistics"]
        grad = gradient["blocks"][str(block)]
        block_rows.append(
            (
                str(block),
                f"{sp['apple']['reference_box_enrichment_over_uniform']:.3f}x / {sp['cup']['reference_box_enrichment_over_uniform']:.3f}x",
                f"{obj['layout_a']['total_energy']:.4f} / {obj['layout_b']['total_energy']:.4f}",
                f"{obj['comparison']['energy_margin_b_minus_a']:.4f} ({obj['comparison']['energy_ratio_b_over_a']:.2f}x)",
                f"{grad['gradient_norm_l2']:.5f}",
                "Useful mechanism-level spatial signal; block 18 gives the larger objective margin and gradient norm." if block == 18 else "Useful mechanism-level control; retained as comparison block.",
            )
        )

    def table(headers, rows):
        return "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n" + "".join(
            "| " + " | ".join(str(value) for value in row) + " |\n" for row in rows
        )

    markdown = """# Preliminary ablation from existing experiments

No new sweep was run. These comparisons reuse Stage 1–7 outputs and are preliminary, mostly single-prompt/single-seed. Timing/strategy changes are not a strictly isolated ablation because the update parameterization also changes at Stage 6C.

## 1. Guidance timing and strategy

""" + table(
        ["Variant", "Existing configuration", "Key measurement", "Conclusion"], timing_rows
    ) + """

The existing evidence identifies early denoising as decisive for visible geometry: late and sparse-middle interventions move the attention objective without propagating to final object position, whereas early sigma-squared guidance produces a visible swap.

## 2. Eta strength

All variants use the same seed-42 half-region task, block 18, indices 0–9, sigma-squared update, K<=5, and threshold 0.2.

""" + table(
        ["Scale", "Eta", "Updates", "Pixel MAE / RMSE", "Guided runtime (s)", "Existing visual conclusion"],
        strength_rows,
    ) + """

All three strengths retain relocation. Lower eta improves the measured and visible fidelity tradeoff, while the fixed threshold requires more updates and runtime. `0.25x` is the tested low-strength choice used by Stage 6E/7, not a globally optimized value.

## 3. Region definition

This is the cleanest existing comparison: seed, latent, eta 0.25x, block, timing, threshold, and objective are held fixed; only the target region changes.

""" + table(
        ["Region", "Definition", "Key measurement", "Conclusion"], region_rows
    ) + f"""

Tight boxes reduce pixel MAE by **{half_diff['mean_absolute_pixel_difference'] - tight_diff['mean_absolute_pixel_difference']:.2f}** and RMSE by **{half_diff['rmse'] - tight_diff['rmse']:.2f}** versus half regions, while preserving relocation. They improve scale/extent but do not solve semantic or background preservation.

## 4. Block 9 versus Block 18 (mechanism-level only)

Values are from the existing index-35 (`t=499.842`) spatial-signal, layout-objective, and gradient probes.

""" + table(
        ["Double block", "Observed-box enrichment apple/cup", "Observed A / swapped B energy", "B-A margin (ratio)", "Gradient L2", "Conclusion"],
        block_rows,
    ) + """

This is **not a complete end-to-end block ablation**: no matched full Backward Guidance generation was run with block 9. Both blocks encode useful spatial separation, while block 18 provides the stronger objective margin and gradient in the existing probe and is therefore the current guidance block.
"""
    path = EVAL_DIR / "preliminary_ablation.md"
    path.write_text(markdown)
    return path


def write_group_summary(rows, quantitative, ablation_path):
    q_table = (EVAL_DIR / "tight_box_quantitative_summary.md").read_text().split("## Aggregate")[0]
    q_table = q_table[q_table.index("| Case") :].rstrip()
    summary = f"""# FLUX Layout Control — Group Meeting Summary

## Final method/configuration used in the current tight-box checkpoint

- Model: `black-forest-labs/FLUX.1-dev`; 768x768, 50 FlowMatch steps, guidance scale 3.5, BF16, batch size 1, sequential CPU offload.
- Signal/objective: Double Stream block 18 native joint attention, image-query to target-text-key probability; Eq. (2) target-box attention-mass energy.
- Backward Guidance: early denoising indices 0–9, `x_t <- x_t - eta * sigma_t^2 * grad_x E`, `eta_scale=0.25` (`eta=24.154335`), K<=5, objective threshold 0.2.
- Only packed latents receive gradients. Text encoders/VAE do not backpropagate and model parameters remain frozen.
- Target regions are finite object-sized boxes. This is the frozen Stage 6E/7 checkpoint configuration, not a globally finalized optimum.

## Basic quantitative evaluation

{q_table}

Aggregate: eight-object mean/min box-inside attention ratio = **{quantitative['mean_ratio']:.4f}/{quantitative['min_ratio']:.4f}**. Coarse relocation is observed in **{quantitative['relocation_count']}/{len(rows)} preliminary tested cases**; this is not a formal benchmark success rate. Stage 7 mean runtime overhead is **+{quantitative['mean_overhead_seconds']:.2f} s**, or **{quantitative['mean_multiplier']:.3f}x (+{quantitative['mean_overhead_percent']:.2f}%)** versus matching baselines.

## Preliminary ablation summary

| Axis | Existing evidence | Preliminary conclusion |
|---|---|---|
| Timing/strategy | Stage 5 late single update and Stage 6A/6B late/sparse updates reduce energy and move attention but do not visibly relocate objects; Stage 6C early sigma-squared guidance visibly swaps them. | Early high-noise intervention is the decisive existing factor for geometry propagation. |
| Eta | 1.0x/0.5x/0.25x all swap; MAE falls 94.89 -> 66.71 -> 61.98, while runtime rises 262.83 -> 302.85 -> 334.95 s. | 0.25x gives the best tested half-region fidelity/control tradeoff, but is not globally tuned. |
| Region | Tight boxes versus 0.25x half regions reduce MAE 61.98 -> 53.46 and RMSE 77.90 -> 74.52. | Tight boxes reduce scale inflation and improve extent; identity/background drift remains. |
| Block | At index 35, block 9/18 observed-vs-swapped energy margin is 1.2713/1.3435; gradient L2 is 0.03748/0.06371. | Both are useful; block 18 is stronger in existing mechanism probes. This is not an end-to-end block ablation. |

Full extracted tables: [preliminary_ablation.md](../outputs/evaluation/{ablation_path.name}).

## Main findings

1. Native FLUX joint attention supplies a differentiable text-to-image spatial signal; block 18 is the strongest current probe target, with block 9 also spatially meaningful.
2. Objective reduction or attention-centroid movement alone is insufficient: late single-step, same-timestep inner-loop, and sparse multi-timestep strategies did not yield visible relocation.
3. Early sigma-squared Backward Guidance is the first existing strategy that consistently changes object geometry and produces the requested swap.
4. The frozen tight-box method relocates both objects in all four preliminary tested cases; mean/min attention box-inside ratios are {quantitative['mean_ratio']:.4f}/{quantitative['min_ratio']:.4f}.
5. Tight boxes and lower eta improve fidelity, but background/composition and object identity/shape drift remain, and cross-prompt compute cost is variable.

## Limitations

- Four cases, two prompts, and four seeds are a sanity set, not a benchmark; no confidence interval or statistical success rate is justified.
- Box-inside values measure attention mass, not detected/segmented visible-object IoU. All four cases have some visible object extent outside the tight targets.
- Manual coarse-relocation and quality judgements are not yet backed by detector, segmentation, CLIP/DINO, or human-rater metrics.
- Eta and region comparisons are seed-42/apple-cup only. The timing comparison is historically sequential rather than fully factorial and controlled.
- Block 9 versus 18 is mechanism-level only; there is no matched block-9 full-generation guidance run.
- Fidelity remains unresolved: backgrounds change, cup identity can drift, and the banana/bottle case changes instance count and bottle material/shape.

## Next work

- Do not add a new algorithm branch before the group-meeting checkpoint. First define a representative evaluation protocol with object localization/segmentation, semantic preservation, image quality, runtime, and VRAM metrics.
- After agreeing on that protocol, run controlled single-variable ablations for timing, block, eta, K/threshold, and region definition across multiple prompts/seeds.
- Treat fidelity preservation as the next method problem only after the basic evaluation protocol is fixed. Forward Guidance remains deferred.
"""
    path = NOTES_DIR / "group_meeting_summary.md"
    path.write_text(summary)
    return path


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = quantitative_rows()
    quantitative = write_quantitative(rows)
    ablation = write_ablation()
    group = write_group_summary(rows, quantitative, ablation)
    print(json.dumps({
        "generated": [
            str(quantitative["csv_path"]),
            str(quantitative["md_path"]),
            str(ablation),
            str(group),
        ],
        "mean_box_inside_ratio": quantitative["mean_ratio"],
        "min_box_inside_ratio": quantitative["min_ratio"],
        "preliminary_relocation_count": f"{quantitative['relocation_count']}/{len(rows)}",
        "stage7_mean_runtime_overhead_seconds": quantitative["mean_overhead_seconds"],
        "stage7_mean_runtime_multiplier": quantitative["mean_multiplier"],
    }, indent=2))


if __name__ == "__main__":
    main()
