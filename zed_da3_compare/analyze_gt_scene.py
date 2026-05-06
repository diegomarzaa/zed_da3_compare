#!/usr/bin/env python3
"""Analyze GT ROI capture scenes produced by gt_annotation_tool.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zed_da3_compare.depth_metrics import (
    joint_valid_mask,
    median_scale_to_reference,
    pairwise_depth_metrics,
    resize_depth_to_shape,
    supervised_depth_metrics,
    valid_depth_mask,
)


METHODS = {
    "zed": "zed_depth.npy",
    "da3_mono": "da3_mono_depth.npy",
    "da3_multiview": "da3_multiview_depth.npy",
}

DA3_METHODS = {
    "da3_mono": "da3_mono_depth.npy",
    "da3_multiview": "da3_multiview_depth.npy",
}


def main() -> None:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser()
    analysis_dir = scene_dir / "analysis"
    plots_dir = analysis_dir / "plots"
    visuals_dir = analysis_dir / "visuals"
    plots_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = scene_dir / "scene_annotations.csv"
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations file: {annotations_path}")

    annotations = pd.read_csv(annotations_path)
    for col in ["gt_distance_m", "x1", "y1", "x2", "y2"]:
        annotations[col] = pd.to_numeric(annotations[col], errors="coerce")

    long, wide = recompute_metrics(scene_dir, annotations)
    existing_long = long[long["sample_exists"] & long["has_depth"] & np.isfinite(long["pred_m"])].copy()
    existing_wide = wide[wide["sample_exists"]].copy()
    zed_ref_roi = build_zed_reference_roi_metrics(existing_wide)
    zed_ref_samples = build_zed_reference_sample_metrics(scene_dir)

    long.to_csv(analysis_dir / "roi_metrics_long.csv", index=False)
    wide.to_csv(analysis_dir / "roi_metrics_wide.csv", index=False)
    existing_long.to_csv(analysis_dir / "roi_metrics_existing_long.csv", index=False)
    existing_wide.to_csv(analysis_dir / "roi_metrics_existing_wide.csv", index=False)
    zed_ref_roi.to_csv(analysis_dir / "zed_reference_roi_metrics.csv", index=False)
    zed_ref_samples.to_csv(analysis_dir / "zed_reference_sample_metrics.csv", index=False)

    summary = summarize_by_method(existing_long)
    summary.to_csv(analysis_dir / "summary_by_method.csv", index=False)

    if existing_wide.empty:
        winner_counts = pd.DataFrame(columns=["method", "wins"])
    else:
        winner_counts = existing_wide["winner_recomputed"].value_counts().rename_axis("method").reset_index(name="wins")
    winner_counts.to_csv(analysis_dir / "winner_counts.csv", index=False)

    write_plots(existing_long, existing_wide, winner_counts, zed_ref_roi, zed_ref_samples, plots_dir)
    write_visual_overlays(scene_dir, wide, visuals_dir)
    write_report(scene_dir, analysis_dir, annotations, existing_wide, summary, winner_counts, zed_ref_roi, zed_ref_samples)

    print(f"analysis_dir={analysis_dir}")
    if summary.empty:
        print("No valid depth metrics found.")
    else:
        print(summary.to_string(index=False))
    print("winner_counts")
    print(winner_counts.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_dir", help="Path to captures/<scene_name>")
    return parser.parse_args()


def recompute_metrics(scene_dir: Path, annotations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    wide_rows: list[dict] = []

    for _, ann in annotations.iterrows():
        sample_id = str(ann["sample_id"])
        sample_dir = scene_dir / sample_id
        sample_exists = sample_dir.is_dir()
        base = {
            "scene": ann.get("scene", scene_dir.name),
            "sample_id": sample_id,
            "sample_exists": sample_exists,
            "object_id": ann.get("object_id", ""),
            "object_name": ann.get("object_name", ""),
            "gt_distance_m": ann["gt_distance_m"],
            "x1": ann["x1"],
            "y1": ann["y1"],
            "x2": ann["x2"],
            "y2": ann["y2"],
            "notes": ann.get("notes", ""),
        }
        wide = dict(base)
        errors: dict[str, float] = {}

        for method, filename in METHODS.items():
            depth_path = sample_dir / filename
            has_depth = sample_exists and depth_path.exists()
            stats = empty_stats()
            if has_depth and np.isfinite(ann["x1"]):
                depth = np.load(depth_path)
                stats = roi_stats(depth, ann["x1"], ann["y1"], ann["x2"], ann["y2"])

            pred = stats["median"]
            abs_err = abs(pred - ann["gt_distance_m"]) if np.isfinite(pred) and np.isfinite(ann["gt_distance_m"]) else np.nan
            rel_err = abs_err / ann["gt_distance_m"] if np.isfinite(abs_err) and ann["gt_distance_m"] > 0 else np.nan
            errors[method] = abs_err

            rows.append(
                {
                    **base,
                    "method": method,
                    "has_depth": has_depth,
                    "pred_m": pred,
                    "mean_m": stats["mean"],
                    "std_m": stats["std"],
                    "p10_m": stats["p10"],
                    "p90_m": stats["p90"],
                    "valid_px": stats["valid_px"],
                    "total_px": stats["total_px"],
                    "valid_ratio": stats["valid_ratio"],
                    "abs_error_m": abs_err,
                    "rel_error": rel_err,
                }
            )
            wide[f"{method}_median_m"] = pred
            wide[f"{method}_mean_m"] = stats["mean"]
            wide[f"{method}_abs_error_m"] = abs_err
            wide[f"{method}_rel_error"] = rel_err
            wide[f"{method}_valid_ratio"] = stats["valid_ratio"]

        finite_errors = {name: value for name, value in errors.items() if np.isfinite(value)}
        wide["winner_recomputed"] = min(finite_errors, key=finite_errors.get) if finite_errors else "none"
        wide_rows.append(wide)

    return pd.DataFrame(rows), pd.DataFrame(wide_rows)


def empty_stats() -> dict[str, float | int]:
    return {
        "median": np.nan,
        "mean": np.nan,
        "std": np.nan,
        "p10": np.nan,
        "p90": np.nan,
        "valid_px": 0,
        "total_px": 0,
        "valid_ratio": np.nan,
    }


def roi_stats(depth: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> dict[str, float | int]:
    h, w = depth.shape[:2]
    ix1 = int(max(0, min(w - 1, round(x1))))
    ix2 = int(max(ix1 + 1, min(w, round(x2))))
    iy1 = int(max(0, min(h - 1, round(y1))))
    iy2 = int(max(iy1 + 1, min(h, round(y2))))
    crop = depth[iy1:iy2, ix1:ix2]
    valid = crop[np.isfinite(crop) & (crop > 0)]
    if valid.size == 0:
        stats = empty_stats()
        stats["total_px"] = int(crop.size)
        stats["valid_ratio"] = 0.0
        return stats
    return {
        "median": float(np.median(valid)),
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "p10": float(np.percentile(valid, 10)),
        "p90": float(np.percentile(valid, 90)),
        "valid_px": int(valid.size),
        "total_px": int(crop.size),
        "valid_ratio": float(valid.size / max(crop.size, 1)),
    }


def summarize_by_method(existing_long: pd.DataFrame) -> pd.DataFrame:
    if existing_long.empty:
        return pd.DataFrame()
    return (
        existing_long.groupby("method")
        .agg(
            n=("abs_error_m", "count"),
            mean_pred_m=("pred_m", "mean"),
            mean_abs_error_m=("abs_error_m", "mean"),
            median_abs_error_m=("abs_error_m", "median"),
            max_abs_error_m=("abs_error_m", "max"),
            mean_rel_error=("rel_error", "mean"),
            median_rel_error=("rel_error", "median"),
            mean_valid_ratio=("valid_ratio", "mean"),
        )
        .reset_index()
    )


def build_zed_reference_roi_metrics(existing_wide: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if existing_wide.empty:
        return pd.DataFrame()

    for _, row in existing_wide.iterrows():
        zed = row.get("zed_median_m", np.nan)
        if not np.isfinite(zed) or zed <= 0:
            continue
        for method in DA3_METHODS:
            pred = row.get(f"{method}_median_m", np.nan)
            if not np.isfinite(pred) or pred <= 0:
                continue
            diff = pred - zed
            abs_diff = abs(diff)
            rows.append(
                {
                    "scene": row.get("scene", ""),
                    "sample_id": row.get("sample_id", ""),
                    "object_id": row.get("object_id", ""),
                    "object_name": row.get("object_name", ""),
                    "method": method,
                    "zed_median_m": zed,
                    "method_median_m": pred,
                    "diff_m": diff,
                    "abs_diff_m": abs_diff,
                    "rel_abs_diff_to_zed": abs_diff / zed,
                    "ratio_method_over_zed": pred / zed,
                    "zed_valid_ratio": row.get("zed_valid_ratio", np.nan),
                    "method_valid_ratio": row.get(f"{method}_valid_ratio", np.nan),
                    "gt_distance_m": row.get("gt_distance_m", np.nan),
                    "notes": row.get("notes", ""),
                }
            )
    return pd.DataFrame(rows)


def build_zed_reference_sample_metrics(scene_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for sample_dir in sorted(scene_dir.glob("sample_*")):
        zed_path = sample_dir / "zed_depth.npy"
        if not zed_path.exists():
            continue
        zed = np.load(zed_path).astype(np.float32)
        zed_valid = valid_depth_mask(zed)
        for method, filename in DA3_METHODS.items():
            depth_path = sample_dir / filename
            if not depth_path.exists():
                continue
            pred = np.load(depth_path).astype(np.float32)
            pred = resize_depth_to_shape(pred, zed.shape[:2])
            mask = joint_valid_mask(zed, pred)
            raw = supervised_depth_metrics(pred, zed, mask)
            pair = pairwise_depth_metrics(zed, pred, mask)
            scale = median_scale_to_reference(pred, zed, mask)
            scaled = supervised_depth_metrics(pred * scale, zed, mask) if np.isfinite(scale) else {}
            pred_valid = valid_depth_mask(pred)
            valid_px = int(mask.sum())
            total_px = int(mask.size)
            row = {
                "sample_id": sample_dir.name,
                "method": method,
                "valid_px": valid_px,
                "total_px": total_px,
                "joint_valid_ratio": valid_px / max(total_px, 1),
                "zed_valid_ratio": float(zed_valid.mean()),
                "method_valid_ratio": float(pred_valid.mean()),
                "zed_median_m": float(np.median(zed[zed_valid])) if zed_valid.any() else np.nan,
                "method_median_m": float(np.median(pred[pred_valid])) if pred_valid.any() else np.nan,
                "median_scale_to_zed": scale,
            }
            row.update({f"raw_{key}": value for key, value in raw.items()})
            row.update({f"pair_{key}": value for key, value in pair.items()})
            row.update({f"scaled_{key}": value for key, value in scaled.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def write_plots(
    existing_long: pd.DataFrame,
    existing_wide: pd.DataFrame,
    winner_counts: pd.DataFrame,
    zed_ref_roi: pd.DataFrame,
    zed_ref_samples: pd.DataFrame,
    plots_dir: Path,
) -> None:
    if existing_long.empty:
        plt.style.use("seaborn-v0_8-whitegrid")
        write_zed_reference_plots(zed_ref_roi, zed_ref_samples, plots_dir)
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"zed": "#1f77b4", "da3_mono": "#ff7f0e", "da3_multiview": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(7, 7))
    max_v = max(float(existing_long["gt_distance_m"].max()), float(existing_long["pred_m"].max())) * 1.08
    ax.plot([0, max_v], [0, max_v], "k--", lw=1, label="perfect")
    for method, group in existing_long.groupby("method"):
        ax.scatter(group["gt_distance_m"], group["pred_m"], label=method, s=70, color=colors.get(method))
        for _, row in group.iterrows():
            ax.annotate(str(row["object_name"])[:12], (row["gt_distance_m"], row["pred_m"]), fontsize=8, alpha=0.75)
    ax.set_xlabel("Ground truth distance (m)")
    ax.set_ylabel("Predicted median ROI depth (m)")
    ax.set_title("GT vs predicted ROI depth")
    ax.set_xlim(0, max_v)
    ax.set_ylim(0, max_v)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "gt_vs_pred_scatter.png", dpi=160)
    plt.close(fig)

    object_order = existing_wide.sort_values("gt_distance_m")["object_name"]
    pivot_abs = existing_long.pivot_table(index="object_name", columns="method", values="abs_error_m", aggfunc="mean")
    pivot_abs = pivot_abs.reindex(object_order)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_abs.plot(kind="bar", ax=ax, color=[colors.get(c) for c in pivot_abs.columns])
    ax.set_ylabel("Absolute error (m)")
    ax.set_title("Absolute error by object")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plots_dir / "abs_error_by_object.png", dpi=160)
    plt.close(fig)

    pivot_rel = existing_long.pivot_table(index="object_name", columns="method", values="rel_error", aggfunc="mean") * 100.0
    pivot_rel = pivot_rel.reindex(object_order)
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_rel.plot(kind="bar", ax=ax, color=[colors.get(c) for c in pivot_rel.columns])
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Relative error by object")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(plots_dir / "relative_error_by_object.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, group in existing_long.groupby("method"):
        ordered = group.sort_values("gt_distance_m")
        ax.plot(ordered["gt_distance_m"], ordered["abs_error_m"], "o-", label=method, color=colors.get(method))
    ax.set_xlabel("Ground truth distance (m)")
    ax.set_ylabel("Absolute error (m)")
    ax.set_title("Error vs distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "error_vs_distance.png", dpi=160)
    plt.close(fig)

    if "zed_abs_error_m" in existing_wide and "da3_mono_abs_error_m" in existing_wide:
        comp = existing_wide[np.isfinite(existing_wide["zed_abs_error_m"]) & np.isfinite(existing_wide["da3_mono_abs_error_m"])].copy()
        if not comp.empty:
            comp["da3_minus_zed_error_m"] = comp["da3_mono_abs_error_m"] - comp["zed_abs_error_m"]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.axhline(0, color="k", lw=1)
            ax.bar(
                comp["object_name"],
                comp["da3_minus_zed_error_m"],
                color=["#d62728" if value > 0 else "#2ca02c" for value in comp["da3_minus_zed_error_m"]],
            )
            ax.set_ylabel("DA3 mono abs error - ZED abs error (m)")
            ax.set_title("Positive means ZED was closer to GT")
            ax.tick_params(axis="x", rotation=35)
            fig.tight_layout()
            fig.savefig(plots_dir / "da3_minus_zed_error.png", dpi=160)
            plt.close(fig)

    if not winner_counts.empty:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(winner_counts["method"], winner_counts["wins"], color=[colors.get(method, "gray") for method in winner_counts["method"]])
        ax.set_ylabel("ROI wins")
        ax.set_title("Winner by absolute error")
        fig.tight_layout()
        fig.savefig(plots_dir / "winner_counts.png", dpi=160)
        plt.close(fig)

    write_zed_reference_plots(zed_ref_roi, zed_ref_samples, plots_dir)


def write_zed_reference_plots(zed_ref_roi: pd.DataFrame, zed_ref_samples: pd.DataFrame, plots_dir: Path) -> None:
    colors = {"da3_mono": "#ff7f0e", "da3_multiview": "#2ca02c"}

    if not zed_ref_roi.empty:
        fig, ax = plt.subplots(figsize=(7, 7))
        max_v = max(float(zed_ref_roi["zed_median_m"].max()), float(zed_ref_roi["method_median_m"].max())) * 1.08
        ax.plot([0, max_v], [0, max_v], "k--", lw=1, label="DA3 = ZED")
        for method, group in zed_ref_roi.groupby("method"):
            ax.scatter(group["zed_median_m"], group["method_median_m"], s=70, label=method, color=colors.get(method))
            for _, row in group.iterrows():
                ax.annotate(str(row["object_name"])[:12], (row["zed_median_m"], row["method_median_m"]), fontsize=8, alpha=0.75)
        ax.set_xlabel("ZED median ROI depth (m)")
        ax.set_ylabel("DA3 median ROI depth (m)")
        ax.set_title("ROI depth: DA3 vs ZED reference")
        ax.set_xlim(0, max_v)
        ax.set_ylim(0, max_v)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "zed_ref_roi_scatter.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_data = zed_ref_roi.sort_values("abs_diff_m", ascending=False)
        labels = plot_data["sample_id"].astype(str) + ":" + plot_data["object_name"].astype(str)
        ax.bar(labels, plot_data["diff_m"], color=["#d62728" if value > 0 else "#1f77b4" for value in plot_data["diff_m"]])
        ax.axhline(0, color="k", lw=1)
        ax.set_ylabel("DA3 median - ZED median (m)")
        ax.set_title("ROI signed difference against ZED")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(plots_dir / "zed_ref_roi_signed_diff.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        zed_ref_roi.boxplot(column="abs_diff_m", by="method", ax=ax)
        ax.set_title("ROI absolute difference against ZED")
        ax.set_xlabel("method")
        ax.set_ylabel("|DA3 - ZED| median depth (m)")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(plots_dir / "zed_ref_roi_abs_diff_boxplot.png", dpi=160)
        plt.close(fig)

    if not zed_ref_samples.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, group in zed_ref_samples.groupby("method"):
            ax.scatter(group["joint_valid_ratio"], group["raw_mae"], s=80, label=method, color=colors.get(method))
        ax.set_xlabel("Joint valid pixel ratio")
        ax.set_ylabel("Raw MAE vs ZED (m)")
        ax.set_title("Per-sample DA3/ZED disagreement")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "zed_ref_sample_mae_vs_validity.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        labels = zed_ref_samples["sample_id"].astype(str) + ":" + zed_ref_samples["method"].astype(str)
        ax.bar(labels, zed_ref_samples["median_scale_to_zed"], color=[colors.get(method, "gray") for method in zed_ref_samples["method"]])
        ax.axhline(1.0, color="k", lw=1)
        ax.set_ylabel("Median scale to ZED")
        ax.set_title("Scale needed to align DA3 median depth to ZED")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(plots_dir / "zed_ref_sample_scale.png", dpi=160)
        plt.close(fig)


def write_visual_overlays(scene_dir: Path, wide: pd.DataFrame, visuals_dir: Path) -> None:
    for sample_dir in sorted(scene_dir.glob("sample_*")):
        left_path = sample_dir / "left.png"
        if not left_path.exists():
            continue
        image = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        sample_rows = wide[wide["sample_id"] == sample_dir.name]
        for _, row in sample_rows.iterrows():
            x1, y1, x2, y2 = [int(row[col]) for col in ["x1", "y1", "x2", "y2"]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (20, 220, 80), 2)
            label = (
                f"{row['object_name']} GT {row['gt_distance_m']:.2f} "
                f"z {row.get('zed_median_m', np.nan):.2f} "
                f"da3 {row.get('da3_mono_median_m', np.nan):.2f}"
            )
            cv2.putText(image, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 220, 80), 1, cv2.LINE_AA)
        cv2.imwrite(str(visuals_dir / f"{sample_dir.name}_roi_overlay.png"), image)


def write_report(
    scene_dir: Path,
    analysis_dir: Path,
    annotations: pd.DataFrame,
    existing_wide: pd.DataFrame,
    summary: pd.DataFrame,
    winner_counts: pd.DataFrame,
    zed_ref_roi: pd.DataFrame,
    zed_ref_samples: pd.DataFrame,
) -> None:
    plots_dir = analysis_dir / "plots"
    visuals_dir = analysis_dir / "visuals"
    sample_dirs = {path.name for path in scene_dir.glob("sample_*") if path.is_dir()}
    missing_samples = sorted(set(annotations["sample_id"].astype(str)) - sample_dirs)

    lines: list[str] = []
    lines.append("# scene_00_tests analysis")
    lines.append("")
    lines.append(f"Annotations rows: {len(annotations)}")
    lines.append(f"Existing sample folders: {len(sample_dirs)}")
    lines.append(f"Rows with existing samples: {int(existing_wide.shape[0])}")
    if missing_samples:
        lines.append(f"Missing/stale samples referenced by scene_annotations.csv: {', '.join(missing_samples)}")
    lines.append("")
    lines.append("## Summary by method")
    lines.append("")
    lines.append(dataframe_to_markdown(summary, floatfmt=".4f") if not summary.empty else "No valid depth metrics found.")
    lines.append("")
    lines.append("## Winner counts")
    lines.append("")
    lines.append(dataframe_to_markdown(winner_counts) if not winner_counts.empty else "No winner data.")
    lines.append("")
    lines.append("## ZED reference summary")
    lines.append("")
    if zed_ref_samples.empty:
        lines.append("No DA3/ZED sample-level comparisons found.")
    else:
        sample_cols = [
            "sample_id",
            "method",
            "joint_valid_ratio",
            "zed_valid_ratio",
            "method_valid_ratio",
            "raw_mae",
            "raw_rmse",
            "raw_abs_rel",
            "raw_bias",
            "median_scale_to_zed",
            "scaled_mae",
            "pair_corr",
            "pair_grad_mae",
        ]
        available_sample_cols = [col for col in sample_cols if col in zed_ref_samples.columns]
        lines.append(dataframe_to_markdown(zed_ref_samples[available_sample_cols], floatfmt=".4f"))
    lines.append("")
    lines.append("## ZED reference ROI metrics")
    lines.append("")
    if zed_ref_roi.empty:
        lines.append("No DA3/ZED ROI comparisons found.")
    else:
        roi_cols = [
            "sample_id",
            "object_name",
            "method",
            "zed_median_m",
            "method_median_m",
            "diff_m",
            "abs_diff_m",
            "rel_abs_diff_to_zed",
            "ratio_method_over_zed",
            "zed_valid_ratio",
            "method_valid_ratio",
            "notes",
        ]
        available_roi_cols = [col for col in roi_cols if col in zed_ref_roi.columns]
        lines.append(dataframe_to_markdown(zed_ref_roi[available_roi_cols], floatfmt=".4f"))
    lines.append("")
    lines.append("## Per-object recomputed metrics")
    lines.append("")
    cols = [
        "sample_id",
        "object_name",
        "gt_distance_m",
        "zed_median_m",
        "zed_abs_error_m",
        "da3_mono_median_m",
        "da3_mono_abs_error_m",
        "winner_recomputed",
        "notes",
    ]
    available_cols = [col for col in cols if col in existing_wide.columns]
    lines.append(dataframe_to_markdown(existing_wide[available_cols], floatfmt=".4f") if not existing_wide.empty else "No existing sample rows.")
    lines.append("")
    lines.append("## Generated plots")
    lines.append("")
    for plot in sorted(plots_dir.glob("*.png")):
        lines.append(f"- plots/{plot.name}")
    lines.append("")
    lines.append("## Generated visual overlays")
    lines.append("")
    for visual in sorted(visuals_dir.glob("*.png")):
        lines.append(f"- visuals/{visual.name}")
    (analysis_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame, *, floatfmt: str = "") -> str:
    if df.empty:
        return ""
    columns = list(df.columns)
    rows = [[format_markdown_value(value, floatfmt=floatfmt) for value in row] for row in df.itertuples(index=False, name=None)]
    widths = []
    for idx, column in enumerate(columns):
        values = [str(row[idx]) for row in rows]
        widths.append(max(len(str(column)), *(len(value) for value in values)))
    header = "| " + " | ".join(str(column).ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = ["| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def format_markdown_value(value, *, floatfmt: str = "") -> str:
    if isinstance(value, float) or isinstance(value, np.floating):
        if not np.isfinite(value):
            return "nan"
        return format(float(value), floatfmt) if floatfmt else str(float(value))
    return str(value)


if __name__ == "__main__":
    main()
