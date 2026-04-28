"""Formant-based evaluation of articulatory inversion results.

Usage:
    python evaluate_formants.py --target data/valentine.wav --resynth logs/.../2000.wav
    python evaluate_formants.py --target data/valentine.wav --log_dir logs/.../

When --log_dir is given instead of --resynth, the script automatically picks
the highest-iteration .wav file in that directory as the resynthesized audio.
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import soundfile

from utils.formants import (
    PETERSON_BARNEY_VOWELS,
    extract_formants,
    formant_trajectory_error,
    nearest_vowel,
)


def _find_best_resynth(log_dir):
    """Return the path to the highest-iteration .wav in log_dir."""
    wavs = glob.glob(os.path.join(log_dir, "*.wav"))
    numbered = []
    for w in wavs:
        m = re.search(r"(\d+)\.wav$", w)
        if m:
            numbered.append((int(m.group(1)), w))
    if not numbered:
        raise FileNotFoundError(f"No numbered .wav files found in {log_dir}")
    numbered.sort(key=lambda x: x[0])
    return numbered[-1][1]


def plot_f1f2(target_formants, resynth_formants, output_path, title="F1/F2 Vowel Space"):
    """Scatter plot of F1 vs F2 with Peterson & Barney reference ellipses."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot P&B reference vowels
    for label, coords in PETERSON_BARNEY_VOWELS.items():
        ax.scatter(coords["f2"], coords["f1"], marker="D", s=80,
                   color="gray", zorder=5, edgecolors="black", linewidths=0.5)
        ax.annotate(f"/{label}/", (coords["f2"], coords["f1"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, color="gray")

    # Target formants (valid frames only)
    t_valid = (target_formants["f1"] > 0) & (target_formants["f2"] > 0)
    ax.scatter(target_formants["f2"][t_valid], target_formants["f1"][t_valid],
               alpha=0.4, s=15, color="tab:blue", label="Target")

    # Resynthesized formants
    r_valid = (resynth_formants["f1"] > 0) & (resynth_formants["f2"] > 0)
    ax.scatter(resynth_formants["f2"][r_valid], resynth_formants["f1"][r_valid],
               alpha=0.4, s=15, color="tab:orange", label="Resynthesized")

    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.set_title(title)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved F1/F2 plot to {output_path}")


def plot_formant_trajectories(target_formants, resynth_formants, output_path):
    """Time-aligned F1 and F2 trajectory comparison."""
    n = min(len(target_formants["f1"]), len(resynth_formants["f1"]))
    t_times = target_formants["times"][:n]

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, fn, label in zip(axes, ["f1", "f2"], ["F1", "F2"]):
        t_vals = target_formants[fn][:n]
        r_vals = resynth_formants[fn][:n]
        ax.plot(t_times, t_vals, linewidth=1, alpha=0.7, label=f"Target {label}")
        ax.plot(t_times, r_vals, linewidth=1, alpha=0.7, label=f"Resynth {label}")
        ax.set_ylabel(f"{label} (Hz)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Formant Trajectories")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Formant-based evaluation")
    parser.add_argument("--target", required=True, help="Path to target audio file")
    parser.add_argument("--resynth", default=None, help="Path to resynthesized audio file")
    parser.add_argument("--log_dir", default=None,
                        help="Log directory (picks highest-iteration .wav)")
    parser.add_argument("--max_formant", type=float, default=5500.0,
                        help="Max formant frequency for Praat analysis")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for output plots/metrics (defaults to --log_dir or cwd)")
    args = parser.parse_args()

    if args.resynth is None and args.log_dir is None:
        parser.error("Provide either --resynth or --log_dir")

    resynth_path = args.resynth or _find_best_resynth(args.log_dir)
    output_dir = args.output_dir or args.log_dir or os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    # Load audio
    target_audio, sr_target = soundfile.read(args.target)
    resynth_audio, sr_resynth = soundfile.read(resynth_path)
    if len(target_audio.shape) == 2:
        target_audio = target_audio.mean(axis=1)
    if len(resynth_audio.shape) == 2:
        resynth_audio = resynth_audio.mean(axis=1)

    print(f"Target:  {args.target}  ({len(target_audio)} samples, {sr_target} Hz)")
    print(f"Resynth: {resynth_path}  ({len(resynth_audio)} samples, {sr_resynth} Hz)")

    # Extract formants
    print("Extracting formants from target...")
    target_fmt = extract_formants(target_audio, sr_target,
                                  max_formant=args.max_formant)
    print("Extracting formants from resynthesis...")
    resynth_fmt = extract_formants(resynth_audio, sr_resynth,
                                   max_formant=args.max_formant)

    # Compute metrics
    metrics = formant_trajectory_error(target_fmt, resynth_fmt)
    print("\n--- Formant Trajectory Error ---")
    print(f"  F1 RMSE:    {metrics['f1_rmse']:.1f} Hz")
    print(f"  F2 RMSE:    {metrics['f2_rmse']:.1f} Hz")
    print(f"  F1+F2 RMSE: {metrics['f1f2_rmse']:.1f} Hz")
    print(f"  F1 corr:    {metrics['f1_corr']:.4f}")
    print(f"  F2 corr:    {metrics['f2_corr']:.4f}")
    print(f"  Frames:     {metrics['n_frames']}")

    # Nearest-vowel classification on median formants
    valid = (resynth_fmt["f1"] > 0) & (resynth_fmt["f2"] > 0)
    if valid.sum() > 0:
        med_f1 = float(np.median(resynth_fmt["f1"][valid]))
        med_f2 = float(np.median(resynth_fmt["f2"][valid]))
        vowel, dist = nearest_vowel(med_f1, med_f2)
        print(f"\n  Median resynth formants: F1={med_f1:.0f} F2={med_f2:.0f}")
        print(f"  Nearest P&B vowel: /{vowel}/ (distance {dist:.0f} Hz)")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "formant_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Plots
    plot_f1f2(target_fmt, resynth_fmt,
              os.path.join(output_dir, "f1f2_vowel_space.png"))
    plot_formant_trajectories(target_fmt, resynth_fmt,
                              os.path.join(output_dir, "formant_trajectories.png"))


if __name__ == "__main__":
    main()
