import json
import numpy as np
import soundfile
import glob
import os
from scipy.signal import stft

def get_all_arrays(d):
    """Recursively extract all leaf arrays from nested dict"""
    arrays = []
    if isinstance(d, dict):
        for v in d.values():
            arrays.extend(get_all_arrays(v))
    else:
        arr = np.array(d)
        if arr.ndim > 0 and arr.shape[0] > 1:
            arrays.append(arr)
    return arrays

def smoothness(params_path):
    with open(params_path) as f:
        params = json.load(f)
    arrays = get_all_arrays(params)
    return sum(np.sum(np.diff(a, axis=0) ** 2) for a in arrays)

def spec_mse(target_path, synth_path):
    t, sr = soundfile.read(target_path)
    s, _  = soundfile.read(synth_path)
    if t.ndim == 2: t = np.mean(t, axis=1)
    if s.ndim == 2: s = np.mean(s, axis=1)
    min_len = min(len(t), len(s))
    t, s = t[:min_len], s[:min_len]
    nperseg = min(512, min_len)
    _, _, T = stft(t, sr, nperseg=nperseg)
    _, _, S = stft(s, sr, nperseg=nperseg)
    return np.mean(np.abs(np.abs(T) - np.abs(S))**2)

def get_lambda(run_dir):
    """Try to read lambda from hydra config"""
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            for line in f:
                if "temporal_lambda" in line:
                    return line.strip().split(":")[-1].strip()
                if "temporal_regularization" in line and "false" in line.lower():
                    return "no_reg"
    return "unknown"

TARGET = "/Users/Jasmine/vocaltrax/vocaltrax/data/valentine.wav"

print(f"{'Lambda':<12} {'Run':<45} {'Smoothness':>12} {'Spec MSE':>10}")
print("-" * 82)

for params_path in sorted(glob.glob("logs/**/params.json", recursive=True)):
    run_dir = os.path.dirname(params_path)
    lam = get_lambda(run_dir)

    wavs = sorted(
        [w for w in glob.glob(os.path.join(run_dir, "*.wav"))
         if os.path.splitext(os.path.basename(w))[0].isdigit()],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    final_wav = wavs[-1] if wavs else None

    smooth = smoothness(params_path)
    mse_val = spec_mse(TARGET, final_wav) if final_wav else float('nan')
    label = "/".join(params_path.split("/")[-4:-1])
    print(f"{lam:<12} {label:<45} {smooth:>12.4f} {mse_val:>10.6f}")