import json
import numpy as np
import glob
import os
import librosa

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

def mel_mse(target_path, synth_path, sr=22050, n_mels=64):
    t, _ = librosa.load(target_path, sr=sr, mono=True)
    s, _ = librosa.load(synth_path, sr=sr, mono=True)
    min_len = min(len(t), len(s))
    t, s = t[:min_len], s[:min_len]
    T = librosa.feature.melspectrogram(y=t, sr=sr, n_mels=n_mels)
    S = librosa.feature.melspectrogram(y=s, sr=sr, n_mels=n_mels)
    return np.mean((np.log1p(T) - np.log1p(S))**2)

def get_lambda(run_dir):
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path) as f:
            content = f.read()
            if "temporal_regularization: false" in content:
                return "no_reg"
            for line in content.splitlines():
                if "temporal_lambda" in line:
                    return line.strip().split(":")[-1].strip()
    return "unknown"

TARGET = "/Users/Jasmine/vocaltrax/vocaltrax/data/valentine.wav"

print(f"{'Lambda':<12} {'Run':<45} {'Smoothness':>12} {'Mel MSE':>10}")
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
    mse_val = mel_mse(TARGET, final_wav) if final_wav else float('nan')
    label = "/".join(params_path.split("/")[-4:-1])
    print(f"{lam:<12} {label:<45} {smooth:>12.4f} {mse_val:>10.6f}")