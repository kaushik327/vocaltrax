import json, numpy as np, soundfile, glob, os, librosa
import parselmouth
from parselmouth.praat import call
import sys

def mel_mse(target_path, synth_path, sr=22050, n_mels=64):
    t, _ = librosa.load(target_path, sr=sr, mono=True)
    s, _ = librosa.load(synth_path, sr=sr, mono=True)
    min_len = min(len(t), len(s))
    t, s = t[:min_len], s[:min_len]
    T = librosa.feature.melspectrogram(y=t, sr=sr, n_mels=n_mels)
    S = librosa.feature.melspectrogram(y=s, sr=sr, n_mels=n_mels)
    return float(np.mean((np.log1p(T) - np.log1p(S))**2))

def formant_distance(target_path, synth_path):
    """Mean absolute F1+F2 distance in Hz between target and synth."""
    def get_formants(path):
        snd = parselmouth.Sound(path)
        formants = call(snd, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
        times = np.arange(0.05, snd.duration - 0.05, 0.01)
        f1 = [call(formants, "Get value at time", 1, t, "Hertz", "Linear") for t in times]
        f2 = [call(formants, "Get value at time", 2, t, "Hertz", "Linear") for t in times]
        f1 = np.array([x for x in f1 if not np.isnan(x)])
        f2 = np.array([x for x in f2 if not np.isnan(x)])
        return f1, f2

    try:
        t1, t2 = get_formants(target_path)
        s1, s2 = get_formants(synth_path)
        min_len = min(len(t1), len(s1))
        if min_len == 0:
            return float('nan')
        f1_dist = np.mean(np.abs(t1[:min_len] - s1[:min_len]))
        f2_dist = np.mean(np.abs(t2[:min_len] - s2[:min_len]))
        return float((f1_dist + f2_dist) / 2)
    except:
        return float('nan')

def smoothness(params_path):
    with open(params_path) as f:
        params = json.load(f)
    total = 0
    def recurse(d):
        nonlocal total
        if isinstance(d, dict):
            for v in d.values(): recurse(v)
        else:
            arr = np.array(d)
            if arr.ndim > 0 and arr.shape[0] > 1:
                total += np.sum(np.diff(arr, axis=0) ** 2)
    recurse(params)
    return total

TARGET = "data/valentine_22k.wav"  # change for experiment

print(f"{'Run':<45} {'Smooth':>8} {'MelMSE':>10} {'FormantΔHz':>12}")
print("-" * 78)

filter_str = sys.argv[1] if len(sys.argv) > 1 else ""


for params_path in sorted(glob.glob("logs/**/params.json", recursive=True)):
    if filter_str and filter_str not in params_path:
        continue
    run_dir = os.path.dirname(params_path)
    wavs = sorted(
        [w for w in glob.glob(os.path.join(run_dir, "*.wav"))
         if os.path.basename(w).split('.')[0].isdigit()],
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )
    if not wavs:
        continue
    final_wav = wavs[-1]
    label = "/".join(params_path.split("/")[-3:-1])
    s = smoothness(params_path)
    m = mel_mse(TARGET, final_wav)
    f = formant_distance(TARGET, final_wav)
    print(f"{label:<45} {s:>8.2f} {m:>10.6f} {f:>12.1f}")