#!/usr/bin/env python3
"""
dataset_gen.py — generate synthetic energy-signal dataset + 7-D feature vectors
Classes: Normal, Spike, Noise, Harmonic
Features: Vrms, Crest, THD, Kurtosis, SpectralFlatness, H3, H5

Outputs:
  <out_dir>/
    X_train.npy, y_train.npy
    X_val.npy,   y_val.npy
    X_test.npy,  y_test.npy
    feature_scaler.json  # per-feature mins/maxs for [-1,1] normalization
"""
import argparse, json
import numpy as np

CLASSES = ["Normal", "Spike", "Noise", "Harmonic"]


# -----------------------------
# Signal generation helpers
# -----------------------------
def gen_time(win, fs):
    return np.arange(win) / fs


def add_harmonics(x, t, f0, rng, base_amp=1.0):
    for k in [3, 5]:
        if rng.random() < 0.8:
            amp = rng.uniform(0.05, 0.3) * base_amp
            phase = rng.uniform(0, 2 * np.pi)
            x += amp * np.sin(2 * np.pi * (k * f0) * t + phase)
    return x


def gen_normal(win, fs, f0, rng):
    t = gen_time(win, fs)
    amp = rng.uniform(0.9, 1.1)
    phase = rng.uniform(0, 2 * np.pi)
    return amp * np.sin(2 * np.pi * f0 * t + phase)


def gen_spike(win, fs, f0, rng):
    x = gen_normal(win, fs, f0, rng)
    idx = rng.integers(0, win)
    magnitude = rng.uniform(2.0, 5.0) * (np.max(np.abs(x)) + 1e-6)
    width = rng.integers(1, 3)  # 1–2 samples
    x[idx : min(idx + width, win)] += magnitude * (1 if rng.random() < 0.5 else -1)
    return x


def gen_noise(win, fs, f0, rng):
    x = gen_normal(win, fs, f0, rng)
    snr_db = rng.uniform(10, 30)
    noise_w = rng.normal(0.0, 1.0, size=win)
    noise_p = np.cumsum(rng.normal(0.0, 1.0, size=win))
    noise_p = noise_p / (np.std(noise_p) + 1e-12)
    alpha = rng.uniform(0.3, 0.7)
    noise = alpha * noise_w + (1 - alpha) * noise_p
    sig_rms = np.sqrt(np.mean(x**2)) + 1e-12
    noise_rms = np.sqrt(np.mean(noise**2)) + 1e-12
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    noise *= target_noise_rms / noise_rms
    return x + noise


def gen_harmonic(win, fs, f0, rng):
    t = gen_time(win, fs)
    amp = rng.uniform(0.9, 1.1)
    phase = rng.uniform(0, 2 * np.pi)
    x = amp * np.sin(2 * np.pi * f0 * t + phase)
    return add_harmonics(x, t, f0, rng, base_amp=amp)


# -----------------------------
# Feature extraction
# -----------------------------
def goertzel_mag(x, k_freq, fs):
    N = len(x)
    w = 2.0 * np.pi * k_freq / fs
    cw = np.cos(w)
    coeff = 2.0 * cw
    s0 = s1 = s2 = 0.0
    for n in range(N):
        s0 = x[n] + coeff * s1 - s2
        s2 = s1
        s1 = s0
    real = s1 - s2 * cw
    imag = s2 * np.sin(w)
    return np.sqrt(real * real + imag * imag) / N


def crest_factor(x):
    rms = np.sqrt(np.mean(x**2)) + 1e-12
    peak = np.max(np.abs(x)) + 1e-12
    return peak / rms


def kurtosis_excess(x):
    mu = np.mean(x)
    s = np.std(x) + 1e-12
    z = (x - mu) / s
    return np.mean(z**4) - 3.0


def spectral_flatness(x):
    X = np.fft.rfft(x * np.hanning(len(x)))
    mag = np.abs(X) + 1e-12
    geo = np.exp(np.mean(np.log(mag)))
    arith = np.mean(mag)
    return float(geo / arith)


def thd_features(x, f0, fs):
    a1 = goertzel_mag(x, f0, fs)
    a3 = goertzel_mag(x, 3 * f0, fs)
    a5 = goertzel_mag(x, 5 * f0, fs)
    harm = np.sqrt(a3 * a3 + a5 * a5)
    thd = float(harm / (a1 + 1e-12))
    return thd, float(a3), float(a5)


def extract_features(x, f0, fs):
    vrms = float(np.sqrt(np.mean(x**2)))
    cf = float(crest_factor(x))
    thd_val, h3, h5 = thd_features(x, f0, fs)
    kur = float(kurtosis_excess(x))
    sfm = float(spectral_flatness(x))
    return np.array([vrms, cf, thd_val, kur, sfm, h3, h5], dtype=np.float32)


# -----------------------------
# Dataset builder
# -----------------------------
def sample_one(win, fs, f0, rng):
    c = rng.integers(0, 4)
    if c == 0:
        x = gen_normal(win, fs, f0, rng)
    elif c == 1:
        x = gen_spike(win, fs, f0, rng)
    elif c == 2:
        x = gen_noise(win, fs, f0, rng)
    else:
        x = gen_harmonic(win, fs, f0, rng)
    return extract_features(x, f0, fs), c


def build_dataset(n, win, fs, f0, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, 7), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    for i in range(n):
        f, c = sample_one(win, fs, f0, rng)
        X[i] = f
        y[i] = c
    return X, y


def normalize_to_pm1(X_train, margin=0.10):
    mins = X_train.min(axis=0)
    maxs = X_train.max(axis=0)
    span = maxs - mins + 1e-9
    mins2 = mins - margin * span
    maxs2 = maxs + margin * span

    def norm(X):
        return np.clip(2.0 * (X - mins2) / (maxs2 - mins2) - 1.0, -1.0, 1.0)

    scaler = {"mins2": mins2.tolist(), "maxs2": maxs2.tolist()}
    return norm, scaler


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="./ml_out", help="Output directory")
    ap.add_argument("--fs", type=int, default=5000, help="Sampling rate (Hz)")
    ap.add_argument("--win", type=int, default=256, help="Samples per window")
    ap.add_argument("--f0", type=float, default=50.0, help="Fundamental frequency (Hz)")
    ap.add_argument("--n_train", type=int, default=12000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)

    print("Generating datasets …")
    Xtr, ytr = build_dataset(args.n_train, args.win, args.fs, args.f0, args.seed + 0)
    Xva, yva = build_dataset(args.n_val, args.win, args.fs, args.f0, args.seed + 1)
    Xte, yte = build_dataset(args.n_test, args.win, args.fs, args.f0, args.seed + 2)

    norm_fn, scaler = normalize_to_pm1(Xtr)
    Xtr_n = norm_fn(Xtr)
    Xva_n = norm_fn(Xva)
    Xte_n = norm_fn(Xte)

    out = args.out_dir
    import os

    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/X_train.npy", Xtr_n)
    np.save(f"{out}/y_train.npy", ytr)
    np.save(f"{out}/X_val.npy", Xva_n)
    np.save(f"{out}/y_val.npy", yva)
    np.save(f"{out}/X_test.npy", Xte_n)
    np.save(f"{out}/y_test.npy", yte)

    with open(f"{out}/feature_scaler.json", "w") as f:
        json.dump(scaler, f, indent=2)

    print("Done.")
    print(f"Saved to: {out}")
    print("Files:")
    for fn in [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
        "feature_scaler.json",
    ]:
        print(" -", f"{out}/{fn}")


if __name__ == "__main__":
    main()
