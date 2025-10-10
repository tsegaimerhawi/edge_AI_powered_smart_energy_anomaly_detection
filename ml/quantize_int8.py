#!/usr/bin/env python3
"""
quantize_int8.py — Post-training int8 quantization + golden reference.

Inputs (from previous steps):
  <data_dir>/model_f32.npz       # W1,b1,W2,b2,W3,b3 (float32) from train_mlp.py
  <data_dir>/X_train.npy         # used to estimate activation scales
  <data_dir>/X_test.npy          # optional, used to save test vectors

Outputs:
  <out_dir>/model_int8.npz       # int8 weights, int16 biases: W1,b1,W2,b2,W3,b3
  <out_dir>/scales.json          # {s_x0,s_a1,s_a2,s_w1,s_w2,s_w3,s_y1,s_y2,s_y3}
  <out_dir>/golden.py            # pure-Python int8 inference matching RTL math
  <out_dir>/test_vectors.npy     # a small batch of normalized feature vectors

Conventions:
  - Inputs (features) are already normalized to [-1,1] by dataset_gen.py.
  - We quantize:
      * activations and inputs to int8 with per-tensor scale s_x (real ≈ int8 * s_x)
      * weights to int8 with per-tensor scale s_w (real ≈ int8 * s_w)
      * biases to int16 in the *output* domain scale s_y = s_w * s_in
  - MAC is int32; we rescale to float using s_w * s_in, add bias*(s_y), then ReLU.
"""

import os, json, argparse
import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def forward_f32(X, W1, b1, W2, b2, W3, b3):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    return (z1, a1, z2, a2, z3)


def quantize_per_tensor_sym(weight_f32, max_abs=None):
    if max_abs is None:
        max_abs = np.max(np.abs(weight_f32)) + 1e-12
    scale = max_abs / 127.0
    q = np.clip(np.round(weight_f32 / scale), -128, 127).astype(np.int8)
    return q, float(scale)


def qbias_from_float(b_float, s_w, s_in):
    # bias stored as int16 with scale s_y = s_w * s_in
    s_y = float(s_w * s_in)
    q = np.clip(np.round(b_float / s_y), -32768, 32767).astype(np.int16)
    return q, s_y


def estimate_activation_scales(X_sample, W1, b1, W2, b2):
    # Compute max-abs of input and hidden activations to set s_x0, s_a1, s_a2
    z1, a1, z2, a2, _ = forward_f32(
        X_sample,
        W1,
        b1,
        W2,
        b2,
        np.zeros((W2.shape[1], 4), np.float32),
        np.zeros((4,), np.float32),
    )
    s_x0 = np.max(np.abs(X_sample)) / 127.0 + 1e-12
    s_a1 = np.max(np.abs(a1)) / 127.0 + 1e-12
    s_a2 = np.max(np.abs(a2)) / 127.0 + 1e-12
    return s_x0, s_a1, s_a2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir", default="./ml_out", help="Where model_f32.npz & X_*.npy live"
    )
    ap.add_argument(
        "--out_dir", default="./ml_out", help="Where to write int8 artifacts"
    )
    ap.add_argument(
        "--act_sample",
        type=int,
        default=2000,
        help="How many samples to estimate activation scales",
    )
    ap.add_argument(
        "--test_vecs", type=int, default=256, help="How many test vectors to export"
    )
    args = ap.parse_args()

    # Load float32 model
    mpath = os.path.join(args.data_dir, "model_f32.npz")
    data = np.load(mpath)
    W1, b1, W2, b2, W3, b3 = (
        data["W1"],
        data["b1"],
        data["W2"],
        data["b2"],
        data["W3"],
        data["b3"],
    )

    # Load some train samples to estimate activation scales
    Xtr = np.load(os.path.join(args.data_dir, "X_train.npy")).astype(np.float32)
    if args.act_sample < len(Xtr):
        Xs = Xtr[: args.act_sample]
    else:
        Xs = Xtr

    # Estimate activation scales
    s_x0, s_a1, s_a2 = estimate_activation_scales(Xs, W1, b1, W2, b2)

    # Quantize weights
    QW1, s_w1 = quantize_per_tensor_sym(W1)
    QW2, s_w2 = quantize_per_tensor_sym(W2)
    QW3, s_w3 = quantize_per_tensor_sym(W3)

    # Quantize biases into output domain per layer
    Qb1, s_y1 = qbias_from_float(b1, s_w1, s_x0)
    Qb2, s_y2 = qbias_from_float(b2, s_w2, s_a1)
    Qb3, s_y3 = qbias_from_float(b3, s_w3, s_a2)

    scales = {
        "s_x0": s_x0,
        "s_a1": s_a1,
        "s_a2": s_a2,
        "s_w1": s_w1,
        "s_w2": s_w2,
        "s_w3": s_w3,
        "s_y1": s_y1,
        "s_y2": s_y2,
        "s_y3": s_y3,
    }

    # Save int8 model
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(
        os.path.join(args.out_dir, "model_int8.npz"),
        W1=QW1,
        b1=Qb1,
        W2=QW2,
        b2=Qb2,
        W3=QW3,
        b3=Qb3,
    )

    with open(os.path.join(args.out_dir, "scales.json"), "w") as f:
        json.dump(scales, f, indent=2)

    # Emit a small test vector set
    Xte = np.load(os.path.join(args.data_dir, "X_test.npy")).astype(np.float32)
    np.save(os.path.join(args.out_dir, "test_vectors.npy"), Xte[: args.test_vecs])

    # Write golden.py (int8 reference)
    golden_src = r"""# golden.py — int8 reference inference matching quantize_int8.py
import numpy as np

def infer_logits(Xn, data, scales):
    QW1 = data["W1"]; Qb1 = data["b1"]
    QW2 = data["W2"]; Qb2 = data["b2"]
    QW3 = data["W3"]; Qb3 = data["b3"]

    s_x0 = scales["s_x0"]; s_a1 = scales["s_a1"]; s_a2 = scales["s_a2"]
    s_w1 = scales["s_w1"]; s_w2 = scales["s_w2"]; s_w3 = scales["s_w3"]
    s_y1 = scales["s_y1"]; s_y2 = scales["s_y2"]; s_y3 = scales["s_y3"]

    # Quantize inputs to int8
    Xq = np.clip(np.round(Xn / s_x0), -128, 127).astype(np.int8)

    # Layer 1
    z1_acc = Xq.astype(np.int32) @ QW1.astype(np.int32)  # (N, H1)
    z1 = z1_acc.astype(np.float32) * (s_w1 * s_x0) + (Qb1.astype(np.float32) * s_y1)
    a1 = np.maximum(0.0, z1)
    a1q = np.clip(np.round(a1 / s_a1), -128, 127).astype(np.int8)

    # Layer 2
    z2_acc = a1q.astype(np.int32) @ QW2.astype(np.int32)  # (N, H2)
    z2 = z2_acc.astype(np.float32) * (s_w2 * s_a1) + (Qb2.astype(np.float32) * s_y2)
    a2 = np.maximum(0.0, z2)
    a2q = np.clip(np.round(a2 / s_a2), -128, 127).astype(np.int8)

    # Layer 3 (logits)
    z3_acc = a2q.astype(np.int32) @ QW3.astype(np.int32)  # (N, 4)
    z3 = z3_acc.astype(np.float32) * (s_w3 * s_a2) + (Qb3.astype(np.float32) * s_y3)
    return z3

def infer_class(Xn, data, scales):
    return np.argmax(infer_logits(Xn, data, scales), axis=1)
"""
    with open(os.path.join(args.out_dir, "golden.py"), "w") as f:
        f.write(golden_src)

    print("Wrote:")
    print(" -", os.path.join(args.out_dir, "model_int8.npz"))
    print(" -", os.path.join(args.out_dir, "scales.json"))
    print(" -", os.path.join(args.out_dir, "golden.py"))
    print(" -", os.path.join(args.out_dir, "test_vectors.npy"))

    # Optional sanity: compare float vs int8 argmax on a small batch
    # (kept here for quick feedback; remove if you prefer a silent run)
    try:
        from types import SimpleNamespace

        # float logits
        _, _, _, _, z3f = forward_f32(Xte[:64], W1, b1, W2, b2, W3, b3)
        pf = np.argmax(z3f, axis=1)
        # int8 logits
        data_q = np.load(os.path.join(args.out_dir, "model_int8.npz"))
        with open(os.path.join(args.out_dir, "scales.json")) as jf:
            sc = json.load(jf)
        from importlib.machinery import SourceFileLoader

        golden = SourceFileLoader(
            "golden", os.path.join(args.out_dir, "golden.py")
        ).load_module()
        pq = golden.infer_class(Xte[:64], data_q, sc)
        agree = np.mean(pf == pq)
        print(
            f"Sanity check: float vs int8 argmax agreement on 64 samples = {agree:.3f}"
        )
    except Exception as e:
        print("Sanity check skipped:", e)


if __name__ == "__main__":
    main()
