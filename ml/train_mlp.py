#!/usr/bin/env python3
"""
train_mlp.py — Train a small MLP on the 7-D features.
Inputs come from dataset_gen.py (normalized to [-1,1]).
Network: 7 → H1 → H2 → 4 with ReLU (default H1=32, H2=16)
Saves:
  <out_dir>/model_f32.npz  # W1,b1,W2,b2,W3,b3 (float32)
  <out_dir>/metrics.json   # train/val/test accuracy & settings
"""

import argparse, json, os, time
import numpy as np

CLASSES = ["Normal", "Spike", "Noise", "Harmonic"]


# -----------------------------
# MLP utilities (NumPy)
# -----------------------------
def relu(x):
    return np.maximum(0, x)


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def one_hot(y, ncls):
    Y = np.zeros((y.shape[0], ncls), dtype=np.float32)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def init_params(rng, in_dim=7, h1=32, h2=16, out_dim=4, wscale=0.1):
    W1 = rng.normal(0, wscale, size=(in_dim, h1)).astype(np.float32)
    b1 = np.zeros((h1,), dtype=np.float32)
    W2 = rng.normal(0, wscale, size=(h1, h2)).astype(np.float32)
    b2 = np.zeros((h2,), dtype=np.float32)
    W3 = rng.normal(0, wscale, size=(h2, out_dim)).astype(np.float32)
    b3 = np.zeros((out_dim,), dtype=np.float32)
    return (W1, b1, W2, b2, W3, b3)


def forward(X, params):
    W1, b1, W2, b2, W3, b3 = params
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    return (z1, a1, z2, a2, z3)


def loss_and_grads(X, y, params, l2=1e-4):
    W1, b1, W2, b2, W3, b3 = params
    z1, a1, z2, a2, z3 = forward(X, params)
    P = softmax(z3)
    Y = one_hot(y, ncls=4)
    eps = 1e-9
    ce = -np.mean(np.sum(Y * np.log(P + eps), axis=1))
    reg = l2 * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    loss = ce + reg

    dZ3 = (P - Y) / X.shape[0]
    dW3 = a2.T @ dZ3 + 2 * l2 * W3
    db3 = np.sum(dZ3, axis=0)

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * (z2 > 0)
    dW2 = a1.T @ dZ2 + 2 * l2 * W2
    db2 = np.sum(dZ2, axis=0)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (z1 > 0)
    dW1 = X.T @ dZ1 + 2 * l2 * W1
    db1 = np.sum(dZ1, axis=0)

    grads = (dW1, db1, dW2, db2, dW3, db3)
    return loss, grads


def accuracy(X, y, params):
    _, _, _, _, z3 = forward(X, params)
    pred = np.argmax(z3, axis=1)
    return float(np.mean(pred == y))


# -----------------------------
# Training loop
# -----------------------------
def train(Xtr, ytr, Xva, yva, args):
    rng = np.random.default_rng(args.seed)
    params = init_params(
        rng, in_dim=7, h1=args.h1, h2=args.h2, out_dim=4, wscale=args.wscale
    )
    N = Xtr.shape[0]
    lr = args.lr

    for ep in range(1, args.epochs + 1):
        # shuffle
        idx = rng.permutation(N)
        for i in range(0, N, args.batch):
            b = idx[i : i + args.batch]
            loss, grads = loss_and_grads(Xtr[b], ytr[b], params, l2=args.l2)
            dW1, db1, dW2, db2, dW3, db3 = grads
            W1, b1, W2, b2, W3, b3 = params
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2
            W3 -= lr * dW3
            b3 -= lr * db3
            params = (W1, b1, W2, b2, W3, b3)

        acc_tr = accuracy(Xtr, ytr, params)
        acc_va = accuracy(Xva, yva, params)
        if ep % 5 == 0 or ep == 1:
            print(
                f"Epoch {ep:03d} | train_acc={acc_tr:.3f} val_acc={acc_va:.3f} lr={lr:.4f}"
            )

        # simple LR decay
        if args.lr_decay > 0 and (ep % args.lr_decay_every == 0):
            lr *= args.lr_decay

    return params


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        default="./ml_out",
        help="Folder with X_*.npy, y_*.npy from dataset_gen.py",
    )
    ap.add_argument("--out_dir", default="./ml_out", help="Where to save model/metrics")
    # Model
    ap.add_argument("--h1", type=int, default=32)
    ap.add_argument("--h2", type=int, default=16)
    ap.add_argument("--wscale", type=float, default=0.1)
    # Training
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument(
        "--lr_decay", type=float, default=0.5, help="Multiply lr by this factor"
    )
    ap.add_argument("--lr_decay_every", type=int, default=15)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    # Load data
    Xtr = np.load(os.path.join(args.data_dir, "X_train.npy"))
    ytr = np.load(os.path.join(args.data_dir, "y_train.npy"))
    Xva = np.load(os.path.join(args.data_dir, "X_val.npy"))
    yva = np.load(os.path.join(args.data_dir, "y_val.npy"))
    Xte = np.load(os.path.join(args.data_dir, "X_test.npy"))
    yte = np.load(os.path.join(args.data_dir, "y_test.npy"))

    t0 = time.time()
    params = train(Xtr, ytr, Xva, yva, args)
    t1 = time.time()

    # Evaluate
    acc_tr = accuracy(Xtr, ytr, params)
    acc_va = accuracy(Xva, yva, params)
    acc_te = accuracy(Xte, yte, params)
    print(
        f"Done in {t1-t0:.1f}s | train={acc_tr:.3f} val={acc_va:.3f} test={acc_te:.3f}"
    )

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    W1, b1, W2, b2, W3, b3 = params
    np.savez(
        os.path.join(args.out_dir, "model_f32.npz"),
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        W3=W3,
        b3=b3,
    )

    metrics = {
        "h1": args.h1,
        "h2": args.h2,
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "l2": args.l2,
        "seed": args.seed,
        "acc_train": acc_tr,
        "acc_val": acc_va,
        "acc_test": acc_te,
        "classes": CLASSES,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved:")
    print(" -", os.path.join(args.out_dir, "model_f32.npz"))
    print(" -", os.path.join(args.out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
