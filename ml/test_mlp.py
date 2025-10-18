#!/usr/bin/env python3
"""
test_model_with_plots.py — evaluate and visualize your model performance.

Requires:
  - model_f32.npz (optional)
  - model_int8.npz
  - scales.json
  - golden.py
  - X_test.npy, y_test.npy

Outputs:
  <data_dir>/plots/
      confusion_matrix.png
      class_metrics.png
      float_vs_int8_scatter.png
"""

import os, argparse, json, numpy as np
from importlib.machinery import SourceFileLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

CLASSES = ["Normal", "Spike", "Noise", "Harmonic"]


# ---------------- Utility funcs ----------------
def relu(x):
    return np.maximum(0, x)


def forward_float(X, model):
    W1, b1, W2, b2, W3, b3 = model
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    return z3


def plot_confusion(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_class_metrics(report_dict, path):
    labels = list(report_dict.keys())[:-3]  # exclude accuracy/macro/weighted
    precision = [report_dict[l]["precision"] for l in labels]
    recall = [report_dict[l]["recall"] for l in labels]
    f1 = [report_dict[l]["f1-score"] for l in labels]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class Metrics (Int8 Model)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_float_vs_int8(logits_f, logits_q, path):
    # Compare one logit dimension (class 0) as example
    lf = logits_f[:, 0]
    lq = logits_q[:, 0]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(lf, lq, alpha=0.4, s=10)
    ax.set_xlabel("Float32 logits (class0)")
    ax.set_ylabel("Int8 logits (class0)")
    ax.set_title("Float vs Int8 Logit Agreement")
    lim = [min(lf.min(), lq.min()), max(lf.max(), lq.max())]
    ax.plot(lim, lim, "r--")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./ml_out", help="Folder with model and data")
    args = ap.parse_args()
    out_plot_dir = os.path.join(args.data_dir, "plots")
    os.makedirs(out_plot_dir, exist_ok=True)

    # Load data
    X_test = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test = np.load(os.path.join(args.data_dir, "y_test.npy"))

    # Load models
    f32_path = os.path.join(args.data_dir, "model_f32.npz")
    model_f32 = None
    if os.path.exists(f32_path):
        m = np.load(f32_path)
        model_f32 = (m["W1"], m["b1"], m["W2"], m["b2"], m["W3"], m["b3"])
        print("Loaded float32 model.")

    data_q = np.load(os.path.join(args.data_dir, "model_int8.npz"))
    with open(os.path.join(args.data_dir, "scales.json")) as f:
        scales = json.load(f)
    golden = SourceFileLoader(
        "golden", os.path.join(args.data_dir, "golden.py")
    ).load_module()
    print("Loaded quantized model + scales.")

    # Float inference
    if model_f32 is not None:
        logits_f = forward_float(X_test, model_f32)
        pred_f = np.argmax(logits_f, axis=1)
        acc_f = np.mean(pred_f == y_test)
        print(f"Float32 accuracy: {acc_f*100:.2f}%")
    else:
        logits_f = None
        pred_f = None

    # Quantized inference
    logits_q = golden.infer_logits(X_test, data_q, scales)
    pred_q = np.argmax(logits_q, axis=1)
    acc_q = np.mean(pred_q == y_test)
    print(f"Int8 quantized accuracy: {acc_q*100:.2f}%")

    if pred_f is not None:
        agree = np.mean(pred_f == pred_q)
        print(f"Float vs Int8 argmax agreement: {agree*100:.2f}%")

    # Confusion matrix & report
    cm = confusion_matrix(y_test, pred_q)
    report_str = classification_report(
        y_test, pred_q, target_names=CLASSES, digits=3, output_dict=False
    )
    report_dict = classification_report(
        y_test, pred_q, target_names=CLASSES, digits=3, output_dict=True
    )
    print("\nClassification Report (Int8):")
    print(report_str)
    print("Confusion matrix:\n", cm)

    # ----- Plots -----
    plot_confusion(cm, CLASSES, os.path.join(out_plot_dir, "confusion_matrix.png"))
    plot_class_metrics(report_dict, os.path.join(out_plot_dir, "class_metrics.png"))
    if logits_f is not None:
        plot_float_vs_int8(
            logits_f, logits_q, os.path.join(out_plot_dir, "float_vs_int8_scatter.png")
        )

    print("\n✅ Saved plots in", out_plot_dir)
    for fn in os.listdir(out_plot_dir):
        print(" -", fn)


if __name__ == "__main__":
    main()
