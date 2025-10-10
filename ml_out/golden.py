# golden.py — int8 reference inference matching quantize_int8.py
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
