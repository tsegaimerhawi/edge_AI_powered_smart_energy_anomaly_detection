#!/usr/bin/env python3
"""
pack_weights_rom.py — emit $readmemh hex files for each layer (neuron-major).

Inputs (from quantize_int8.py):
  <data_dir>/model_int8.npz   # int8 W*, int16 b*
  (Optional) <data_dir>/scales.json (not used here; firmware writes regs)

Outputs (to out_dir):
  layer1_w.memh   # int8,  one 8-bit hex per line, length = IN1 * H1
  layer1_b.memh   # int16, one 16-bit hex per line, length = H1
  layer2_w.memh   # int8,  length = H1 * H2
  layer2_b.memh   # int16, length = H2
  layer3_w.memh   # int8,  length = H2 * 4
  layer3_b.memh   # int16, length = 4
  layout.json     # dims + ordering docs

Conventions:
  - Weights are neuron-major: for j in 0..H-1, write W[0,j], W[1,j], ..., W[IN-1,j]
  - $readmemh lines are **hex without 0x**, uppercase. Two's complement encoding.
    * int8  -> 2 hex digits (00..FF)
    * int16 -> 4 hex digits (0000..FFFF)
  - Endianness: each line is **one element**; no multi-byte packing, so endianness is a non-issue.
"""

import os, json, argparse
import numpy as np


def to_hex_i8(v):
    # v is int8 numpy scalar -> two's complement 8-bit
    return f"{(int(v) & 0xFF):02X}"


def to_hex_i16(v):
    # v is int16 numpy scalar -> two's complement 16-bit
    return f"{(int(v) & 0xFFFF):04X}"


def write_memh_weights_neuron_major(path, W):
    """
    W shape = (IN, H). Emit lines in neuron-major:
      for j in 0..H-1:
        for i in 0..IN-1: write W[i,j]
    """
    IN, H = W.shape
    with open(path, "w") as f:
        for j in range(H):
            for i in range(IN):
                f.write(to_hex_i8(W[i, j]) + "\n")


def write_memh_bias(path, b):
    """
    b shape = (H,)
    """
    with open(path, "w") as f:
        for j in range(b.shape[0]):
            f.write(to_hex_i16(b[j]) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./ml_out", help="Folder with model_int8.npz")
    ap.add_argument("--out_dir", default="./ml_out/rom", help="Where to write *.memh")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    npz = np.load(os.path.join(args.data_dir, "model_int8.npz"))
    W1 = npz["W1"].astype(np.int8)  # shape (7, H1)
    b1 = npz["b1"].astype(np.int16)  # shape (H1,)
    W2 = npz["W2"].astype(np.int8)  # shape (H1, H2)
    b2 = npz["b2"].astype(np.int16)  # shape (H2,)
    W3 = npz["W3"].astype(np.int8)  # shape (H2, 4)
    b3 = npz["b3"].astype(np.int16)  # shape (4,)

    # Sanity dims
    IN1, H1 = W1.shape
    H1b = b1.shape[0]
    H1_, H2 = W2.shape
    H2b = b2.shape[0]
    H2_, O = W3.shape
    O_ = b3.shape[0]
    assert H1 == H1b and H1 == H1_, "Layer 1 dim mismatch"
    assert H2 == H2b and H2 == H2_, "Layer 2 dim mismatch"
    assert O == O_, "Output dim mismatch"
    assert IN1 == 7, "Expected 7 input features"

    # Write memh files
    write_memh_weights_neuron_major(os.path.join(args.out_dir, "layer1_w.memh"), W1)
    write_memh_bias(os.path.join(args.out_dir, "layer1_b.memh"), b1)

    write_memh_weights_neuron_major(os.path.join(args.out_dir, "layer2_w.memh"), W2)
    write_memh_bias(os.path.join(args.out_dir, "layer2_b.memh"), b2)

    write_memh_weights_neuron_major(os.path.join(args.out_dir, "layer3_w.memh"), W3)
    write_memh_bias(os.path.join(args.out_dir, "layer3_b.memh"), b3)

    # Document layout
    layout = {
        "layer1": {
            "inputs": IN1,
            "neurons": H1,
            "weights_order": "neuron-major (W[0..IN-1, j] for j=0..H1-1)",
        },
        "layer2": {
            "inputs": H1,
            "neurons": H2,
            "weights_order": "neuron-major (W[0..H1-1, j] for j=0..H2-1)",
        },
        "layer3": {
            "inputs": H2,
            "neurons": O,
            "weights_order": "neuron-major (W[0..H2-1, j] for j=0..O-1)",
        },
        "element_format": {
            "int8_weight": "2-digit HEX per line, two's complement",
            "int16_bias": "4-digit HEX per line, two's complement",
        },
    }
    with open(os.path.join(args.out_dir, "layout.json"), "w") as f:
        json.dump(layout, f, indent=2)

    print("Wrote ROM images to:", args.out_dir)
    for fn in [
        "layer1_w.memh",
        "layer1_b.memh",
        "layer2_w.memh",
        "layer2_b.memh",
        "layer3_w.memh",
        "layer3_b.memh",
        "layout.json",
    ]:
        print(" -", os.path.join(args.out_dir, fn))


if __name__ == "__main__":
    main()
