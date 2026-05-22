#!/usr/bin/env python3
"""
Convert a sparrow-engine-compatible ONNX model from FP32 to FP16 for use with sparrow-engine's
`[inference] precision = "fp16"` manifest field.

Uses ORT's `transformers.float16` converter (more YOLO-aware than the standard
`onnxconverter-common` variant, which has Cast-node issues on YOLOv10's Resize
output). Keeps input/output as FP32 (`keep_io_types=True`) so sparrow-engine's preprocess
+ postprocess code is unchanged.

Default op block list keeps `Resize` in FP32 (required for MDv6/DeepFaune YOLOv10
graphs). Add more ops via `--block-op` if a model needs it.

Usage:
    python tools/convert_fp16.py <model_dir>
    python tools/convert_fp16.py <model_dir> --block-op Resize --block-op Cast

The script:
  1. Reads the model file from `<model_dir>/manifest.toml`'s `[model] file` field
  2. Converts to FP16 with op_block_list
  3. Writes the FP16 model to `<model_dir>/<stem>_fp16.onnx`
  4. Prints the suggested manifest patch:
       [model]
       file_fp16 = "<stem>_fp16.onnx"
       [inference]
       precision = "fp16"

Reproducibility: bench results in
docs/research/phase3.7/track_b/experiments/results.md were obtained with the
same call.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path,
                    help="Directory containing manifest.toml + the FP32 .onnx")
    ap.add_argument("--block-op", action="append", default=["Resize"],
                    help="Op type to keep in FP32 (default: Resize). May repeat.")
    ap.add_argument("--out-suffix", default="_fp16",
                    help="Suffix appended to FP32 stem for output (default: _fp16)")
    args = ap.parse_args()

    if not args.model_dir.is_dir():
        print(f"error: {args.model_dir} is not a directory", file=sys.stderr)
        return 1

    manifest_path = args.model_dir / "manifest.toml"
    if not manifest_path.exists():
        print(f"error: {manifest_path} not found", file=sys.stderr)
        return 1

    # Parse manifest's [model] file = "..."
    fp32_filename = None
    in_model = False
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("[model]"):
            in_model = True
            continue
        if line.startswith("[") and line != "[model]":
            in_model = False
            continue
        if in_model and line.startswith("file") and "=" in line and "fp16" not in line:
            fp32_filename = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
    if fp32_filename is None:
        print(f"error: could not find [model] file = ... in {manifest_path}", file=sys.stderr)
        return 1

    fp32_path = args.model_dir / fp32_filename
    if not fp32_path.exists():
        print(f"error: {fp32_path} not found", file=sys.stderr)
        return 1

    fp16_path = args.model_dir / f"{fp32_path.stem}{args.out_suffix}.onnx"
    print(f"Source:    {fp32_path}")
    print(f"Output:    {fp16_path}")
    print(f"Block-ops: {args.block_op}")

    import onnx  # type: ignore[import-not-found]
    from onnxruntime.transformers.float16 import (  # type: ignore[import-not-found]
        convert_float_to_float16,
    )

    print(f"Loading {fp32_path.name}...")
    model = onnx.load(str(fp32_path))
    print(f"Converting to FP16...")
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=args.block_op,
        force_fp16_initializers=True,
    )
    # ORT's `transformers.float16` converter sometimes emits multiple Cast
    # nodes with the same auto-generated name AND the same output tensor name
    # when one Constant feeds multiple blocked-op consumers. ORT's load-time
    # validator rejects this on two distinct axes:
    #   (1) "two nodes with same node name" — node.name uniqueness
    #   (2) "Duplicate definition of name (...)" — tensor name uniqueness
    # (a tensor produced by two different nodes is forbidden).
    #
    # Observed on MDv6 (1 dup) and DeepFaune (2 dups). Fix: keep the first
    # Cast that emits a given output tensor; for each subsequent Cast that
    # would emit the same tensor name, rewire its consumers to read from the
    # first Cast's output and drop the duplicate Cast node entirely. This
    # collapses functionally-identical Casts (same input → same output dtype)
    # without changing the graph's semantics.
    output_to_first_node: dict[str, str] = {}  # output tensor name → producing node name (kept)
    nodes_to_drop: list[int] = []
    for idx, node in enumerate(model_fp16.graph.node):
        for out_name in node.output:
            if out_name in output_to_first_node:
                # Duplicate. Drop this node; consumers continue to reference
                # the same tensor name, which is now produced solely by the
                # first node.
                nodes_to_drop.append(idx)
                break
            else:
                output_to_first_node[out_name] = node.name
    if nodes_to_drop:
        # Drop in reverse so indexes stay valid.
        for idx in reversed(nodes_to_drop):
            del model_fp16.graph.node[idx]
        print(f"Dropped {len(nodes_to_drop)} duplicate-output Cast node(s)")
    # Even after dropping output-duplicate nodes, defensively dedupe any
    # remaining node-name collisions (uncommon but cheap to check).
    seen: dict[str, int] = {}
    n_renamed = 0
    for node in model_fp16.graph.node:
        if node.name in seen:
            seen[node.name] += 1
            node.name = f"{node.name}_dup{seen[node.name]}"
            n_renamed += 1
        else:
            seen[node.name] = 0
    if n_renamed:
        print(f"Renamed {n_renamed} duplicate node name(s)")
    print(f"Saving {fp16_path.name}...")
    onnx.save(model_fp16, str(fp16_path))
    print(f"Done.")
    print()
    print("Suggested manifest patch (add to manifest.toml):")
    print()
    print(f'[model]')
    print(f'file_fp16 = "{fp16_path.name}"')
    print()
    print('[inference]')
    print('precision = "fp16"')
    print()
    print("Then run a parity verification before activating in production:")
    print("  python scripts/find_divergent_images.py  # compare detection counts")
    print("  python scripts/diagnose_stages.py        # per-stage numerical diff")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
