"""Weight singular-value spectra from checkpoints (E7).

Usage:
    san spectra checkpoints/san_muon_base_105B_s42*.pkl [--out spectra]

Per checkpoint, for every Dense kernel (scanned kernels have shape
(L, din, dout)), computes the full SV spectrum per layer plus SV-entropy
effective rank and stable rank. Writes <out>/<ckpt_stem>.npz with keys
"<param_path>/sv"          (L, min(din, dout))
"<param_path>/eff_rank"    (L,)
"<param_path>/stable_rank" (L,)
and prints a per-matrix summary. Milestone files (…_step<k>.pkl) give the
0/25/50/75/100% trajectory; the money figure compares end-of-training
per-layer ranks across arms. Globs resolve against LOCAL files — bulk-fetch
milestones first: hf download <repo> --include "checkpoints/<name>_step*".
"""
import os
import pickle
import sys

import numpy as np


def _walk(tree, prefix=()):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _walk(v, prefix + (k,))
    else:
        yield prefix, tree


def spectra(params):
    out = {}
    for path, leaf in _walk(params):
        if path[-1] != "kernel":
            continue
        arr = np.asarray(leaf, dtype=np.float32)
        if arr.ndim not in (2, 3):
            continue
        mats = arr if arr.ndim == 3 else arr[None]
        sv = np.linalg.svd(mats, compute_uv=False)
        p = sv / np.clip(sv.sum(-1, keepdims=True), 1e-12, None)
        eff = np.exp(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum(-1))
        stable = (sv ** 2).sum(-1) / np.clip(sv[..., 0] ** 2, 1e-12, None)
        out["/".join(path[:-1])] = {"sv": sv, "eff_rank": eff, "stable_rank": stable}
    return out


def main(args):
    from .eval import tee_stdout, upload_result
    os.makedirs(args.out, exist_ok=True)
    summary_path = os.path.join(args.out, "spectra_summary.txt")
    with tee_stdout(summary_path):
        _run(args)
    if getattr(args, "upload_results", False):
        upload_result(summary_path, dest_dir="results/spectra")
        for f in sorted(os.listdir(args.out)):
            if f.endswith(".npz"):
                upload_result(os.path.join(args.out, f), dest_dir="results/spectra")


def _run(args):

    for path in args.checkpoints:
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        if not isinstance(ckpt, dict) or "params" not in ckpt:
            print(f"skip {path}: not a checkpoint", file=sys.stderr)
            continue
        result = spectra(ckpt["params"])
        stem = os.path.splitext(os.path.basename(path))[0]
        arrays = {f"{name}/{k}": v for name, d in result.items() for k, v in d.items()}
        np.savez(os.path.join(args.out, stem + ".npz"), **arrays)

        print(f"\n{stem}  (step {ckpt.get('step', '?')})")
        print(f"  {'matrix':<44}{'eff_rank':>16}{'stable_rank':>14}")
        for name, d in sorted(result.items()):
            e, s = d["eff_rank"], d["stable_rank"]
            print(f"  {name:<44}{e.mean():>9.1f} ±{e.std():>5.1f}{s.mean():>14.1f}")
    print(f"\nSpectra written to {args.out}/", file=sys.stderr)
