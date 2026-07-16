import argparse
import os
import re
import sys
import threading

HELP = """Check the readme"""


_ABSL_LOG_START = re.compile(rb"^[EIWF]\d{4} \d\d:\d\d:\d\d")
_NOISY_LOG_HEADER = re.compile(
    rb"\] (?:Fusion: .*gemm_fusion|Computation: .*_computation|Delay kernel timed out)"
)

_log_filter_installed = False


def _install_xla_log_filter():
    """Drop XLA Triton autotuner noise from stderr.

    XLA's Triton GEMM autotuner logs failed candidate fusions via LOG(ERROR)
    in xtile_compiler.cc and cuda_timer.cc. These are unconditional and do
    not respect TF_CPP_MIN_LOG_LEVEL, so we filter them at the file
    descriptor level.

    Strategy:
      - Rebind Python's sys.stderr to a fresh file object over the real
        terminal fd, so tqdm and print() writes go straight to the terminal
        and never enter our pipe. This keeps progress bars (which use \\r
        without trailing \\n) from stalling the filter's line parser.
      - Replace fd 2 with a pipe. Only C-level writes (absl / XLA LOG(...))
        now flow through the pipe, and they are always \\n-terminated and
        well-formed, so a simple line-based filter is reliable.
    """
    global _log_filter_installed
    if _log_filter_installed:
        return
    _log_filter_installed = True

    py_stderr_fd = os.dup(2)
    try:
        sys.stderr.flush()
    except Exception:
        pass
    sys.stderr = os.fdopen(py_stderr_fd, "w", encoding="utf-8",
                           errors="replace", buffering=1)

    out_fd = os.dup(2) 

    r_fd, w_fd = os.pipe()
    os.dup2(w_fd, 2)
    os.close(w_fd)

    def pump():
        reader = os.fdopen(r_fd, "rb", buffering=0)
        out = os.fdopen(out_fd, "wb", buffering=0)
        buf = b""
        skipping = False
        try:
            while True:
                chunk = reader.read(65536)
                if not chunk:
                    break
                buf += chunk
                while True:
                    idx = buf.find(b"\n")
                    if idx == -1:
                        break
                    line = bytes(buf[:idx])
                    buf = buf[idx + 1:]
                    is_log_start = bool(_ABSL_LOG_START.match(line))
                    if skipping:
                        if is_log_start:
                            if _NOISY_LOG_HEADER.search(line):
                                continue
                            skipping = False
                            out.write(line + b"\n")
                        # else: continuation body of a skipped log block — drop
                    else:
                        if is_log_start and _NOISY_LOG_HEADER.search(line):
                            skipping = True
                            continue
                        out.write(line + b"\n")
        except Exception:
            pass

    t = threading.Thread(target=pump, daemon=True, name="xla-log-filter")
    t.start()


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
      
if "latency_hiding_scheduler" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_latency_hiding_scheduler=true"
    ).strip()
_install_xla_log_filter()

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        sys.exit(0)

    parser = argparse.ArgumentParser(prog="san", add_help=False)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("pretrain", add_help=False)
    p.add_argument("--name", type=str, default="san",
                   help="Experiment name for checkpoints and wandb (default: san)")
    p.add_argument("--dataset", type=str, default="synth",
                   help="Registry name (synth, fineweb-edu) or any HF repo id (default: synth)")
    p.add_argument("--text-field", type=str, default=None,
                   help="Text field name for ad-hoc HF datasets (default: text)")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Pre-tokenized memmap corpus dir (from `san tokenize-corpus`). "
                        "When set, replaces HF streaming: faster, exact global shuffle")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Resume from a format-v2 checkpoint")
    p.add_argument("--resume-step", type=int, default=None,
                   help="Override resume step (skip this many batches)")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Per-device batch size in packed blocks (default: 64)")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--max-steps", type=int, default=100_000,
                   help="Total training steps — also the WSD schedule horizon (default: 100000). "
                        "Token budget = max_steps x batch_size x num_gpus x seq_len")
    p.add_argument("--max-docs", type=int, default=None,
                   help="Cap unique training documents per epoch; the stream cycles over "
                        "this fixed subset (default: full dataset). For data-scaling runs")
    p.add_argument("--optimizer", type=str, default="muon", choices=["muon", "adamw"])
    p.add_argument("--ffn", action=argparse.BooleanOptionalAction, default=False,
                   help="Include FFN sublayers (control arm). Default: attention-only SAN")
    p.add_argument("--lr", type=float, default=None,
                   help="Adam LR. Default: E0 locks — adamw runs: SAN 6e-4 / FFN 2.4e-3; "
                        "muon runs keep the adam side at 3e-4 (as swept)")
    p.add_argument("--muon-lr", type=float, default=None,
                   help="Muon LR. Default: E0 locks — SAN 0.02 / FFN 0.04")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=20)
    p.add_argument("--d-ff", type=int, default=None,
                   help="FFN width when --ffn (default: 4*d_model)")
    p.add_argument("--activation", type=str, default="swiglu", choices=["swiglu", "geglu", "drelu"])
    p.add_argument("--residual", type=str, default="gated",
                   choices=["gated", "rezero", "standard", "none"],
                   help="Residual variant (E3): gated sigmoid (default), ReZero alpha, "
                        "standard skip, or no skip")
    p.add_argument("--norm", type=str, default="zcrms", choices=["zcrms", "rms"],
                   help="Norm variant (E3): zero-centered RMSNorm (default) or standard RMSNorm")
    p.add_argument("--no-qk-norm", action="store_true",
                   help="Disable QK-norm (E3)")
    p.add_argument("--post-attn-norm", action="store_true",
                   help="Sandwich norm: extra norm on sublayer outputs before the gate (E3)")
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--decay-ratio", type=float, default=0.15)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1000,
                   help="Save checkpoint every N steps (default: 1000)")
    p.add_argument("--upload-checkpoints", action="store_true",
                   help="Upload checkpoints to HF hub (default: local only). Uploads: "
                        "rolling ckpt every --upload-every steps, all milestones, final")
    p.add_argument("--upload-every", type=int, default=10_000,
                   help="Rolling-checkpoint upload cadence in steps (default: 10000)")
    p.add_argument("--eval-every", type=int, default=500,
                   help="Val loss + gate logging cadence (default: 500)")
    p.add_argument("--val-blocks", type=int, default=16,
                   help="Held-out packed blocks for val loss (default: 16)")
    p.add_argument("--log-rank-every", type=int, default=0,
                   help="Representation-rank logging cadence, 0 = off (default: 0)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    p = sub.add_parser("tokenize-corpus", add_help=False)
    p.add_argument("--dataset", type=str, default="synth")
    p.add_argument("--text-field", type=str, default=None)
    p.add_argument("--out-dir", type=str, default=None,
                   help="Output dir (default: data/<dataset>). Use the network "
                        "volume, e.g. /workspace/data/synth, so it survives pod loss")
    p.add_argument("--max-files", type=int, default=None,
                   help="Only process the first N parquet shards (smoke test)")
    p.add_argument("--force", action="store_true",
                   help="Discard existing progress and restart from scratch")
    p.add_argument("--upload", action="store_true",
                   help="After tokenizing, upload the corpus to HF (split into 10GB parts)")

    p = sub.add_parser("download-corpus", add_help=False)
    p.add_argument("--dataset", type=str, default="synth")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Destination dir (default: data/<dataset>)")

    p = sub.add_parser("tokenizer-train", add_help=False)
    p.add_argument("--dataset", type=str, default="synth")
    p.add_argument("--text-field", type=str, default=None)
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--max-docs", type=int, default=2_000_000)
    p.add_argument("--force", action="store_true")
    p.add_argument("--upload", action="store_true",
                   help="Upload trained tokenizer to HF hub (to share across machines)")

    p = sub.add_parser("spectra", add_help=False)
    p.add_argument("checkpoints", nargs="+",
                   help="Checkpoint .pkl paths (milestone globs: <name>_step*.pkl; "
                        "globs need local files — hf download the milestones first)")
    p.add_argument("--out", type=str, default="spectra",
                   help="Output dir for per-checkpoint .npz (default: spectra)")

    p = sub.add_parser("sample", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, default=None, help="Prompt text to continue")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("eval", add_help=False)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, default="synth",
                   help="Dataset for held-out val loss (default: synth)")
    p.add_argument("--text-field", type=str, default=None,
                   help="Text field name for ad-hoc HF datasets")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-blocks", type=int, default=64,
                   help="Number of packed validation blocks (default: 64)")
    p.add_argument("--throughput-runs", type=int, default=10)
    p.add_argument("--by-exercise", action="store_true",
                   help="E4: slice val loss by document group (see --group-field)")
    p.add_argument("--by-region", action="store_true",
                   help="E4: slice val loss by token region (query/trace/answer)")
    p.add_argument("--group-field", type=str, default="exercise",
                   help="Metadata field for --by-exercise grouping (default: exercise; "
                        "e.g. language)")
    p.add_argument("--val-docs", type=int, default=20_000,
                   help="Documents for the E4 decomposition (default: 20000)")
    p.add_argument("--tasks", type=str, nargs="*", default=None,
                   help="Reserved: downstream tasks (lm-eval-harness adapter, future)")

    args = parser.parse_args()

    if not args.command:
        print(HELP)
        sys.exit(0)

    if args.command == "pretrain":
        from .pretrain import pretrain
        pretrain(args)
    elif args.command == "tokenize-corpus":
        from .data import tokenize_corpus, upload_corpus
        out = tokenize_corpus(
            dataset=args.dataset, text_field=args.text_field,
            out_dir=args.out_dir, force=args.force, max_files=args.max_files,
        )
        if args.upload:
            upload_corpus(out, dataset=args.dataset)
    elif args.command == "download-corpus":
        from .data import download_corpus
        download_corpus(dataset=args.dataset, out_dir=args.out_dir)
    elif args.command == "tokenizer-train":
        from .tokenizer import train_tokenizer
        train_tokenizer(
            dataset=args.dataset, text_field=args.text_field,
            vocab_size=args.vocab_size, max_docs=args.max_docs,
            force=args.force, upload=args.upload,
        )
    elif args.command == "spectra":
        from .spectra import main as spectra_main
        spectra_main(args)
    elif args.command == "sample":
        from .run import main as run_main
        run_main(args)
    elif args.command == "eval":
        from .eval import main as eval_main_fn
        eval_main_fn(args)
