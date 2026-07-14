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

    # 1. Dup the original stderr fd; rebind Python's sys.stderr to it so
    #    all Python-level writes bypass the pipe.
    py_stderr_fd = os.dup(2)
    try:
        sys.stderr.flush()
    except Exception:
        pass
    sys.stderr = os.fdopen(py_stderr_fd, "w", encoding="utf-8",
                           errors="replace", buffering=1)

    # 2. Dup another copy for the filter thread to write filtered C-level
    #    output to.
    out_fd = os.dup(2)  # before dup2 below, fd 2 is still the real terminal

    # 3. Redirect fd 2 to the write end of a new pipe. All C-level writes
    #    from XLA now land in the pipe.
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
_install_xla_log_filter()

from .dataset.tokenizer import DEFAULT_MAX_ENC_LEN, DEFAULT_MAX_DEC_LEN, DEFAULT_MAX_GEN_LEN


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(HELP)
        sys.exit(0)

    parser = argparse.ArgumentParser(prog="needle", add_help=False)
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("pretrain", add_help=False)
    p.add_argument("--name", type=str, default="pretrain",
                   help="Experiment name for wandb (default: pretrain)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Resume from checkpoint (e.g. checkpoints/needle_base.pkl)")
    p.add_argument("--resume-step", type=int, default=None,
                   help="Override resume step (skip this many batches)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-kv-heads", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--num-dec-layers", type=int, default=8)
    p.add_argument("--max-enc-len", type=int, default=DEFAULT_MAX_ENC_LEN)
    p.add_argument("--max-dec-len", type=int, default=DEFAULT_MAX_DEC_LEN)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--decay-ratio", type=float, default=0.05)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=None,
                   help="Stop after N steps (default: full epoch)")
    p.add_argument("--save-every", type=int, default=1000,
                   help="Save and upload checkpoint every N steps (default: 1000)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")

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
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--val-blocks", type=int, default=64,
                   help="Number of packed validation blocks (default: 64)")
    p.add_argument("--throughput-runs", type=int, default=10)
    p.add_argument("--tasks", type=str, nargs="*", default=None,
                   help="Reserved: downstream tasks (lm-eval-harness adapter, future)")

    p = sub.add_parser("tpu", add_help=False)
    tpu_sub = p.add_subparsers(dest="tpu_action")

    tp = tpu_sub.add_parser("create", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--type", dest="accel_type", type=str, default="v6e-8")
    tp.add_argument("--version", type=str, default=None,
                    help="Software version (auto-detected from --type if omitted)")
    tp.add_argument("--preemptible", action="store_true", default=False,
                    help="Create a preemptible (spot) TPU VM")

    tp = tpu_sub.add_parser("connect", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("setup", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("sync", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("pretrain", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)
    tp.add_argument("train_args", nargs=argparse.REMAINDER,
                    help="Extra args passed to needle pretrain")

    tp = tpu_sub.add_parser("claude", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("stop", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("start", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tp = tpu_sub.add_parser("delete", add_help=False)
    tp.add_argument("name", type=str)
    tp.add_argument("--zone", type=str, default=None)

    tpu_sub.add_parser("list", add_help=False)

    args = parser.parse_args()

    if not args.command:
        print(HELP)
        sys.exit(0)

    if args.command == "pretrain":
        import jax
        if os.path.exists("/dev/accel0"):
            jax.distributed.initialize()
        from .training.pretrain import pretrain
        pretrain(args)
    elif args.command == "sample":
        from .model.run import main as run_main
        run_main(args)
    elif args.command == "eval":
        from .training.eval import main as eval_main_fn
        eval_main_fn(args)
    elif args.command == "tpu":
        from .utils.tpu import tpu_dispatch
        tpu_dispatch(args)
