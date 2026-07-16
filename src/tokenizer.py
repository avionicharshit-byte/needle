import os

import sentencepiece as spm

TOKENIZER_DIR = os.path.dirname(__file__)
TOKENIZER_PREFIX = os.path.join(TOKENIZER_DIR, "tokenizer")

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3

# Single-turn ChatML markers (PleIAs Monad/Baguettotron-compatible format).
# Registered as atomic tokenizer symbols so trace/answer regions can be
# located exactly in token space for per-region loss analysis.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_START = "<think>"
THINK_END = "</think>"
CHAT_MARKERS = [IM_START, IM_END, THINK_START, THINK_END]
IM_START_ID, IM_END_ID, THINK_START_ID, THINK_END_ID = 4, 5, 6, 7

HF_REPO = "Cactus-Compute/simple-attention-networks"
_HF_TOKENIZER_DIR = "tokenizer"


class SANTokenizer:
    """Wrapper around SentencePiece providing the interface the codebase expects."""

    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    @property
    def pad_token_id(self):
        return PAD_ID

    @property
    def eos_token_id(self):
        return EOS_ID

    @property
    def bos_token_id(self):
        return BOS_ID

    @property
    def vocab_size(self):
        return self.sp.GetPieceSize()

    def encode(self, text):
        return self.sp.Encode(text, out_type=int)

    def decode(self, ids):
        if isinstance(ids, (list, tuple)) and len(ids) > 0 and isinstance(ids[0], (list, tuple)):
            return [self.sp.Decode(seq) for seq in ids]
        return self.sp.Decode(list(ids))

    def __call__(self, texts, truncation=True, max_length=None, **kwargs):
        all_ids = []
        for text in texts:
            ids = self.sp.Encode(text, out_type=int)
            if truncation and max_length:
                ids = ids[:max_length]
            all_ids.append(ids)
        return {"input_ids": all_ids}


def _download_tokenizer_from_hf():
    """Download tokenizer files from HuggingFace Hub into TOKENIZER_DIR."""
    from huggingface_hub import hf_hub_download

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    for fname in ["tokenizer.model", "tokenizer.vocab"]:
        hf_hub_download(
            repo_id=HF_REPO,
            filename=f"{_HF_TOKENIZER_DIR}/{fname}",
            repo_type="model",
            local_dir=TOKENIZER_DIR,
        )
        nested = os.path.join(TOKENIZER_DIR, _HF_TOKENIZER_DIR, fname)
        dst = os.path.join(TOKENIZER_DIR, fname)
        if os.path.exists(nested) and not os.path.exists(dst):
            os.rename(nested, dst)


def get_tokenizer():
    model_path = TOKENIZER_PREFIX + ".model"
    if not os.path.exists(model_path):
        try:
            print("Downloading pretraining tokenizer from HuggingFace...")
            _download_tokenizer_from_hf()
        except Exception as e:
            raise RuntimeError(
                f"No pretraining tokenizer at {model_path} and HF download failed ({e}). "
                f"Run `san tokenizer-train` (add --upload to share it via HF hub)."
            ) from e
    return SANTokenizer(model_path)


def train_tokenizer(dataset="synth", text_field=None, vocab_size=16384,
                    max_docs=2_000_000, force=False, upload=False):
    """Train a SentencePiece BPE tokenizer on the pretraining corpus.

    Streams text through the same formatter registry as training, so the
    tokenizer sees exactly the distribution the model will train on.
    """
    from tqdm import tqdm
    from .data import resolve_dataset, stream_texts

    model_path = TOKENIZER_PREFIX + ".model"
    if os.path.exists(model_path) and not force:
        print(f"Tokenizer already exists at {model_path} (use --force to retrain)")
        return model_path

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    spec = resolve_dataset(dataset, text_field)
    print(f"Training SentencePiece BPE tokenizer (vocab_size={vocab_size}, "
          f"corpus={spec.repo}, max_docs={max_docs:,})...")

    corpus_path = os.path.join(TOKENIZER_DIR, "corpus.txt")
    try:
        with open(corpus_path, "w") as f:
            texts = stream_texts(spec, shuffle=False, take=max_docs)
            for text in tqdm(texts, total=max_docs, desc="Writing corpus"):
                # docs with internal newlines simply span multiple corpus lines
                f.write(text + "\n")

        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=TOKENIZER_PREFIX,
            vocab_size=vocab_size,
            model_type="bpe",
            pad_id=PAD_ID,
            eos_id=EOS_ID,
            bos_id=BOS_ID,
            unk_id=UNK_ID,
            user_defined_symbols=CHAT_MARKERS,
            byte_fallback=True,
            normalization_rule_name="identity",
            remove_extra_whitespaces=False,
            input_sentence_size=2_000_000,
            shuffle_input_sentence=True,
            num_threads=min(128, max(1, (os.cpu_count() or 1) * 3 // 4)),
            train_extremely_large_corpus=False,
            minloglevel=2,
        )
    finally:
        if os.path.exists(corpus_path):
            os.remove(corpus_path)
    print(f"Tokenizer saved to {model_path}")

    if upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
        for ext in (".model", ".vocab"):
            api.upload_file(
                path_or_fileobj=TOKENIZER_PREFIX + ext,
                path_in_repo=f"{_HF_TOKENIZER_DIR}/tokenizer{ext}",
                repo_id=HF_REPO,
                repo_type="model",
            )
        print(f"Uploaded tokenizer to {HF_REPO}/{_HF_TOKENIZER_DIR}/")
    return model_path
