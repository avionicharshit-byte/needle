import os

import sentencepiece as spm

# Artifacts live flat next to the code. The artifact is named needle_lm (not
# needle) on purpose: the old tool-calling tokenizer ships as needle.model on
# dev machines and on HF, and must never be silently picked up by this branch —
# its vocab was fit to tool-call JSON.
TOKENIZER_DIR = os.path.dirname(__file__)
TOKENIZER_PREFIX = os.path.join(TOKENIZER_DIR, "needle_lm")

PAD_ID = 0
EOS_ID = 1
BOS_ID = 2
UNK_ID = 3

_HF_MODEL_REPO = "Cactus-Compute/needle"
_HF_TOKENIZER_DIR = "tokenizer_lm"


class NeedleTokenizer:
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
    for fname in ["needle_lm.model", "needle_lm.vocab"]:
        hf_hub_download(
            repo_id=_HF_MODEL_REPO,
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
                f"Run `needle tokenizer-train` (add --upload for multi-host TPU use)."
            ) from e
    return NeedleTokenizer(model_path)


def train_tokenizer(dataset="synth", text_field=None, vocab_size=8192,
                    max_docs=2_000_000, force=False, upload=False):
    """Train a SentencePiece BPE tokenizer on the pretraining corpus.

    Streams text through the same formatter registry as training, so the
    tokenizer sees exactly the distribution the model will train on.
    """
    from tqdm import tqdm
    from .pretraining import resolve_dataset, stream_texts

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
            byte_fallback=True,
            normalization_rule_name="identity",
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
        for ext in (".model", ".vocab"):
            api.upload_file(
                path_or_fileobj=TOKENIZER_PREFIX + ext,
                path_in_repo=f"{_HF_TOKENIZER_DIR}/needle_lm{ext}",
                repo_id=_HF_MODEL_REPO,
                repo_type="model",
            )
        print(f"Uploaded tokenizer to {_HF_MODEL_REPO}/{_HF_TOKENIZER_DIR}/")
    return model_path
