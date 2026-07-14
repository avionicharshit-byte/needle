import os

import jax
import jax.numpy as jnp
import numpy as np

from .tokenizer import HF_REPO


def _replicate(tree):
    """Replicate a pytree across all local devices for pmap."""
    devices = np.array(jax.local_devices())
    n = len(devices)
    mesh = jax.sharding.Mesh(devices, ("d",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("d"))

    def _rep(x):
        x = jnp.asarray(x)
        return jax.device_put(jnp.broadcast_to(x, (n,) + x.shape), sharding)

    return jax.tree.map(_rep, tree)


def _unreplicate(tree):
    """Get a single copy from a pmap-replicated pytree."""
    return jax.tree.map(lambda x: jax.device_get(x[0]), tree)


def shard_batch(batch, num_devices):
    """Reshape a batch array so leading dim is (num_devices, per_device_batch, ...)."""
    return batch.reshape(num_devices, -1, *batch.shape[1:])


def _upload_checkpoint(ckpt_path, wait=False):
    """Upload a checkpoint file to HuggingFace Hub in a background thread.

    wait=True blocks until the upload finishes — required for the final
    checkpoint, because daemon threads are killed at interpreter exit and a
    fire-and-forget upload seconds before process end can be truncated.
    """
    import threading

    def _upload():
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
            filename = os.path.basename(ckpt_path)
            print(f"[hf] Uploading {filename} to {HF_REPO}/checkpoints/ ...")
            api.upload_file(
                path_or_fileobj=ckpt_path,
                path_in_repo=f"checkpoints/{filename}",
                repo_id=HF_REPO,
                repo_type="model",
            )
            print(f"[hf] Checkpoint uploaded: {HF_REPO}/checkpoints/{filename}")
        except Exception as e:
            print(f"[hf] Warning: checkpoint upload failed: {e}")

    t = threading.Thread(target=_upload, daemon=True)
    t.start()
    if wait:
        t.join(timeout=900)
