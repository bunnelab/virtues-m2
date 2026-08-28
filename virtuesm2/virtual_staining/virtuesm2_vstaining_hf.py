import os

from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from virtuesm2.virtual_staining.virtuesm2_vstaining import VM2_VirtualStaining
from virtuesm2.models.virtuesm2 import VirTuesM2

class VM2_VirtualStaining_HF(PyTorchModelHubMixin, VM2_VirtualStaining):
    """VirTues-M2 virtual staining model.

    Loads the frozen VirTues-M2 backbone from HuggingFace (bunnelab/virtues-m2)
    and trains a VirtualStainingDecoder on top to predict target MX channels
    from H&E patch tokens and ESM protein embeddings.

    Parameters
    ----------
    vm2_encoder:
        The pre-trained VirTuesM2 encoder.
    esm_embeddings:
        Float tensor of shape (N_proteins, emb_dim) with one row per protein.
    uniprot_ids:
        Ordered list of UniProt ID strings corresponding to rows in esm_embeddings.
    mode:
        "he" for H&E to MX, "mx" for MX to H&E (default "he").
    decoder_pattern:
        Attention pattern for the decoder (default "ffff").
    decoder_feedforward_dim:
        Feedforward dimension for the decoder (default 2048).
    decoder_num_heads:
        Number of attention heads for the decoder (default 8).
    """
    SAFETENSORS_CKPT = "virtuesm2_vstaining.safetensors"

    def __init__(
        self,
        vm2_encoder: VirTuesM2,
        marker_embedding_dir: str,
        mode: str = "he",
    ):
        super().__init__(
            vm2_encoder=vm2_encoder,
            marker_embedding_dir=marker_embedding_dir,
            mode=mode,
        )

    @classmethod
    def _from_pretrained(
        cls, *, model_id, revision, cache_dir, force_download, proxies,
        resume_download, local_files_only, token,
        subfolder=None, map_location="cpu", strict=False, **model_kwargs,
    ):
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            path = os.path.join(model_id, subfolder or "", VM2_VirtualStaining_HF.SAFETENSORS_CKPT)
        else:
            path = hf_hub_download(
                repo_id=model_id, filename=VM2_VirtualStaining_HF.SAFETENSORS_CKPT,
                subfolder=subfolder, revision=revision, cache_dir=cache_dir,
                force_download=force_download, proxies=proxies,
                resume_download=resume_download, token=token,
                local_files_only=local_files_only,
            )
        return cls._load_as_safetensor(model, path, map_location, strict)