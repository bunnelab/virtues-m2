import os
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from virtuesm2.segmentation.decoder import VM2_Segmentation
from instanseg.utils.loss.instanseg_loss import InstanSeg

class VM2_Segmentation_HF(VM2_Segmentation, PyTorchModelHubMixin):
    """
    HuggingFace compatible version of VirTuesM2-segmentation. This allows to directly use "from_pretrained" but expects the marker embedding directory to be prepared before.
    """
    SAFETENSORS_CKPT = "virtuesm2_multimodal_segmentation.safetensors"

    def __init__(
        self,
        virtuesm2_model,
        num_celltypes=9, 
        extract_layers=[6, 12, 18, 24], 
        model_dim=2048,
    ):
        instanseg_processor = InstanSeg(
            n_sigma=2,
            window_size=128,
        )

        super().__init__(
            virtuesm2_model=virtuesm2_model,
            dim_out=instanseg_processor.dim_out,
            num_celltypes=num_celltypes,
            extract_layers=extract_layers,
            model_dim=model_dim,
            mode="multimodal",
        )

        instanseg_processor.initialize_pixel_classifier(self, MLP_width=5)
        self.eval()
        self.instance_processor = instanseg_processor
        # return segmentation_model
    
    @classmethod
    def _from_pretrained(
        cls, *, model_id, revision, cache_dir, force_download, proxies,
        resume_download, local_files_only, token,
        subfolder=None, map_location="cpu", strict=False, **model_kwargs,
    ):
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            path = os.path.join(model_id, subfolder or "", VM2_Segmentation_HF.SAFETENSORS_CKPT)
        else:
            path = hf_hub_download(
                repo_id=model_id, filename=VM2_Segmentation_HF.SAFETENSORS_CKPT,
                subfolder=subfolder, revision=revision, cache_dir=cache_dir,
                force_download=force_download, proxies=proxies,
                resume_download=resume_download, token=token,
                local_files_only=local_files_only,
            )
        return cls._load_as_safetensor(model, path, map_location, strict)