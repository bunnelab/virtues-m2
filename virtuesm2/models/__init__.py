import logging

from . import virtuesm2 as vits

from .multiplex_tokenizer import build_multiplex_tokenizer
from ..utils.utils import load_marker_embeddings
import copy
import torch


logger = logging.getLogger("virtues-m2")


def build_virtuesm2_model(cfg, only_teacher=False, ckpt_path=None):
    esm_embds = load_marker_embeddings(cfg.marker_embedding_dir)
    esm_embds = esm_embds.to(dtype=torch.float16)
    match cfg.student.arch:
        case "vit_base":
            model_vit_dim = 768
        case "vit_large":
            model_vit_dim = 1024
        case "vit_huge":
            model_vit_dim = 1280
        case "vit_huge_uni":
            model_vit_dim = 1536
        case "vit_giant":
            model_vit_dim = 1408

    assert cfg.model.model_dim == model_vit_dim, (
        f"Model for multiplex embedding dimension {cfg.model.model_dim} does not match the expected dimension {model_vit_dim} for architecture {cfg.student.arch}"
    )
    multiplex_tokenizer = build_multiplex_tokenizer(cfg, esm_embds, only_teacher=only_teacher)
    model, _ = build_model_from_cfg(
        cfg, only_teacher=only_teacher, mx_embed_layer=multiplex_tokenizer, norm_after_he_tokenizer=cfg.model.norm_after_he_tokenizer
    )

    if ckpt_path:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        state_dict = state_dict["teacher"]
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        load_status = model.load_state_dict(state_dict, strict=False)
        print("Loaded model with status:", load_status)
    return model


def build_model_from_cfg(cfg, only_teacher=False, mx_embed_layer=None, norm_after_he_tokenizer=False, he_embed_layer=None):
    return build_model(
        cfg.student,
        only_teacher=only_teacher,
        img_size=cfg.crops.global_crops_size,
        mx_embed_layer=mx_embed_layer,
        norm_after_he_tokenizer=norm_after_he_tokenizer,
        he_embed_layer=he_embed_layer,
    )


def build_model(args, only_teacher=False, img_size=224, mx_embed_layer=None, he_embed_layer=None, norm_after_he_tokenizer=False):
    args.arch = args.arch.removesuffix("_memeff")
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=args.patch_size,
            init_values=args.layerscale,
            ffn_layer=args.ffn_layer,
            block_chunks=args.block_chunks,
            qkv_bias=args.qkv_bias,
            proj_bias=args.proj_bias,
            ffn_bias=args.ffn_bias,
            num_register_tokens=args.num_register_tokens,
            interpolate_offset=args.interpolate_offset,
            interpolate_antialias=args.interpolate_antialias,
            mx_embed_layer=mx_embed_layer,
            he_embed_layer=he_embed_layer,
            norm_after_he_tokenizer=norm_after_he_tokenizer,
        )

        if mx_embed_layer is None:
            teacher = vits.__dict__[args.arch](**vit_kwargs)
            student = vits.__dict__[args.arch](
                **vit_kwargs,
                drop_path_rate=args.drop_path_rate,
                drop_path_uniform=args.drop_path_uniform,
            )
        else:
            model_vit_size = args.arch
            match model_vit_size:
                case "vit_base":
                    model_vit_dim = 768
                case "vit_large":
                    model_vit_dim = 1024
                case "vit_huge":
                    model_vit_dim = 1280
                case "vit_huge_uni":
                    model_vit_dim = 1536
                case "vit_giant":
                    model_vit_dim = 1408

            teacher = vits.__dict__[args.arch](**vit_kwargs)
            teacher.mx_embed_layer = copy.deepcopy(mx_embed_layer)
            if he_embed_layer is not None:
                teacher.he_embed_layer = copy.deepcopy(he_embed_layer)

            student = vits.__dict__[args.arch](
                **vit_kwargs,
                drop_path_rate=args.drop_path_rate,
                drop_path_uniform=args.drop_path_uniform,
            )
            student.mx_embed_layer = copy.deepcopy(mx_embed_layer)
            if he_embed_layer is not None:
                student.he_embed_layer = copy.deepcopy(he_embed_layer)
        embed_dim = student.embed_dim
    if only_teacher:
        return teacher, teacher.embed_dim
    else:
        return student, teacher, embed_dim
