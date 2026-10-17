import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from monai.networks.nets import resnet50, vista3d132
from monai.networks.nets.segresnet_ds import SegResEncoder
from safetensors.torch import load_file
from torch import nn

from .context import ROOT, context, read_json


# Architecture presets and Primus argument handling follow MIC-DKFZ/image_classification
# (glovita/models/img_encoder/nnfoundation.py, commit 025e5942f5b6471889e5b8d1e6355ab4a331e402).
RESENC_L = {"n_stages": 6, "features_per_stage": [32, 64, 128, 256, 320, 320], "conv_op": nn.Conv3d, "kernel_sizes": [[3, 3, 3]] * 6,
            "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]], "n_blocks_per_stage": [1, 3, 4, 6, 6, 6],
            "n_conv_per_stage_decoder": [1, 1, 1, 1, 1], "conv_bias": True, "norm_op": nn.InstanceNorm3d, "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None, "dropout_op_kwargs": None, "nonlin": nn.LeakyReLU, "nonlin_kwargs": {"inplace": True}}
PRIMUS_KWARGS = ("embed_dim", "patch_embed_size", "eva_depth", "eva_numheads", "input_shape", "use_rot_pos_emb", "use_abs_pos_embed", "drop_path_rate", "init_values", "scale_attn_inner", "num_register_tokens")
PRIMUS_DEFAULTS = {"use_rot_pos_emb": True, "use_abs_pos_embed": True, "drop_path_rate": 0.2, "init_values": 0.1, "scale_attn_inner": True, "num_register_tokens": 0}


def published(name):
    storage, _, _ = context()
    return read_json(Path(storage["checkpoints"]) / "published" / f"{name}.json")


def load_nnfoundation(name, shape):
    from dynamic_network_architectures.architectures.primus import Primus
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

    manifest = published(name)
    checkpoint = torch.load(Path(manifest["directory"]) / "checkpoint_final.pth", map_location="cpu", weights_only=True)
    plan, weights = checkpoint["nnssl_adaptation_plan"], checkpoint["network_weights"]
    architecture = plan["architecture_plans"]
    if architecture["arch_class_name"] == "ResEncL":
        encoder = ResidualEncoderUNet(input_channels=1, num_classes=1, deep_supervision=False, **RESENC_L).encoder
        encoder.load_state_dict({key.removeprefix("encoder."): value for key, value in weights.items() if key.startswith("encoder.")}, strict=True)
        encoder.return_skips = True
        return encoder.eval().requires_grad_(False), manifest
    assert architecture["arch_class_name"] == "PrimusX", architecture["arch_class_name"]
    kwargs = {key.removeprefix("encoder_"): value for key, value in architecture["arch_kwargs"].items()}
    kwargs = {key: value for key, value in {**PRIMUS_DEFAULTS, **kwargs}.items() if key in PRIMUS_KWARGS}
    kwargs.update(patch_embed_size=tuple(kwargs["patch_embed_size"]), input_shape=tuple(shape))
    model = Primus(input_channels=1, num_classes=1, **kwargs)
    prefixes = (plan["key_to_stem"], plan["key_to_encoder"])
    pretrained = {key: value for key, value in weights.items() if key.startswith(prefixes)}
    source = tuple(size // patch for size, patch in zip(next(iter(plan["pretrain_plan"]["configurations"].values()))["patch_size"], kwargs["patch_embed_size"]))
    target = tuple(size // patch for size, patch in zip(shape, kwargs["patch_embed_size"]))
    embedding = pretrained[plan["key_to_lpe"]]
    prefix, grid = embedding[:, :embedding.shape[1] - source[0] * source[1] * source[2]], embedding[:, embedding.shape[1] - source[0] * source[1] * source[2]:]
    grid = F.interpolate(grid.transpose(1, 2).reshape(1, -1, *source).float(), size=target, mode="trilinear", align_corners=False)
    pretrained[plan["key_to_lpe"]] = torch.cat([prefix, grid.flatten(2).transpose(1, 2).to(embedding.dtype)], 1)
    missing, unexpected = model.load_state_dict(pretrained, strict=False)
    assert not unexpected and not [key for key in missing if key.startswith(prefixes)], (unexpected, missing)
    return model.eval().requires_grad_(False), manifest


def load_flexict(checkpoint):
    manifest = published("flexict")
    sys.path.insert(0, str(Path(manifest["code_directory"])))
    from flexi_ct.flexi_ct_2d import _BACKBONE_KWARGS, _load_teacher_into_backbone
    from flexi_ct.models import flexi_ct_backbone_base

    backbone = flexi_ct_backbone_base(**_BACKBONE_KWARGS)
    path = Path(manifest["directory"]) / checkpoint
    if checkpoint == "ct_3d_teacher.pth":
        _load_teacher_into_backbone(backbone, str(path))
    else:
        weights = torch.load(path, map_location="cpu", weights_only=True)["model"]
        backbone.load_state_dict({key.removeprefix("vision_model."): value for key, value in weights.items() if key.startswith("vision_model.")}, strict=True)
    return backbone.eval().requires_grad_(False), manifest


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_encoder(name):
    storage, _, _ = context()
    manifest = read_json(Path(storage["checkpoints"]) / "published" / f"{name}.json")
    if name == "fmcib":
        model = resnet50(pretrained=False, n_input_channels=1, widen_factor=2, conv1_t_stride=2, feed_forward=False, bias_downsample=True)
        checkpoint = torch.load(manifest["path"], map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["trunk_state_dict"], strict=True)
    elif name == "ctfm":
        directory = Path(manifest["directory"])
        model = SegResEncoder(**read_json(directory / "config.json"))
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
    elif name == "genesis":
        module = source_module("genesis_unet", ROOT / "third_party/genesis/pytorch/unet3d.py")
        model = module.UNet3D()
        checkpoint = torch.load(Path(manifest["directory"]) / "Genesis_Chest_CT.pt", map_location="cpu", weights_only=True)
        weights = {key.removeprefix("module."): value for key, value in checkpoint["state_dict"].items()}
        model.load_state_dict(weights, strict=True)
    elif name == "vista":
        model = vista3d132(encoder_embed_dim=48, in_channels=1)
        weights = torch.load(Path(manifest["directory"]) / "vista3d_pretrained_model/model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
    else:
        raise ValueError(name)
    model.eval().requires_grad_(False)
    return model, manifest


def spatial_features(model, name, image):
    if name == "fmcib":
        image = model.maxpool(model.act(model.bn1(model.conv1(image))))
        for block in [model.layer1, model.layer2, model.layer3, model.layer4]:
            image = block(image)
        return image
    if name == "ctfm":
        return model(image)[-1]
    if name == "genesis":
        for block in [model.down_tr64, model.down_tr128, model.down_tr256, model.down_tr512]:
            image, _ = block(image)
        return image
    if name == "vista":
        return model.image_encoder.encoder(image)[-1]
    raise ValueError(name)
