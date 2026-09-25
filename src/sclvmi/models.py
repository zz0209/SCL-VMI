import importlib.util
from pathlib import Path

import torch
from monai.networks.nets import resnet50, vista3d132
from monai.networks.nets.segresnet_ds import SegResEncoder
from safetensors.torch import load_file

from .context import ROOT, context, read_json


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
