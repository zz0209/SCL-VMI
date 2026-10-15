import hashlib

import torch
from monai.networks.nets import resnet50, vista3d132

from .models import spatial_features


def random_encoder(name, seed):
    torch.manual_seed(seed)
    if name == "fmcib":
        model = resnet50(pretrained=False, n_input_channels=1, widen_factor=2, conv1_t_stride=2, feed_forward=False, bias_downsample=True)
    else:
        model = vista3d132(encoder_embed_dim=48, in_channels=1)
    return model.eval().requires_grad_(False)


def saved_random_encoder(name, path, digest):
    model = random_encoder(name, 0)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    assert parameter_digest(model) == digest, f"Saved random {name} encoder differs from the recorded parameters"
    return model.eval().requires_grad_(False)


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fmcib_prefix(model, tensor, last):
    value = model.maxpool(model.act(model.bn1(model.conv1(tensor))))
    for name in ["layer1", "layer2"][:["layer1", "layer2"].index(last) + 1]:
        value = getattr(model, name)(value)
    return value


def captured_sites(model, name, tensor, modules):
    captures = {}
    handles = [model.get_submodule(module).register_forward_hook(lambda _, inputs, output, key=site: captures.__setitem__(key, output.detach().clone())) for site, module in modules.items()]
    spatial_features(model, name, tensor)
    for handle in handles:
        handle.remove()
    return captures
