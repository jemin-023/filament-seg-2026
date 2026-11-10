"""U-Net with a pretrained encoder, 1-channel input, 3 outputs (semantic mask, spine, boundary)."""
import segmentation_models_pytorch as smp
import torch

ENCODERS = {
    "effv2s": "tu-tf_efficientnetv2_s",
    "r34": "resnet34",
}


def build_model(arch="effv2s", pretrained=True) -> torch.nn.Module:
    return smp.Unet(
        encoder_name=ENCODERS[arch],
        encoder_weights="imagenet" if pretrained else None,
        in_channels=1,
        classes=3,
        decoder_attention_type="scse",
    )


def load_model(path, device="cuda") -> torch.nn.Module:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(ck["cfg"]["arch"], pretrained=False)
    m.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
    m.cfg = ck["cfg"]
    return m.to(device).eval()
