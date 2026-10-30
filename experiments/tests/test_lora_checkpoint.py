from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.checkpoint.state_dict import get_model_state_dict

from olmo.train.checkpointer import _slice_state_dict, merge_and_save_unsharded


def linear_block():
    return nn.Sequential(nn.Linear(4, 4))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.ModuleDict({"blocks": linear_block()})
        self.vision_backbone = nn.ModuleDict({
            "image_vit": linear_block(),
            "image_pooling_2d": linear_block(),
            "image_projector": linear_block(),
        })
        self.action_expert = nn.Linear(4, 2)

    def forward(self, x):
        vision = self.vision_backbone
        image = vision.image_projector(vision.image_pooling_2d(vision.image_vit(x)))
        return self.action_expert(self.transformer.blocks(x) + image)


@dataclass
class TinyModelConfig:
    def build_model(self):
        return TinyModel()


@dataclass
class TinyTrainConfig:
    model: TinyModelConfig = field(default_factory=TinyModelConfig)


class LoraCheckpointTest(unittest.TestCase):
    def make_checkpoint(self, root, *, checkpointing, freeze_vit):
        torch.manual_seed(1)
        model = TinyModel()
        model.vision_backbone["image_vit"].requires_grad_(not freeze_vit)
        if checkpointing:
            model.transformer["blocks"] = checkpoint_wrapper(model.transformer["blocks"])
            for name in ("image_pooling_2d", "image_projector"):
                model.vision_backbone[name] = checkpoint_wrapper(model.vision_backbone[name])
        for name in ("transformer", "vision_backbone"):
            module = getattr(model, name)
            targets = [key for key, layer in module.named_modules()
                       if isinstance(layer, nn.Linear) and layer.weight.requires_grad]
            wrapped = get_peft_model(module, LoraConfig(r=2, target_modules=targets))
            for key, parameter in wrapped.named_parameters():
                if "lora_" in key:
                    nn.init.normal_(parameter, std=0.1)
            setattr(model, name, wrapped)
        model.eval()
        state = get_model_state_dict(model)
        for name in ("transformer", "vision_backbone"):
            getattr(model, name).save_pretrained(
                root / name, state_dict=_slice_state_dict(state, name), safe_serialization=False
            )
        return model, state

    def test_merge_preserves_full_model(self):
        for checkpointing in (False, True):
            for freeze_vit in (False, True):
                with self.subTest(checkpointing=checkpointing, freeze_vit=freeze_vit), TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    model, state = self.make_checkpoint(root, checkpointing=checkpointing, freeze_vit=freeze_vit)
                    x = torch.randn(3, 4)
                    expected = model(x).detach()
                    merge_and_save_unsharded(
                        root / "merged", model, TinyTrainConfig(),
                        root / "transformer", root / "vision_backbone", full_state=state,
                    )
                    merged = TinyModel()
                    saved = torch.load(root / "merged/model.pt", weights_only=True)
                    merged.load_state_dict(saved, strict=True)
                    torch.testing.assert_close(merged(x), expected)
                    torch.testing.assert_close(merged.action_expert.weight, model.action_expert.weight)
                    self.assertFalse(any("lora_" in key or "_checkpoint_wrapped_module" in key for key in saved))
                    self.assertTrue((root / "merged/config.yaml").is_file())

    def test_merge_rejects_missing_base_weights(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, state = self.make_checkpoint(root, checkpointing=True, freeze_vit=True)
            del state["vision_backbone.base_model.model.image_vit.0.weight"]
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                merge_and_save_unsharded(
                    root / "merged", model, TinyTrainConfig(),
                    root / "transformer", root / "vision_backbone", full_state=state,
                )
            self.assertFalse((root / "merged/model.pt").exists())
