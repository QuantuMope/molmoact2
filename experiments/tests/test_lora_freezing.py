from itertools import product

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer

from olmo.train.run_trainer import _collect_lora_targets


def linear_block():
    module = torch.nn.Module()
    module.proj = torch.nn.Linear(4, 4)
    return module


@pytest.mark.parametrize("ft_llm,ft_vit,ft_connector", product((False, True), repeat=3))
def test_lora_respects_module_freezing(ft_llm, ft_vit, ft_connector):
    transformer = torch.nn.Module()
    transformer.blocks = linear_block()
    transformer.requires_grad_(ft_llm)

    vision = torch.nn.Module()
    # Repeated leaf names must not cause connector targets to match ViT layers.
    vision.image_vit = linear_block().requires_grad_(ft_vit)
    vision.image_pooling_2d = linear_block().requires_grad_(ft_connector)
    vision.image_projector = linear_block().requires_grad_(ft_connector)

    for module, expected in (
        (transformer, {"blocks.proj"} if ft_llm else set()),
        (vision, ({"image_vit.proj"} if ft_vit else set()) |
         ({"image_pooling_2d.proj", "image_projector.proj"} if ft_connector else set())),
    ):
        frozen = [parameter for parameter in module.parameters() if not parameter.requires_grad]
        before = [parameter.detach().clone() for parameter in frozen]
        targets = _collect_lora_targets(module)
        assert set(targets) == expected
        if targets:
            wrapped = get_peft_model(module, LoraConfig(r=2, target_modules=targets))
            assert wrapped.get_base_model() is module
        adapters = {name for name, layer in module.named_modules() if isinstance(layer, LoraLayer)}
        assert adapters == expected

        if targets:
            optimizer = torch.optim.AdamW([p for p in module.parameters() if p.requires_grad], lr=.01)
            x = torch.randn(2, 4)
            loss = sum(module.get_submodule(name)(x).square().mean() for name in targets)
            loss.backward()
            optimizer.step()
            for name in targets:
                assert module.get_submodule(name).lora_B["default"].weight.grad is not None
        for parameter, original in zip(frozen, before):
            assert not parameter.requires_grad
            assert parameter.grad is None
            torch.testing.assert_close(parameter, original, rtol=0, atol=0)


def test_lora_does_not_reenable_frozen_output_head():
    transformer = torch.nn.Module()
    transformer.blocks = linear_block()
    transformer.ff_out = torch.nn.Linear(4, 4).requires_grad_(False)
    targets = _collect_lora_targets(transformer)
    assert targets == ["blocks.proj"]
    get_peft_model(transformer, LoraConfig(r=2, target_modules=targets))
    assert not isinstance(transformer.ff_out, LoraLayer)
    assert all(not parameter.requires_grad for parameter in transformer.ff_out.parameters())
