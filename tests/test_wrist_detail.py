import torch

from slim.model.wrist_detail import HighResolutionWristAdapter


def test_wrist_detail_adapter_is_exact_noop_at_initialization():
    adapter = HighResolutionWristAdapter(dim=384, bottleneck_dim=96)
    base = torch.randn(2, 196, 384)
    high_resolution = torch.randn(2, 256, 384)

    output = adapter(base, high_resolution, (14, 14), (16, 16))

    assert output.shape == base.shape
    assert torch.equal(output, base)


def test_wrist_detail_zero_output_projection_receives_gradient():
    adapter = HighResolutionWristAdapter(dim=32, bottleneck_dim=8)
    base = torch.randn(2, 4, 32)
    high_resolution = torch.randn(2, 9, 32)

    adapter(base, high_resolution, (2, 2), (3, 3)).square().mean().backward()

    assert adapter.up.weight.grad is not None
    assert torch.count_nonzero(adapter.up.weight.grad) > 0
