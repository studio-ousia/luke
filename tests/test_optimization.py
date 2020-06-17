import torch

from luke.optimization import LukeAdamW


def test_luke_adam_w():
    w = torch.tensor([0.1, -0.2, -0.1], requires_grad=True)
    target = torch.tensor([0.4, 0.2, -0.5])
    criterion = torch.nn.MSELoss()
    optimizer = LukeAdamW(params=[w], lr=2e-1, weight_decay=0.0)
    for _ in range(100):
        loss = criterion(w, target)
        loss.backward()
        optimizer.step()
        w.grad.detach_()  # No zero_grad() function on simple tensors. we do it ourselves.
        w.grad.zero_()

    assert torch.allclose(w, target, atol=0.01)
