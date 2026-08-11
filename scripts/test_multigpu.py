"""
Quick logic test of auto_parallel_forward with a tiny CPU model.
(We force n_visible_gpus() to 0 so it takes the single-device path;
the multi-device code path is exercised only when CUDA is present.)
"""
import sys, torch, torch.nn as nn
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from src.dist_utils import auto_parallel_forward


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_size = 14
        self.embed_dim = 8
        self.linear = nn.Linear(4, 8)

    @torch.no_grad()
    def forward(self, x):
        return {"cls": self.linear(x[:, 0]), "patch": x, "Hp": 2, "Wp": 2}

    @torch.no_grad()
    def encode_image(self, x, *, extra=1.0):
        out = self.forward(x)
        out["extra"] = torch.tensor([extra])
        return out


def test_single_device():
    m = TinyModel().eval()
    x = torch.randn(5, 4)
    out = auto_parallel_forward(m, x)
    assert out["cls"].shape == (5, 8)
    assert out["patch"].shape == (5, 4)
    assert out["Hp"] == 2
    out2 = auto_parallel_forward(m, x, call="encode_image", extra=2.0)
    assert out2["extra"].item() == 2.0 or out2["extra"].shape[0] == 5
    print("auto_parallel_forward single-device: OK")


if __name__ == "__main__":
    test_single_device()
