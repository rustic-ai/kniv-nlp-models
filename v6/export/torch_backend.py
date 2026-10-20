"""The export graphs in PyTorch, behind ``kniv_v7``'s backend interface: the
reference the ONNX runtime is checked against, and the way to run a release
with its ``model.pt``.

    from kniv_v7 import Kniv
    from v6.export.torch_backend import TorchBackend
    k = Kniv(path, backend=TorchBackend.from_release(path))    # a release directory
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .graphs import GRAPHS
from .onnx_v7 import SPEC


class TorchBackend:
    def __init__(self, model, device=None):
        torch.backends.mha.set_fastpath_enabled(False)
        self.device = torch.device(device or "cpu")
        self.g = {k: v(model).eval().to(self.device) for k, v in GRAPHS.items()}

    @classmethod
    def load(cls, run: str, local: Path | None = Path("data/v6-eval/checkpoints"), device=None):
        from ..train.gold_eval import HF_REPO, load_run
        model, *_ = load_run(run, HF_REPO, torch.device("cpu"), local)
        return cls(model, device)

    @classmethod
    def from_release(cls, path, device=None):
        """From a release directory's ``model.pt``."""
        from ..train.gold_eval import model_from_checkpoint
        ck = torch.load(Path(path) / "model.pt", map_location="cpu", weights_only=False)
        model, *_ = model_from_checkpoint(ck, torch.device("cpu"))
        return cls(model, device)

    @torch.no_grad()
    def run(self, name: str, **inputs) -> list[np.ndarray]:
        args = []
        for k in SPEC[name][0]:
            v = inputs.get(k)
            if v is None:                     # an input this encoder's graph does not use
                v = np.zeros((1, 1), dtype=np.int64)
            args.append(torch.from_numpy(np.asarray(v)).to(self.device))
        out = self.g[name](*args)
        out = out if isinstance(out, tuple) else (out,)
        return [o.float().cpu().numpy() if o.is_floating_point() else o.cpu().numpy() for o in out]
