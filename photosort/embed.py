from __future__ import annotations
import numpy as np, torch, open_clip
from PIL import Image
from .config import CLIP_MODEL, CLIP_PRETRAINED, EMBED_BATCH

class Embedder:
    def __init__(self, device: str | None = None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self._model = self._pre = self._tok = None

    def _load(self):
        if self._model is None:
            m, _, pre = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
            self._model = m.eval().to(self.device); self._pre = pre
            self._tok = open_clip.get_tokenizer(CLIP_MODEL)

    @torch.no_grad()
    def encode_images(self, ims: list[Image.Image]) -> np.ndarray:
        self._load()
        out = []
        for i in range(0, len(ims), EMBED_BATCH):
            x = torch.stack([self._pre(im.convert("RGB")) for im in ims[i:i + EMBED_BATCH]]).to(self.device)
            f = self._model.encode_image(x)
            out.append((f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        self._load()
        texts = [t if t.lower().startswith("a photo") else f"a photo of {t}" for t in texts]
        f = self._model.encode_text(self._tok(texts).to(self.device))
        return (f / f.norm(dim=-1, keepdim=True)).float().cpu().numpy()

_E: Embedder | None = None
def get_embedder() -> Embedder:
    global _E
    if _E is None:
        _E = Embedder()
    return _E
