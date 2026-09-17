import numpy as np
import pytest
from PIL import Image, ImageFilter

def make_image(tmp_path, name="a.jpg", size=(1600, 1200), kind="sharp", seed=0):
    """Random high-contrast texture (sharp) or the same blurred (blurry)."""
    rng = np.random.default_rng(seed)
    arr = (rng.random((size[1] // 8, size[0] // 8, 3)) * 255).astype("uint8")
    im = Image.fromarray(arr).resize(size, Image.NEAREST)
    if kind == "blurry":
        im = im.filter(ImageFilter.GaussianBlur(12))
    p = tmp_path / name
    im.save(p, quality=90) if p.suffix.lower() in (".jpg", ".jpeg") else im.save(p)
    return p

@pytest.fixture
def make_img(tmp_path):
    return lambda **kw: make_image(tmp_path, **kw)
