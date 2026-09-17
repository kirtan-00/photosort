import numpy as np
from PIL import Image
from photosort.embed import get_embedder

def test_text_and_image_agree():
    e = get_embedder()
    red = Image.new("RGB", (256, 256), (220, 20, 20)); blue = Image.new("RGB", (256, 256), (20, 20, 220))
    I = e.encode_images([red, blue]); T = e.encode_text(["a red square", "a blue square"])
    assert I.shape == (2, 512) and abs(np.linalg.norm(I[0]) - 1) < 1e-3
    S = T @ I.T
    assert S[0, 0] > S[0, 1] and S[1, 1] > S[1, 0]

def test_offline_flag_set_before_hub_import():
    import os, photosort.embed  # noqa: F401
    import huggingface_hub.constants as C
    assert os.environ.get("HF_HUB_OFFLINE") in ("1", "0")   # "0" only after a one-time online fallback
    assert isinstance(C.HF_HUB_OFFLINE, bool)

def test_embedder_singleton_and_lock():
    import threading
    from photosort import embed as E
    seen = set()
    def grab(): seen.add(id(E.get_embedder()))
    ts = [threading.Thread(target=grab) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert len(seen) == 1 and E._LOCK is not None
