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
