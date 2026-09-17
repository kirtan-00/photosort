import numpy as np
from PIL import Image
from photosort.decode import load_preview
from photosort.features import phash, exif_info, sharpness_tiles, eye_sharpness, to_gray

def test_sharp_beats_blurry(make_img):
    s = to_gray(load_preview(make_img(name="s.jpg", kind="sharp")))
    b = to_gray(load_preview(make_img(name="b.jpg", kind="blurry")))
    assert sharpness_tiles(s)[0] > 5 * sharpness_tiles(b)[0]

def test_phash_stable_under_resize(make_img):
    p = make_img(name="p.jpg")
    im = Image.open(p)
    import imagehash
    h1 = phash(im); h2 = phash(im.resize((800, 600)))
    assert len(h1) == 16 and (imagehash.hex_to_hash(h1) - imagehash.hex_to_hash(h2)) <= 4

def test_exif_missing_is_none(make_img):
    info = exif_info(make_img(name="e.jpg"))
    assert info["taken_at"] is None and info["width"] == 1600

def test_eye_sharpness_uses_eye_region():
    gray = np.zeros((400, 400), np.uint8)
    gray[180:220, 120:280] = (np.random.default_rng(0).random((40, 160)) * 255).astype(np.uint8)
    lm = np.array([[150, 200], [250, 200], [200, 260], [170, 320], [230, 320]], float)
    assert eye_sharpness(gray, lm) > 100
    assert eye_sharpness(np.zeros((400, 400), np.uint8), lm) == 0.0
