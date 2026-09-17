from photosort import config
def test_exts():
    assert ".jpg" in config.IMAGE_EXTS and ".arw" in config.RAW_EXTS
    assert config.PREVIEW_EDGE == 1024
