from photosort import config
def test_exts():
    assert ".jpg" in config.IMAGE_EXTS and ".arw" in config.RAW_EXTS
    assert config.PREVIEW_EDGE == 1024


def test_settings_round_trip_and_unreadable_file(tmp_path):
    from photosort import settings
    from photosort.config import settings_path, app_home
    assert settings_path() == app_home() / "settings.json"
    assert settings.load() == {} and settings.get_export_base() is None
    settings.set_export_base(tmp_path / "disk")
    assert settings.get_export_base() == tmp_path / "disk"
    assert settings.load()["export_base"] == str(tmp_path / "disk")
    settings.set_export_base(None)
    assert settings.get_export_base() is None and "export_base" not in settings.load()
    settings_path().write_text("{not json")
    assert settings.load() == {} and settings.get_export_base() is None
