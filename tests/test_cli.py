from photosort.cli import main
from conftest import make_image

def test_index_cmd(tmp_path, capsys):
    make_image(tmp_path, "a.jpg")
    main(["index", str(tmp_path), "--no-faces", "--workers", "1"])
    out = capsys.readouterr().out
    assert "indexed 1" in out
