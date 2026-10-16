"""Tests of the command line options that do not open a window."""
import runpy

import pytest

from neuroviewer import __version__, cli, config


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_write_config_roundtrip(tmp_path, monkeypatch):
    target = tmp_path / "cfg" / "config.py"
    monkeypatch.setenv("NEUROVIEWER_CONFIG", str(target))
    cli.main(["--write-config"])
    assert target.is_file()
    values = runpy.run_path(str(target))
    names = {k for k in values if k.isupper()}
    defaults = {k for k in vars(config) if k.isupper()}
    assert names == defaults
    assert all(values[k] == getattr(config, k) for k in names)
    target.read_text(encoding="ascii")  # plain ASCII


def test_write_config_never_overwrites(tmp_path, monkeypatch, capsys):
    target = tmp_path / "config.py"
    target.write_text("RADIOLOGICAL = True\n", encoding="ascii")
    monkeypatch.setenv("NEUROVIEWER_CONFIG", str(target))
    cli.main(["--write-config"])
    assert target.read_text(encoding="ascii") == "RADIOLOGICAL = True\n"
    cli.main(["--write-config", "--force"])
    assert (tmp_path / "config.py.bak").read_text(encoding="ascii") == "RADIOLOGICAL = True\n"
    assert "APP_TITLE" in target.read_text(encoding="ascii")


def test_config_path(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NEUROVIEWER_CONFIG", str(tmp_path / "x.py"))
    cli.main(["--config-path"])
    assert str(tmp_path / "x.py") in capsys.readouterr().out
