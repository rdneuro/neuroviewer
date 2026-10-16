"""
cli.py - command line entry point of Neuroviewer.

    neuroviewer                          open the viewer
    neuroviewer T1.mgz aseg.mgz lh.pial  open the viewer with these files
    neuroviewer --write-config           create the user config file (once)
    neuroviewer --config-path            print where the user config file lives
    neuroviewer --install-desktop        add a menu entry + icon (Linux)
    neuroviewer --version

The option handling runs before Qt/VTK are imported, so the maintenance
options are instant and work without a display.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

from . import __version__

_HERE = os.path.dirname(os.path.abspath(__file__))
_MARKER = "# ======================================================== USER OVERRIDES"


def _parser():
    p = argparse.ArgumentParser(
        prog="neuroviewer",
        description="Freeview-like neuroimaging viewer (PySide6 + vedo/VTK + nibabel).")
    p.add_argument("files", nargs="*",
                   help="volumes (.nii/.nii.gz/.mgz/...), surfaces (lh.pial, .surf.gii) "
                        "or streamlines (.trk/.tck) to open at start")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--write-config", action="store_true",
                   help="write the user config file with all defaults and exit")
    p.add_argument("--force", action="store_true",
                   help="with --write-config: overwrite an existing user config file")
    p.add_argument("--config-path", action="store_true",
                   help="print the user config file location and exit")
    p.add_argument("--install-desktop", action="store_true",
                   help="install a desktop menu entry and icon for this user (Linux) and exit")
    return p


def _user_config_path():
    # Imported lazily: importing config applies the user file and prints a line.
    from .config import user_config_path  # noqa: PLC0415
    return user_config_path()


def _template_text():
    """The settings block of the packaged config.py (between the first section and the marker)."""
    with open(os.path.join(_HERE, "config.py"), "r", encoding="ascii") as fh:
        text = fh.read()
    start = text.index("# -----")
    end = text.index(_MARKER)
    header = (
        '"""\n'
        "Neuroviewer user config. Overrides the packaged defaults at start-up.\n"
        "Keep only the lines you change; deleted names fall back to the defaults.\n"
        f"Written by neuroviewer {__version__}.\n"
        '"""\n\n')
    return header + text[start:end].rstrip() + "\n"


def write_config(force=False):
    path = _user_config_path()
    if os.path.exists(path) and not force:
        print(f"[config] {path} already exists; left untouched (use --force to overwrite)")
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        backup = path + ".bak"
        shutil.copy2(path, backup)
        print(f"[config] previous file saved as {backup}")
    with open(path, "w", encoding="ascii") as fh:
        fh.write(_template_text())
    print(f"[config] written: {path}")
    return path


def install_desktop():
    """Freedesktop menu entry + icon in ~/.local/share (no root needed)."""
    if not sys.platform.startswith("linux"):
        print("[desktop] only implemented for Linux; nothing done")
        return
    data = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
    icon_dir = os.path.join(data, "icons", "hicolor", "256x256", "apps")
    svg_dir = os.path.join(data, "icons", "hicolor", "scalable", "apps")
    app_dir = os.path.join(data, "applications")
    for d in (icon_dir, svg_dir, app_dir):
        os.makedirs(d, exist_ok=True)
    shutil.copy2(os.path.join(_HERE, "resources", "neuroviewer.png"), os.path.join(icon_dir, "neuroviewer.png"))
    shutil.copy2(os.path.join(_HERE, "resources", "neuroviewer.svg"), os.path.join(svg_dir, "neuroviewer.svg"))
    with open(os.path.join(_HERE, "resources", "neuroviewer.desktop"), "r", encoding="ascii") as fh:
        entry = fh.read()
    script = os.path.join(os.path.dirname(sys.executable), "neuroviewer")
    exe = script if os.path.isfile(script) else shutil.which("neuroviewer")
    cmd = f'"{exe}"' if exe else f'"{sys.executable}" -m neuroviewer'
    entry = entry.replace("Exec=neuroviewer %F", f"Exec={cmd} %F")
    target = os.path.join(app_dir, "neuroviewer.desktop")
    with open(target, "w", encoding="ascii") as fh:
        fh.write(entry)
    print(f"[desktop] menu entry: {target}")
    print(f"[desktop] icon: {os.path.join(icon_dir, 'neuroviewer.png')}")
    print(f"[desktop] command: {cmd}")


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.config_path:
        print(_user_config_path())
        return
    if args.write_config:
        write_config(force=args.force)
        return
    if args.install_desktop:
        install_desktop()
        return
    missing = [f for f in args.files if not os.path.exists(f)]
    for f in missing:
        print(f"[open] not found, skipped: {f}")
    files = [f for f in args.files if f not in missing]
    from .app import run  # noqa: PLC0415  (heavy imports only for the GUI)
    sys.exit(run(files))


if __name__ == "__main__":
    main()
