# AppImage

A single-file Linux build that bundles its own Python and every dependency.
Users download it, `chmod +x`, and run it; nothing is installed.

```bash
bash packaging/appimage/build_appimage.sh          # from this source tree
bash packaging/appimage/build_appimage.sh --pypi   # from the PyPI release
PY_VERSION=3.11 bash packaging/appimage/build_appimage.sh
```

How it works: [python-appimage](https://github.com/niess/python-appimage)
downloads a relocatable manylinux Python AppImage, `pip install`s the
Neuroviewer wheel into it (PySide6-Essentials, VTK, vedo, nibabel, numpy,
matplotlib come along as binary wheels), and the script adds the small X11
libraries Qt 6.5+ needs (`libxcb-cursor` and three helpers) before packing
with appimagetool.

Notes:

- The GitHub workflow `.github/workflows/release.yml` builds it on Ubuntu
  22.04 for every `v*` tag and attaches it to the GitHub release.
- Target machines need glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+,
  RHEL/Rocky 8+) and a working OpenGL driver. The driver always comes from the
  host; it is never bundled.
- Running an AppImage needs FUSE 2 (`libfuse2` / `libfuse2t64` on Ubuntu).
  Without it: `./Neuroviewer-*.AppImage --appimage-extract-and-run`.
- Expected size: roughly 200-300 MB (VTK and Qt are most of it).
