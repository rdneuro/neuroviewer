<p align="center">
  <img src="https://raw.githubusercontent.com/rdneuro/neuroviewer/main/src/neuroviewer/resources/neuroviewer.png" width="128" alt="Neuroviewer icon">
</p>

<h1 align="center">Neuroviewer</h1>

<p align="center">
A freeview-like neuroimaging viewer written in pure Python.<br>
PySide6 interface, vedo/VTK rendering, nibabel reading. Native desktop app, no browser, no server.
</p>

<p align="center">
  <a href="https://pypi.org/project/neuroviewer/"><img src="https://img.shields.io/pypi/v/neuroviewer.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/neuroviewer/"><img src="https://img.shields.io/pypi/pyversions/neuroviewer.svg" alt="Python versions"></a>
  <a href="https://github.com/rdneuro/neuroviewer/actions/workflows/tests.yml"><img src="https://github.com/rdneuro/neuroviewer/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
</p>

<!--
Screenshot: save one with the toolbar "Screenshot" button (real T1 + aseg,
1+3 layout) as docs/screenshot.png, commit it, then uncomment:
<p align="center"><img src="https://raw.githubusercontent.com/rdneuro/neuroviewer/main/docs/screenshot.png" width="900" alt="Neuroviewer"></p>
-->

## Install

```bash
pip install neuroviewer              # core viewer
pip install "neuroviewer[open3d]"    # + Open3D point-cloud tools
```

Then:

```bash
neuroviewer                          # empty window
neuroviewer T1.mgz aseg.mgz lh.pial  # open files at start
python -m neuroviewer                # same thing
neuroviewer --install-desktop        # Linux: add a menu entry with icon
```

Requirements: Python 3.9+, an OpenGL-capable display. On Linux, Qt 6.5+
needs `libxcb-cursor0` (`sudo apt install libxcb-cursor0`). A self-contained
Linux AppImage is attached to each GitHub release.

## What it does

- **Files** (left): folder tree filtered to neuroimaging files, plus a
  recursive search under the root folder (name filter such as `aseg`,
  `*T1*.nii.gz`; blank = everything). Double click opens a file.
- **Views** (centre): one main view and three side views (layout `1+3`,
  `2x2` or `1x1`). Click a panel title, a toolbar button or `Ctrl+1..4` to put
  3D / sagittal / coronal / axial in the main position.
- **Layers** (right): one entry per volume, top of the list drawn on top.
  Colormap, window min/max, opacity, "transparent below min", frame (4D).
- **3D scene** (right, second tab): slices / bounding box / volume rendering
  toggles, the **Tools** menu and the list of 3D objects (surfaces,
  streamlines, isosurfaces, point clouds) with visibility, colour, opacity.
- **Time course** (bottom): signal of the active 4D layer at the cursor.
- **Status bar**: cursor in RAS mm, voxel index and value of every visible
  layer (with the FreeSurfer label name for label maps).

### Mouse and keys

| where | action |
|---|---|
| 2D | left click / drag: move cursor; shift+left or middle drag: pan; right drag or wheel: zoom |
| 2D | Up / PageUp, Down / PageDown: next / previous slice; `r`: fit slice |
| 3D | left drag: rotate; middle: pan; right / wheel: zoom; **double click** on a slice or surface: move cursor; `r`: reset camera |

### Formats

- Volumes: `.nii`, `.nii.gz`, `.mgz`, `.mgh`, Analyze `.img/.hdr`, MINC `.mnc`, PAR/REC.
- Surfaces: FreeSurfer `lh.pial`, `rh.white`, ... (shifted to scanner RAS with
  `c_ras`) and GIfTI `.surf.gii`.
- Streamlines: `.trk`, `.tck` (direction-coloured, subsampled to
  `TRACT_MAX_STREAMLINES`).

### Coordinates

Everything is kept in world coordinates (scanner RAS, mm). Volumes are
reoriented to RAS+ voxel order on load (FreeSurfer conformed `.mgz` is LIA on
disk) and keep their full affine. The first loaded volume defines the display
grid; every other layer is sampled onto it through its own affine, so layers
with different resolutions or fields of view overlay correctly. Oblique
volumes are resampled for display.

Label maps are detected by file name (`aseg`, `aparc`, `ribbon`, ...) plus
integer values. Their colours come from `$FREESURFER_HOME/FreeSurferColorLUT.txt`
(or `FREESURFER_LUT_PATH` in the config), with generated colours as fallback.

## Configuration

The packaged defaults live in `neuroviewer/config.py`. Do not edit that copy;
create your own once and edit it:

```bash
neuroviewer --write-config   # writes ~/.config/neuroviewer/config.py (Linux/macOS)
neuroviewer --config-path    # shows where it is (%APPDATA%\neuroviewer on Windows)
```

It is plain Python with the same names (radiological convention, colours,
layout, label hints, point-cloud size, Open3D parameters, screenshot scale,
...). Keep only the lines you change. Set `NEUROVIEWER_CONFIG` to use another
file. An existing file is never overwritten unless you pass `--force`.

## Tools (extension point)

`neuroviewer/tools.py` holds a registry. A tool is one function that receives
a context (layers, active layer, cursor, grid, dialogs) and returns 3D objects:

```python
from neuroviewer.tools import register_tool

@register_tool("My tool", "What it does.")
def my_tool(ctx):
    layer = ctx.require_active()
    arr, origin, spacing = ctx.aligned(layer)
    ...
    return [("result name", vedo_object, "mesh")]
```

Built-in tools: isosurface, label surfaces (e.g. `17 53`), cortex point cloud
from FreeSurfer labels, threshold point cloud, point cloud from the last
surface, Open3D voxel downsampling and Open3D Poisson reconstruction. Open3D
is optional; it is used as a compute engine and the result is shown in vedo.

## Troubleshooting

- **`Could not find the Qt platform plugin "xcb"`**: another Qt install is
  exporting its own plugin path (Spyder/PyQt5, opencv-python, conda qt).
  Neuroviewer resets it to the PySide6 plugins and prints the folder it used
  (`[qt] ...`). Start it from a terminal, not from the Spyder/IPython console.
  If it still fails, run `QT_DEBUG_PLUGINS=1 neuroviewer`.
- **Black views on Wayland**: `QT_QPA_PLATFORM=xcb neuroviewer`.
- **Missing `libxcb-cursor0`** (Qt 6.5+ on Linux): install it with your
  package manager.

## Development

```bash
git clone https://github.com/rdneuro/neuroviewer.git
cd neuroviewer
pip install -e ".[dev]"
pytest
```

See [RELEASING.md](RELEASING.md) for the PyPI and AppImage release steps.

## License

MIT. See [LICENSE](LICENSE). Neuroviewer depends on PySide6 (LGPLv3), VTK
(BSD), vedo (MIT), nibabel (MIT) and matplotlib (PSF-based); the AppImage
redistributes them under their own licenses.
