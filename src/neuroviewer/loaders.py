"""
loaders.py - file discovery and loading for Neuroviewer.

Everything returned here lives in world coordinates (scanner RAS, mm):
  * volumes are reoriented to the closest canonical (RAS+) voxel order and
    keep their full affine (FreeSurfer .mgz conformed volumes are LIA on disk);
  * FreeSurfer surfaces are shifted from tkregister space to scanner RAS with
    the c_ras stored in their header;
  * streamlines (.trk/.tck) are read in RASmm by nibabel.
"""
from __future__ import annotations

import colorsys
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import nibabel as nib
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation

from . import config

_FS_SUFFIXES_LOWER = {s.lower() for s in config.FS_SURFACE_SUFFIXES}
_RAS_ORNT = np.array([[0, 1], [1, 1], [2, 1]], dtype=float)


# =========================================================================
# file kinds
# =========================================================================
def path_kind(path):
    """Return "volume", "surface", "tracts" or None for a file path."""
    name = os.path.basename(path).lower()
    if name.endswith(config.TRACT_EXTENSIONS):
        return "tracts"
    if name.endswith(".gii"):
        if name.endswith(config.GIFTI_SKIP_SUFFIXES):
            return None
        return "surface"
    if name[:3] in ("lh.", "rh.") and name[3:] in _FS_SUFFIXES_LOWER:
        return "surface"
    if name.endswith(config.VOLUME_EXTENSIONS):
        return "volume"
    return None


def tree_name_filters():
    """Wildcards for the file browser (QFileSystemModel name filters)."""
    pats = ["*" + e for e in config.VOLUME_EXTENSIONS]
    pats += ["*" + e for e in config.TRACT_EXTENSIONS]
    pats += ["*.gii"]
    pats += [p.upper() for p in list(pats)]
    pats += [f"{h}.{s}" for h in ("lh", "rh") for s in config.FS_SURFACE_SUFFIXES]
    return sorted(set(pats))


def display_name(path):
    """Short layer name; keeps the subject folder for FreeSurfer/BIDS trees."""
    base = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    if parent in ("mri", "surf", "anat", "func", "dwi", "label"):
        grand = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))
        if grand:
            return f"{grand}/{base}"
    return base


# =========================================================================
# layer model
# =========================================================================
@dataclass(eq=False)
class Layer:
    """One volume layer: canonical RAS data + display properties."""
    name: str
    path: str
    data: np.ndarray                 # (X, Y, Z) or (X, Y, Z, T), RAS+ voxel order
    affine: np.ndarray               # 4x4 voxel -> scanner RAS (mm)
    is_label: bool = False
    cmap: str = "gray"
    vmin: float = 0.0
    vmax: float = 1.0
    opacity: float = 1.0
    visible: bool = True
    clear_below: bool = False
    frame: int = 0
    inv_affine: np.ndarray = field(init=False, repr=False)
    aligned_cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.affine = np.asarray(self.affine, dtype=float)
        self.inv_affine = np.linalg.inv(self.affine)

    @property
    def shape3(self):
        return tuple(int(s) for s in self.data.shape[:3])

    @property
    def n_frames(self):
        return int(self.data.shape[3]) if self.data.ndim == 4 else 1

    def frame_data(self):
        if self.data.ndim == 3:
            return self.data
        return self.data[..., int(np.clip(self.frame, 0, self.n_frames - 1))]

    def voxel_sizes(self):
        return np.sqrt((self.affine[:3, :3] ** 2).sum(axis=0))

    def is_axis_aligned(self):
        m = self.affine[:3, :3]
        diag = np.diag(m)
        off = m - np.diag(diag)
        return bool(np.abs(off).max() <= 1e-4 * np.abs(diag).max() and np.all(diag > 0))

    def info_text(self):
        vs = ", ".join(f"{v:.2f}" for v in self.voxel_sizes())
        kind = "label map" if self.is_label else "intensity"
        oblique = "" if self.is_axis_aligned() else " | oblique (resampled for display)"
        return (f"{self.path}\nshape {tuple(self.data.shape)} | voxel ({vs}) mm | "
                f"{self.data.dtype} | {kind}{oblique}")


# =========================================================================
# volumes
# =========================================================================
_CACHE = OrderedDict()


def _cache_key(path):
    st = os.stat(path)
    return (os.path.abspath(path), st.st_mtime_ns, st.st_size)


def load_canonical(path):
    """Load a volume and reorient it to RAS+ voxel order (cached, LRU)."""
    key = _cache_key(path)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    img = nib.load(path)
    data = np.asanyarray(img.dataobj)
    affine = np.asarray(img.affine, dtype=float).copy()

    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim > 4:
        print(f"[load] {os.path.basename(path)}: {data.ndim}D array, extra dims flattened into frames")
        data = data.reshape(data.shape[:3] + (-1,))
    if data.ndim < 3:
        raise ValueError(f"{path}: expected a 3D or 4D volume, got shape {data.shape}")

    ornt = io_orientation(affine)
    if not np.array_equal(ornt, _RAS_ORNT):
        affine = affine @ inv_ornt_aff(ornt, data.shape[:3])
        data = apply_orientation(data, ornt)

    if data.dtype == np.float64:
        data = data.astype(np.float32)
    elif data.dtype == np.bool_:
        data = data.astype(np.uint8)
    elif data.dtype == np.int64 and data.size and np.abs(data).max() < 2 ** 31:
        data = data.astype(np.int32)

    _CACHE[key] = (data, affine)
    while len(_CACHE) > config.CACHE_MAX_VOLUMES:
        _CACHE.popitem(last=False)
    return data, affine


def guess_is_label(path, data):
    name = os.path.basename(path).lower()
    if not any(h in name for h in config.LABEL_NAME_HINTS):
        return False
    if data.ndim != 3:
        return False
    sample = np.asarray(data[::3, ::3, ::3])
    if np.issubdtype(sample.dtype, np.integer):
        integer = True
    else:
        finite = sample[np.isfinite(sample)]
        integer = finite.size > 0 and np.allclose(finite, np.round(finite))
    if not integer:
        return False
    return np.unique(sample).size <= config.LABEL_MAX_UNIQUE


def robust_range(layer):
    """Percentile window computed on non-zero voxels of the current frame."""
    d = np.asarray(layer.frame_data()[::2, ::2, ::2], dtype=np.float32)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    if nz.size > 100:
        d = nz
    if d.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(d, config.WINDOW_PERCENTILES)
    if hi <= lo:
        lo, hi = float(d.min()), float(d.max())
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def load_volume_layer(path, as_overlay=False, report=None):
    t0 = time.time()
    if report:
        report(("status", f"Loading {os.path.basename(path)} ..."))
    data, affine = load_canonical(path)
    layer = Layer(name=display_name(path), path=path, data=data, affine=affine)
    layer.is_label = guess_is_label(path, data)
    if layer.is_label:
        layer.cmap = "lut"
        layer.vmin = 0.0
        layer.vmax = float(np.nanmax(np.asarray(data[::2, ::2, ::2])))
        layer.opacity = config.LABEL_OPACITY if as_overlay else 1.0
        layer.clear_below = False
    else:
        layer.vmin, layer.vmax = robust_range(layer)
        if as_overlay:
            layer.cmap = config.OVERLAY_CMAP
            layer.opacity = config.OVERLAY_OPACITY
            layer.clear_below = True
        else:
            layer.cmap = config.BASE_CMAP
    print(f"[load] {path} | shape {data.shape} | {data.dtype} | "
          f"{'label' if layer.is_label else 'intensity'} | {time.time() - t0:.1f} s")
    return layer


# =========================================================================
# surfaces and streamlines
# =========================================================================
def load_surface(path, report=None):
    """Return (vertices (N,3) in scanner RAS mm, faces (M,3))."""
    if report:
        report(("status", f"Loading surface {os.path.basename(path)} ..."))
    if path.lower().endswith(".gii"):
        g = nib.load(path)
        verts = g.agg_data("NIFTI_INTENT_POINTSET")
        faces = g.agg_data("NIFTI_INTENT_TRIANGLE")
        if verts is None or faces is None or len(verts) == 0 or len(faces) == 0:
            raise ValueError(f"{path}: GIfTI file without pointset/triangle arrays")
    else:
        verts, faces, meta = nib.freesurfer.read_geometry(path, read_metadata=True)
        if config.FS_SURFACE_ADD_CRAS:
            cras = meta.get("cras") if meta else None
            if cras is not None:
                verts = verts + np.asarray(cras, dtype=float)
            else:
                print(f"[surface] {os.path.basename(path)}: no c_ras in header, "
                      "vertices kept in tkregister space")
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def surface_color(path):
    name = os.path.basename(path).lower()
    for key, rgb in config.SURFACE_COLORS.items():
        if key in name:
            return rgb
    return config.SURFACE_DEFAULT_COLOR


def load_streamlines(path, report=None):
    """Return a list of (n_i, 3) float32 arrays in RASmm (subsampled)."""
    if report:
        report(("status", f"Loading streamlines {os.path.basename(path)} ..."))
    sl = nib.streamlines.load(path).streamlines
    n = len(sl)
    if n > config.TRACT_MAX_STREAMLINES:
        rng = np.random.default_rng(config.RANDOM_SEED)
        idx = np.sort(rng.choice(n, config.TRACT_MAX_STREAMLINES, replace=False))
        print(f"[tracts] {os.path.basename(path)}: {n} streamlines, showing "
              f"{config.TRACT_MAX_STREAMLINES}")
    else:
        idx = np.arange(n)
    return [np.asarray(sl[int(i)], dtype=np.float32) for i in idx]


# =========================================================================
# label look-up table (FreeSurferColorLUT.txt when available)
# =========================================================================
_LUT = None
_TABLE = np.zeros((0, 3), dtype=np.float32)
_TABLE_CAP = 1 << 20


def _lut_paths():
    paths = []
    if config.FREESURFER_LUT_PATH:
        paths.append(config.FREESURFER_LUT_PATH)
    fs_home = os.environ.get("FREESURFER_HOME", "")
    if fs_home:
        paths.append(os.path.join(fs_home, "FreeSurferColorLUT.txt"))
    return paths


def get_lut():
    """dict label -> (name, (r, g, b) uint8). Empty dict if no LUT is found."""
    global _LUT
    if _LUT is not None:
        return _LUT
    _LUT = {}
    for p in _lut_paths():
        if not os.path.isfile(p):
            continue
        with open(p, "r", errors="ignore") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 5 and parts[0].isdigit():
                    try:
                        rgb = tuple(int(v) for v in parts[2:5])
                    except ValueError:
                        continue
                    _LUT[int(parts[0])] = (parts[1], rgb)
        print(f"[lut] {len(_LUT)} labels read from {p}")
        break
    if not _LUT:
        print("[lut] FreeSurferColorLUT.txt not found, using generated label colors")
    return _LUT


def _fallback_color(i):
    h = (i * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(h, 0.65, 0.95)


def label_color_table(max_label):
    """float32 (n, 3) table of label colors in [0, 1], covering max_label."""
    global _TABLE
    n = int(min(max(int(max_label), 0) + 1, _TABLE_CAP))
    if _TABLE.shape[0] >= n:
        return _TABLE
    n = max(n, 256, _TABLE.shape[0] * 2)
    n = min(n, _TABLE_CAP)
    lut = get_lut()
    table = np.empty((n, 3), dtype=np.float32)
    for i in range(n):
        if i in lut:
            table[i] = np.asarray(lut[i][1], dtype=np.float32) / 255.0
        else:
            table[i] = _fallback_color(i)
    _TABLE = table
    return _TABLE


def label_name(value):
    lut = get_lut()
    v = int(round(value))
    if v in lut:
        return lut[v][0]
    return f"label {v}"
