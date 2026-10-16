"""
slicing.py - display grid, slice sampling, colour compositing and the VTK
objects that carry the three orthogonal planes.

The display grid is an axis-aligned world grid defined by the first loaded
volume (as in freeview). Every layer is sampled onto that grid through its own
affine (nearest neighbour), coloured with its colormap and alpha-blended in
numpy; the result is one RGB image per plane. Each plane image is a
vtkImageData living at its true world position, so the same data feeds the 2D
panels and the three planes drawn in the 3D panel.
"""
from __future__ import annotations

import numpy as np
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkFiltersSources import vtkLineSource
from vtkmodules.vtkRenderingCore import (vtkActor, vtkImageActor,
                                         vtkPolyDataMapper, vtkTextActor)

from . import config
from . import loaders

AXIS_NAMES = {0: "sagittal", 1: "coronal", 2: "axial"}
NAME_TO_AXIS = {v: k for k, v in AXIS_NAMES.items()}


def inplane_axes(axis):
    return [a for a in range(3) if a != axis]


# =========================================================================
# display grid
# =========================================================================
class Grid:
    """Axis-aligned world grid: world = origin + index * spacing."""

    def __init__(self, origin, spacing, dims):
        self.origin = np.asarray(origin, dtype=float)
        self.spacing = np.asarray(spacing, dtype=float)
        self.dims = np.asarray(dims, dtype=int)
        self.matrix = np.eye(4)
        self.matrix[:3, :3] = np.diag(self.spacing)
        self.matrix[:3, 3] = self.origin

    @classmethod
    def from_layer(cls, layer):
        a = layer.affine
        if layer.is_axis_aligned():
            return cls(a[:3, 3], np.diag(a)[:3], layer.shape3)
        sh = np.asarray(layer.shape3) - 1
        corners = np.array([[i, j, k, 1.0] for i in (0, sh[0])
                            for j in (0, sh[1]) for k in (0, sh[2])])
        w = (a @ corners.T)[:3].T
        lo, hi = w.min(axis=0), w.max(axis=0)
        sp = layer.voxel_sizes()
        dims = np.floor((hi - lo) / sp + 1e-6).astype(int) + 1
        return cls(lo, sp, dims)

    @property
    def lo(self):
        return self.origin.copy()

    @property
    def hi(self):
        return self.origin + (self.dims - 1) * self.spacing

    def bounds(self):
        lo, hi = self.lo, self.hi
        return [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]

    def center(self):
        return 0.5 * (self.lo + self.hi)

    def index_of(self, world):
        idx = np.rint((np.asarray(world, dtype=float) - self.origin) / self.spacing).astype(int)
        return np.clip(idx, 0, self.dims - 1)

    def world_of(self, idx):
        return self.origin + np.asarray(idx, dtype=float) * self.spacing

    def snap(self, world):
        return self.world_of(self.index_of(world))


# =========================================================================
# sampling
# =========================================================================
def _is_diagonal(m, tol=1e-6):
    return np.abs(m - np.diag(np.diag(m))).max() < tol


def sample_plane(layer, grid, axis, k):
    """Values of `layer` on plane `axis`=k of `grid` (NaN outside the FOV).

    Returns float32 (n_u, n_v) with (u, v) the in-plane axes in increasing
    order: (x, y) axial, (x, z) coronal, (y, z) sagittal.
    """
    a, b = inplane_axes(axis)
    nu, nv = int(grid.dims[a]), int(grid.dims[b])
    out = np.full((nu, nv), np.nan, dtype=np.float32)
    data = layer.frame_data()
    sh = layer.shape3
    m = layer.inv_affine @ grid.matrix

    if _is_diagonal(m[:3, :3]):
        ia = np.rint(m[a, a] * np.arange(nu) + m[a, 3]).astype(np.int64)
        ib = np.rint(m[b, b] * np.arange(nv) + m[b, 3]).astype(np.int64)
        ic = int(np.rint(m[axis, axis] * k + m[axis, 3]))
        if not 0 <= ic < sh[axis]:
            return out
        ma = (ia >= 0) & (ia < sh[a])
        mb = (ib >= 0) & (ib < sh[b])
        if not ma.any() or not mb.any():
            return out
        sub = np.take(data, ic, axis=axis)
        out[np.ix_(ma, mb)] = sub[np.ix_(ia[ma], ib[mb])]
        return out

    g = np.empty((3, nu, nv), dtype=np.float64)
    g[axis] = k
    g[a] = np.arange(nu)[:, None]
    g[b] = np.arange(nv)[None, :]
    idx = np.einsum("ij,jkl->ikl", m[:3, :3], g) + m[:3, 3][:, None, None]
    idx = np.rint(idx).astype(np.int64)
    valid = np.ones((nu, nv), dtype=bool)
    for d in range(3):
        valid &= (idx[d] >= 0) & (idx[d] < sh[d])
    if valid.any():
        out[valid] = data[idx[0][valid], idx[1][valid], idx[2][valid]]
    return out


def value_at(layer, world):
    """(value or None, voxel index) of `layer` at a world point."""
    ijk = np.rint((layer.inv_affine @ np.r_[np.asarray(world, dtype=float), 1.0])[:3]).astype(int)
    if np.any(ijk < 0) or np.any(ijk >= np.asarray(layer.shape3)):
        return None, ijk
    return float(layer.frame_data()[tuple(ijk)]), ijk


def timecourse_at(layer, world):
    if layer.data.ndim != 4:
        return None
    ijk = np.rint((layer.inv_affine @ np.r_[np.asarray(world, dtype=float), 1.0])[:3]).astype(int)
    if np.any(ijk < 0) or np.any(ijk >= np.asarray(layer.shape3)):
        return None
    return np.asarray(layer.data[ijk[0], ijk[1], ijk[2], :], dtype=np.float64)


def aligned_array(layer):
    """(array, origin, spacing) of the current frame on an axis-aligned grid.

    Axis-aligned layers are returned as they are; oblique layers are resampled
    (nearest neighbour) slab by slab. Results are cached per frame.
    """
    key = layer.frame
    if key in layer.aligned_cache:
        return layer.aligned_cache[key]
    data = np.asarray(layer.frame_data())
    if layer.is_axis_aligned():
        res = (data, layer.affine[:3, 3].copy(), np.diag(layer.affine)[:3].copy())
    else:
        g = Grid.from_layer(layer)
        out = np.zeros(tuple(g.dims), dtype=data.dtype)
        nz = int(g.dims[2])
        step = max(nz // 4, 1)
        for z in range(nz):
            vals = sample_plane(layer, g, 2, z)
            out[:, :, z] = np.nan_to_num(vals, nan=0.0).astype(data.dtype)
            if z % step == 0:
                print(f"[resample] {layer.name}: {100 * z / nz:.0f}%")
        print(f"[resample] {layer.name}: done")
        res = (out, g.origin.copy(), g.spacing.copy())
    if len(layer.aligned_cache) >= 2:
        layer.aligned_cache.clear()
    layer.aligned_cache[key] = res
    return res


# =========================================================================
# colour
# =========================================================================
_TABLES = {}


def cmap_table(name):
    if name in _TABLES:
        return _TABLES[name]
    if name == "heat":
        cm = LinearSegmentedColormap.from_list(
            "heat", [(0.55, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 0.55, 0.0), (1.0, 1.0, 0.25)])
    else:
        try:
            cm = matplotlib.colormaps[name]
        except KeyError:
            cm = matplotlib.colormaps["gray"]
    t = cm(np.linspace(0.0, 1.0, 256))[:, :3].astype(np.float32)
    _TABLES[name] = t
    return t


def colorize(layer, vals):
    """RGB float32 (..., 3) in [0, 1] and alpha float32 (...) for sampled values."""
    valid = np.isfinite(vals)
    if layer.cmap == "lut":
        lab = np.where(valid, vals, 0.0).astype(np.int64)
        lab = np.clip(lab, 0, None)
        table = loaders.label_color_table(int(lab.max()) if lab.size else 0)
        rgb = table[np.clip(lab, 0, table.shape[0] - 1)]
        alpha = (valid & (lab != 0)).astype(np.float32) * layer.opacity
        return rgb, alpha
    table = cmap_table(layer.cmap)
    span = layer.vmax - layer.vmin
    span = span if span != 0 else 1.0
    t = np.where(valid, (vals - layer.vmin) / span, 0.0)
    idx = np.clip(np.rint(t * 255.0), 0, 255).astype(np.int64)
    rgb = table[idx]
    mask = valid
    if layer.clear_below:
        mask = mask & (np.where(valid, vals, -np.inf) >= layer.vmin)
    alpha = mask.astype(np.float32) * layer.opacity
    return rgb, alpha


def composite(layers, grid, axis, k):
    """Alpha-blend the visible layers (bottom to top) on one plane -> uint8 RGB."""
    a, b = inplane_axes(axis)
    rgb = np.zeros((int(grid.dims[a]), int(grid.dims[b]), 3), dtype=np.float32)
    for layer in layers:
        if not layer.visible or layer.opacity <= 0:
            continue
        vals = sample_plane(layer, grid, axis, k)
        col, alpha = colorize(layer, vals)
        alpha = alpha[..., None]
        rgb = rgb * (1.0 - alpha) + col * alpha
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


# =========================================================================
# VTK carriers
# =========================================================================
class PlaneImage:
    """vtkImageData holding one composited plane at its world position."""

    def __init__(self, axis):
        self.axis = axis
        self.image = vtkImageData()
        self.extent = (0, 0, 0, 0, 0, 0)
        self.ready = False

    def configure(self, grid):
        dims = [int(d) for d in grid.dims]
        dims[self.axis] = 1
        self.image = vtkImageData()
        self.image.SetDimensions(*dims)
        self.image.SetSpacing(*[float(s) for s in grid.spacing])
        self.image.SetOrigin(*[float(o) for o in grid.origin])
        self.extent = (0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
        self.ready = False

    def update(self, rgb_uv, grid, k):
        origin = grid.origin.copy()
        origin[self.axis] = grid.origin[self.axis] + k * grid.spacing[self.axis]
        self.image.SetOrigin(*[float(o) for o in origin])
        flat = np.ascontiguousarray(rgb_uv.transpose(1, 0, 2).reshape(-1, 3))
        arr = numpy_to_vtk(flat, deep=True)
        arr.SetName("rgb")
        self.image.GetPointData().SetScalars(arr)
        self.image.Modified()
        self.ready = True

    def make_actor(self, interpolation="nearest"):
        act = vtkImageActor()
        act.GetMapper().SetInputData(self.image)
        act.SetDisplayExtent(*self.extent)
        if interpolation == "linear":
            act.GetProperty().SetInterpolationTypeToLinear()
        else:
            act.GetProperty().SetInterpolationTypeToNearest()
        return act


class Crosshair:
    """Two line actors crossing at the cursor inside one plane."""

    def __init__(self, color, width):
        self.sources = [vtkLineSource(), vtkLineSource()]
        self.actors = []
        for src in self.sources:
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(src.GetOutputPort())
            act = vtkActor()
            act.SetMapper(mapper)
            act.GetProperty().SetColor(*color)
            act.GetProperty().SetLineWidth(width)
            act.PickableOff()
            self.actors.append(act)

    def update(self, cursor, grid, axis, normal_offset):
        lo = grid.lo - 0.5 * grid.spacing
        hi = grid.hi + 0.5 * grid.spacing
        for src, ax in zip(self.sources, inplane_axes(axis)):
            p0 = np.array(cursor, dtype=float)
            p1 = p0.copy()
            p0[ax], p1[ax] = lo[ax], hi[ax]
            p0[axis] += normal_offset
            p1[axis] += normal_offset
            src.SetPoint1(*p0)
            src.SetPoint2(*p1)
            src.Modified()

    def set_visible(self, flag):
        for act in self.actors:
            act.SetVisibility(bool(flag))


def text_actor(text="", size=None, color=None, bold=False):
    t = vtkTextActor()
    t.SetInput(text)
    prop = t.GetTextProperty()
    prop.SetFontSize(size or config.TEXT_SIZE)
    prop.SetColor(*(color or config.TEXT_COLOR))
    prop.SetBold(bool(bold))
    prop.SetFontFamilyToArial()
    prop.ShadowOn()
    t.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    return t


def place_text(t, x, y, hjust="left", vjust="bottom"):
    t.GetPositionCoordinate().SetValue(float(x), float(y))
    prop = t.GetTextProperty()
    {"left": prop.SetJustificationToLeft, "center": prop.SetJustificationToCentered,
     "right": prop.SetJustificationToRight}[hjust]()
    {"bottom": prop.SetVerticalJustificationToBottom, "center": prop.SetVerticalJustificationToCentered,
     "top": prop.SetVerticalJustificationToTop}[vjust]()
