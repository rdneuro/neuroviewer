"""
tools.py - the extension point of Neuroviewer.

A tool is a function that receives a ToolContext and returns a list of
(name, object, kind) tuples; each object is added to the 3D scene, where it
can be shown/hidden, recoloured or removed. `kind` is a free tag ("mesh",
"points", "surface", ...) that later tools use to find their inputs
(e.g. the Open3D tools work on the last "points" object).

Add a new feature by writing one function here:

    @register_tool("My tool", "What it does.")
    def my_tool(ctx):
        layer = ctx.require_active()
        arr, origin, spacing = ctx.aligned(layer)
        ...
        return [("my result", vedo_object, "mesh")]

Everything is in world coordinates (scanner RAS, mm), so results line up
with the slices, surfaces and streamlines already on screen.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import vedo
from PySide6 import QtWidgets

from . import config
from . import loaders
from . import slicing


class ToolCancelled(Exception):
    """Raised when the user cancels a dialog."""


@dataclass
class ToolSpec:
    name: str
    fn: object
    description: str = ""


TOOLS = {}


def register_tool(name, description=""):
    def deco(fn):
        TOOLS[name] = ToolSpec(name, fn, description)
        return fn
    return deco


# =========================================================================
# context handed to every tool
# =========================================================================
class ToolContext:

    def __init__(self, window):
        self.window = window
        self.layers = list(window.layers)
        self.active = window.active
        self.cursor = np.array(window.cursor, dtype=float)
        self.grid = window.grid

    # ------------------------------------------------------------ data
    def require_active(self):
        if self.active is None:
            raise RuntimeError("Load a volume and select it in the Layers panel first.")
        return self.active

    def aligned(self, layer):
        """(array, origin, spacing) of the current frame on an axis-aligned grid."""
        return slicing.aligned_array(layer)

    def last_object(self, kinds):
        kinds = (kinds,) if isinstance(kinds, str) else tuple(kinds)
        for item in reversed(self.window.scene):
            if item.kind in kinds:
                return item
        return None

    # ------------------------------------------------------------ dialogs
    def ask_float(self, title, label, value, lo=-1e12, hi=1e12, decimals=4):
        val, ok = QtWidgets.QInputDialog.getDouble(self.window, title, label, float(value),
                                                   float(lo), float(hi), int(decimals))
        if not ok:
            raise ToolCancelled()
        return float(val)

    def ask_int(self, title, label, value, lo=0, hi=10 ** 9):
        val, ok = QtWidgets.QInputDialog.getInt(self.window, title, label, int(value), int(lo), int(hi))
        if not ok:
            raise ToolCancelled()
        return int(val)

    def ask_text(self, title, label, value=""):
        val, ok = QtWidgets.QInputDialog.getText(self.window, title, label, text=str(value))
        if not ok:
            raise ToolCancelled()
        return str(val)

    def log(self, msg):
        print(f"[tool] {msg}")
        self.window.statusBar().showMessage(msg, 10000)


# =========================================================================
# helpers
# =========================================================================
def points_of(obj):
    """Vertex array of a vedo object across vedo versions."""
    v = getattr(obj, "vertices", None)
    if v is not None and not callable(v):
        return np.asarray(v)
    p = getattr(obj, "points", None)
    if callable(p):
        return np.asarray(p())
    if p is not None:
        return np.asarray(p)
    raise RuntimeError("Object has no vertices.")


def _smooth(mesh):
    if config.SURF_SMOOTH_ITER <= 0:
        return mesh
    try:
        mesh.smooth(niter=config.SURF_SMOOTH_ITER)
    except Exception as exc:  # noqa: BLE001
        print(f"[tool] smoothing skipped: {exc}")
    return mesh


def _crop_mask(mask, origin, spacing, pad=2):
    """Crop a boolean volume to its bounding box (plus padding)."""
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(axis=0) - pad, 0)
    hi = np.minimum(idx.max(axis=0) + pad + 1, np.asarray(mask.shape))
    sub = mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    return sub, np.asarray(origin) + lo * np.asarray(spacing)


def _points_cloud(xyz, scalars, cmap):
    pts = vedo.Points(np.asarray(xyz, dtype=np.float32), r=config.PC_POINT_SIZE)
    try:
        pts.cmap(cmap, np.asarray(scalars, dtype=np.float32))
    except Exception as exc:  # noqa: BLE001
        print(f"[tool] point colouring skipped: {exc}")
    return pts


def _subsample(n, n_max):
    if n <= n_max:
        return np.arange(n)
    rng = np.random.default_rng(config.RANDOM_SEED)
    return np.sort(rng.choice(n, n_max, replace=False))


def _voxel_cloud(layer, mask, ctx, what):
    data = np.asarray(layer.frame_data())
    ijk = np.argwhere(mask)
    if ijk.shape[0] == 0:
        raise RuntimeError(f"No voxels selected for {what}.")
    labels = data[mask]
    keep = _subsample(ijk.shape[0], config.PC_MAX_POINTS)
    ijk, labels = ijk[keep], labels[keep]
    a = layer.affine
    xyz = ijk @ a[:3, :3].T + a[:3, 3]
    if config.PC_COLOR_BY == "label":
        scal, cmap = np.mod(labels, 20), "tab20"
    else:
        scal, cmap = xyz[:, 2], config.PC_CMAP
    ctx.log(f"{what}: {len(xyz)} points (of {int(mask.sum())} voxels)")
    return _points_cloud(xyz, scal, cmap)


def _import_open3d():
    try:
        import open3d as o3d  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError("Open3D is not installed (pip install open3d).") from exc
    return o3d


# =========================================================================
# built-in tools
# =========================================================================
@register_tool("Isosurface (active layer)",
               "Threshold the active layer and extract a smoothed isosurface.")
def tool_isosurface(ctx):
    layer = ctx.require_active()
    thr = ctx.ask_float("Isosurface", f"Threshold for {layer.name}:", layer.vmin)
    arr, origin, spacing = ctx.aligned(layer)
    mask = np.asarray(arr) >= thr
    if not mask.any():
        raise RuntimeError(f"No voxel >= {thr:g} in {layer.name}.")
    sub, sub_origin = _crop_mask(mask, origin, spacing)
    vol = vedo.Volume(sub.astype(np.uint8), spacing=spacing, origin=sub_origin)
    mesh = _smooth(vol.isosurface(0.5))
    mesh.c(config.ISO_COLOR).alpha(config.ISO_ALPHA)
    ctx.log(f"isosurface of {layer.name} at {thr:g}: {len(points_of(mesh))} vertices")
    return [(f"iso >= {thr:g} | {layer.name}", mesh, "mesh")]


@register_tool("Label surfaces (active layer)",
               "One smoothed surface per label value of a segmentation (e.g. 17 53).")
def tool_label_surfaces(ctx):
    layer = ctx.require_active()
    text = ctx.ask_text("Label surfaces", "Label values (space separated):",
                        config.DEFAULT_LABEL_SURFACES)
    labels = [int(t) for t in text.replace(",", " ").split() if t.strip().lstrip("-").isdigit()]
    if not labels:
        raise RuntimeError("No label values given.")
    arr, origin, spacing = ctx.aligned(layer)
    arr = np.asarray(arr)
    table = loaders.label_color_table(max(labels))
    out = []
    for lab in labels:
        mask = arr == lab
        if not mask.any():
            ctx.log(f"label {lab} not present in {layer.name}")
            continue
        sub, sub_origin = _crop_mask(mask, origin, spacing)
        vol = vedo.Volume(sub.astype(np.uint8), spacing=spacing, origin=sub_origin)
        mesh = _smooth(vol.isosurface(0.5))
        mesh.c(tuple(float(c) for c in table[lab]))
        out.append((f"{loaders.label_name(lab)} ({lab})", mesh, "mesh"))
    ctx.log(f"{len(out)} label surfaces from {layer.name}")
    return out


@register_tool("Cortex point cloud (FreeSurfer labels)",
               "Voxel centres of cortical labels (ribbon / aseg / aparc+aseg) as a point cloud.")
def tool_cortex_cloud(ctx):
    layer = ctx.require_active()
    data = np.asarray(layer.frame_data())
    mask = np.isin(data, np.asarray(config.CORTEX_LABELS))
    if not mask.any():
        raise RuntimeError("No cortical label found in the active layer "
                           "(expected ribbon.mgz, aseg.mgz or aparc+aseg.mgz).")
    return [(f"cortex cloud | {layer.name}", _voxel_cloud(layer, mask, ctx, "cortex"), "points")]


@register_tool("Threshold point cloud (active layer)",
               "Voxel centres above a threshold as a point cloud.")
def tool_threshold_cloud(ctx):
    layer = ctx.require_active()
    thr = ctx.ask_float("Threshold cloud", f"Keep voxels >= (in {layer.name}):", layer.vmin)
    mask = np.asarray(layer.frame_data()) >= thr
    return [(f"cloud >= {thr:g} | {layer.name}", _voxel_cloud(layer, mask, ctx, "threshold"), "points")]


@register_tool("Point cloud from last surface",
               "Vertices of the last surface/mesh in the scene as a point cloud.")
def tool_surface_cloud(ctx):
    item = ctx.last_object(("surface", "mesh"))
    if item is None:
        raise RuntimeError("Load a surface or create a mesh first.")
    xyz = points_of(item.obj)
    keep = _subsample(len(xyz), config.PC_MAX_POINTS)
    xyz = xyz[keep]
    ctx.log(f"{len(xyz)} points from {item.name}")
    return [(f"cloud | {item.name}", _points_cloud(xyz, xyz[:, 2], config.PC_CMAP), "points")]


@register_tool("Open3D: voxel downsample last cloud",
               "Voxel-grid downsampling of the last point cloud with Open3D.")
def tool_o3d_downsample(ctx):
    o3d = _import_open3d()
    item = ctx.last_object("points")
    if item is None:
        raise RuntimeError("Create a point cloud first.")
    vs = ctx.ask_float("Open3D downsample", "Voxel size (mm):", config.O3D_VOXEL_SIZE, 0.01, 100.0, 2)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_of(item.obj).astype(np.float64))
    down = pcd.voxel_down_sample(vs)
    xyz = np.asarray(down.points)
    ctx.log(f"Open3D downsample {vs:g} mm: {len(pcd.points)} -> {len(xyz)} points")
    return [(f"down {vs:g}mm | {item.name}", _points_cloud(xyz, xyz[:, 2], config.PC_CMAP), "points")]


@register_tool("Open3D: Poisson surface from last cloud",
               "Normals + screened Poisson reconstruction of the last point cloud (Open3D).")
def tool_o3d_poisson(ctx):
    o3d = _import_open3d()
    item = ctx.last_object("points")
    if item is None:
        raise RuntimeError("Create a point cloud first.")
    depth = ctx.ask_int("Poisson", "Octree depth:", config.O3D_POISSON_DEPTH, 4, 12)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_of(item.obj).astype(np.float64))
    if len(pcd.points) > config.O3D_POISSON_MAX_POINTS:
        span = np.ptp(np.asarray(pcd.points), axis=0).max()
        vs = span / 200.0
        while len(pcd.points) > config.O3D_POISSON_MAX_POINTS:
            pcd = pcd.voxel_down_sample(vs)
            vs *= 1.3
        ctx.log(f"cloud reduced to {len(pcd.points)} points before Poisson")
    nn = np.asarray(pcd.compute_nearest_neighbor_distance())
    radius = 3.0 * float(np.median(nn)) if nn.size else 2.0
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(config.O3D_ORIENT_K)
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    dens = np.asarray(dens)
    mesh.remove_vertices_by_mask(dens < np.quantile(dens, config.O3D_DENSITY_QUANTILE))
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    out = vedo.Mesh([verts, tris]).c(config.ISO_COLOR)
    ctx.log(f"Poisson depth {depth}: {len(verts)} vertices, {len(tris)} triangles")
    return [(f"poisson d{depth} | {item.name}", out, "mesh")]
