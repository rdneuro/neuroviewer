"""Headless tests of the numeric core (no display needed)."""
import numpy as np
import nibabel as nib
import pytest

from neuroviewer import loaders, slicing

LIA = np.array([[-1, 0, 0, 32], [0, 0, 1, -32], [0, -1, 0, 32], [0, 0, 0, 1]], float)


@pytest.fixture()
def lia_volume(tmp_path):
    """FreeSurfer-like LIA volume with one bright voxel at a known world point."""
    data = np.zeros((64, 64, 64), np.uint8)
    marker = np.array([10.0, -5.0, 7.0])
    ijk = np.rint(np.linalg.inv(LIA) @ np.r_[marker, 1.0])[:3].astype(int)
    data[tuple(ijk)] = 200
    path = tmp_path / "T1.mgz"
    nib.save(nib.MGHImage(data, LIA), str(path))
    return path, marker


def test_reorientation_keeps_world_position(lia_volume):
    path, marker = lia_volume
    layer = loaders.load_volume_layer(str(path))
    assert layer.is_axis_aligned()
    assert np.allclose(np.diag(layer.affine)[:3], 1.0)
    value, _ = slicing.value_at(layer, marker)
    assert value == 200.0


def test_three_planes_cross_at_marker(lia_volume):
    path, marker = lia_volume
    layer = loaders.load_volume_layer(str(path))
    grid = slicing.Grid.from_layer(layer)
    k = grid.index_of(marker)
    for axis in range(3):
        a, b = slicing.inplane_axes(axis)
        plane = slicing.sample_plane(layer, grid, axis, k[axis])
        assert plane[k[a], k[b]] == 200.0


def test_label_detection_and_composite(tmp_path):
    lab = np.zeros((32, 32, 32), np.int32)
    lab[10:20, 10:20, 10:20] = 17
    path = tmp_path / "aseg.mgz"
    nib.save(nib.MGHImage(lab, LIA), str(path))
    layer = loaders.load_volume_layer(str(path), as_overlay=True)
    assert layer.is_label and layer.cmap == "lut"
    grid = slicing.Grid.from_layer(layer)
    rgb = slicing.composite([layer], grid, 2, 15)
    assert rgb.dtype == np.uint8 and rgb.shape == (32, 32, 3)
    assert rgb.max() > 0


def test_oblique_4d_layer(tmp_path):
    rot = np.array([[np.cos(0.2), -np.sin(0.2), 0], [np.sin(0.2), np.cos(0.2), 0], [0, 0, 1]])
    aff = np.eye(4)
    aff[:3, :3] = rot * 2
    aff[:3, 3] = [-30, -30, -30]
    data = np.random.default_rng(0).random((16, 16, 16, 4)).astype(np.float32)
    path = tmp_path / "bold.nii.gz"
    nib.save(nib.Nifti1Image(data, aff), str(path))
    layer = loaders.load_volume_layer(str(path))
    assert layer.n_frames == 4 and not layer.is_axis_aligned()
    arr, origin, spacing = slicing.aligned_array(layer)
    assert arr.ndim == 3 and np.allclose(spacing, 2.0, atol=1e-4)
    center = (aff @ np.r_[8, 8, 8, 1.0])[:3]
    ts = slicing.timecourse_at(layer, center)
    assert ts.shape == (4,)


@pytest.mark.parametrize("name,kind", [
    ("a.nii.gz", "volume"), ("b.mgz", "volume"), ("lh.pial", "surface"),
    ("x.surf.gii", "surface"), ("x.func.gii", None), ("t.tck", "tracts"), ("readme.txt", None),
])
def test_path_kind(name, kind):
    assert loaders.path_kind(name) == kind
