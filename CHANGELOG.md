# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-15

First public release.

### Added
- Desktop viewer (PySide6 + vedo/VTK) with one main view and three side views
  (layouts 1+3, 2x2, 1x1); click a panel title or Ctrl+1..4 to swap.
- File panel with a filtered folder tree and recursive name search.
- Volumes: NIfTI, MGH/MGZ, Analyze, MINC, PAR/REC, reoriented to RAS+ with
  their full affine; layers with colormap, window, opacity, threshold and
  4D frame; FreeSurfer LUT colours for label maps.
- 3D scene: slices, bounding box, volume rendering, FreeSurfer/GIfTI surfaces
  (shifted to scanner RAS) and .trk/.tck streamlines.
- Tools registry: isosurface, label surfaces, cortex and threshold point
  clouds, point cloud from a surface, Open3D downsampling and Poisson
  reconstruction (optional extra).
- Time course of 4D layers at the cursor; cursor readout with label names.
- `neuroviewer` command with `--write-config`, `--config-path`,
  `--install-desktop` and `--version`; user config file that overrides the
  packaged defaults.
