"""
views.py - the four render panels of Neuroviewer (3D, sagittal, coronal, axial).

Each panel is a Qt frame with a clickable title bar (click -> main view) and a
VTK render window driven by a vedo.Plotter embedded through
QVTKRenderWindowInteractor.

2D panels: parallel-projection camera along the slice normal.
  left click / drag ......... move the cursor (all views follow)
  shift + left drag, middle . pan
  right drag, wheel ......... zoom
  Up/PageUp, Down/PageDown .. next / previous slice
  r ......................... fit the slice to the panel

3D panel: trackball camera.
  left drag rotate, middle pan, right/wheel zoom
  double click on a slice or surface ... move the cursor there
  r ......................... reset the camera
"""
from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets

import vtkmodules.qt
vtkmodules.qt.PyQtImpl = "PySide6"
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor  # noqa: E402
import vtkmodules.vtkRenderingOpenGL2  # noqa: E402,F401  (OpenGL backend)
import vtkmodules.vtkRenderingFreeType  # noqa: E402,F401  (text rendering)
from vtkmodules.vtkInteractionStyle import (vtkInteractorStyleImage,  # noqa: E402
                                            vtkInteractorStyleTrackballCamera)
from vtkmodules.vtkFiltersSources import vtkOutlineSource, vtkSphereSource  # noqa: E402
from vtkmodules.vtkRenderingCore import (vtkActor, vtkCellPicker,  # noqa: E402
                                         vtkPolyDataMapper, vtkProp,
                                         vtkWindowToImageFilter)
from vtkmodules.vtkIOImage import vtkPNGWriter  # noqa: E402
import vedo  # noqa: E402

from . import config  # noqa: E402
from . import slicing  # noqa: E402

_LETTERS = {0: ("L", "R"), 1: ("P", "A"), 2: ("I", "S")}


# =========================================================================
# helpers for objects that may be vedo objects or raw VTK props
# =========================================================================
def _prop_of(obj):
    if isinstance(obj, vtkProp):
        return obj
    return getattr(obj, "actor", obj)


def set_visible(obj, flag):
    _prop_of(obj).SetVisibility(bool(flag))


def set_color(obj, rgb):
    if hasattr(obj, "c") and callable(getattr(obj, "c")):
        obj.c(tuple(rgb))
        return
    prop = _prop_of(obj)
    prop.GetProperty().SetColor(*rgb)
    mapper = prop.GetMapper()
    if mapper is not None:
        mapper.ScalarVisibilityOff()


def set_opacity(obj, value):
    if hasattr(obj, "alpha") and callable(getattr(obj, "alpha")):
        obj.alpha(float(value))
        return
    _prop_of(obj).GetProperty().SetOpacity(float(value))


def _letter(vec):
    ax = int(np.argmax(np.abs(vec)))
    return _LETTERS[ax][1 if vec[ax] > 0 else 0]


def view_geometry(axis):
    """(sign, view_up): camera sits at centre + sign * normal * distance."""
    if axis == 2:
        return (-1 if config.RADIOLOGICAL else 1), np.array([0.0, 1.0, 0.0])
    if axis == 1:
        return (1 if config.RADIOLOGICAL else -1), np.array([0.0, 0.0, 1.0])
    return (-1 if config.SAGITTAL_ANTERIOR_LEFT else 1), np.array([0.0, 0.0, 1.0])


# =========================================================================
# interactor styles
# =========================================================================
class SliceStyle(vtkInteractorStyleImage):
    """Image style where the left button drives the cursor."""

    def __init__(self):
        super().__init__()
        self.panel = None
        self._dragging = False
        self.SetInteractionModeToImage2D()
        self.AddObserver("LeftButtonPressEvent", self._left_down)
        self.AddObserver("LeftButtonReleaseEvent", self._left_up)
        self.AddObserver("MouseMoveEvent", self._move)
        self.AddObserver("KeyPressEvent", self._key)
        self.AddObserver("CharEvent", self._char)

    def _left_down(self, obj, event):
        iren = self.GetInteractor()
        if iren.GetShiftKey():
            self.OnLeftButtonDown()
            return
        self._dragging = True
        self.panel.pick_at(*iren.GetEventPosition())

    def _left_up(self, obj, event):
        if self._dragging:
            self._dragging = False
            return
        self.OnLeftButtonUp()

    def _move(self, obj, event):
        if self._dragging:
            self.panel.pick_at(*self.GetInteractor().GetEventPosition())
            return
        self.OnMouseMove()

    def _key(self, obj, event):
        sym = self.GetInteractor().GetKeySym()
        if sym in ("Up", "Prior", "Right"):
            self.panel.controller.step_slice(self.panel.axis, +1)
        elif sym in ("Down", "Next", "Left"):
            self.panel.controller.step_slice(self.panel.axis, -1)

    def _char(self, obj, event):
        if self.GetInteractor().GetKeyCode() in ("r", "R"):
            self.panel.reset_camera()


class Style3D(vtkInteractorStyleTrackballCamera):
    """Trackball style; double click picks a point and moves the cursor."""

    def __init__(self):
        super().__init__()
        self.panel = None
        self.AddObserver("LeftButtonPressEvent", self._left_down)
        self.AddObserver("CharEvent", self._char)

    def _left_down(self, obj, event):
        iren = self.GetInteractor()
        if iren.GetRepeatCount() > 0:
            self.panel.pick_at(*iren.GetEventPosition())
            return
        self.OnLeftButtonDown()

    def _char(self, obj, event):
        if self.GetInteractor().GetKeyCode() in ("r", "R"):
            self.panel.reset_camera()


# =========================================================================
# panels
# =========================================================================
class ViewPanel(QtWidgets.QFrame):
    """Title bar + embedded vedo.Plotter."""

    def __init__(self, key, title, controller, background, parent):
        super().__init__(parent)
        self.key = key
        self.title = title
        self.controller = controller
        self.grid = None
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = QtWidgets.QPushButton(title, self)
        self.header.setFlat(True)
        self.header.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.header.setToolTip("Click to show this view in the main position")
        self.header.clicked.connect(lambda: controller.promote_view(self.key))
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.header)
        layout.addWidget(self.vtk_widget, 1)

        self.plotter = vedo.Plotter(qt_widget=self.vtk_widget, bg=background, axes=0)
        self.plotter.show(interactive=False)
        self.renderer = self.plotter.renderer
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.interactor = self.render_window.GetInteractor()
        # drop vedo's keyboard shortcuts (q/e/s... are meant for standalone windows)
        for ev in ("KeyPressEvent", "KeyReleaseEvent", "CharEvent"):
            self.interactor.RemoveObservers(ev)
        self._style = None
        self.set_highlight(False)

    def set_style(self, style):
        style.panel = self
        self.interactor.SetInteractorStyle(style)
        style.SetDefaultRenderer(self.renderer)
        self._style = style

    def set_highlight(self, is_main):
        if is_main:
            self.header.setStyleSheet(
                "QPushButton {font-weight: bold; color: #ffd27f; background: #2b2b33;"
                " border: none; padding: 3px; text-align: left;}")
        else:
            self.header.setStyleSheet(
                "QPushButton {color: #c8c8d0; background: #1e1e24; border: none;"
                " padding: 3px; text-align: left;}"
                "QPushButton:hover {color: #ffd27f;}")

    def render(self):
        if not self.isVisible():
            return
        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()

    def screenshot(self, path, scale=1):
        """Save this panel as PNG (own render window, not vedo's global plotter)."""
        self.render_window.Render()
        w2i = vtkWindowToImageFilter()
        w2i.SetInput(self.render_window)
        w2i.SetScale(int(scale))
        w2i.SetInputBufferTypeToRGB()
        w2i.ReadFrontBufferOff()
        w2i.Update()
        writer = vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        self.render_window.Render()

    def shutdown(self):
        try:
            self.vtk_widget.Finalize()
        except Exception as exc:  # noqa: BLE001
            print(f"[shutdown] {self.key}: {exc}")


class Slice2DPanel(ViewPanel):

    def __init__(self, axis, controller, parent):
        name = slicing.AXIS_NAMES[axis]
        super().__init__(name, name.capitalize(), controller, config.BG_2D, parent)
        self.axis = axis
        self.normal = np.eye(3)[axis]
        self.sign, self.up = view_geometry(axis)
        self.set_style(SliceStyle())
        self.image_actor = None
        self.crosshair = Crosshair2D()
        for act in self.crosshair.actors:
            self.renderer.AddActor(act)
        self.crosshair.set_visible(False)
        self.orient = {k: slicing.text_actor(bold=True) for k in ("left", "right", "top", "bottom")}
        slicing.place_text(self.orient["left"], 0.015, 0.5, "left", "center")
        slicing.place_text(self.orient["right"], 0.985, 0.5, "right", "center")
        slicing.place_text(self.orient["top"], 0.5, 0.985, "center", "top")
        slicing.place_text(self.orient["bottom"], 0.5, 0.015, "center", "bottom")
        self.info = slicing.text_actor(size=config.TEXT_SIZE - 2)
        slicing.place_text(self.info, 0.015, 0.015, "left", "bottom")
        for t in list(self.orient.values()) + [self.info]:
            self.renderer.AddViewProp(t)
        self._update_orientation_labels()

    def _update_orientation_labels(self):
        d = -self.sign * self.normal
        right = np.cross(d, self.up)
        self.orient["right"].SetInput(_letter(right))
        self.orient["left"].SetInput(_letter(-right))
        self.orient["top"].SetInput(_letter(self.up))
        self.orient["bottom"].SetInput(_letter(-self.up))

    def set_scene(self, plane_image, grid):
        self.clear_scene()
        self.grid = grid
        self.image_actor = plane_image.make_actor(config.SLICE_INTERP_2D)
        self.renderer.AddViewProp(self.image_actor)
        self.crosshair.set_visible(True)
        self.reset_camera(render=False)

    def clear_scene(self):
        if self.image_actor is not None:
            self.renderer.RemoveViewProp(self.image_actor)
            self.image_actor = None
        self.crosshair.set_visible(False)
        self.info.SetInput("")
        self.grid = None

    def reset_camera(self, render=True):
        if self.grid is None:
            return
        g = self.grid
        c = g.center()
        extent = (g.hi - g.lo) + g.spacing
        dist = float(np.linalg.norm(extent)) * 2.0 + 10.0
        cam = self.renderer.GetActiveCamera()
        cam.ParallelProjectionOn()
        cam.SetFocalPoint(*c)
        cam.SetPosition(*(c + self.sign * self.normal * dist))
        cam.SetViewUp(*self.up)
        up_ax = int(np.argmax(np.abs(self.up)))
        right_ax = [a for a in slicing.inplane_axes(self.axis) if a != up_ax][0]
        w, h = self.render_window.GetSize()
        aspect = (w / h) if (w > 0 and h > 0) else 1.0
        half_h = 0.5 * extent[up_ax]
        half_w = 0.5 * extent[right_ax]
        cam.SetParallelScale(1.04 * max(half_h, half_w / aspect))
        self.renderer.ResetCameraClippingRange()
        if render:
            self.render()

    def pick_at(self, x, y):
        if self.grid is None:
            return
        ren = self.renderer
        ren.SetDisplayPoint(float(x), float(y), 0.0)
        ren.DisplayToWorld()
        w = ren.GetWorldPoint()
        if w[3] == 0:
            return
        p = np.array(w[:3], dtype=float) / w[3]
        p[self.axis] = self.controller.cursor[self.axis]
        self.controller.set_cursor(p)

    def update_cursor(self, cursor, idx):
        if self.grid is None:
            return
        offset = self.sign * 0.5 * float(self.grid.spacing[self.axis])
        self.crosshair.update(cursor, self.grid, self.axis, offset)
        self.info.SetInput(f"{self.title} {int(idx[self.axis]) + 1}/{int(self.grid.dims[self.axis])}")

    def set_crosshair_visible(self, flag):
        self.crosshair.set_visible(flag and self.grid is not None)


class Crosshair2D(slicing.Crosshair):
    def __init__(self):
        super().__init__(config.CROSSHAIR_COLOR, config.CROSSHAIR_WIDTH)


class View3DPanel(ViewPanel):

    def __init__(self, controller, parent):
        super().__init__("3d", "3D", controller, config.BG_3D, parent)
        self.set_style(Style3D())
        self.plane_actors = {}
        self.slices_visible = True
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(0.0005)

        self.outline_src = vtkOutlineSource()
        m = vtkPolyDataMapper()
        m.SetInputConnection(self.outline_src.GetOutputPort())
        self.outline = vtkActor()
        self.outline.SetMapper(m)
        self.outline.GetProperty().SetColor(*config.OUTLINE_COLOR)
        self.outline.PickableOff()
        self.outline.SetVisibility(False)
        self.renderer.AddActor(self.outline)

        self.sphere_src = vtkSphereSource()
        self.sphere_src.SetRadius(config.CURSOR_SPHERE_RADIUS)
        self.sphere_src.SetThetaResolution(24)
        self.sphere_src.SetPhiResolution(16)
        m2 = vtkPolyDataMapper()
        m2.SetInputConnection(self.sphere_src.GetOutputPort())
        self.sphere = vtkActor()
        self.sphere.SetMapper(m2)
        self.sphere.GetProperty().SetColor(*config.CURSOR_SPHERE_COLOR)
        self.sphere.PickableOff()
        self.sphere.SetVisibility(False)
        self.renderer.AddActor(self.sphere)

        self.info = slicing.text_actor(size=config.TEXT_SIZE - 2)
        slicing.place_text(self.info, 0.015, 0.015, "left", "bottom")
        self.renderer.AddViewProp(self.info)

    def set_scene(self, plane_images, grid):
        self.clear_scene()
        self.grid = grid
        for axis, plane in plane_images.items():
            act = plane.make_actor(config.SLICE_INTERP_3D)
            act.SetVisibility(self.slices_visible)
            self.plane_actors[axis] = act
            self.renderer.AddViewProp(act)
        self.outline_src.SetBounds(*grid.bounds())
        self.outline.SetVisibility(True)
        self.sphere.SetVisibility(True)
        self.reset_camera(render=False)

    def clear_scene(self):
        for act in self.plane_actors.values():
            self.renderer.RemoveViewProp(act)
        self.plane_actors = {}
        self.outline.SetVisibility(False)
        self.sphere.SetVisibility(False)
        self.info.SetInput("")
        self.grid = None

    def reset_camera(self, render=True):
        cam = self.renderer.GetActiveCamera()
        cam.ParallelProjectionOff()
        if self.grid is not None:
            c = self.grid.center()
            diag = float(np.linalg.norm(self.grid.hi - self.grid.lo)) + 1.0
            d = np.asarray(config.VIEW3D_DIRECTION, dtype=float)
            d = d / np.linalg.norm(d)
            cam.SetFocalPoint(*c)
            cam.SetPosition(*(c + d * diag * 1.9))
            cam.SetViewUp(0.0, 0.0, 1.0)
        self.renderer.ResetCamera()
        self.renderer.ResetCameraClippingRange()
        if render:
            self.render()

    def update_cursor(self, cursor, idx):
        self.sphere_src.SetCenter(*[float(v) for v in cursor])
        self.sphere_src.Modified()
        self.info.SetInput(f"RAS {cursor[0]:.1f}, {cursor[1]:.1f}, {cursor[2]:.1f} mm")

    def pick_at(self, x, y):
        if self.picker.Pick(float(x), float(y), 0.0, self.renderer):
            pos = np.array(self.picker.GetPickPosition(), dtype=float)
            self.controller.set_cursor(pos)

    def set_slices_visible(self, flag):
        self.slices_visible = bool(flag)
        for act in self.plane_actors.values():
            act.SetVisibility(self.slices_visible)
        self.render()

    def set_outline_visible(self, flag):
        self.outline.SetVisibility(bool(flag) and self.grid is not None)
        self.render()

    def set_cursor_visible(self, flag):
        self.sphere.SetVisibility(bool(flag) and self.grid is not None)

    def add_object(self, obj):
        if isinstance(obj, vtkProp):
            self.renderer.AddViewProp(obj)
        else:
            self.plotter.add(obj)
        self.render()

    def remove_object(self, obj):
        if isinstance(obj, vtkProp):
            self.renderer.RemoveViewProp(obj)
        else:
            self.plotter.remove(obj)
        self.render()
