"""
app.py - Neuroviewer main window.

    neuroviewer [file1.nii.gz file2.mgz lh.pial ...]
    python -m neuroviewer [files ...]

A freeview-like viewer in pure Python: PySide6 for the interface, vedo/VTK
for rendering, nibabel for reading. Files are found in the left panel (folder
tree + recursive search), volumes become layers (right panel), and the main
view can be swapped among 3D / sagittal / coronal / axial by clicking a
panel title (or Ctrl+1..4).
"""
from __future__ import annotations

import fnmatch
import os
import sys
import time
import traceback
from dataclasses import dataclass

os.environ["QT_API"] = "pyside6"


def _setup_qt_plugins():
    """Point Qt to the platform plugins of the PySide6 that is actually imported.

    Other Qt installs export QT_QPA_PLATFORM_PLUGIN_PATH / QT_PLUGIN_PATH for
    their own Qt (Spyder/PyQt5 consoles, opencv-python, conda qt packages).
    PySide6 then fails with: Could not find the Qt platform plugin "xcb" in "".
    This runs before any QApplication exists and only prints what it changed.
    """
    others = [m for m in ("PyQt5", "PyQt6", "PySide2") if m in sys.modules]
    if others:
        print(f"[qt] WARNING: {', '.join(others)} already loaded in this process "
              "(Spyder/IPython console?). Run Neuroviewer from a terminal: neuroviewer")
    from PySide6 import QtCore as _qc  # noqa: PLC0415  (no QApplication yet)
    candidates = []
    try:
        candidates.append(_qc.QLibraryInfo.path(_qc.QLibraryInfo.LibraryPath.PluginsPath))
    except Exception as exc:  # noqa: BLE001
        print(f"[qt] QLibraryInfo lookup failed: {exc}")
    root = os.path.dirname(os.path.abspath(sys.modules["PySide6"].__file__))
    candidates += [os.path.join(root, "Qt", "plugins"), os.path.join(root, "plugins")]
    prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
    candidates += [os.path.join(prefix, "lib", "qt6", "plugins"),
                   os.path.join(prefix, "Library", "lib", "qt6", "plugins"),
                   os.path.join(prefix, "plugins")]
    for cand in candidates:
        platforms = os.path.join(cand, "platforms")
        if cand and os.path.isdir(platforms) and os.listdir(platforms):
            for var, val in (("QT_PLUGIN_PATH", cand), ("QT_QPA_PLATFORM_PLUGIN_PATH", platforms)):
                old = os.environ.get(var)
                if old and os.path.normpath(old) != os.path.normpath(val):
                    print(f"[qt] {var} was {old!r}, replaced by the PySide6 path")
                os.environ[var] = val
            print(f"[qt] PySide6 {_qc.qVersion()} | platform plugins: {platforms} | "
                  f"{sorted(os.listdir(platforms))}")
            return
    print("[qt] WARNING: no PySide6 platform plugin folder found; searched:")
    for cand in candidates:
        print(f"      {cand}")


_setup_qt_plugins()

import numpy as np  # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from . import config  # noqa: E402
from . import loaders  # noqa: E402
from . import slicing  # noqa: E402
from . import tools  # noqa: E402
from . import views  # noqa: E402

import vedo  # noqa: E402
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray  # noqa: E402
from vtkmodules.vtkCommonCore import vtkPoints  # noqa: E402
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData  # noqa: E402
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper  # noqa: E402


# =========================================================================
# background jobs
# =========================================================================
class WorkerSignals(QtCore.QObject):
    done = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    progress = QtCore.Signal(object)


class FuncWorker(QtCore.QRunnable):
    """Runs fn(*args, report=emit) in the thread pool."""

    def __init__(self, fn, *args):
        super().__init__()
        self.fn = fn
        self.args = args
        self.signals = WorkerSignals()
        self.setAutoDelete(False)

    def run(self):
        try:
            res = self.fn(*self.args, report=self.signals.progress.emit)
        except Exception:  # noqa: BLE001
            self.signals.failed.emit(traceback.format_exc())
        else:
            self.signals.done.emit(res)


def search_files(root, pattern, flag, report=None):
    """Walk `root` and report every neuroimaging file whose name matches."""
    pattern = (pattern or "").strip()
    if pattern and not any(ch in pattern for ch in "*?["):
        pattern = f"*{pattern}*"
    pattern = pattern.lower()
    hits, scanned, t_last = [], 0, 0.0
    for dirpath, dirnames, filenames in os.walk(root):
        if flag.get("stop"):
            break
        if config.SEARCH_SKIP_HIDDEN:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        dirnames.sort()
        for fn in sorted(filenames):
            scanned += 1
            full = os.path.join(dirpath, fn)
            if loaders.path_kind(full) is None:
                continue
            if pattern and not fnmatch.fnmatch(fn.lower(), pattern):
                continue
            hits.append(full)
            if report:
                report(("hit", full))
            if len(hits) >= config.SEARCH_MAX_RESULTS:
                return hits
        now = time.time()
        if report and now - t_last > 0.2:
            t_last = now
            report(("scan", scanned, dirpath))
    return hits


# =========================================================================
# streamlines -> VTK actor
# =========================================================================
def streamlines_actor(lines):
    lines = [ln for ln in lines if len(ln) >= 2]
    if not lines:
        raise RuntimeError("No streamline with at least two points.")
    pts = np.concatenate(lines).astype(np.float32)
    cols = np.empty((len(pts), 3), dtype=np.uint8)
    conn = []
    start = 0
    for ln in lines:
        n = len(ln)
        d = np.gradient(ln.astype(np.float64), axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        cols[start:start + n] = np.abs(d) * 255.0
        conn.append(np.r_[n, np.arange(start, start + n)])
        start += n
    vtk_pts = vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(pts, deep=True))
    cells = vtkCellArray()
    cells.SetCells(len(lines), numpy_to_vtkIdTypeArray(np.concatenate(conn).astype(np.int64), deep=True))
    poly = vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.SetLines(cells)
    carr = numpy_to_vtk(cols, deep=True)
    carr.SetName("direction")
    poly.GetPointData().SetScalars(carr)
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(poly)
    mapper.SetScalarModeToUsePointData()
    mapper.SetColorModeToDirectScalars()
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(config.TRACT_LINE_WIDTH)
    return actor


@dataclass(eq=False)
class SceneItem:
    name: str
    obj: object
    kind: str
    visible: bool = True


# =========================================================================
# main window
# =========================================================================
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.APP_TITLE)
        self.resize(*config.WINDOW_SIZE)

        self.layers = []
        self.active = None
        self.grid = None
        self.grid_layer = None
        self.cursor = np.zeros(3)
        self.planes = {a: slicing.PlaneImage(a) for a in range(3)}
        self.plane_k = {a: None for a in range(3)}
        self.scene = []
        self.vr_obj = None
        self.vr_key = None
        self.main_key = config.DEFAULT_MAIN_VIEW
        self.layout_mode = config.DEFAULT_LAYOUT
        self.pool = QtCore.QThreadPool.globalInstance()
        self._jobs = set()
        self._busy = 0
        self._search_flag = None
        self._search_count = 0
        self._volume_queue = []
        self._volume_loading = False

        self._build_views()
        self._build_toolbar()
        self._build_file_dock()
        self._build_layer_dock()
        self._build_scene_dock()
        self._build_timecourse_dock()
        self._build_statusbar()
        self._build_shortcuts()
        self.resizeDocks([self.files_dock, self.layer_dock], [config.FILES_DOCK_WIDTH,
                         config.RIGHT_DOCK_WIDTH], QtCore.Qt.Orientation.Horizontal)
        self.resizeDocks([self.tc_dock], [config.TIMECOURSE_HEIGHT], QtCore.Qt.Orientation.Vertical)
        self.apply_layout()

    # =====================================================================
    # UI construction
    # =====================================================================
    def _build_views(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        self.view_grid = QtWidgets.QGridLayout(central)
        self.view_grid.setContentsMargins(2, 2, 2, 2)
        self.view_grid.setSpacing(3)
        self.panels = {"3d": views.View3DPanel(self, central)}
        for axis in (0, 1, 2):
            p = views.Slice2DPanel(axis, self, central)
            self.panels[p.key] = p

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        tb.setIconSize(QtCore.QSize(16, 16))
        act = QtGui.QAction("Open...", self)
        act.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        act.triggered.connect(self.open_dialog)
        tb.addAction(act)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Layout "))
        self.layout_combo = QtWidgets.QComboBox()
        self.layout_combo.addItems(["1+3", "2x2", "1x1"])
        self.layout_combo.setCurrentText(self.layout_mode)
        self.layout_combo.currentTextChanged.connect(self.set_layout)
        tb.addWidget(self.layout_combo)
        tb.addWidget(QtWidgets.QLabel("  Main view "))
        for i, (key, label) in enumerate((("3d", "3D"), ("sagittal", "Sag"),
                                          ("coronal", "Cor"), ("axial", "Ax"))):
            a = QtGui.QAction(label, self)
            a.setToolTip(f"Show {key} in the main position (Ctrl+{i + 1})")
            a.triggered.connect(lambda checked=False, k=key: self.promote_view(k))
            tb.addAction(a)
        tb.addSeparator()
        a = QtGui.QAction("Reset views", self)
        a.triggered.connect(self.reset_views)
        tb.addAction(a)
        a = QtGui.QAction("Screenshot", self)
        a.setToolTip(f"Save the main view as PNG (x{config.SCREENSHOT_SCALE})")
        a.triggered.connect(self.screenshot)
        tb.addAction(a)
        self.chk_cross = QtWidgets.QCheckBox("Cursor")
        self.chk_cross.setChecked(True)
        self.chk_cross.toggled.connect(self._toggle_crosshair)
        tb.addWidget(self.chk_cross)
        tb.addSeparator()
        tb.addWidget(QtWidgets.QLabel(" Go to RAS "))
        self.goto_edit = QtWidgets.QLineEdit()
        self.goto_edit.setPlaceholderText("x y z (mm)")
        self.goto_edit.setFixedWidth(150)
        self.goto_edit.returnPressed.connect(self._goto)
        tb.addWidget(self.goto_edit)

    def _build_file_dock(self):
        dock = QtWidgets.QDockWidget("Files", self)
        dock.setObjectName("files")
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        row = QtWidgets.QHBoxLayout()
        self.root_edit = QtWidgets.QLineEdit()
        self.root_edit.returnPressed.connect(lambda: self.set_root(self.root_edit.text()))
        b_browse = QtWidgets.QToolButton(text="...")
        b_browse.setToolTip("Choose the root folder")
        b_browse.clicked.connect(self._browse_root)
        b_up = QtWidgets.QToolButton(text="Up")
        b_up.clicked.connect(lambda: self.set_root(os.path.dirname(self.root_edit.text().rstrip(os.sep)) or os.sep))
        row.addWidget(self.root_edit, 1)
        row.addWidget(b_browse)
        row.addWidget(b_up)
        v.addLayout(row)

        self.fs_model = QtWidgets.QFileSystemModel(self)
        self.fs_model.setFilter(QtCore.QDir.Filter.AllDirs | QtCore.QDir.Filter.Files
                                | QtCore.QDir.Filter.NoDotAndDotDot)
        self.fs_model.setNameFilters(loaders.tree_name_filters())
        self.fs_model.setNameFilterDisables(False)
        self.tree = QtWidgets.QTreeView()
        self.tree.setModel(self.fs_model)
        self.tree.setColumnHidden(2, True)
        self.tree.setColumnHidden(3, True)
        self.tree.setColumnWidth(0, 260)
        self.tree.setSortingEnabled(True)
        self.tree.sortByColumn(0, QtCore.Qt.SortOrder.AscendingOrder)
        self.tree.doubleClicked.connect(self._tree_double_clicked)
        self.tree.setToolTip("Double click a file to open it")
        v.addWidget(self.tree, 3)

        v.addWidget(QtWidgets.QLabel("Recursive search under the root folder"))
        row2 = QtWidgets.QHBoxLayout()
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("name filter, e.g. aseg or *T1*.nii.gz (blank = all)")
        self.search_edit.returnPressed.connect(self.start_search)
        b_search = QtWidgets.QPushButton("Search")
        b_search.clicked.connect(self.start_search)
        b_stop = QtWidgets.QPushButton("Stop")
        b_stop.clicked.connect(self.stop_search)
        row2.addWidget(self.search_edit, 1)
        row2.addWidget(b_search)
        row2.addWidget(b_stop)
        v.addLayout(row2)
        self.search_list = QtWidgets.QListWidget()
        self.search_list.itemDoubleClicked.connect(
            lambda it: self.open_path(it.data(QtCore.Qt.ItemDataRole.UserRole)))
        v.addWidget(self.search_list, 2)
        self.search_info = QtWidgets.QLabel("")
        self.search_info.setWordWrap(True)
        v.addWidget(self.search_info)

        dock.setWidget(w)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.files_dock = dock
        start = config.START_DIR or os.path.expanduser("~")
        self.set_root(start)

    def _build_layer_dock(self):
        dock = QtWidgets.QDockWidget("Layers", self)
        dock.setObjectName("layers")
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        self.layer_list = QtWidgets.QListWidget()
        self.layer_list.setToolTip("Top of the list is drawn on top. Checkbox = visible.")
        self.layer_list.itemChanged.connect(self._layer_item_changed)
        self.layer_list.currentRowChanged.connect(self._layer_selected)
        v.addWidget(self.layer_list, 2)

        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Up", lambda: self._move_layer(+1)), ("Down", lambda: self._move_layer(-1)),
                         ("Remove", self._remove_layer)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)

        form = QtWidgets.QFormLayout()
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(config.COLORMAPS)
        self.cmap_combo.currentTextChanged.connect(self._prop_changed)
        form.addRow("Colormap", self.cmap_combo)
        self.vmin_spin = self._make_spin()
        self.vmax_spin = self._make_spin()
        form.addRow("Min", self.vmin_spin)
        form.addRow("Max", self.vmax_spin)
        b_auto = QtWidgets.QPushButton("Auto window (percentiles)")
        b_auto.clicked.connect(self._auto_window)
        form.addRow("", b_auto)
        orow = QtWidgets.QHBoxLayout()
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.valueChanged.connect(self._prop_changed)
        self.opacity_label = QtWidgets.QLabel("1.00")
        self.opacity_label.setFixedWidth(36)
        orow.addWidget(self.opacity_slider, 1)
        orow.addWidget(self.opacity_label)
        form.addRow("Opacity", orow)
        self.clear_chk = QtWidgets.QCheckBox("Transparent below min")
        self.clear_chk.toggled.connect(self._prop_changed)
        form.addRow("", self.clear_chk)
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setKeyboardTracking(False)
        self.frame_spin.valueChanged.connect(self._prop_changed)
        form.addRow("Frame", self.frame_spin)
        v.addLayout(form)
        self.layer_info = QtWidgets.QLabel("")
        self.layer_info.setWordWrap(True)
        self.layer_info.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        v.addWidget(self.layer_info)
        v.addStretch(1)

        dock.setWidget(w)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.layer_dock = dock
        self._set_props_enabled(False)

    def _make_spin(self):
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(-1e12, 1e12)
        s.setDecimals(config.SPIN_DECIMALS)
        s.setKeyboardTracking(False)
        s.valueChanged.connect(self._prop_changed)
        return s

    def _build_scene_dock(self):
        dock = QtWidgets.QDockWidget("3D scene", self)
        dock.setObjectName("scene")
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(4, 4, 4, 4)

        self.chk_slices = QtWidgets.QCheckBox("Show slices in 3D")
        self.chk_slices.setChecked(True)
        self.chk_slices.toggled.connect(lambda f: self.panels["3d"].set_slices_visible(f))
        self.chk_outline = QtWidgets.QCheckBox("Show bounding box")
        self.chk_outline.setChecked(True)
        self.chk_outline.toggled.connect(lambda f: self.panels["3d"].set_outline_visible(f))
        self.chk_vr = QtWidgets.QCheckBox("Volume rendering of the active layer")
        self.chk_vr.toggled.connect(lambda f: self._update_vr())
        for c in (self.chk_slices, self.chk_outline, self.chk_vr):
            v.addWidget(c)

        box = QtWidgets.QGroupBox("Tools")
        bv = QtWidgets.QVBoxLayout(box)
        self.tool_combo = QtWidgets.QComboBox()
        for name, spec in tools.TOOLS.items():
            self.tool_combo.addItem(name)
            self.tool_combo.setItemData(self.tool_combo.count() - 1, spec.description,
                                        QtCore.Qt.ItemDataRole.ToolTipRole)
        b_run = QtWidgets.QPushButton("Run tool")
        b_run.clicked.connect(self.run_tool)
        bv.addWidget(self.tool_combo)
        bv.addWidget(b_run)
        v.addWidget(box)

        v.addWidget(QtWidgets.QLabel("Objects (checkbox = visible)"))
        self.scene_list = QtWidgets.QListWidget()
        self.scene_list.itemChanged.connect(self._scene_item_changed)
        v.addWidget(self.scene_list, 1)
        row = QtWidgets.QHBoxLayout()
        b_load = QtWidgets.QPushButton("Load surface/tracts...")
        b_load.clicked.connect(self._load_scene_file)
        b_color = QtWidgets.QPushButton("Color...")
        b_color.clicked.connect(self._color_scene_item)
        row.addWidget(b_load)
        row.addWidget(b_color)
        v.addLayout(row)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Opacity"))
        self.scene_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.scene_opacity.setRange(0, 100)
        self.scene_opacity.setValue(100)
        self.scene_opacity.valueChanged.connect(self._opacity_scene_item)
        row2.addWidget(self.scene_opacity, 1)
        v.addLayout(row2)
        row3 = QtWidgets.QHBoxLayout()
        b_rm = QtWidgets.QPushButton("Remove")
        b_rm.clicked.connect(self._remove_scene_item)
        b_clear = QtWidgets.QPushButton("Clear all")
        b_clear.clicked.connect(self._clear_scene)
        row3.addWidget(b_rm)
        row3.addWidget(b_clear)
        v.addLayout(row3)

        dock.setWidget(w)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self.tabifyDockWidget(self.layer_dock, dock)
        self.layer_dock.raise_()

    def _build_timecourse_dock(self):
        dock = QtWidgets.QDockWidget("Time course", self)
        dock.setObjectName("timecourse")
        self.tc_fig = Figure(figsize=(8, 1.9), facecolor="#1b1b20")
        self.tc_ax = self.tc_fig.add_subplot(111)
        self.tc_canvas = FigureCanvasQTAgg(self.tc_fig)
        self.tc_canvas.setMinimumHeight(120)
        dock.setWidget(self.tc_canvas)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.tc_dock = dock
        self._draw_timecourse(None, None)

    def _build_statusbar(self):
        sb = self.statusBar()
        self.status_label = QtWidgets.QLabel("Open a volume (double click in the Files panel).")
        self.status_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        sb.addWidget(self.status_label, 1)
        self.busy_bar = QtWidgets.QProgressBar()
        self.busy_bar.setRange(0, 0)
        self.busy_bar.setFixedWidth(140)
        self.busy_bar.setVisible(False)
        sb.addPermanentWidget(self.busy_bar)

    def _build_shortcuts(self):
        for i, key in enumerate(config.VIEW_ORDER):
            sc = QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{i + 1}"), self)
            sc.activated.connect(lambda k=key: self.promote_view(k))

    # =====================================================================
    # layout of the four views
    # =====================================================================
    def set_layout(self, mode):
        self.layout_mode = mode
        self.apply_layout()

    def promote_view(self, key):
        if key not in self.panels:
            return
        self.main_key = key
        self.apply_layout()

    def apply_layout(self):
        g = self.view_grid
        for p in self.panels.values():
            g.removeWidget(p)
        for i in range(4):
            g.setRowStretch(i, 0)
            g.setColumnStretch(i, 0)
        order = [self.main_key] + [k for k in config.VIEW_ORDER if k != self.main_key]
        if self.layout_mode == "2x2":
            for (r, c), k in zip(((0, 0), (0, 1), (1, 0), (1, 1)), order):
                g.addWidget(self.panels[k], r, c)
            for i in range(2):
                g.setRowStretch(i, 1)
                g.setColumnStretch(i, 1)
            visible = set(order)
        elif self.layout_mode == "1x1":
            g.addWidget(self.panels[order[0]], 0, 0)
            g.setRowStretch(0, 1)
            g.setColumnStretch(0, 1)
            visible = {order[0]}
        else:
            g.addWidget(self.panels[order[0]], 0, 0, 3, 1)
            for r, k in enumerate(order[1:]):
                g.addWidget(self.panels[k], r, 1)
                g.setRowStretch(r, 1)
            g.setColumnStretch(0, config.MAIN_VIEW_STRETCH)
            g.setColumnStretch(1, 1)
            visible = set(order)
        for k, p in self.panels.items():
            p.setVisible(k in visible)
            p.set_highlight(k == self.main_key)
        QtCore.QTimer.singleShot(60, self._after_relayout)

    def _after_relayout(self):
        for k, p in self.panels.items():
            if k != "3d":
                p.reset_camera(render=False)
        self.render_all()

    def reset_views(self):
        for p in self.panels.values():
            p.reset_camera(render=False)
        self.render_all()

    def render_all(self):
        for p in self.panels.values():
            p.render()

    def _toggle_crosshair(self, flag):
        for k, p in self.panels.items():
            if k == "3d":
                p.set_cursor_visible(flag)
            else:
                p.set_crosshair_visible(flag)
        self.render_all()

    # =====================================================================
    # files
    # =====================================================================
    def set_root(self, path):
        path = os.path.abspath(os.path.expanduser(path or "~"))
        if not os.path.isdir(path):
            self.statusBar().showMessage(f"Not a folder: {path}", 5000)
            return
        self.root_edit.setText(path)
        self.fs_model.setRootPath(path)
        self.tree.setRootIndex(self.fs_model.index(path))

    def _browse_root(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Root folder", self.root_edit.text())
        if path:
            self.set_root(path)

    def _tree_double_clicked(self, index):
        path = self.fs_model.filePath(index)
        if os.path.isfile(path):
            self.open_path(path)

    def open_dialog(self):
        exts = " ".join("*" + e for e in config.VOLUME_EXTENSIONS + config.TRACT_EXTENSIONS) + " *.gii"
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open", self.root_edit.text(),
            f"Neuroimaging ({exts});;FreeSurfer surfaces (lh.* rh.*);;All files (*)")
        for p in paths:
            self.open_path(p)

    def open_path(self, path):
        if not path:
            return
        kind = loaders.path_kind(path)
        if kind is None:
            kind = "volume"
            print(f"[open] unknown extension, trying as a volume: {path}")
        if kind == "volume":
            # Volumes load one at a time, in the order requested: the first one
            # defines the display grid and the stacking order is predictable.
            self._volume_queue.append(path)
            self._load_next_volume()
        elif kind == "surface":
            self._run(loaders.load_surface, path,
                      done=lambda res, p=path: self._surface_loaded(p, res))
        elif kind == "tracts":
            self._run(loaders.load_streamlines, path,
                      done=lambda res, p=path: self._tracts_loaded(p, res))

    def _load_scene_file(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Load surfaces or streamlines", self.root_edit.text(),
            "Surfaces/tracts (lh.* rh.* *.gii *.trk *.tck);;All files (*)")
        for p in paths:
            self.open_path(p)

    # ------------------------------------------------------------ search
    def start_search(self):
        self.stop_search()
        root = self.root_edit.text()
        if not os.path.isdir(root):
            return
        self.search_list.clear()
        self._search_count = 0
        self._search_root = root
        self._search_flag = {"stop": False}
        self.search_info.setText(f"Searching {root} ...")
        self._run(search_files, root, self.search_edit.text(), self._search_flag,
                  done=self._search_done, progress=self._search_progress)

    def stop_search(self):
        if self._search_flag is not None:
            self._search_flag["stop"] = True

    def _search_progress(self, msg):
        if msg[0] == "hit":
            path = msg[1]
            rel = os.path.relpath(path, self._search_root)
            item = QtWidgets.QListWidgetItem(rel)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.search_list.addItem(item)
            self._search_count += 1
        elif msg[0] == "scan":
            tail = msg[2][-70:]
            self.search_info.setText(f"{self._search_count} found | {msg[1]} files scanned | ...{tail}")

    def _search_done(self, hits):
        stopped = self._search_flag is not None and self._search_flag.get("stop")
        cap = " (limit reached)" if len(hits) >= config.SEARCH_MAX_RESULTS else ""
        self.search_info.setText(f"{len(hits)} files found{cap}{' (stopped)' if stopped else ''}. "
                                 "Double click to open.")

    # =====================================================================
    # job runner
    # =====================================================================
    def _run(self, fn, *args, done=None, failed=None, progress=None):
        w = FuncWorker(fn, *args)
        self._jobs.add(w)
        w.signals.done.connect(lambda res, w=w: self._job_done(w, res, done))
        w.signals.failed.connect(lambda msg, w=w: self._job_failed(w, msg, failed))
        w.signals.progress.connect(progress if progress else self._job_status)
        self._set_busy(+1)
        self.pool.start(w)

    def _job_status(self, msg):
        if isinstance(msg, tuple) and msg and msg[0] == "status":
            self.statusBar().showMessage(msg[1], 5000)

    def _job_done(self, w, res, done):
        self._jobs.discard(w)
        self._set_busy(-1)
        if done is None:
            return
        try:
            done(res)
        except Exception:  # noqa: BLE001
            self._error(traceback.format_exc())

    def _job_failed(self, w, msg, failed):
        self._jobs.discard(w)
        self._set_busy(-1)
        if failed is not None:
            failed(msg)
        self._error(msg)

    def _set_busy(self, delta):
        self._busy = max(self._busy + delta, 0)
        self.busy_bar.setVisible(self._busy > 0)

    def _error(self, text):
        print(text)
        last = text.strip().splitlines()[-1] if text.strip() else text
        QtWidgets.QMessageBox.warning(self, config.APP_TITLE, last)

    # =====================================================================
    # layers
    # =====================================================================
    def _load_next_volume(self):
        if self._volume_loading or not self._volume_queue:
            return
        path = self._volume_queue.pop(0)
        self._volume_loading = True
        self._run(loaders.load_volume_layer, path, bool(self.layers),
                  done=self._volume_loaded, failed=self._volume_failed)

    def _volume_failed(self, msg):
        self._volume_loading = False
        QtCore.QTimer.singleShot(0, self._load_next_volume)

    def _volume_loaded(self, layer):
        self._volume_loading = False
        try:
            self.layers.append(layer)
            if self.grid is None:
                self._set_grid_from(layer)
            self.active = layer
            self._refresh_layer_list()
            self.refresh(force=True)
            self.statusBar().showMessage(f"Loaded {layer.path}", 5000)
        finally:
            QtCore.QTimer.singleShot(0, self._load_next_volume)

    def _set_grid_from(self, layer):
        if layer is None:
            self.grid = None
            self.grid_layer = None
            for p in self.panels.values():
                p.clear_scene()
            self.render_all()
            return
        self.grid_layer = layer
        self.grid = slicing.Grid.from_layer(layer)
        self.cursor = self.grid.snap(self.grid.center())
        for plane in self.planes.values():
            plane.configure(self.grid)
        self.plane_k = {a: None for a in range(3)}
        self._recompute_planes(force=True)
        self.panels["3d"].set_scene(self.planes, self.grid)
        for axis in (0, 1, 2):
            self.panels[slicing.AXIS_NAMES[axis]].set_scene(self.planes[axis], self.grid)
        self._toggle_crosshair(self.chk_cross.isChecked())

    def _refresh_layer_list(self):
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        current_row = -1
        for row, layer in enumerate(reversed(self.layers)):
            item = QtWidgets.QListWidgetItem(layer.name)
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked if layer.visible
                               else QtCore.Qt.CheckState.Unchecked)
            item.setToolTip(layer.path)
            self.layer_list.addItem(item)
            if layer is self.active:
                current_row = row
        self.layer_list.setCurrentRow(current_row)
        self.layer_list.blockSignals(False)
        self._populate_props()

    def _layer_at_row(self, row):
        if row < 0 or row >= len(self.layers):
            return None
        return self.layers[len(self.layers) - 1 - row]

    def _layer_selected(self, row):
        layer = self._layer_at_row(row)
        if layer is None:
            return
        self.active = layer
        self._populate_props()
        self._update_vr()
        self._update_readouts()

    def _layer_item_changed(self, item):
        row = self.layer_list.row(item)
        layer = self._layer_at_row(row)
        if layer is None:
            return
        layer.visible = item.checkState() == QtCore.Qt.CheckState.Checked
        self.refresh(force=True)

    def _move_layer(self, delta):
        if self.active is None:
            return
        i = self.layers.index(self.active)
        j = i + delta
        if not 0 <= j < len(self.layers):
            return
        self.layers[i], self.layers[j] = self.layers[j], self.layers[i]
        self._refresh_layer_list()
        self.refresh(force=True)

    def _remove_layer(self):
        layer = self.active
        if layer is None:
            return
        self.layers.remove(layer)
        if self.vr_key is not None and self.vr_key[0] is layer:
            self._remove_vr()
        self.active = self.layers[-1] if self.layers else None
        if layer is self.grid_layer:
            self._set_grid_from(self.layers[0] if self.layers else None)
        self._refresh_layer_list()
        self.refresh(force=True)
        self._update_vr()

    def _set_props_enabled(self, flag):
        for wdg in (self.cmap_combo, self.vmin_spin, self.vmax_spin, self.opacity_slider,
                    self.clear_chk, self.frame_spin):
            wdg.setEnabled(flag)

    def _populate_props(self):
        layer = self.active
        widgets = (self.cmap_combo, self.vmin_spin, self.vmax_spin, self.opacity_slider,
                   self.clear_chk, self.frame_spin)
        for wdg in widgets:
            wdg.blockSignals(True)
        try:
            if layer is None:
                self._set_props_enabled(False)
                self.layer_info.setText("")
                return
            self._set_props_enabled(True)
            if self.cmap_combo.findText(layer.cmap) < 0:
                self.cmap_combo.addItem(layer.cmap)
            self.cmap_combo.setCurrentText(layer.cmap)
            step = max(abs(layer.vmax - layer.vmin) / 100.0, 10 ** (-config.SPIN_DECIMALS))
            for s, val in ((self.vmin_spin, layer.vmin), (self.vmax_spin, layer.vmax)):
                s.setSingleStep(step)
                s.setValue(val)
            self.opacity_slider.setValue(int(round(layer.opacity * 100)))
            self.opacity_label.setText(f"{layer.opacity:.2f}")
            self.clear_chk.setChecked(layer.clear_below)
            self.frame_spin.setRange(0, layer.n_frames - 1)
            self.frame_spin.setValue(layer.frame)
            self.frame_spin.setEnabled(layer.n_frames > 1)
            self.layer_info.setText(layer.info_text())
        finally:
            for wdg in widgets:
                wdg.blockSignals(False)

    def _prop_changed(self, *_):
        layer = self.active
        if layer is None:
            return
        layer.cmap = self.cmap_combo.currentText()
        layer.vmin = float(self.vmin_spin.value())
        layer.vmax = float(self.vmax_spin.value())
        layer.opacity = self.opacity_slider.value() / 100.0
        self.opacity_label.setText(f"{layer.opacity:.2f}")
        layer.clear_below = self.clear_chk.isChecked()
        layer.frame = int(self.frame_spin.value())
        self.refresh(force=True)
        self._update_vr()

    def _auto_window(self):
        layer = self.active
        if layer is None:
            return
        layer.vmin, layer.vmax = loaders.robust_range(layer)
        self._populate_props()
        self.refresh(force=True)
        self._update_vr()

    # =====================================================================
    # cursor and slices
    # =====================================================================
    def set_cursor(self, world):
        if self.grid is None:
            return
        self.cursor = self.grid.snap(world)
        self.refresh()

    def step_slice(self, axis, delta):
        if self.grid is None:
            return
        idx = self.grid.index_of(self.cursor)
        idx[axis] = int(np.clip(idx[axis] + delta, 0, self.grid.dims[axis] - 1))
        self.cursor = self.grid.world_of(idx)
        self.refresh()

    def _goto(self):
        txt = self.goto_edit.text().replace(",", " ").split()
        try:
            p = np.array([float(t) for t in txt[:3]])
        except ValueError:
            return
        if p.size == 3:
            self.set_cursor(p)

    def _recompute_planes(self, force=False):
        idx = self.grid.index_of(self.cursor)
        for axis in range(3):
            k = int(idx[axis])
            if force or self.plane_k[axis] != k:
                rgb = slicing.composite(self.layers, self.grid, axis, k)
                self.planes[axis].update(rgb, self.grid, k)
                self.plane_k[axis] = k
        return idx

    def refresh(self, force=False):
        if self.grid is None:
            self._update_readouts()
            return
        idx = self._recompute_planes(force=force)
        for p in self.panels.values():
            p.update_cursor(self.cursor, idx)
        self.render_all()
        self._update_readouts()

    def _update_readouts(self):
        if self.grid is None:
            self.status_label.setText("Open a volume (double click in the Files panel).")
            self._draw_timecourse(None, None)
            return
        x, y, z = self.cursor
        parts = [f"RAS ({x:.1f}, {y:.1f}, {z:.1f}) mm"]
        for layer in reversed(self.layers):
            if not layer.visible:
                continue
            val, ijk = slicing.value_at(layer, self.cursor)
            if val is None:
                parts.append(f"{layer.name}: outside")
                continue
            txt = f"{layer.name} [{ijk[0]} {ijk[1]} {ijk[2]}] = {val:.5g}"
            if layer.cmap == "lut":
                txt += f" ({loaders.label_name(val)})"
            parts.append(txt)
        self.status_label.setText("   |   ".join(parts))
        layer = self.active
        if layer is not None and layer.n_frames > 1:
            self._draw_timecourse(layer, slicing.timecourse_at(layer, self.cursor))
        else:
            self._draw_timecourse(None, None)

    def _draw_timecourse(self, layer, ts):
        ax = self.tc_ax
        ax.clear()
        ax.set_facecolor("#1b1b20")
        for s in ax.spines.values():
            s.set_color("#777777")
        ax.tick_params(colors="#bbbbbb", labelsize=8)
        if layer is None or ts is None:
            ax.text(0.5, 0.5, "Time course: select a 4D layer", color="#888888",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.plot(np.arange(len(ts)), ts, color="#ffb347", lw=1.0)
            ax.axvline(layer.frame, color="#ff5050", lw=1.0)
            ax.set_xlim(0, max(len(ts) - 1, 1))
            ax.set_xlabel("Frame", color="#bbbbbb", fontsize=8)
            ax.set_ylabel("Signal", color="#bbbbbb", fontsize=8)
        self.tc_fig.tight_layout(pad=0.6)
        self.tc_canvas.draw_idle()

    # =====================================================================
    # 3D scene
    # =====================================================================
    def _add_scene_object(self, name, obj, kind):
        item = SceneItem(name, obj, kind)
        self.scene.append(item)
        self.panels["3d"].add_object(obj)
        li = QtWidgets.QListWidgetItem(f"[{kind}] {name}")
        li.setFlags(li.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        li.setCheckState(QtCore.Qt.CheckState.Checked)
        self.scene_list.blockSignals(True)
        self.scene_list.addItem(li)
        self.scene_list.setCurrentItem(li)
        self.scene_list.blockSignals(False)

    def _surface_loaded(self, path, res):
        verts, faces = res
        mesh = vedo.Mesh([verts, faces]).c(loaders.surface_color(path))
        try:
            mesh.phong()
        except Exception:  # noqa: BLE001
            pass
        self._add_scene_object(loaders.display_name(path), mesh, "surface")

    def _tracts_loaded(self, path, lines):
        actor = streamlines_actor(lines)
        self._add_scene_object(f"{loaders.display_name(path)} ({len(lines)})", actor, "tracts")

    def _scene_item_changed(self, li):
        row = self.scene_list.row(li)
        if 0 <= row < len(self.scene):
            item = self.scene[row]
            item.visible = li.checkState() == QtCore.Qt.CheckState.Checked
            views.set_visible(item.obj, item.visible)
            self.panels["3d"].render()

    def _selected_scene_item(self):
        row = self.scene_list.currentRow()
        if 0 <= row < len(self.scene):
            return row, self.scene[row]
        return -1, None

    def _color_scene_item(self):
        row, item = self._selected_scene_item()
        if item is None:
            return
        col = QtWidgets.QColorDialog.getColor(parent=self)
        if col.isValid():
            views.set_color(item.obj, (col.redF(), col.greenF(), col.blueF()))
            self.panels["3d"].render()

    def _opacity_scene_item(self, value):
        row, item = self._selected_scene_item()
        if item is None:
            return
        views.set_opacity(item.obj, value / 100.0)
        self.panels["3d"].render()

    def _remove_scene_item(self):
        row, item = self._selected_scene_item()
        if item is None:
            return
        self.panels["3d"].remove_object(item.obj)
        self.scene.pop(row)
        self.scene_list.takeItem(row)

    def _clear_scene(self):
        for item in self.scene:
            self.panels["3d"].remove_object(item.obj)
        self.scene = []
        self.scene_list.clear()

    def run_tool(self):
        name = self.tool_combo.currentText()
        spec = tools.TOOLS.get(name)
        if spec is None:
            return
        ctx = tools.ToolContext(self)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            out = spec.fn(ctx) or []
        except tools.ToolCancelled:
            return
        except Exception:  # noqa: BLE001
            self._error(traceback.format_exc())
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        for name_i, obj, kind in out:
            self._add_scene_object(name_i, obj, kind)

    # ------------------------------------------------------------ volume rendering
    def _remove_vr(self):
        if self.vr_obj is not None:
            self.panels["3d"].remove_object(self.vr_obj)
        self.vr_obj = None
        self.vr_key = None

    def _update_vr(self):
        layer = self.active
        if not self.chk_vr.isChecked() or layer is None:
            if self.vr_obj is not None:
                self._remove_vr()
            return
        key = (layer, layer.frame)
        try:
            if self.vr_obj is None or self.vr_key is None or self.vr_key[0] is not key[0] \
                    or self.vr_key[1] != key[1]:
                QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
                try:
                    self._remove_vr()
                    arr, origin, spacing = slicing.aligned_array(layer)
                    vol = vedo.Volume(np.asarray(arr, dtype=np.float32), spacing=spacing, origin=origin)
                    try:
                        vol.mode(0)
                    except Exception:  # noqa: BLE001
                        pass
                    self.vr_obj = vol
                    self.vr_key = key
                    self.panels["3d"].add_object(vol)
                finally:
                    QtWidgets.QApplication.restoreOverrideCursor()
            cmap = config.VR_CMAP_FOR_LABELS if layer.cmap == "lut" else layer.cmap
            cmap = "hot" if cmap == "heat" else cmap
            try:
                self.vr_obj.cmap(cmap, vmin=layer.vmin, vmax=layer.vmax)
            except TypeError:
                self.vr_obj.cmap(cmap)
            try:
                self.vr_obj.alpha(list(config.VR_ALPHA_RAMP), vmin=layer.vmin, vmax=layer.vmax)
            except TypeError:
                self.vr_obj.alpha(list(config.VR_ALPHA_RAMP))
            self.panels["3d"].render()
        except Exception:  # noqa: BLE001
            self._error(traceback.format_exc())

    # =====================================================================
    # misc
    # =====================================================================
    def screenshot(self):
        p = self.panels[self.main_key]
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save screenshot", os.path.join(self.root_edit.text(), f"neuroviewer_{self.main_key}.png"),
            "PNG (*.png)")
        if not path:
            return
        try:
            p.screenshot(path, scale=config.SCREENSHOT_SCALE)
            self.statusBar().showMessage(f"Saved {path}", 5000)
        except Exception:  # noqa: BLE001
            self._error(traceback.format_exc())

    def closeEvent(self, event):
        self.stop_search()
        for p in self.panels.values():
            p.shutdown()
        super().closeEvent(event)


# =========================================================================
# entry point
# =========================================================================
def _dark_palette(app):
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    c = QtGui.QColor
    roles = QtGui.QPalette.ColorRole
    pal.setColor(roles.Window, c(37, 37, 42))
    pal.setColor(roles.WindowText, c(220, 220, 225))
    pal.setColor(roles.Base, c(27, 27, 32))
    pal.setColor(roles.AlternateBase, c(42, 42, 48))
    pal.setColor(roles.ToolTipBase, c(50, 50, 56))
    pal.setColor(roles.ToolTipText, c(230, 230, 230))
    pal.setColor(roles.Text, c(220, 220, 225))
    pal.setColor(roles.Button, c(45, 45, 52))
    pal.setColor(roles.ButtonText, c(220, 220, 225))
    pal.setColor(roles.Highlight, c(200, 140, 40))
    pal.setColor(roles.HighlightedText, c(20, 20, 20))
    app.setPalette(pal)


ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "neuroviewer.png")


def run(files=()):
    """Start the GUI, optionally opening `files`; returns the Qt exit code."""
    QtWidgets.QApplication.setApplicationName("Neuroviewer")
    QtWidgets.QApplication.setOrganizationName("rdneuro")
    QtGui.QGuiApplication.setDesktopFileName("neuroviewer")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([sys.argv[0]])
    if os.path.isfile(ICON_PATH):
        app.setWindowIcon(QtGui.QIcon(ICON_PATH))
    if config.DARK_THEME:
        _dark_palette(app)
    win = MainWindow()
    win.show()
    for path in files:
        QtCore.QTimer.singleShot(200, lambda p=path: win.open_path(os.path.abspath(p)))
    return app.exec()


def main():
    """Kept for `python -m neuroviewer.app`; the console command uses cli.main."""
    from .cli import main as cli_main  # noqa: PLC0415
    cli_main()


if __name__ == "__main__":
    main()
