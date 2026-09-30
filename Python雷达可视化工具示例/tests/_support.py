"""Python 雷达可视化工具测试的无图形界面加载辅助函数。"""

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
import types


os.environ.setdefault("MPLBACKEND", "Agg")
EXAMPLE_DIR = Path(__file__).resolve().parents[1]
_MISSING = object()


def _ensure_mock_tkinter():
    try:
        import tkinter
        return False
    except (ImportError, ModuleNotFoundError):
        class MockVar:
            def __init__(self, value=None):
                self._value = value

            def get(self):
                return self._value

            def set(self, val):
                self._value = val

        fake_tk = types.ModuleType("tkinter")
        fake_tk.BooleanVar = lambda value=False: MockVar(bool(value))
        fake_tk.DoubleVar = lambda value=0.0: MockVar(float(value))
        fake_tk.IntVar = lambda value=0: MockVar(int(value))
        fake_tk.StringVar = lambda value="": MockVar(str(value))
        fake_tk.Tk = object
        fake_tk.W = "w"
        fake_tk.E = "e"
        fake_tk.X = "x"
        fake_tk.Y = "y"
        fake_tk.BOTH = "both"
        fake_tk.LEFT = "left"
        fake_tk.RIGHT = "right"
        fake_tk.TOP = "top"
        fake_tk.BOTTOM = "bottom"
        fake_tk._default_root = None
        fake_tk.Tcl = lambda: None
        fake_tk.__path__ = []

        fake_ttk = types.ModuleType("tkinter.ttk")
        fake_ttk.Frame = fake_ttk.Label = fake_ttk.Button = fake_ttk.Checkbutton = (
            fake_ttk.Entry
        ) = fake_ttk.LabelFrame = fake_ttk.Notebook = fake_ttk.Combobox = (
            lambda *a, **kw: types.SimpleNamespace(
                pack=lambda *a, **kw: None,
                grid=lambda *a, **kw: None,
                add=lambda *a, **kw: None,
            )
        )
        fake_mb = types.ModuleType("tkinter.messagebox")
        fake_fd = types.ModuleType("tkinter.filedialog")
        fake_font = types.ModuleType("tkinter.font")

        sys.modules["tkinter"] = fake_tk
        sys.modules["tkinter.ttk"] = fake_ttk
        sys.modules["tkinter.messagebox"] = fake_mb
        sys.modules["tkinter.filedialog"] = fake_fd
        sys.modules["tkinter.font"] = fake_font
        return True


_ensure_mock_tkinter()
import tkinter as tk


def load_radar_viewer(module_name, *, without_scipy=False):
    """隔离加载示例脚本，并恢复临时替换的后端与可选依赖。"""
    fake_backend = types.ModuleType("matplotlib.backends.backend_tkagg")
    fake_backend.FigureCanvasTkAgg = object
    replacements = {"matplotlib.backends.backend_tkagg": fake_backend}
    if without_scipy:
        replacements["scipy.optimize"] = types.ModuleType("scipy.optimize")

    originals = {name: sys.modules.get(name, _MISSING) for name in replacements}
    sys.modules.update(replacements)
    try:
        script_path = EXAMPLE_DIR / "radarViewer.py"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in originals.items():
            if original is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


@contextmanager
def tkinter_root():
    """为 Tk 变量提供临时 Tcl 根，并在测试后恢复全局状态。"""
    if hasattr(tk, "_default_root"):
        original_root = tk._default_root
        try:
            tk._default_root = tk.Tcl()
        except Exception:
            tk._default_root = None
        try:
            yield
        finally:
            tk._default_root = original_root
    else:
        yield


class Value:
    """提供测试所需的最小 Tk 变量接口。"""

    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, val):
        self.value = val
