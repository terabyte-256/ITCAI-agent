from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_legacy_main_path = Path(__file__).resolve().parent.parent / "main.py"
_spec = spec_from_file_location("app._api_main", _legacy_main_path)
if _spec is None or _spec.loader is None:
    raise RuntimeError("Failed to load FastAPI app module.")
_module = module_from_spec(_spec)
_spec.loader.exec_module(_module)
app = _module.app
