"""VisionServe checkpoint converter (runs in the converter image only; the server stays Python-free).

CLI: `visionserve-convert <format> <source> --name NAME [...]`. Python: `export(...)` (api.py),
returning a `Report` (report.py). Heavy frameworks are imported lazily.
"""


def export(*args, **kwargs):
    from .api import export as _export
    return _export(*args, **kwargs)


export.__doc__ = "See visionserve.convert.api.export."


def __getattr__(name):  # noqa: N807 — PEP 562
    if name in ("Report", "TierResult"):
        from . import report
        return getattr(report, name)
    if name == "ConvertError":
        from .common import ConvertError
        return ConvertError
    raise AttributeError(name)
