"""Render docs/architecture.png from the DOT source used by the dashboard.

    python scripts/make_architecture_diagram.py      (needs Graphviz `dot` on PATH)
"""
import subprocess

import _bootstrap  # noqa: F401

from dashboard.p_architecture import DOT

with open("docs/architecture.dot", "w") as fh:
    fh.write(DOT)
subprocess.run(["dot", "-Tpng", "-Gdpi=150", "docs/architecture.dot", "-o", "docs/architecture.png"], check=True)
print("wrote docs/architecture.png")
