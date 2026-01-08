"""
Generate a comprehensive API Index page listing all public functions
across hossam modules with links to their detailed docs.

Writes to: api/index.md
Requires: mkdocs-gen-files
"""
from __future__ import annotations

import inspect
import importlib
import sys
import os
from typing import List, Tuple

# Add parent directories to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import mkdocs_gen_files

MODULES: List[str] = [
    "hossam.analysis",
    "hossam.data_loader",
    "hossam.gis",
    "hossam.plot",
    "hossam.prep",
    "hossam.util",
]

rows: List[Tuple[str, str, str, str]] = []

for mod_name in MODULES:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        print(f"Failed to import {mod_name}: {e}", file=sys.stderr)
        continue
    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_"):
            continue
        # Only include functions defined in this module, not imported ones
        if obj.__module__ != mod_name:
            continue
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "(…)"
        doc = inspect.getdoc(obj) or ""
        summary = doc.splitlines()[0] if doc else ""
        module_short = mod_name.split(".")[-1]
        rows.append((module_short, name, sig, summary))

rows.sort(key=lambda r: (r[0], r[1]))

md_lines: List[str] = []
md_lines.append("---")
md_lines.append("title: API Index")
md_lines.append("---")
md_lines.append("")
md_lines.append("# Full API Index\n")
md_lines.append("모듈별 공개 함수 전체 목록과 간단 설명입니다. 각 항목은 상세 API 페이지로 연결됩니다.\n")
md_lines.append("")
md_lines.append("| Module | Function | Signature | Summary |")
md_lines.append("|--------|----------|-----------|---------|")

for module_short, name, sig, summary in rows:
    link = f"[{name}](../api/{module_short}.md#hossam.{module_short}.{name})"
    # Escape pipe characters in signature and summary to avoid breaking markdown table
    sig_escaped = sig.replace("|", "\\|")
    summary_escaped = summary.replace("|", "\\|")
    md_lines.append(f"| {module_short} | {link} | `{sig_escaped}` | {summary_escaped} |")

with mkdocs_gen_files.open("api/index.md", "w") as f:
    f.write("\n".join(md_lines))
