# -*- coding: utf-8 -*-
"""MCP wrappers for hossam.hs_plot

시각화 함수는 파일 저장 경로(`save_path`)를 활용하는 사용을 권장합니다.
"""
from typing import Any
from pandas import DataFrame

from hossam.hs_plot import (
    lineplot as _lineplot,
    boxplot as _boxplot,
    kdeplot as _kdeplot,
)


def register(mcp):
    # 자동 등록: 언더바로 시작하지 않는 모든 공개 함수 노출(중복 방지)
    import inspect as _inspect
    import functools as _functools
    import hossam.hs_plot as _mod

    for _name, _fn in _inspect.getmembers(_mod, _inspect.isfunction):
        if _name.startswith("_"):
            continue
        _tool_name = f"hs_plot_{_name}"
        if mcp.get_tool_info(_tool_name):
            continue

        def _make_tool(fn=_fn, tool_name=_tool_name):
            @mcp.tool(tool_name, description=fn.__doc__)
            @_functools.wraps(fn)
            def _tool(**kwargs):
                return fn(**kwargs)

        _make_tool()
