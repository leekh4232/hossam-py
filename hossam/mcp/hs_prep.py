# -*- coding: utf-8 -*-
"""MCP wrappers for hossam.hs_prep"""
from typing import List
from pandas import DataFrame

from hossam.hs_prep import (
    standard_scaler as _standard_scaler,
    minmax_scaler as _minmax_scaler,
    set_category as _set_category,
    get_dummies as _get_dummies,
    replace_outliner as _replace_outliner,
)


def register(mcp):
    # 자동 등록: 언더바로 시작하지 않는 모든 공개 함수 노출(중복 방지)
    import inspect as _inspect
    import functools as _functools
    import hossam.hs_prep as _mod

    for _name, _fn in _inspect.getmembers(_mod, _inspect.isfunction):
        if _name.startswith("_"):
            continue
        _tool_name = f"hs_prep_{_name}"
        if mcp.get_tool_info(_tool_name):
            continue

        def _make_tool(fn=_fn, tool_name=_tool_name):
            @mcp.tool(tool_name, description=fn.__doc__)
            @_functools.wraps(fn)
            def _tool(**kwargs):
                return fn(**kwargs)

        _make_tool()
