# -*- coding: utf-8 -*-
"""MCP auto-wrappers for hossam.data_loader

공개 함수(언더바 미사용) 전체를 MCP tool로 자동 등록합니다.
단, load_data는 hs_util.load_data로 대체되므로 제외합니다.
"""
import inspect as _inspect
import hossam.data_loader as _mod


def register(mcp):
    # load_data는 제외 (hs_util.load_data를 사용)
    excluded_functions = ["load_data"]

    for _name, _fn in _inspect.getmembers(_mod, _inspect.isfunction):
        if _name.startswith("_"):
            continue
        if _name in excluded_functions:
            continue
        _tool_name = f"hs_data_{_name}"
        if mcp.get_tool_info(_tool_name):
            continue

        def _make_tool(fn=_fn, name=_name):
            @mcp.tool(name)
            def _tool(**kwargs):
                return fn(**kwargs)

        _make_tool()
