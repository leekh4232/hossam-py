# -*- coding: utf-8 -*-
"""MCP auto-wrappers for hossam.data_loader

공개 함수(언더바 미사용) 전체를 MCP tool로 자동 등록합니다.
"""
import inspect as _inspect
import hossam.data_loader as _mod


def register(mcp):
    for _name, _fn in _inspect.getmembers(_mod, _inspect.isfunction):
        if _name.startswith("_"):
            continue
        _tool_name = f"hs_{_name}"
        if mcp.get_tool_info(_tool_name):
            continue

        def _make_tool(fn=_fn, name=_name):
            @mcp.tool(name)
            def _tool(**kwargs):
                return fn(**kwargs)

        _make_tool()
