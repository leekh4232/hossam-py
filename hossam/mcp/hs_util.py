# -*- coding: utf-8 -*-
"""MCP wrappers for hossam.hs_util"""
def register(mcp):
    # @mcp.tool("hs_pretty_table")
    # def hs_pretty_table(df, tablefmt: str = "simple", headers: str | list = "keys"):
    #     """DataFrame을 단순 표 형태로 출력합니다. 원격 환경에서는 표 문자열을 반환합니다."""
    #     # pretty_table은 print만 수행하므로, 문자열을 반환하도록 보정
    #     from tabulate import tabulate
    #     s = tabulate(df, headers=headers, tablefmt=tablefmt, showindex=True, numalign="right")
    #     return s

    # 자동 등록: 언더바로 시작하지 않는 모든 공개 함수 노출(중복 방지)
    import inspect as _inspect
    import functools as _functools
    import hossam.hs_util as _mod

    for _name, _fn in _inspect.getmembers(_mod, _inspect.isfunction):
        if _name.startswith("_"):
            continue
        _tool_name = f"hs_util_{_name}"
        if mcp.get_tool_info(_tool_name):
            continue

        def _make_tool(fn=_fn, tool_name=_tool_name):
            @mcp.tool(tool_name, description=fn.__doc__)
            @_functools.wraps(fn)
            def _tool(**kwargs):
                return fn(**kwargs)

        _make_tool()
