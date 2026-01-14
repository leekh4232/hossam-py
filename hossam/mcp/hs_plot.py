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
    # @mcp.tool("hs_lineplot")
    # def hs_lineplot(df: DataFrame, xname: str | None = None, yname: str | None = None, hue: str | None = None, save_path: str | None = None, **params: Any):
    #     """선 그래프를 그립니다. 원격 환경에서는 `save_path`로 저장하여 활용하세요."""
    #     _lineplot(df=df, xname=xname, yname=yname, hue=hue, save_path=save_path, **params)
    #     return {"saved": bool(save_path), "path": save_path}

    # @mcp.tool("hs_boxplot")
    # def hs_boxplot(df: DataFrame, xname: str | None = None, yname: str | None = None, orient: str = "v", save_path: str | None = None, **params: Any):
    #     """상자그림(boxplot)을 그립니다. `save_path` 지정 시 파일로 저장합니다."""
    #     _boxplot(df=df, xname=xname, yname=yname, orient=orient, save_path=save_path, **params)
    #     return {"saved": bool(save_path), "path": save_path}

    # @mcp.tool("hs_kdeplot")
    # def hs_kdeplot(df: DataFrame, xname: str | None = None, yname: str | None = None, hue: str | None = None, fill: bool = False, quartile_split: bool = False, save_path: str | None = None, **params: Any):
    #     """KDE(커널 밀도) 그래프를 그립니다. 1D KDE는 `quartile_split`로 사분위별 분할 가능."""
    #     _kdeplot(df=df, xname=xname, yname=yname, hue=hue, fill=fill, quartile_split=quartile_split, save_path=save_path, **params)
    #     return {"saved": bool(save_path), "path": save_path}

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
