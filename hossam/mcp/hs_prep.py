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
    @mcp.tool("hs_standard_scaler")
    def hs_standard_scaler(data: DataFrame | list | dict, yname: str | None = None, save_path: str | None = None, load_path: str | None = None):
        """연속형 변수에 대해 Z-Score(Standard) 스케일링을 수행합니다."""
        return _standard_scaler(data, yname=yname, save_path=save_path, load_path=load_path)

    @mcp.tool("hs_minmax_scaler")
    def hs_minmax_scaler(data: DataFrame | list | dict, yname: str | None = None, save_path: str | None = None, load_path: str | None = None):
        """연속형 변수를 0~1로 정규화(MinMax Scaling)합니다."""
        return _minmax_scaler(data, yname=yname, save_path=save_path, load_path=load_path)

    @mcp.tool("hs_set_category")
    def hs_set_category(df: DataFrame, fields: List[str]):
        """지정된 컬럼을 pandas Categorical로 설정합니다."""
        return _set_category(df, *fields)

    @mcp.tool("hs_get_dummies")
    def hs_get_dummies(df: DataFrame, fields: List[str] | None = None, drop_first: bool = True, dtype: str = "int"):
        """명목형 변수를 더미 변수(원-핫)로 변환합니다."""
        fields = fields or []
        return _get_dummies(df, *fields, drop_first=drop_first, dtype=dtype)

    @mcp.tool("hs_replace_outliner")
    def hs_replace_outliner(df: DataFrame, method: str = "nan", fields: List[str] | None = None):
        """IQR 기준 이상치를 지정된 방식으로 대체합니다."""
        fields = fields or []
        return _replace_outliner(df, method=method, *fields)

    # 자동 등록: 언더바로 시작하지 않는 모든 공개 함수 노출(중복 방지)
    import inspect as _inspect
    import hossam.hs_prep as _mod

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
