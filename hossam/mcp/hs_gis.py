# -*- coding: utf-8 -*-
"""MCP wrappers for hossam.hs_gis"""
def register(mcp):
    from hossam.hs_gis import save_shape as _save_shape

    @mcp.tool("hs_gis_save_shape")
    def hs_save_shape(gdf_or_df, path: str, crs: str | None = None, lat_col: str = "latitude", lon_col: str = "longitude"):
        """GeoDataFrame 또는 DataFrame을 Shapefile/GeoPackage로 저장합니다."""
        _save_shape(gdf_or_df, path=path, crs=crs, lat_col=lat_col, lon_col=lon_col)
        return {"saved": True, "path": path}

    # 자동 등록: 언더바로 시작하지 않는 모든 공개 함수 노출(중복 방지)
    import inspect as _inspect
    import functools as _functools
    import hossam.hs_gis as _mod

    for _name, _fn in _inspect.getmembers(_mod, _inspect.isfunction):
        if _name.startswith("_"):
            continue
        _tool_name = f"hs_gis_{_name}"
        if mcp.get_tool_info(_tool_name):
            continue

        def _make_tool(fn=_fn, tool_name=_tool_name):
            @mcp.tool(tool_name, description=fn.__doc__)
            @_functools.wraps(fn)
            def _tool(**kwargs):
                return fn(**kwargs)

        _make_tool()
