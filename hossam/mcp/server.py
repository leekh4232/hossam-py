# -*- coding: utf-8 -*-
"""
Hossam MCP Server - VSCode/Copilot Compatible

표준 MCP(Model Context Protocol) 호환 서버입니다.
- StdIO (표준입출력) 기반 JSON 라인 프로토콜
- VSCode Copilot Chat, Cline, Cursor 등과 호환
- 모든 hossam 도구를 MCP tool로 등록

실행:
  python -m hossam.mcp.server
  또는
  hossam-mcp (CLI 엔트리포인트)
"""
import sys
import json
import logging
from typing import Any, Callable, Dict, Optional

# 로깅 설정 (stderr로 출력, stdout은 MCP 프로토콜 전용)
logging.basicConfig(
    level=logging.WARNING,
    format="[hossam-mcp] %(levelname)s: %(message)s",
    stream=sys.stderr,
)

try:
    import pandas as pd
    from pandas import DataFrame
except Exception:
    pd = None
    DataFrame = Any


class SimpleMCP:
    """경량 MCP 서버 구현"""

    def __init__(self, name: str = "hossam"):
        self.name = name
        self._tools: Dict[str, Dict[str, Any]] = {}

    def tool(self, name: str | None = None, description: str = ""):
        """tool 데코레이터: MCP에 함수를 도구로 등록"""
        def decorator(fn: Callable[..., Any]):
            tool_name = name or fn.__name__
            if not tool_name.startswith("hs_"):
                tool_name = f"hs_{tool_name}"

            # docstring 추출
            doc = (description or fn.__doc__ or "No description").split('\n')[0]

            self._tools[tool_name] = {
                "fn": fn,
                "description": doc,
                "doc": fn.__doc__ or "",
                "module": getattr(fn, "__module__", None),
            }
            return fn
        return decorator

    def list_tools(self) -> list[str]:
        """등록된 모든 tool 이름 반환"""
        return sorted(self._tools.keys())

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """특정 tool 정보 반환"""
        return self._tools.get(name)

    def call(self, tool: str, **kwargs) -> Any:
        """tool 호출 또는 코드 생성

        기본: 코드 생성(mode='code'). 실행이 필요하면 mode='run' 또는 run/execute/result 플래그를 사용.
        """
        if tool not in self._tools:
            raise KeyError(f"Unknown tool: {tool}")
        meta = self._tools[tool]
        mode = kwargs.pop("mode", None) or kwargs.pop("return", None)
        # 추가 플래그 해석
        run_flag = kwargs.pop("run", None) or kwargs.pop("execute", None) or kwargs.pop("result", None)
        code_flag = kwargs.pop("code", None) or kwargs.pop("code_only", None)

        # 기본값: 코드
        if mode is None:
            if run_flag:
                mode = "run"
            elif code_flag:
                mode = "code"
            else:
                mode = "code"

        # normalize
        mode = str(mode).lower() if mode else "code"

        if mode == "code":
            return _generate_code(tool, meta, kwargs)
        fn = meta["fn"]
        return fn(**kwargs)




def _df_from_any(obj: Any):
    """DataFrame 변환 헬퍼"""
    if pd is None:
        raise RuntimeError("pandas 필요: pip install pandas")

    if isinstance(obj, pd.DataFrame):
        return obj

    if isinstance(obj, str):
        s = obj.lower()
        if s.endswith(".csv"):
            return pd.read_csv(obj)
        if s.endswith(".xlsx"):
            return pd.read_excel(obj)
        raise ValueError("CSV/XLSX 경로만 지원")

    try:
        return pd.DataFrame(obj)
    except Exception:
        raise ValueError("DataFrame으로 변환 불가")


def _serialize(obj: Any) -> Any:
    """JSON 직렬화"""
    import numpy as np

    if pd is not None and isinstance(obj, pd.DataFrame):
        return {
            "index": obj.index.tolist(),
            "columns": obj.columns.tolist(),
            "data": obj.where(pd.notnull(obj), None).values.tolist(),
        }
    if pd is not None and isinstance(obj, pd.Series):
        return {
            "index": obj.index.tolist(),
            "name": obj.name,
            "data": obj.where(pd.notnull(obj), None).tolist(),
        }
    if isinstance(obj, (list, dict, str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    return str(obj)


def _py_repr(val: Any) -> str:
    import json as _json
    if isinstance(val, str):
        return repr(val)
    try:
        return _json.dumps(val, ensure_ascii=False)
    except Exception:
        return repr(val)


def _base_module_for_tool(tool: str, meta: Dict[str, Any]) -> tuple[str, str]:
    """tool명과 등록 메타에서 import 경로와 함수명 추정"""
    fn = meta.get("fn")
    mod = meta.get("module") or getattr(fn, "__module__", "")
    func = tool[3:] if tool.startswith("hs_") else tool
    if mod.startswith("hossam.mcp."):
        # mcp 래퍼에서 온 경우: 실제 모듈은 hossam.뒤꼬리
        tail = mod.split("hossam.mcp.", 1)[1]
        base_mod = f"hossam.{tail}"
    elif mod.startswith("hossam."):
        base_mod = mod
        func = getattr(fn, "__name__", func)
    else:
        # 폴백: 툴명으로 추정
        base_mod = f"hossam.{tool.split('_', 1)[1].split('.')[0]}"
    return base_mod, func


def _generate_code(tool: str, meta: Dict[str, Any], args: Dict[str, Any]) -> str:
    """요청된 도구 호출에 대한 파이썬 예제 코드를 생성"""
    base_mod, func = _base_module_for_tool(tool, meta)

    lines: list[str] = []
    needs_pd = False

    # df 전처리 코드 스니펫 구성
    call_args = []
    for k, v in list(args.items()):
        if k == "df":
            needs_pd = True
            if isinstance(v, str) and v.lower().endswith(".csv"):
                lines.append("import pandas as pd")
                lines.append(f"df = pd.read_csv({repr(v)})")
                call_args.append("df=df")
            elif isinstance(v, str) and v.lower().endswith(".xlsx"):
                lines.append("import pandas as pd")
                lines.append(f"df = pd.read_excel({repr(v)})")
                call_args.append("df=df")
            else:
                lines.append("import pandas as pd")
                lines.append(f"df = pd.DataFrame({_py_repr(v)})")
                call_args.append("df=df")
            args.pop(k, None)
        else:
            call_args.append(f"{k}={_py_repr(v)}")

    # import 라인
    lines.append(f"from {base_mod} import {func}")
    # 호출 라인
    args_str = ", ".join(call_args)
    call_line = f"result = {func}({args_str})" if call_args else f"result = {func}()"
    lines.append(call_line)
    lines.append("print(result)")

    return "\n".join(lines)


def _register_all(mcp: SimpleMCP):
    """모든 모듈 등록"""
    from . import hs_stats as mcp_stats
    mcp_stats.register(mcp)
    from . import hs_plot as mcp_plot
    mcp_plot.register(mcp)
    from . import hs_prep as mcp_prep
    mcp_prep.register(mcp)
    from . import hs_gis as mcp_gis
    mcp_gis.register(mcp)
    from . import hs_timeserise as mcp_ts
    mcp_ts.register(mcp)
    from . import hs_classroom as mcp_classroom
    mcp_classroom.register(mcp)
    from . import hs_util as mcp_util
    mcp_util.register(mcp)
    # data_loader 공개 함수도 노출
    try:
        from . import loader as mcp_loader
        mcp_loader.register(mcp)
    except Exception:
        # 선택 모듈 실패는 전체 서버 동작에 영향 없도록 무시
        pass


def _send_response(ok: bool, data: Any = None, error: str = None):
    """MCP 응답 전송 (stdout 사용)"""
    response = {"ok": ok}
    if ok and data is not None:
        response["result"] = _serialize(data)
    elif not ok and error:
        response["error"] = error

    print(json.dumps(response, ensure_ascii=False))
    sys.stdout.flush()


def run():
    """MCP 서버 메인 루프"""
    mcp = SimpleMCP(name="hossam")
    _register_all(mcp)

    # 초기화 응답
    _send_response(True, {
        "server": mcp.name,
        "tools": mcp.list_tools(),
        "version": "1.0"
    })

    # JSON 라인 처리 루프
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            tool = req.get("tool")
            args = req.get("args", {})

            if not tool:
                _send_response(False, error="'tool' 필수")
                continue

            # DataFrame 변환 (코드 생성 모드가 아닐 때만)
            mode = args.get("mode") or args.get("return")
            run_flag = args.get("run") or args.get("execute") or args.get("result")
            code_flag = args.get("code") or args.get("code_only")
            if mode is None:
                if run_flag:
                    mode = "run"
                elif code_flag:
                    mode = "code"
                else:
                    mode = "code"  # 기본 코드 생성
            mode = str(mode).lower()

            if mode != "code" and "df" in args:
                args["df"] = _df_from_any(args["df"])

            result = mcp.call(tool, **args)
            _send_response(True, result)

        except json.JSONDecodeError:
            _send_response(False, error="Invalid JSON")
        except KeyError as e:
            _send_response(False, error=f"Unknown tool: {str(e)}")
        except Exception as e:
            _send_response(False, error=str(e))


if __name__ == "__main__":
    run()
