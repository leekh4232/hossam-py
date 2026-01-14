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
import os
import json
import logging
import inspect
import time
from typing import Any, Callable, Dict, Optional
import contextlib
import io
from typing import List, Tuple

# 로깅 설정 (stderr로 출력, stdout은 MCP 프로토콜 전용)
logging.basicConfig(
    level=logging.INFO,  # INFO로 변경하여 요청/응답 로그 표시
    format="[%(asctime)s] [hossam-mcp] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)

try:
    import pandas as pd
    from pandas import DataFrame
except Exception:
    pd = None
    DataFrame = Any


class HossamMCP:
    """경량 MCP 서버 구현 클래스.

    이 클래스는 MCP(모델 컨텍스트 프로토콜)에서 사용할 도구를 등록하고
    조회/호출하는 기능을 제공합니다.

    Attributes:
        name (str): 서버/네임스페이스 이름.
        _tools (Dict[str, Dict[str, Any]]): 등록된 도구 메타데이터.
    """

    def __init__(self, name: str = "hossam"):
        """초기화.

        Args:
            name (str): MCP 서버 이름.
        """
        self.name = name
        self._tools: Dict[str, Dict[str, Any]] = {}

    def tool(self, name: Optional[str] = None, description: str = ""):
        """도구 등록용 데코레이터.

        MCP에서 호출 가능한 함수를 등록합니다. 등록 시 도구명은 `hs_` 접두사로
        정규화되며, 시그니처와 파라미터 메타데이터를 함께 저장합니다.

        Args:
            name (Optional[str]): 명시적 도구명. 미지정 시 함수명을 사용.
            description (str): 도구 설명. 미지정 시 함수 docstring 1행.

        Returns:
            Callable: 원본 함수 데코레이터.
        """

        def decorator(fn: Callable[..., Any]):
            tool_name = name or fn.__name__
            if not tool_name.startswith("hs_"):
                tool_name = f"hs_{tool_name}"

            sig = inspect.signature(fn)
            doc = (description or fn.__doc__ or "No description").split("\n")[0]

            self._tools[tool_name] = {
                "fn": fn,
                "description": doc,
                "doc": fn.__doc__ or "",
                "module": getattr(fn, "__module__", None),
                "signature": str(sig),
                "params": {
                    pname: {
                        "kind": str(param.kind),
                        "required": param.default is inspect._empty,
                    }
                    for pname, param in sig.parameters.items()
                },
                "returns": "python_code",
                "mode": "codegen_only",
            }
            return fn

        return decorator

    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """등록된 도구 명세를 반환합니다.

        Returns:
            Dict[str, Dict[str, Any]]: 도구 이름별 설명/시그니처/파라미터/리턴 타입.
        """
        return {
            name: {
                "description": spec["description"],
                "signature": spec["signature"],
                "params": spec["params"],
                "returns": spec["returns"],
                "mode": spec.get("mode", "codegen_only"),
            }
            for name, spec in self._tools.items()
        }

    def get_tool_info(self, name: str) -> Optional[Dict[str, Any]]:
        """특정 도구의 상세 정보를 조회합니다.

        Args:
            name (str): 도구명.

        Returns:
            Optional[Dict[str, Any]]: 도구 메타데이터 또는 None.
        """
        return self._tools.get(name)

    def call(self, tool: str, **kwargs) -> Any:
        """도구 호출 또는 코드 생성.

        기본 동작은 코드 생성(`mode='code'`). 실행이 필요하면 `mode='run'`을 지정하거나
        `run/execute/result` 플래그를 사용합니다.

        Args:
            tool (str): 도구명.
            **kwargs: 도구 인자 및 실행 모드 지정.

        Returns:
            Any: 코드 문자열 또는 실제 실행 결과.
        """
        if tool not in self._tools:
            raise KeyError(f"Unknown tool: {tool}")

        meta = self._tools[tool]
        mode = kwargs.pop("mode", None) or kwargs.pop("return", None)

        # 실행/코드 플래그 해석
        run_flag = kwargs.pop("run", None) or kwargs.pop("execute", None) or kwargs.pop("result", None)
        code_flag = kwargs.pop("code", None) or kwargs.pop("code_only", None)

        if mode is None:
            mode = "run" if run_flag else "code"

        mode = str(mode).lower() if mode else "code"

        if mode == "code":
            return _generate_code(tool, meta, kwargs)

        fn = meta["fn"]
        return fn(**kwargs)




def _df_from_any(obj: Any) -> Any:
    """입력 객체를 `pandas.DataFrame`으로 변환합니다.

    CSV/XLSX 경로 문자열과 시퀀스/매핑 객체를 지원합니다.

    Args:
        obj (Any): 입력 데이터 또는 파일 경로.

    Returns:
        DataFrame: 변환된 데이터프레임.

    Raises:
        RuntimeError: pandas 미설치 시.
        ValueError: 지원하지 않는 경로 또는 변환 실패.
    """
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
    """MCP 응답을 위한 직렬화 헬퍼.

    pandas 객체와 numpy 배열을 JSON 호환 형태로 변환합니다.

    Args:
        obj (Any): 직렬화 대상.

    Returns:
        Any: JSON 호환 객체.
    """
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
    """파이썬 리터럴/JSON 문자열로 안전하게 변환합니다.

    Args:
        val (Any): 값.

    Returns:
        str: 코드 내에서 사용 가능한 표현 문자열.
    """
    import json as _json
    if isinstance(val, str):
        return repr(val)
    try:
        return _json.dumps(val, ensure_ascii=False)
    except Exception:
        return repr(val)


def _log_request(request_id: Any, method: Optional[str], params: Dict[str, Any]) -> None:
    """요청 로깅 헬퍼.

    Args:
        request_id (Any): 요청 ID.
        method (Optional[str]): 메서드 이름.
        params (Dict[str, Any]): 파라미터.
    """
    logging.info("=" * 80)
    logging.info(f"📥 Request [id: {request_id}]")
    logging.info(f"   Method: {method}")
    if params:
        logging.info(f"   Params: {str(params)[:200]}...")
    logging.info("=" * 80)


def _build_tools_list(mcp: HossamMCP) -> List[Dict[str, Any]]:
    """도구 목록을 MCP 형식으로 구성합니다.

    Args:
        mcp (HossamMCP): 서버 인스턴스.

    Returns:
        List[Dict[str, Any]]: MCP tools/list 응답용 배열.
    """
    return [
        {
            "name": name,
            "description": spec["description"],
            "inputSchema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
        for name, spec in mcp.list_tools().items()
    ]


# MCP 상수: 프로토콜/서버 정보
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "hossam-mcp"
SERVER_VERSION = "1.0.0"


def _base_module_for_tool(tool: str, meta: Dict[str, Any]) -> Tuple[str, str]:
    """도구명/메타데이터를 기반으로 import 경로와 함수명을 추정합니다.

    규칙: `hs_모듈_함수` 형태의 도구명 사용을 권장합니다.

    Args:
        tool (str): 도구명.
        meta (Dict[str, Any]): 등록 메타데이터.

    Returns:
        Tuple[str, str]: 베이스 모듈 경로, 함수명.
    """
    fn = meta.get("fn")
    mod = meta.get("module") or getattr(fn, "__module__", "")

    # 도구명에서 모듈명과 함수명 파싱
    # 예: hs_util_load_data -> (hs_util, load_data)
    # 예: hs_plot_histplot -> (hs_plot, histplot)
    if tool.startswith("hs_"):
        parts = tool.split("_", 2)  # ['hs', 'util', 'load_data']
        if len(parts) >= 3:
            module_name = f"{parts[0]}_{parts[1]}"  # hs_util
            func = parts[2]  # load_data
        else:
            # 폴백 (구버전 호환)
            module_name = "hs_util"
            func = tool[3:]
    else:
        module_name = "hs_util"
        func = tool

    if mod.startswith("hossam.mcp."):
        # mcp 래퍼에서 온 경우: 실제 모듈은 hossam.뒤꼬리
        tail = mod.split("hossam.mcp.", 1)[1]
        base_mod = f"hossam.{tail}"
    elif mod.startswith("hossam."):
        base_mod = mod
        func = getattr(fn, "__name__", func)
    else:
        # 폴백: 툴명의 모듈 부분 사용
        base_mod = f"hossam.{module_name[3:]}"  # hs_util -> hossam.util (잘못된 경우)
        # 수정: data_loader는 data_loader로
        if module_name == "hs_data":
            base_mod = "hossam.data_loader"
        elif module_name.startswith("hs_"):
            base_mod = f"hossam.{module_name}"

    return base_mod, func


def _generate_code(tool: str, meta: Dict[str, Any], args: Dict[str, Any]) -> str:
    """도구 호출 예제 파이썬 코드를 생성합니다.

    DataFrame 인자를 자동으로 적절한 로드 코드로 변환합니다.

    Args:
        tool (str): 도구명.
        meta (Dict[str, Any]): 도구 메타데이터.
        args (Dict[str, Any]): 호출 인자.

    Returns:
        str: 실행 가능한 예제 코드 문자열.
    """
    base_mod, func = _base_module_for_tool(tool, meta)

    lines: List[str] = []

    # df 전처리 코드 스니펫 구성
    call_args = []
    for k, v in list(args.items()):
        if k == "df":
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
    # 도구 import 라인
    lines.append(f"from {base_mod} import {func}")
    # 호출 라인
    args_str = ", ".join(call_args)
    call_line = f"result = {func}({args_str})" if call_args else f"result = {func}()"
    lines.append(call_line)
    lines.append("print(result)")

    return "\n".join(lines)


@contextlib.contextmanager
def _suppress_import_stdout():
    """모듈 임포트 중 stdout 배너 출력 억제.

    MCP 표준 출력 채널을 보호하기 위해 임포트 중 발생하는 배너/프린트를 차단합니다.

    Yields:
        None: 컨텍스트 종료 시 stdout 복구.
    """
    original = sys.stdout
    try:
        sys.stdout = io.StringIO()  # 배너를 버립니다
        yield
    finally:
        sys.stdout = original


def _register_all(mcp: HossamMCP):
    """모든 hossam MCP 도구를 서버에 등록합니다.

    Args:
        mcp (HossamMCP): MCP 서버 인스턴스.
    """
    with _suppress_import_stdout():
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



def _write_message(obj: Dict[str, Any]):
    """JSON-RPC 2.0 메시지를 Content-Length 헤더로 프레이밍하여 전송합니다.

    Args:
        obj (Dict[str, Any]): 전송할 JSON 객체.
    """
    payload = json.dumps(obj, ensure_ascii=False)
    data = payload.encode("utf-8")
    # 표준 MCP/LSP 스타일 헤더 프레이밍
    sys.stdout.write(f"Content-Length: {len(data)}\r\n\r\n")
    sys.stdout.write(payload)
    sys.stdout.flush()


def _send_response(response: Dict[str, Any]):
    """JSON-RPC 2.0 형식의 MCP 응답을 전송합니다.

    Args:
        response (Dict[str, Any]): 응답 객체(`jsonrpc`/`id`/`result` 또는 `error`).
    """
    # 응답 로깅 (stderr)
    if "result" in response:
        result_preview = str(response.get("result", ""))[:80]
        logging.info(f"📤 Response [id: {response.get('id')}] - Result: {result_preview}...")
    elif "error" in response:
        logging.error(f"📤 Response [id: {response.get('id')}] - Error: {response['error']}")

    _write_message(response)


def _send_error(request_id: Any, code: int, message: str):
    """JSON-RPC 2.0 에러 응답을 전송합니다.

    Args:
        request_id (Any): 요청 식별자.
        code (int): 에러 코드.
        message (str): 에러 메시지.
    """
    _send_response({
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": code,
            "message": message
        }
    })


def _read_json_rpc_message() -> Optional[Dict[str, Any]]:
    """STDIO에서 JSON-RPC 2.0 메시지를 읽습니다.

    MCP/LSP 호환을 위해 Content-Length 헤더 프레이밍을 우선 사용하고
    필요 시 라인 기반 폴백을 지원합니다.

    Returns:
        Optional[Dict[str, Any]]: 파싱된 요청 객체 또는 None.
    """
    buf = sys.stdin.buffer

    # 첫 라인 확인: 바로 JSON이면 폴백 처리
    first = buf.readline()
    if not first:
        return None
    first_text = first.decode("utf-8", errors="ignore").strip()
    if first_text.startswith("{"):
        try:
            return json.loads(first_text)
        except Exception:
            return None

    # 헤더 파싱
    headers: Dict[str, str] = {}
    line = first_text
    while True:
        if not line:
            # 빈 라인: 헤더 종료
            break
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
        # 다음 라인
        nxt = buf.readline()
        if not nxt:
            return None
        line = nxt.decode("utf-8", errors="ignore").strip()
        if line == "":
            break

    content_length = int(headers.get("content-length", "0") or 0)
    if content_length <= 0:
        # 라인 기반 폴백: 다음 라인을 JSON으로 시도
        nxt = buf.readline()
        if not nxt:
            return None
        text = nxt.decode("utf-8", errors="ignore").strip()
        try:
            return json.loads(text)
        except Exception:
            return None

    body = buf.read(content_length)
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def run():
    """MCP 서버 메인 루프.

    JSON-RPC 2.0 프레이밍을 사용하여 VS Code/Copilot 등 MCP 클라이언트와 통신합니다.
    """
    mcp = HossamMCP(name="hossam")
    _register_all(mcp)

    # DEV 모드 토글: 환경변수 `HOSSAM_MCP_DEV`가 "1"이면 DEBUG 레벨
    dev_mode = os.getenv("HOSSAM_MCP_DEV", "0") == "1"
    if dev_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.info("🛠 DEV 모드 활성화 (DEBUG 로그)")

    logging.info(f"🚀 Hossam MCP 서버 시작 (도구 수: {len(mcp.list_tools())})")

    try:
        # JSON-RPC 2.0 메시지 처리 루프
        while True:
            # 메시지 수신
            req = _read_json_rpc_message()
            if req is None:
                break

            try:
                request_id = req.get("id")
                method = req.get("method")
                params = req.get("params", {})

                # 요청 로깅
                _log_request(request_id, method, params)

                # MCP 프로토콜 핸들링
                if method == "initialize":
                    # 초기화 요청 처리
                    _send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "protocolVersion": PROTOCOL_VERSION,
                            "capabilities": {
                                "tools": {},
                            },
                            "serverInfo": {
                                "name": SERVER_NAME,
                                "version": SERVER_VERSION,
                            }
                        }
                    })
                    logging.info("✅ Initialize 응답 전송")

                elif method == "notifications/initialized":
                    # 초기화 완료 알림 (응답 불필요)
                    logging.info("✅ Client initialized")

                elif method == "ping":
                    # 핑 응답 (일부 클라이언트에서 사용)
                    _send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {}
                    })

                elif method == "tools/list":
                    # 도구 목록 요청
                    tools_list = _build_tools_list(mcp)
                    _send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "tools": tools_list
                        }
                    })

                elif method == "tools/call":
                    # 도구 호출
                    tool_name = params.get("name")
                    tool_args = params.get("arguments", {})

                    if not tool_name:
                        _send_error(request_id, -32602, "도구 이름 필요")
                        continue

                    # DataFrame 변환
                    mode = tool_args.get("mode") or tool_args.get("return") or "code"
                    if mode != "code" and "df" in tool_args:
                        tool_args["df"] = _df_from_any(tool_args["df"])

                    result = mcp.call(tool_name, **tool_args)

                    _send_response({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": str(result)
                                }
                            ]
                        }
                    })

                else:
                    _send_error(request_id, -32601, f"Unknown method: {method}")

            except json.JSONDecodeError as e:
                logging.error(f"❌ Invalid JSON: {str(e)}")
                _send_error(None, -32700, "Parse error")
            except Exception as e:
                logging.error(f"❌ Exception: {str(e)}")
                _send_error(request_id, -32603, str(e))

    except KeyboardInterrupt:
        logging.info("👋 서버 종료 (KeyboardInterrupt)")
    except Exception as e:
        logging.error(f"❌ 서버 오류: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    run()
