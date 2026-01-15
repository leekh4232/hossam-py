# -*- coding: utf-8 -*-
"""
Hossam MCP REST API 서버 (Flask 기반)
- /api/mcp 엔드포인트로 JSON-RPC 2.0 요청을 처리
- 기존 HossamMCP와 동일한 도구 등록 및 호출 방식
"""
from flask import Flask, request, jsonify
import logging
import sys
import os
import json
from hossam.mcp.server import HossamMCP, _register_all

app = Flask(__name__)

# MCP 서버 인스턴스 생성 및 도구 등록
mcp = HossamMCP(name="hossam")
_register_all(mcp)

@app.route("/api/mcp", methods=["POST"])
def mcp_api():
    try:
        req = request.get_json(force=True)
        request_id = req.get("id")
        method = req.get("method")
        params = req.get("params", {})

        if method == "initialize":
            return jsonify({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "1.0.0",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "hossam-mcp", "version": "1.0.0"}
                }
            })
        elif method == "tools/list":
            tools_list = [k for k in mcp.list_tools()]
            return jsonify({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"tools": tools_list}
            })
        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            if not tool_name:
                return jsonify({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32602, "message": "도구 이름 필요"}})
            result = mcp.call(tool_name, **tool_args)
            return jsonify({
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {"content": [{"type": "text", "text": str(result)}]}
            })
        else:
            return jsonify({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}})
    except Exception as e:
        return jsonify({"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": str(e)}})

def main():
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)

if __name__ == "__main__":
    main()
