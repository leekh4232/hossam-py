# -*- coding: utf-8 -*-
"""
Hossam MCP Layer

- 기존 공개 API는 유지하며, MCP 관련 코드는 이 패키지 내부에만 위치합니다.
- 서버는 명시적 엔트리포인트로만 실행됩니다 (`hossam-mcp`).
- 각 모듈별 `register(mcp)` 함수를 통해 MCP tool을 등록합니다.
"""

__all__ = [
    "server",
]
