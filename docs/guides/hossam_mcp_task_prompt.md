# hossam MCP 적용 작업 지시문

(이 문서는 AI 개발 에이전트에게 `hossam` 라이브러리를 MCP 기반으로 확장하도록 지시하기 위한 공식 작업 지시서이다.)

---

## 작업 목적 (Goal)

기존 Python 라이브러리 **`hossam`**을 유지보수 친화적으로 확장하여,
다음 두 가지 역할을 동시에 수행하도록 재구성하라.

1. 📦 기존 노트북 및 사용자 코드에서 import 가능한 라이브러리
2. 🤖 Model Context Protocol(MCP) 기반 서버로서 모든 모듈을 tool 형태로 제공

기존 사용자 코드, 노트북, 문서, API 경로는 **절대 깨지지 않아야 한다**.

---

## 필수 제약 조건 (Hard Constraints)

### 1. 공개 API 호환성 유지 (절대 조건)

다음 import 경로는 변경되거나 제거되어서는 안 된다.

```python
from hossam.hs_stats import ...
from hossam.hs_plot import ...
from hossam.hs_prep import ...
from hossam.hs_gis import ...
from hossam.hs_timeserise import ...
from hossam.hs_classroom import ...
from hossam.hs_util import ...
from hossam.data_loader import ...
```

- 기존 노트북과 문서는 수정 없이 그대로 동작해야 한다.
- 함수 이름, 시그니처 변경 금지.

---

### 2. MCP는 추가 레이어로만 구현

- MCP 관련 코드는 반드시 `hossam/mcp/` 하위에 위치시켜라.
- 라이브러리 import 시 MCP 서버가 자동 실행되면 안 된다.
- MCP 서버는 명시적인 엔트리포인트 실행으로만 동작해야 한다.

---

### 3. 디렉토리 구조 규칙

```text
hossam/
├── hs_stats.py
├── hs_plot.py
├── hs_gis.py
├── hs_prep.py
├── hs_timeserise.py
├── hs_classroom.py
├── hs_util.py
├── data_loader.py
│
└── mcp/
    ├── __init__.py
    ├── server.py
    ├── hs_stats.py
    ├── hs_plot.py
    ├── hs_gis.py
    ├── hs_prep.py
    ├── hs_timeserise.py
    ├── hs_classroom.py
    └── hs_util.py
```

---

## MCP Tool 설계 규칙

### 4. 모든 모듈 MCP화의 정의

- 모든 `hs_*` 모듈은 MCP에 등록되어야 한다.
- 그러나 모든 함수를 MCP tool로 노출하지는 않는다.

#### MCP tool로 노출할 대상
- 사용자 관점에서 의미 있는 작업 단위
- 분석, 통계, 전처리, 시각화, 교육 목적 함수

#### MCP tool로 노출하지 말 것
- 내부 헬퍼 함수
- `_` 로 시작하는 함수
- 저수준 유틸리티 함수
- 조합 없이는 의미 없는 함수

---

### 5. MCP Wrapper 작성 규칙

- MCP wrapper는 기존 공개 API만 사용해야 한다.
- 내부 구현을 직접 호출하지 말 것.
- Wrapper는 얇고 명확해야 한다.

```python
from hossam.hs_stats import summary_stats

def register(mcp):
    @mcp.tool()
    def hs_summary_stats(df):
        """데이터프레임 요약 통계"""
        return summary_stats(df)
```

---

### 6. MCP 서버 엔트리포인트

- `hossam/mcp/server.py`에서 모든 모듈을 등록하라.
- `register(mcp)` 패턴을 사용할 것.

---

### 7. 패키지 실행 방식

```toml
[project.scripts]
hossam-mcp = "hossam.mcp.server:run"
```

사용자는 다음 명령으로 MCP 서버를 실행한다.

```bash
hossam-mcp
```

---

## 품질 요구사항 (Quality Requirements)

- MCP tool 이름은 반드시 `hs_` prefix 유지
- Docstring은 AI가 이해 가능한 자연어로 작성
- 모듈당 MCP tool 2~5개 권장
- 중복 기능은 대표 tool로 통합

---

## 최종 산출물 (Deliverables)

1. MCP 레이어가 추가된 hossam 디렉토리 구조
2. 각 hs_* 모듈에 대응하는 MCP wrapper 파일
3. server.py 및 CLI 엔트리포인트
4. 기존 노트북·문서와의 호환성 유지 확인
