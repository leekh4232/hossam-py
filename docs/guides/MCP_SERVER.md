# Hossam MCP 서버 가이드

> MCP (Model Context Protocol) 기반 서버 인터페이스를 통한 hossam 도구 사용 방법입니다.

---

## 개요

Hossam은 두 가지 방식으로 사용할 수 있습니다:

| 방식 | 사용 방법 | 사용자 |
|------|---------|--------|
| **라이브러리** | `from hossam import *` | 개별 개발자, Jupyter 노트북 |
| **MCP 서버** | `hossam-mcp` | VSCode Copilot, Cline, AI 에이전트 |

이 문서는 **MCP 서버 방식**에 대해 설명합니다.

---

## MCP 서버 시작

### 1. 설치

```bash
pip install hossam
```

### 2. 서버 실행

```bash
# 명령어로 시작
hossam-mcp

# 또는 Python 모듈로 시작
python -m hossam.mcp.server
```

### 3. 출력 확인

```
🚀 Hossam MCP 서버 시작 (도구 수: 120+)
════════════════════════════════════════════════════════════════
📚 Hossam MCP 도구 지식 베이스 로드됨
════════════════════════════════════════════════════════════════
```

---

## 통신 프로토콜

### 요청 형식

```json
{
  "tool": "도구이름",
  "args": {
    "파라미터1": "값1",
    "파라미터2": "값2"
  }
}
```

### 응답 형식

**성공:**
```json
{
  "ok": true,
  "result": <결과값 또는 코드>
}
```

**실패:**
```json
{
  "ok": false,
  "error": "에러 메시지"
}
```

---

## 작동 모드

### 코드 생성 모드 (기본)

실행하지 않고 Python 코드만 반환합니다.

**요청:**
```json
{
  "tool": "hs_stats_missing_values",
  "args": {
    "df": "./data.csv",
    "fields": ["age", "income"]
  }
}
```

**응답:**
```json
{
  "ok": true,
  "result": "import pandas as pd\ndf = pd.read_csv('./data.csv')\nfrom hossam.hs_stats import missing_values\nresult = missing_values(df=df, fields=['age', 'income'])\nprint(result)"
}
```

### 실행 모드

실제로 함수를 실행하고 결과를 반환합니다.

**요청:**
```json
{
  "tool": "hs_util_make_normalize_values",
  "args": {
    "mean": 0,
    "std": 1,
    "size": 5,
    "mode": "run"
  }
}
```

**응답:**
```json
{
  "ok": true,
  "result": [-0.19, 1.12, 0.81, -1.53, 0.45]
}
```

---

## 도구 이름 규칙

모든 도구는 `hs_<모듈>_<함수>` 형식입니다:

| 모듈 | 예시 도구 |
|------|---------|
| hs_stats | `hs_stats_missing_values`, `hs_stats_describe` |
| hs_plot | `hs_plot_distribution_plot`, `hs_plot_scatterplot` |
| hs_prep | `hs_prep_standard_scaler`, `hs_prep_get_dummies` |
| hs_util | `hs_util_load_data`, `hs_util_load_info` |
| hs_gis | `hs_gis_geocode`, `hs_gis_load_shape` |
| hs_timeserise | `hs_timeserise_diff`, `hs_timeserise_rolling` |
| hs_classroom | `hs_classroom_cluster_students` |

---

## 데이터 입력 형식

### CSV/Excel 파일 경로

```json
{
  "tool": "hs_stats_missing_values",
  "args": {
    "df": "./data/sales.csv"
  }
}
```

### 딕셔너리 배열 (권장)

```json
{
  "tool": "hs_stats_missing_values",
  "args": {
    "df": [
      {"name": "Alice", "age": 25, "score": 85},
      {"name": "Bob", "age": 30, "score": 92}
    ]
  }
}
```

### JSON 구조

```json
{
  "tool": "hs_stats_missing_values",
  "args": {
    "df": {
      "index": [0, 1, 2],
      "columns": ["x", "y"],
      "data": [[1, 2], [3, 4], [5, 6]]
    }
  }
}
```

---

## 주요 도구 사용 예시

### 예시 1: 결측치 분석

```json
{
  "tool": "hs_stats_missing_values",
  "args": {
    "df": "./insurance.csv",
    "fields": ["age", "bmi", "charges"],
    "mode": "run"
  }
}
```

### 예시 2: 데이터 로드 (명목형 지정)

```json
{
  "tool": "hs_util_load_data",
  "args": {
    "key": "insurance",
    "categories": ["sex", "smoker", "region"],
    "info": true,
    "mode": "run"
  }
}
```

### 예시 3: 기술통계

```json
{
  "tool": "hs_stats_describe",
  "args": {
    "df": [
      {"age": 25, "bmi": 28.5, "charges": 12500},
      {"age": 30, "bmi": 32.1, "charges": 15200}
    ],
    "fields": ["age", "bmi", "charges"],
    "mode": "run"
  }
}
```

### 예시 4: 범주형 기술통계

```json
{
  "tool": "hs_stats_category_describe",
  "args": {
    "df": "./insurance.csv",
    "mode": "run"
  }
}
```

### 예시 5: 시각화 (코드만 반환)

```json
{
  "tool": "hs_plot_scatterplot",
  "args": {
    "df": "./insurance.csv",
    "xname": "bmi",
    "yname": "charges",
    "hue": "smoker"
  }
}
```

### 예시 6: 전처리

```json
{
  "tool": "hs_prep_standard_scaler",
  "args": {
    "df": "./insurance.csv",
    "features": ["age", "bmi", "charges"],
    "mode": "run"
  }
}
```

---

## MCP 지원 도구

### 도구 목록 확인

```json
{
  "tool": "hs_mcp_list_tools",
  "args": {}
}
```

### 사용 가이드

```json
{
  "tool": "hs_mcp_usage_guide",
  "args": {}
}
```

### 도구 지식 (전체 목록 + 설명)

```json
{
  "tool": "hs_mcp_tool_knowledge",
  "args": {}
}
```

### 특정 주제 도움말

```json
{
  "tool": "hs_mcp_help",
  "args": {
    "topic": "load_data"
  }
}
```

---

## 터미널 테스트

### 1. 도구 목록 확인

```bash
echo '{"tool":"hs_mcp_list_tools","args":{}}' | hossam-mcp
```

### 2. 사용 가이드 확인

```bash
echo '{"tool":"hs_mcp_usage_guide","args":{}}' | hossam-mcp
```

### 3. 데이터 생성 및 실행

```bash
echo '{"tool":"hs_util_make_normalize_values","args":{"mean":0,"std":1,"size":5,"mode":"run"}}' | hossam-mcp
```

### 4. 샘플 데이터셋 목록 확인

```bash
echo '{"tool":"hs_util_load_info","args":{"mode":"run"}}' | hossam-mcp
```

---

## 규칙 및 제약

### 명목형 변수 지정 (필수!)

데이터 로드 시 명목형 변수는 반드시 `categories` 파라미터로 지정:

```json
{
  "tool": "hs_util_load_data",
  "args": {
    "key": "insurance",
    "categories": ["sex", "smoker", "region"]  // 필수!
  }
}
```

### 파라미터 명명 규칙

- `df`: DataFrame 또는 파일 경로
- `fields`, `xname`, `yname`: 컬럼명
- `hue`: 보조 범주형 컬럼
- `features`: 전처리할 컬럼 리스트
- `method`: 처리 방식 선택

---

## 클라이언트별 통합

### VSCode Copilot Chat

`.vscode/settings.json`:
```json
{
  "mcp.servers": {
    "hossam": {
      "command": "hossam-mcp"
    }
  }
}
```

### Cline

VSCode에서 Cline 확장 설치 후, MCP 서버 자동 인식

### 커스텀 클라이언트

stdin/stdout을 통해 JSON 라인 프로토콜로 통신하는 클라이언트면 됩니다.

---

## 환경변수

서버 실행 시 다음 환경변수 지원:

```bash
# Python 경로 명시 (필요시)
export PYTHONPATH="/path/to/hossam"

# 로그 레벨 설정 (선택사항)
export LOG_LEVEL=INFO

# 서버 시작
hossam-mcp
```

---

## 문제 해결

### "도구를 찾을 수 없음" 오류

```
오류: Unknown tool: hs_stats_missing_values

원인: 도구명 오타 또는 대소문자 오류
해결: hs_mcp_list_tools()로 정확한 이름 확인
```

### "categories 파라미터" 오류

```
오류: 데이터 타입이 object가 됨

원인: load_data 호출 시 categories 미지정
해결: categories=["sex", "smoker", "region"] 파라미터 추가
```

### 파이프 통신 오류

```bash
# ❌ 잘못된 형식 (줄바꿈 없음)
echo '{"tool":"hs_mcp_list_tools","args":{}}' | hossam-mcp

# ✅ 올바른 형식 (JSON 라인)
echo '{"tool":"hs_mcp_list_tools","args":{}}' | hossam-mcp
```

---

## 관련 문서

- **빠른 시작**: [QUICKSTART.md](./QUICKSTART.md)
- **VSCode 통합**: [VSCode_COPILOT.md](./VSCode_COPILOT.md)
- **GitHub**: https://github.com/leekh4232/hossam-py
