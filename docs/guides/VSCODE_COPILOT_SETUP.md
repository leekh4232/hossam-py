# Hossam MCP + VSCode Copilot 연동 완료

> hossam 라이브러리의 모든 기능을 VSCode의 GitHub Copilot과 Cline을 통해 사용할 수 있습니다.

## 🚀 시작하기 (2단계)

### 1단계: VSCode에서 프로젝트 열기

```bash
code /Users/leekh/hossam-py
```

### 2단계: Copilot Chat에서 hossam 사용

**텍스트 예시:**
```
@hossam
이 CSV 파일의 결측치를 분석해줄래?
```

**Copilot이 자동으로:**
- `hs_missing_values` 도구 호출
- 결과 DataFrame 생성 및 분석
- 코드 제안

## 📦 필수 VSCode 확장

| 확장 | 설명 | 상태 |
|------|------|------|
| **GitHub Copilot** | AI 코드 어시스턴트 | 권장 |
| **GitHub Copilot Chat** | 채팅 인터페이스 | 권장 |
| **Cline** | MCP 클라이언트 | 선택 |

**설치 방법:**
1. VSCode 확장 마켓플레이스 열기 (Cmd+Shift+X)
2. "GitHub Copilot Chat" 검색 후 설치
3. (선택) "Cline" 설치

## 📝 사용 예제

### 예제 1: 데이터 분석

**사용자:**
```
CSV 파일을 로드해서 통계를 보여줄래?
```

**Copilot 응답 (MCP 도구 자동 사용):**
```python
import pandas as pd
from hossam import hs_stats

# 파일 로드
df = pd.read_csv("data.csv")

# hs_missing_values로 결측치 분석
missing_info = hs_stats.missing_values(df)
print(missing_info)

# hs_normal_test로 정규성 검정
normal_results = hs_stats.normal_test(df)
print(normal_results)
```

### 예제 2: 데이터 전처리

**사용자:**
```
이 DataFrame을 표준화해줄래?
```

**Copilot 응답:**
```python
from hossam import hs_prep

# Standard Scaler 적용
scaled_df = hs_prep.standard_scaler(df, save_path="scaled.pkl")
print(scaled_df.head())
```

### 예제 3: 시각화

**사용자:**
```

### 기본: 코드 스니펫 반환, 실행이 필요하면 플래그 추가

기본값은 코드 생성입니다. 실행 결과가 필요하면 아래 중 하나를 추가하세요.

- `mode: "run"` 또는 `return: "result"`
- `run: true` / `execute: true` / `result: true`

예시(Cline/터미널):

```bash
echo '{"tool":"hs_missing_values","args":{"df":[{"a":1,"b":null},{"a":null,"b":2}]}}' | hossam-mcp  # 코드만 반환(기본)
echo '{"tool":"hs_missing_values","args":{"run":true,"df":[{"a":1,"b":null},{"a":null,"b":2}]}}' | hossam-mcp  # 실행 강제
```

Copilot Chat 팁:
- 기본은 코드 반환이므로, 실행을 원할 때는 "실행 결과를 보여줘"처럼 명시하거나 `run: true`를 붙여달라고 요청하세요.
이 데이터를 박스플롯으로 그려줄래. 파일로 저장해.
```

**Copilot 응답:**
```python
from hossam import hs_plot

hs_plot.boxplot(
    df=df,
    xname="category",
    yname="value",
    save_path="./boxplot.png"
)
```

## 🔧 고급 설정

### Python 경로 수동 지정

`.vscode/settings.json`에서:

```json
{
  "[python]": {
    "defaultInterpreterPath": "/usr/local/bin/python3"
  }
}
```

### 환경 변수 추가

```json
{
  "mcpServers": {
    "hossam": {
      "command": "python",
      "args": ["-m", "hossam.mcp.server"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "PYTHONDONTWRITEBYTECODE": "1"
      }
    }
  }
}
```

## 🛠️ 트러블슈팅

### MCP 서버가 시작되지 않음

```bash
# 1. 직접 테스트
python -m hossam.mcp.server

# 2. 간단한 호출 테스트
echo '{"tool":"hs_make_normalize_values","args":{"mean":0,"std":1,"size":5}}' | python -m hossam.mcp.server
```

### VSCode에서 MCP 인식 안 됨

```bash
# VSCode 재시작
Cmd+Shift+P (또는 Ctrl+Shift+P on Windows)
> Developer: Reload Window
```

### Python 경로 오류

```bash
# Python 위치 확인
which python3

# 또는 VSCode 설정에서
{
  "[python]": {
    "defaultInterpreterPath": "/usr/local/bin/python3"
  }
}
```

## 📚 참고 자료

- [전체 MCP 가이드](docs/guides/mcp.md)
- [VSCode 연동 상세 가이드](docs/guides/vscode-copilot-integration.md)
- [Hossam API 문서](docs/api/)
- [GitHub Copilot 공식 문서](https://github.com/features/copilot)

## ✨ 주요 이점

1. **간편한 대화형 인터페이스**: "~해줄래?" 형식으로 자연스럽게 사용
2. **자동 도구 선택**: Copilot이 최적의 hossam 도구 자동 선택
3. **코드 제안**: AI가 분석 코드를 자동 생성
4. **빠른 프로토타이핑**: 복잡한 명령어 대신 자연언어 사용
5. **기존 코드와 호환**: 라이브러리 방식 동시 지원

## 🎓 교육용 활용

데이터 분석 강의에서:

```
"학생들에게 Copilot Chat에서 이렇게 말해보라고 안내:
@hossam
이 수능 점수 데이터의 최다/최소 범주를 보여줄래?
```

자동으로 `hs_category_summary` 실행 → 교육용 분석 결과 제시

---

**상태:** ✅ 완전 준비됨
**마지막 업데이트:** 2026년 1월 14일
**버전:** hossam 0.3.16 + MCP Server 1.0
