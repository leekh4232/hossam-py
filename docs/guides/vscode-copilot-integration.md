# VSCode + Copilot과 함께 Hossam MCP 사용 가이드

> GitHub Copilot Chat을 통해 hossam MCP 서버의 모든 도구를 VSCode에서 직접 사용할 수 있습니다.

## 요구사항

1. **VSCode 확장**
   - GitHub Copilot Chat (https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat)
   - 또는 GitHub Copilot (기본 버전)

2. **Python 환경**
   - Python 3.11+
   - hossam 패키지 (개발 설치 완료)

## 설정 완료 확인

프로젝트에 이미 다음 파일들이 준비되어 있습니다:

1. **`.vscode/settings.json`** - VSCode MCP 서버 설정
2. **`.cline_tools.json`** - Cline 확장 MCP 설정

### 빠른 실행 (이미 설정됨)

```bash
# 1. VSCode 열기
code .

# 2. VSCode 재시작 후 자동 MCP 연결
```

### 설정 파일 내용 확인

**`.vscode/settings.json` (이미 생성됨):**

```json
{
  "mcpServers": {
    "hossam": {
      "command": "python",
      "args": ["-m", "hossam.mcp.server"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      },
      "disabled": false
    }
  }
}
```

**`.cline_tools.json` (이미 생성됨):**

```json
{
  "cline": {
    "mcpServers": [
      {
        "name": "hossam",
        "command": "python",
        "args": ["-m", "hossam.mcp.server"],
        "env": {
          "PYTHONPATH": "${workspaceFolder}"
        },
        "disabled": false
      }
    ]
  }
}
```

## 단계별 설정 (처음 설정하는 경우)

### 1. Cline 확장 설치 (권장)

더 나은 MCP 지원을 위해 **Cline** 확장을 설치하세요:

1. VSCode 확장 마켓플레이스에서 "Cline" 검색
2. **Cline** (by Saoudrizwan) 설치

또는 GitHub Copilot Chat 확장 사용:
- https://marketplace.visualstudio.com/items?itemName=GitHub.copilot-chat

### 2. VSCode 재시작

```bash
# macOS/Linux
Cmd+Shift+P

# Windows
Ctrl+Shift+P

# 검색 후 선택
> Developer: Reload Window
```

## 사용 방법

### Copilot Chat에서

```
/mcp hs_missing_values
```

Copilot이 hossam 도구를 인식하고 제안합니다.

### Cline에서

1. Cline 패널 열기 (좌측 사이드바)
2. MCP 탭에서 "hossam" 서버 확인
3. 도구 목록 보기 및 직접 호출

### 실행 결과가 필요할 때

기본은 코드 스니펫 반환입니다. 실행이 필요하면 아래 플래그 중 하나를 추가하세요:

- `mode: "run"` / `return: "result"`
- 또는 `run: true` / `execute: true` / `result: true`

예시:

```bash
echo '{"tool":"hs_outlier_table","args":{"df":"./data.csv"}}' | hossam-mcp        # 코드만 반환(기본)
echo '{"tool":"hs_outlier_table","args":{"run":true,"df":"./data.csv"}}' | hossam-mcp  # 실행 강제
```

Copilot Chat에서는 기본이 코드 반환입니다. 실행을 원하면 "실행해서 결과를 보여줘" 또는 `run:true`를 포함해 요청하세요.

## 예제 사용 시나리오

### 시나리오 1: 데이터 분석

**사용자:**
```
데이터프레임에서 결측치를 분석해줘. 현재 열려있는 CSV 파일을 사용하자.
```

**Copilot (MCP 도구 사용):**
```python
# hs_missing_values를 사용하여 분석
import pandas as pd
df = pd.read_csv("data.csv")
result = hossam_mcp.call("hs_missing_values", df=df)
print(result)
```

### 시나리오 2: 데이터 전처리

**사용자:**
```
이 DataFrame을 MinMax 스케일링해줘.
```

**Copilot:**
```python
# hs_minmax_scaler 사용
scaled_df = hossam_mcp.call("hs_minmax_scaler", data=df)
```

### 시나리오 3: 시각화

**사용자:**
```
이 데이터로 박스플롯을 그려줘.
```

**Copilot:**
```python
# hs_boxplot으로 시각화 (파일로 저장)
hossam_mcp.call(
    "hs_boxplot",
    df=df,
    xname="category",
    yname="value",
    save_path="./boxplot.png"
)
```

## 트러블슈팅

### MCP 서버가 시작되지 않음

**확인 사항:**
```bash
# 1. hossam이 설치되어 있는지 확인
python -m pip list | grep hossam

# 2. MCP 서버 직접 실행 테스트
python -m hossam.mcp.server

# 3. 간단한 호출 테스트
echo '{"tool":"hs_make_normalize_values","args":{"mean":0,"std":1,"size":5}}' | python -m hossam.mcp.server
```

### VSCode에서 MCP 서버가 인식되지 않음

**해결 방법:**
1. `.vscode/settings.json` 문법 확인 (JSON 유효성)
2. VSCode 재시작
3. Output 패널에서 "MCP" 확인

### Python 경로 오류

`.vscode/settings.json`에 명시적 Python 경로 추가:

```json
{
  "mcpServers": {
    "hossam": {
      "command": "/usr/bin/python3",
      "args": ["-m", "hossam.mcp.server"]
    }
  }
}
```

**macOS/Linux에서 Python 경로 찾기:**
```bash
which python3
```

**Windows에서 Python 경로 찾기:**
```cmd
where python
```

## 고급 설정

### 환경 변수 설정

```json
{
  "mcpServers": {
    "hossam": {
      "command": "python",
      "args": ["-m", "hossam.mcp.server"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}:${workspaceFolder}/hossam",
        "PYTHONDONTWRITEBYTECODE": "1"
      }
    }
  }
}
```

### 다중 MCP 서버

```json
{
  "mcpServers": {
    "hossam": {
      "command": "python",
      "args": ["-m", "hossam.mcp.server"]
    },
    "other-tool": {
      "command": "node",
      "args": ["./server.js"]
    }
  }
}
```

## 권장 VSCode 확장

| 확장명 | 설명 |
|--------|------|
| GitHub Copilot Chat | AI 기반 코드 어시스턴트 |
| Cline | MCP 클라이언트 (자동 서버 관리) |
| REST Client | API 테스트 (필요 시) |
| Python | Python 개발 지원 |
| Pylance | Python 타입 검사 |

## 설정 파일 예제 (완성본)

`.vscode/settings.json`:

```json
{
  "[python]": {
    "defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "linting.enabled": true,
    "linting.pylintEnabled": true,
    "formatting.provider": "black"
  },

  "mcpServers": {
    "hossam": {
      "command": "python",
      "args": ["-m", "hossam.mcp.server"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      },
      "disabled": false
    }
  },

  "github.copilot.chat.localeOverride": "ko",
  "[github-copilot-chat]": {
    "editor.defaultFormatter": "GitHub.copilot"
  }
}
```

## 다음 단계

1. ✅ `.vscode/settings.json` 생성
2. ✅ MCP 서버 등록
3. ✅ VSCode 재시작
4. ✅ Copilot Chat에서 `@hossam` 또는 `/mcp` 호출
5. ✅ 도구 목록 확인 및 사용

---

**마지막 업데이트:** 2026년 1월 14일
**MCP Client 호환성:** Copilot Chat, Cline, Cursor 등
