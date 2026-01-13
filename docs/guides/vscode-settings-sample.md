# VSCode 설정 완성형 샘플

> `pip install hossam` 후 프로젝트에서 바로 사용 가능한 `.vscode/settings.json` 샘플입니다.

## 📁 파일 위치

프로젝트 루트에 `.vscode/settings.json` 파일을 생성하세요:

```
your-project/
├── .vscode/
│   └── settings.json    ← 이 파일 생성
├── data/
└── notebooks/
```

## 📝 완성형 settings.json

다음 내용을 복사해서 `.vscode/settings.json`에 붙여넣으세요:

```json
{
  "github.copilot.chat.codeGeneration.instructions": [
    {
      "file": "hossam-mcp-instructions.md"
    }
  ],
  "github.copilot.chat.tools.vscode": [
    {
      "name": "hossam",
      "description": "Python 데이터 분석 헬퍼 - 통계, 시각화, 전처리, GIS, 시계열 분석 도구",
      "command": "hossam-mcp"
    }
  ],
  "github.copilot.chat.localeOverride": "ko"
}
```

### ⚙️ 주요 설정 설명

| 설정 | 설명 | 수정 필요 여부 |
|------|------|----------------|
| `github.copilot.chat.tools.vscode` | Copilot에서 `@hossam` 도구 활성화 | ❌ 수정 불필요 |
| `command: "hossam-mcp"` | hossam MCP 서버 실행 명령 | ⚠️ Python 경로 이슈 시에만 수정 |
| `localeOverride: "ko"` | Copilot Chat 한국어 사용 | ✅ 선택사항 (en으로 변경 가능) |

## 🔧 환경별 수정이 필요한 경우

### 1. Python 가상환경 사용 시

**문제:** `hossam-mcp` 명령을 찾을 수 없음
**해결:** 가상환경의 Python 경로를 명시

```json
{
  "github.copilot.chat.tools.vscode": [
    {
      "name": "hossam",
      "description": "Python 데이터 분석 헬퍼",
      "command": "/Users/your-name/your-project/.venv/bin/python",
      "args": ["-m", "hossam.mcp.server"]
    }
  ]
}
```

**Python 경로 찾는 방법:**

```bash
# macOS/Linux
which python
# 또는
which python3

# Windows (PowerShell)
(Get-Command python).Path

# Windows (CMD)
where python
```

### 2. Conda 환경 사용 시

```json
{
  "github.copilot.chat.tools.vscode": [
    {
      "name": "hossam",
      "description": "Python 데이터 분석 헬퍼",
      "command": "/Users/your-name/miniconda3/envs/your-env/bin/python",
      "args": ["-m", "hossam.mcp.server"]
    }
  ]
}
```

**Conda Python 경로 찾기:**

```bash
# 활성 환경의 Python 경로 확인
conda activate your-env
which python
```

### 3. Windows 시스템 Python 사용 시

```json
{
  "github.copilot.chat.tools.vscode": [
    {
      "name": "hossam",
      "description": "Python 데이터 분석 헬퍼",
      "command": "C:\\Python311\\python.exe",
      "args": ["-m", "hossam.mcp.server"]
    }
  ]
}
```

### 4. 여러 Python 도구 함께 사용

```json
{
  "github.copilot.chat.tools.vscode": [
    {
      "name": "hossam",
      "description": "Python 데이터 분석 헬퍼",
      "command": "hossam-mcp"
    },
    {
      "name": "other-tool",
      "description": "다른 도구",
      "command": "other-mcp-server"
    }
  ]
}
```

## 🧪 설정 확인

### 1. 터미널에서 테스트

```bash
# hossam-mcp가 실행되는지 확인
hossam-mcp

# 간단한 테스트
echo '{"tool":"hs_make_normalize_values","args":{"mean":0,"std":1,"size":5}}' | hossam-mcp
```

**예상 출력:**
```json
{
  "code": "from hossam import hs_stats\nresult = hs_stats.hs_make_normalize_values(mean=0, std=1, size=5)"
}
```

### 2. VSCode에서 확인

1. **Copilot Chat 열기**: `Cmd+I` (macOS) 또는 `Ctrl+I` (Windows)
2. **테스트 프롬프트 입력**:
   ```
   @hossam 정규분포 난수 5개 생성하는 코드
   ```
3. **예상 응답**:
   ```python
   from hossam import hs_stats
   values = hs_stats.hs_make_normalize_values(mean=0, std=1, size=5)
   print(values)
   ```

## 📚 추가 설정 (선택사항)

### Python 분석 설정 추가

```json
{
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  },
  "python.analysis.typeCheckingMode": "basic",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,

  "github.copilot.chat.tools.vscode": [
    {
      "name": "hossam",
      "description": "Python 데이터 분석 헬퍼",
      "command": "hossam-mcp"
    }
  ]
}
```

### Jupyter Notebook 지원 추가

```json
{
  "jupyter.jupyterServerType": "local",
  "notebook.formatOnSave.enabled": true,

  "github.copilot.chat.tools.vscode": [
    {
      "name": "hossam",
      "description": "Python 데이터 분석 헬퍼",
      "command": "hossam-mcp"
    }
  ]
}
```

## 🐛 트러블슈팅

### "hossam-mcp not found" 오류

**원인:** PATH에 hossam이 설치된 Python 환경이 없음

**해결:**
1. 가상환경 활성화 확인
2. `pip install hossam` 재실행
3. 절대 경로로 Python 명시 (위 예시 참고)

### "@hossam이 인식되지 않음"

**원인:** GitHub Copilot Chat 확장이 설정을 읽지 못함

**해결:**
1. VSCode 재시작 (`Cmd+Shift+P` → "Developer: Reload Window")
2. `settings.json` JSON 문법 오류 확인
3. GitHub Copilot Chat 확장 최신 버전 확인

### "command failed" 오류

**원인:** Python 환경 또는 hossam 설치 문제

**해결:**
```bash
# 1. Python 버전 확인 (3.8 이상 필요)
python --version

# 2. hossam 설치 확인
pip show hossam

# 3. MCP 서버 직접 실행 테스트
python -m hossam.mcp.server
```

## 📖 관련 문서

- [MCP 서버 사용법 전체 가이드](mcp.md)
- [VSCode + Copilot 연동 상세](vscode-copilot-integration.md)
- [Copilot Chat 프롬프트 예시](copilot-prompts.md)
- [전체 API 문서](https://py.hossam.kr)

---

**최종 업데이트:** 2026년 1월 14일
**호환성:** VSCode 1.85+, GitHub Copilot Chat 0.12+
