---
title: VS Code에서 Hossam MCP 연동 가이드
---

# 🧩 Hossam MCP 연동 가이드 (VS Code Copilot)

이 문서는 Hossam 라이브러리를 VS Code에서 MCP(Model Context Protocol) 서버로 연동하여 Copilot Chat에서 데이터 분석 도구들을 직접 호출하는 방법을 정리합니다. 또한, 워크스페이스의 설정 파일 예시를 그대로 인용하여 손쉽게 복사/붙여넣기 할 수 있도록 제공합니다.

## MCP란?

- MCP는 도구(툴)들을 표준화된 프로토콜로 노출해 AI 어시스턴트가 호출할 수 있게 하는 방식입니다.
- Hossam MCP 서버는 `StdIO(JSON Lines)` 기반으로 동작하며, VS Code Copilot Chat과 호환됩니다.
- 서버 실행 엔트리: `python -m hossam.mcp.server` 또는 `hossam-mcp`.

## 사전 준비

### 패키지 설치:

```shell
pip install hossam
```

### VS Code 익스텐션 설치

```
GitHub Copilot Chat
```

## VS Code 설정(.vscode/settings.json)

- 파일: `.vscode/settings.json`
- 역할: Copilot Chat에 외부 MCP 서버(Hossam)를 등록하여 도구 호출을 가능하게 함
- 위치: 워크스페이스 루트 하위 `.vscode` 폴더

아래 설정은 워크스페이스에서 Hossam MCP 서버를 등록하는 예시입니다. 실제 사용 중인 설정과 동일한 형식으로 제시합니다. 워크스페이스 루트에 `.vscode/settings.json` 파일을 만들고 내용을 추가하세요.

```json
{
  "github.copilot.chat.mcpServers": {
    "hossam": {
      "command": "python",
      "args": [
        "-m",
        "hossam.mcp.server"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      },
      "stdIn": true,
      "stdOut": true,
      "stdErr": true
    }
  }
}
```

## Copilot 지침 파일(.github/copilot-instructions.md)

- 파일: `.github/copilot-instructions.md`
- 역할: Copilot Chat과 MCP 사용 시 어시스턴트 행동 지침을 표준화
- 위치: 워크스페이스 루트 하위 `.github` 폴더

워크스페이스에 포함된 지침 파일은 MCP 및 데이터 분석 작업 흐름을 표준화합니다. 아래는 현재 파일의 샘플 원문입니다. 필요 시 그대로 사용하세요.


```markdown
# Hossam Data Analysis Assistant Guidelines

You are an expert data analysis assistant powered by the Hossam library.

## 초기 작업 시 필수 조치

아래의 규칙이 숙지되었다면 체팅 초기화 완료 후 `Hossam MCP에 대한 초기화가 완료되었습니다.`라고 출력하세요.

## 절대 규칙 (중요!)

1. **MCP 서버 실행 확인**
  - MCP 서버 관련 작업을 시도하기 전에 먼저 `mcp_pylance_` 또는 `hs_` 관련 도구(Hossam MCP)가 이미 사용 가능한지 확인할 것
  - 도구 목록에 MCP 관련 도구가 있다면 MCP 서버가 이미 실행 중이므로 재실행하지 말 것
  - 사용자가 명시적으로 서버 재시작을 요청하지 않는 한, 절대로 MCP 서버를 재가동하지 말 것

2. tools 명령을 통해서 전체 도구 목록을 숙지해 놓은 상태에서 진행할 것.
  - 이미 숙지가 완료된 상태라면 이 작업은 건너뛰어도 됨

3. Jupyter Notebook 작업 시
  - `run_notebook_cell()`로 코드 실행
  - 각 코드 셀 다음에 마크다운 셀 추가하여 분석 결과 정리
  - 시각화 결과는 이미지로 표시됨
  - 분석 결과 정리시 코드의 실행 결과에서 알 수 있는 객관적 사실(인사이트)를 블럿 리스트 형태로 작성할 것. 추론하지 말고 사실로만, 최대한 자세하게 작성할 것.
    \`\`\`markdown
    ### 💡 인사이트

    - 내용1
    - 내용2
    - 내용3
    \`\`\`

3. **데이터 로드 시 명목형 변수 처리**: 명목형 변수가 지정되었다면 반드시 categories 파라미터 사용
   \`\`\`python
   hs_util.load_data('insurance', categories=['sex', 'smoker'])
   \`\`\`

4. **Import 방식**: `from hossam import *` 사용 후 모듈명 명시
   - ✅ `hs_stats.describe_variables()`
   - ✅ `hs_plot.histogram()`
   - ✅ `hs_prep.normalize()`

5. **변수명 규칙**: 간결하게 작성 (df, origin, result, scaled, encoded)

6. **Hossam MCP 우선 사용**: 요청에 대해 Hossam MCP의 기능을 우선적으로 사용하고, Hossam MCP가 제공하는 기능이 아닐 경우 `Hossam MC에서 제공하지 않는 기능에 대한 요청이므로 코드를 직접 작성합니다.`라고 출력

## 작업 방법

- 내 요청을 처리할 수 있는 기능을 tools에서 검색해서 수행.
    - 이미 tools를 통해서 목록을 숙지하고 있다면 재검색은 필요 없음
- 요청을 처리할 수 있는 기능이 두 개 이상인 경우 넘버 리스트 형태로 나에게 선택지를 제시하고 내 선택에 따라 진행할 것
- 여러 코드 블록을 동시에 추가할 경우 순차적으로 처리할 것. 블록의 순서가 사고 과정과 일치해야 함
- 내가 특별히 요청하기 전까지는 노트북의 요약을 가져오지 말것.
- 요청에 대해 나에게 알려줄 내용이 있다면(AI의 출력 내용 등) 작업중인 노트북에 빈 마크다운 블록을 추가하고 작성할 것.
- 유니코드 아이콘을 최대한 활용할 것
```

## 제공 기능(툴) 개요

Hossam MCP 서버는 `hs_` 접두사를 사용해 공개 API를 도구로 자동 등록합니다. 주요 네임스페이스와 예시는 아래와 같습니다.

- `hs_util_*`: 파일/노트북 유틸리티, 데이터 로드(`hs_util_load_data`), 테이블/포맷 유틸 등
- `hs_stats_*`: 변수 기술통계(`describe_variables`), 상관/가설검정 등 통계 분석 함수
- `hs_plot_*`: `lineplot`, `boxplot`, `kdeplot`, `histogram` 등 시각화. 원격 환경에서는 `save_path` 활용 권장
- `hs_prep_*`: 전처리(`standard_scaler`, `minmax_scaler`, `get_dummies`, `replace_outliner`, `set_category` 등)
- `hs_gis_*`: 공간 데이터 저장/변환(`hs_gis_save_shape` 등)
- `hs_classroom_*`: 교육/실습 보조 함수들
- `hs_timeserise_*`: 시계열 관련 유틸리티
- `hs_data_*`: `hossam.data_loader` 공개 함수 자동 등록(일부 제외)

각 함수는 원본 시그니처를 따르며, MCP 호출 시 파라미터는 JSON 형태로 전달됩니다.

## 사용 예시(워크플로)

1. VS Code에서 Copilot Chat 열기 → MCP 서버가 자동 기동됨(첫 요청 시)
2. 데이터 로드: `hs_util_load_data`에 `name`, `categories` 등 파라미터 전달
3. 기술통계: `hs_stats_describe_variables` 호출
4. 시각화: `hs_plot_histogram` 또는 `hs_plot_kdeplot` 호출, `save_path`로 결과 저장
5. 전처리: `hs_prep_get_dummies`, `hs_prep_standard_scaler` 등 순차 적용

## 서버 수동 실행(선택)

터미널에서 직접 실행할 수도 있습니다.

```bash
python -m hossam.mcp.server
# 또는
hossam-mcp
```

서버는 `stderr`에 로그를 남기고, `stdout`은 MCP 프로토콜로만 사용됩니다.

## 트러블슈팅

- MCP 서버 미기동: VS Code 설정의 `github.copilot.chat.mcpServers` 항목 확인, Python 경로/가상환경 점검
- 도구 미노출: 함수가 언더바(`_`)로 시작하면 제외됩니다. 공개 API만 노출됨
- 그래프 출력 문제: 원격 환경에서는 화면 표시 대신 `save_path`를 지정해 이미지 파일로 저장하여 확인

---

문의/기여는 리포지토리 이슈 또는 가이드를 참고하세요.
