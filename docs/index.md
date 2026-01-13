---
title: HOSSAM Overview
---

# HOSSAM 패키지 개요

HOSSAM은 데이터 로딩/전처리/분석/시각화 및 GIS 유틸리티를 제공하는 파이썬 패키지입니다.

## 빠른 시작

```python
from hossam import hs_util, data_loader, hs_prep, hs_stats, hs_plot, hs_gis

# 예: 데이터 로드 후 정보 출력
df = data_loader.load_data("sample_key")
hs_util.pretty_table(df.head())
```

## 주요 모듈

- `hossam.data_loader`: 원격/로컬 데이터 조회 및 로딩
- `hossam.hs_prep`: 스케일링, 결측치 처리 등 전처리 유틸
- `hossam.hs_stats`: 통계 분석 유틸 (VIF 필터, 추세선 계산 등)
- `hossam.hs_plot`: 다양한 시각화 함수(kde, box, scatter 등)
- `hossam.hs_gis`: 지오코딩 및 쉐이프 로드/저장
- `hossam.hs_util`: 표 예쁘게 출력, 샘플 데이터 생성 등 공용 유틸

## 🤖 MCP 서버

Hossam은 **Model Context Protocol(MCP)** 서버로도 작동하며, VSCode Copilot/Cline과 통합하여 데이터 분석 코드를 자동 생성할 수 있습니다.

### 빠른 실행

```bash
# 설치 후
pip install hossam

# 서버 실행
hossam-mcp
```

### VSCode + Copilot 사용

```
@hossam 이 DataFrame 결측치 분석 코드만 보여줘
```

### 상세 문서

- **[VSCode settings.json 완성형 샘플](guides/vscode-settings-sample.md)** ⭐ 추천
- [MCP 서버 사용법](guides/mcp.md)
- [VSCode + Copilot 연동 가이드](guides/vscode-copilot-integration.md)
- [Copilot Chat 프롬프트 예시](guides/copilot-prompts.md)
- [VSCode Copilot 설정 가이드](guides/VSCODE_COPILOT_SETUP.md)
- [MCP 작업 가이드 (개발자용)](guides/hossam_mcp_task_prompt.md)

더 자세한 내용은 API 레퍼런스를 참고하세요.
