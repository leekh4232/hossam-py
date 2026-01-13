# 🎓 Hossam Data Helper

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.3.8-green.svg)](https://pypi.org/project/hossam/)
[![Documentation](https://img.shields.io/badge/docs-py.hossam.kr-blue.svg)](https://py.hossam.kr)

**Hossam**은 데이터 분석, 시각화, 통계 처리를 위한 종합 헬퍼 라이브러리입니다.

아이티윌(ITWILL)에서 진행 중인 머신러닝 및 데이터 분석 수업을 위해 개발되었으며, 이광호 강사의 강의에서 활용됩니다.

## ✨ 주요 특징

- 📊 **풍부한 시각화**: 25+ 시각화 함수 (Seaborn/Matplotlib 기반)
- 🎯 **통계 분석**: 회귀, 분류, 시계열 분석 도구
- 📦 **샘플 데이터**: 학습용 데이터셋 즉시 로드
- 🔧 **데이터 전처리**: 결측치 처리, 이상치 탐지, 스케일링
- 🤖 **MCP 서버**: VSCode/Copilot과 통합 가능한 Model Context Protocol 지원
- 📈 **교육용 최적화**: 데이터 분석 교육에 특화된 설계

---

## 📦 설치

```bash
pip install hossam
```

**요구사항**: Python 3.8 이상

---

## 🚀 빠른 시작

### 샘플 데이터 로드

```python
from hossam import load_data, load_info

# 사용 가능한 데이터셋 확인
datasets = load_info()

# 데이터 로드
df = load_data('AD_SALES')
```

### 간단한 시각화

```python
from hossam import hs_plot
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# 산점도
hs_plot.scatterplot(df=df, xname='x', yname='y', hue='category')

# 박스플롯
hs_plot.boxplot(df=df, xname='category', yname='x')
```

---

## 🤖 MCP Server

Hossam은 **Model Context Protocol(MCP)** 기반 서버로도 작동하며, VSCode Copilot/Cline과 통합하여 데이터 분석 코드를 자동 생성할 수 있습니다.

### 빠른 시작

```bash
# 서버 시작
hossam-mcp
```

### VSCode + Copilot 연동

프로젝트의 `.vscode/settings.json`에 hossam MCP 서버가 자동 등록되어 있습니다.

**Copilot Chat에서 사용:**
```
@hossam 이 DataFrame의 결측치를 분석하고 처리하는 코드 작성해줘
```

**자세한 가이드:**
- [MCP 서버 사용법](https://py.hossam.kr/guides/mcp/)
- [VSCode + Copilot 연동](https://py.hossam.kr/guides/vscode-copilot-integration/)
- [Copilot Chat 프롬프트 예시](https://py.hossam.kr/guides/copilot-prompts/)

---

## 📚 전체 문서

**완전한 API 문서와 가이드는 [py.hossam.kr](https://py.hossam.kr)에서 확인하세요.**

### 주요 모듈

- **hs_plot**: 25+ 시각화 함수 (선 그래프, 산점도, 히스토그램, 박스플롯, 히트맵 등)
- **hs_stats**: 회귀/분류 분석, 교차검증, 정규성 검정, 상관분석 등
- **hs_prep**: 결측치 처리, 이상치 탐지, 스케일링, 인코딩
- **hs_gis**: GIS 데이터 로드 및 시각화 (대한민국 지도 지원)
- **hs_classroom**: 학습용 이진분류, 다중분류, 회귀 데이터 생성
- **hs_util**: 예쁜 테이블 출력, 그리드 서치 등

자세한 사용법은 [API 문서](https://py.hossam.kr/api/hossam/)를 참고하세요.

---

## 🎓 예제

### 결측치 분석

```python
from hossam import hs_prep

# 결측치 정보 확인
hs_prep.hs_missing_values(df)

# 결측치 시각화
hs_prep.hs_missing_values_barplot(df)
```

### 회귀 분석

```python
from hossam import hs_stats

# 단순 선형 회귀
result = hs_stats.hs_simple_regression(df, xname='x', yname='y', plot=True)
```

### 상관분석 히트맵

```python
from hossam import hs_plot

hs_plot.heatmap(df=df, annot=True, cmap='coolwarm')
```

더 많은 예제는 [문서 사이트](https://py.hossam.kr)를 참고하세요.

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자유롭게 사용, 수정, 배포할 수 있습니다.

---

## 🔗 링크

- **문서**: [py.hossam.kr](https://py.hossam.kr)
- **PyPI**: [pypi.org/project/hossam](https://pypi.org/project/hossam/)
- **강사**: 이광호 (ITWILL 머신러닝 및 데이터 분석)

---

**Made with ❤️ for Data Science Education**
