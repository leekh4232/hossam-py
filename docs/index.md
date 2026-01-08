---
title: HOSSAM Overview
---

# HOSSAM 패키지 개요

HOSSAM은 데이터 로딩/전처리/분석/시각화 및 GIS 유틸리티를 제공하는 파이썬 패키지입니다.

## 빠른 시작

```python
from hossam import util, data_loader, prep, analysis, plot, gis

# 예: 데이터 로드 후 정보 출력
df = data_loader.load_data("sample_key")
util.hs_pretty_table(df.head())
```

## 주요 모듈

- `hossam.data_loader`: 원격/로컬 데이터 조회 및 로딩
- `hossam.prep`: 스케일링, 결측치 처리 등 전처리 유틸
- `hossam.analysis`: 통계 분석 유틸 (VIF 필터, 추세선 계산 등)
- `hossam.plot`: 다양한 시각화 함수(kde, box, scatter 등)
- `hossam.gis`: 지오코딩 및 쉐이프 로드/저장
- `hossam.util`: 표 예쁘게 출력, 샘플 데이터 생성 등 공용 유틸

더 자세한 내용은 API 레퍼런스를 참고하세요.
