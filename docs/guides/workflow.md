---
title: Workflow Guide
---

# 실전 워크플로: 로드 → 전처리 → 분석 → 시각화 → GIS

아래는 HOSSAM을 활용한 대표적인 작업 흐름 예시입니다.

## 1) 데이터 로드

```python
from hossam import util, data_loader

# 키 목록/설명 확인 (원격 메타데이터)
info = data_loader.load_info()
util.hs_pretty_table(info.head())

# 키로 데이터 로드
df = data_loader.load_data("AD_SALES")
```

## 2) 전처리

```python
from hossam import prep

# 카테고리 지정
from pandas import DataFrame
# df["category"] = df["category"].astype("category")  # 혹은 아래 헬퍼 사용
prep.hs_set_category(df, "category")

# 결측치 대체
df2 = prep.hs_replace_missing_value(df.select_dtypes(include="number"), strategy="mean")

# 스케일링(Standard)
scaled = prep.hs_standard_scaler(df, yname="y", save_path="std.pkl")
```

## 3) 분석

```python
from hossam import analysis

# 다중공선성 제거(VIF)
filtered = analysis.hs_vif_filter(df, yname="y", ignore=["id"], threshold=10.0)

# 추세선 계산
x = df["x"].to_numpy()
y = df["y"].to_numpy()
vx, vy = analysis.hs_trend(x, y, degree=2)
```

## 4) 시각화

```python
from hossam import plot as hs_plot

hs_plot.hs_scatterplot(df=df, xname="x", yname="y", hue="category", palette="Set1")
hs_plot.hs_kdeplot(df=df, xname="x", hue="category", fill=True)
```

## 5) GIS

```python
from hossam import gis

# 지오코딩 (VWorld API 키 필요)
result = gis.hs_geocode(df, addr="address", key="YOUR_VWORLD_KEY")

# Shapefile 로드/저장
gdf = gis.hs_load_shape("path/to/file.shp")
gis.hs_save_shape(gdf, "out.gpkg")
```

## 참고
- 더 자세한 API는 사이드바의 "API Reference"에서 모듈별 문서를 참고하세요.
- 모든 공개 함수 목록은 "API Reference → Index"에서 확인할 수 있습니다.
