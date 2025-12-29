# 🎓 Hossam Data Helper

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.3.0-green.svg)](https://pypi.org/project/hossam/)

**Hossam**은 데이터 분석, 시각화, 통계 처리를 위한 종합 헬퍼 라이브러리입니다.

아이티윌(ITWILL)에서 진행 중인 머신러닝 및 데이터 분석 수업을 위해 개발되었으며, 이광호 강사의 강의에서 활용됩니다.

---

## 📋 목차

- [특징](#-특징)
- [설치](#-설치)
- [빠른 시작](#-빠른-시작)
- [주요 기능](#-주요-기능)
  - [데이터 로더](#1-데이터-로더)
  - [시각화 모듈](#2-시각화-모듈-hossamplot)
  - [분석 모듈](#3-분석-모듈-hossamanalysis)
  - [전처리 모듈](#4-전처리-모듈-hossamprep)
  - [유틸리티 모듈](#5-유틸리티-모듈-hossamutil)
- [의존성](#-의존성)
- [문서](#-문서)
- [라이선스](#-라이선스)
- [저자](#-저자)

---

## ✨ 특징

- 📊 **풍부한 시각화**: Seaborn/Matplotlib 기반의 25+ 시각화 함수
- 🎯 **통계 분석**: 회귀, 분류, 시계열 분석을 위한 통계 도구
- 📦 **샘플 데이터**: 학습용 데이터셋 즉시 로드 기능
- 🔧 **데이터 전처리**: 결측치 처리, 이상치 탐지, 스케일링 등
- 🚀 **간편한 사용**: 직관적인 API로 빠른 프로토타이핑 지원
- 📈 **교육용 최적화**: 데이터 분석 교육에 특화된 설계

---

## 📦 설치

### PyPI를 통한 설치 (권장)

```bash
pip install hossam
```

### 개발 버전 설치

```bash
git clone https://github.com/leekh4232/hossam-data.git
cd hossam-data
pip install -e .
```

### 요구사항

- Python 3.8 이상
- pandas, numpy, matplotlib, seaborn 등 (자동 설치됨)

---

## 🚀 빠른 시작

### 버전 확인

```python
import hossam
print(hossam.__version__)  # 0.3.0
```

### 샘플 데이터 로드

```python
from hossam import load_data, load_info

# 사용 가능한 데이터셋 목록 확인
datasets = load_info()
print(datasets)

# 특정 키워드로 검색
ad_datasets = load_info(search="AD")

# 데이터셋 로드
df = load_data('AD_SALES')
print(df.head())
```

### 간단한 시각화

```python
from hossam import plot as hs_plot
import pandas as pd
import numpy as np

# 샘플 데이터 생성
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

# 산점도 그리기
hs_plot.hs_scatterplot(df=df, xname='x', yname='y', hue='category', palette='Set1')

# 박스플롯 그리기
hs_plot.hs_boxplot(df=df, xname='category', yname='x', palette='pastel')

# KDE 플롯 그리기
hs_plot.hs_kdeplot(df=df, xname='x', hue='category', fill=True, fill_alpha=0.3)
```

---

## 🎯 주요 기능

### 1. 데이터 로더

학습용 샘플 데이터셋을 빠르게 로드할 수 있습니다.

```python
from hossam import load_data, load_info

# 모든 데이터셋 목록 보기
all_datasets = load_info()

# 키워드로 검색
search_results = load_info(search="regression")

# 데이터 로드
df = load_data('DATASET_NAME')
```

**주요 데이터셋** (예시):
- `AD_SALES`: 광고비와 매출 데이터
- 기타 다양한 회귀, 분류, 시계열 데이터셋

---

### 2. 시각화 모듈 (`hossam.plot`)

#### 기본 플롯

##### 선 그래프 (Line Plot)
```python
from hossam import plot as hs_plot

hs_plot.hs_lineplot(
    df=df,
    xname='time',
    yname='value',
    hue='category',
    marker='o',
    palette='Set1'
)
```

##### 산점도 (Scatter Plot)
```python
hs_plot.hs_scatterplot(
    df=df,
    xname='x',
    yname='y',
    hue='group',
    palette='husl'
)
```

##### 히스토그램 (Histogram)
```python
hs_plot.hs_histplot(
    df=df,
    xname='value',
    hue='category',
    bins=30,
    kde=True,
    palette='Set2'
)
```

#### 분포 시각화

##### 박스플롯 (Box Plot)
```python
hs_plot.hs_boxplot(
    df=df,
    xname='category',
    yname='value',
    orient='v',
    palette='pastel'
)
```

##### 바이올린 플롯 (Violin Plot)
```python
hs_plot.hs_violinplot(
    df=df,
    xname='category',
    yname='value',
    palette='muted'
)
```

##### KDE 플롯 (Kernel Density Estimation)
```python
# 1차원 KDE
hs_plot.hs_kdeplot(
    df=df,
    xname='value',
    hue='category',
    fill=True,
    fill_alpha=0.3,
    palette='Set1'
)

# 2차원 KDE
hs_plot.hs_kdeplot(
    df=df,
    xname='x',
    yname='y',
    palette='coolwarm'
)
```

#### 통계적 플롯

##### 회귀선이 포함된 산점도 (Regression Plot)
```python
hs_plot.hs_regplot(
    df=df,
    xname='x',
    yname='y',
    palette='red'
)
```

##### 선형 모델 플롯 (LM Plot)
```python
hs_plot.hs_lmplot(
    df=df,
    xname='x',
    yname='y',
    hue='category'
)
```

##### 잔차 플롯 (Residual Plot)
```python
from sklearn.linear_model import LinearRegression

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 잔차 플롯
hs_plot.hs_residplot(
    y=y_test,
    y_pred=y_pred,
    lowess=True,  # LOWESS 평활화
    mse=True      # MSE 범위 표시
)
```

##### Q-Q 플롯 (Quantile-Quantile Plot)
```python
residuals = y_test - y_pred
hs_plot.hs_qqplot(y_pred=residuals)
```

##### 혼동 행렬 (Confusion Matrix)
```python
hs_plot.hs_confusion_matrix(
    y=y_test,
    y_pred=y_pred,
    cmap='Blues'
)
```

#### 다변량 분석

##### 쌍 관계 플롯 (Pair Plot)
```python
hs_plot.hs_pairplot(
    df=df,
    diag_kind='kde',
    hue='category',
    palette='Set1'
)
```

##### 공동 분포 플롯 (Joint Plot)
```python
hs_plot.hs_jointplot(
    df=df,
    xname='x',
    yname='y',
    palette='viridis'
)
```

##### 히트맵 (Heatmap)
```python
# 상관계수 행렬
corr_matrix = df.corr()
hs_plot.hs_heatmap(
    data=corr_matrix,
    palette='coolwarm'
)
```

#### 고급 시각화

##### 볼록 껍질 산점도 (Convex Hull)
```python
hs_plot.hs_convex_hull(
    data=df,
    xname='x',
    yname='y',
    hue='cluster',
    palette='Set1'
)
```

##### 100% 누적 막대 그래프 (Stacked Bar)
```python
hs_plot.hs_stackplot(
    df=df,
    xname='category',
    hue='subcategory',
    palette='Pastel1'
)
```

##### P-Value 주석 박스플롯
```python
hs_plot.hs_pvalue1_anotation(
    data=df,
    target='value',
    hue='group',
    pairs=[('A', 'B'), ('B', 'C')],
    test='t-test_ind',
    text_format='star'
)
```

##### 클래스별 분포 (Distribution by Class)
```python
hs_plot.hs_distribution_by_class(
    data=df,
    xnames=['feature1', 'feature2'],
    hue='target',
    type='kde',
    fill=True,
    palette='Set1'
)
```

##### 클래스별 산점도 (Scatter by Class)
```python
hs_plot.hs_scatter_by_class(
    data=df,
    group=[['x', 'y'], ['x', 'z']],
    hue='target',
    outline=True,  # 볼록 껍질 표시
    palette='husl'
)
```

#### 공통 매개변수

모든 시각화 함수는 다음 공통 매개변수를 지원합니다:

- **width**: 캔버스 가로 픽셀 (기본값: 1280)
- **height**: 캔버스 세로 픽셀 (기본값: 720)
- **dpi**: 해상도 (기본값: 200)
- **palette**: 색상 팔레트 ('Set1', 'Set2', 'pastel', 'husl', 'coolwarm' 등)
- **ax**: 외부 Axes 객체 전달 가능
- **callback**: Axes 후처리 콜백 함수

#### 캔버스 크기 조정 예제

```python
# 고해상도 큰 차트
hs_plot.hs_scatterplot(
    df=df,
    xname='x',
    yname='y',
    width=1920,
    height=1080,
    dpi=300
)
```

#### 외부 Axes 사용 예제

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

hs_plot.hs_boxplot(df=df, xname='cat', yname='val', ax=axes[0, 0])
hs_plot.hs_violinplot(df=df, xname='cat', yname='val', ax=axes[0, 1])
hs_plot.hs_histplot(df=df, xname='val', ax=axes[1, 0])
hs_plot.hs_kdeplot(df=df, xname='val', ax=axes[1, 1])

plt.tight_layout()
plt.show()
```

#### 콜백 함수 사용 예제

```python
def custom_style(ax):
    ax.set_title('사용자 정의 제목', fontsize=16, fontweight='bold')
    ax.set_xlabel('X축 레이블', fontsize=12)
    ax.set_ylabel('Y축 레이블', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

hs_plot.hs_scatterplot(
    df=df,
    xname='x',
    yname='y',
    callback=custom_style
)
```

---

### 3. 분석 모듈 (`hossam.analysis`)

데이터 분석을 위한 통계 기능들을 제공합니다.

```python
from hossam import analysis as hs_analysis

# 기술 통계 분석
# 회귀 분석 헬퍼
# 분류 성능 평가
# 시계열 분석
# 등등 (상세 문서 참조)
```

---

### 4. 전처리 모듈 (`hossam.prep`)

데이터 전처리 및 정제를 위한 유틸리티입니다.

```python
from hossam import prep as hs_prep

# 결측치 처리
# 이상치 탐지 및 제거
# 스케일링 및 인코딩
# 등등 (상세 문서 참조)
```

---

### 5. 유틸리티 모듈 (`hossam.util`)

기타 편의 기능들을 제공합니다.

```python
from hossam import util as hs_util

# 다양한 헬퍼 함수들
# 데이터 변환
# 파일 I/O 지원
# 등등 (상세 문서 참조)
```

---

## 📚 의존성

Hossam은 다음 라이브러리들을 사용합니다:

### 핵심 의존성
- **pandas**: 데이터 처리 및 분석
- **numpy**: 수치 계산
- **matplotlib**: 기본 시각화
- **seaborn**: 통계 시각화

### 통계 및 머신러닝
- **scipy**: 과학 계산 및 통계
- **scikit-learn**: 머신러닝 알고리즘
- **statsmodels**: 통계 모델링
- **pingouin**: 통계 분석

### 기타
- **tqdm**: 진행률 표시
- **tabulate**: 표 형식 출력
- **requests**: HTTP 요청
- **openpyxl**, **xlrd**: Excel 파일 지원
- **statannotations**: 통계 주석
- **joblib**: 직렬화 및 병렬 처리

모든 의존성은 `pip install hossam` 시 자동으로 설치됩니다.

---

## 🎓 사용 사례

### 교육용

```python
# 수업에서 빠르게 시각화 시연
from hossam import load_data, plot as hs_plot

df = load_data('SAMPLE_DATA')
hs_plot.hs_pairplot(df=df, hue='target', palette='Set1')
```

### 데이터 탐색

```python
# 빠른 EDA (탐색적 데이터 분석)
from hossam import plot as hs_plot

# 분포 확인
hs_plot.hs_distribution_by_class(
    data=df,
    hue='target',
    type='histkde'
)

# 상관관계 확인
hs_plot.hs_heatmap(data=df.corr(), palette='coolwarm')

# 특징 관계 확인
hs_plot.hs_scatter_by_class(
    data=df,
    hue='target',
    outline=True
)
```

### 모델 평가

```python
from sklearn.linear_model import LinearRegression
from hossam import plot as hs_plot

# 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 잔차 분석
hs_plot.hs_residplot(y=y_test, y_pred=y_pred, lowess=True, mse=True)

# 정규성 검증
hs_plot.hs_qqplot(y_pred=y_test - y_pred)
```

---

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 👨‍🏫 저자

**이광호 (Lee Kwang-Ho)**
- 아이티윌(ITWILL) 강사
- 머신러닝 및 데이터 분석 교육 전문
- Email: leekh4232@gmail.com
- Blog: [https://blog.hossam.kr/](https://blog.hossam.kr/)
- GitHub: [https://github.com/leekh4232](https://github.com/leekh4232)
- Youtube: [https://www.youtube.com/@hossam-codingclub](https://www.youtube.com/@hossam-codingclub)

---

## 🙏 감사의 말

이 라이브러리는 아이티윌에서 진행되는 데이터 분석 교육을 위해 개발되었습니다.

수강생 여러분의 학습에 도움이 되기를 바랍니다.

---

## 📞 지원 및 문의

- 이슈 리포트: [GitHub Issues](https://github.com/leekh4232/hossam-data/issues)
- 이메일: leekh4232@gmail.com

---

**Happy Data Analysis! 📊✨**
