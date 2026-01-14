# Hossam MCP 빠른 시작 가이드

> `pip install hossam` 설치 후 바로 사용하는 가이드입니다.

---

## 📦 설치

```bash
pip install hossam
```

**필수 요구사항:**
- Python 3.8+
- pandas, matplotlib, seaborn, scipy, statsmodels

---

## 🎯 기본 사용 (Python 라이브러리)

### 1단계: 데이터 로드

```python
from hossam import *

# ✅ 명목형 변수를 categories 파라미터로 반드시 지정
df = hs_util.load_data("insurance", categories=["sex", "smoker", "region"])
```

**주의**: `astype('category')` 수동 변환 금지 - 반드시 `categories` 파라미터 사용

### 2단계: 기술통계

```python
# 연속형 변수
stats = hs_stats.describe(df, "charges")

# 명목형 변수
cat_stats = hs_stats.category_describe(df)

# 결측치 확인
missing = hs_stats.missing_values(df)

# 이상치 확인
outliers = hs_stats.outlier_table(df, "charges")
```

### 3단계: 시각화

```python
# 분포 시각화
hs_plot.distribution_plot(df, "charges")

# 범주별 막대그래프
hs_plot.countplot(df, "sex")

# 산점도
hs_plot.scatterplot(df, xname="bmi", yname="charges", hue="smoker")

# 상관계수 히트맵
hs_plot.heatmap(df)
```

### 4단계: 전처리

```python
# 표준화
scaled = hs_prep.standard_scaler(df, features=["age", "bmi", "charges"])

# 정규화
normalized = hs_prep.minmax_scaler(df, features=["age", "bmi", "charges"])

# One-Hot 인코딩
encoded = hs_prep.get_dummies(df, fields=["sex", "smoker", "region"])

# 이상치 제거
df_clean = hs_prep.replace_outliner(df, "charges", method="remove")
```

---

## 🤖 VSCode + GitHub Copilot 사용 (권장)

### 1단계: VSCode 확장 설치

1. VSCode 열기
2. 확장 마켓플레이스 (`Cmd+Shift+X`)
3. "GitHub Copilot Chat" 검색 및 설치
4. VSCode 재시작

### 2단계: 프로젝트 설정

프로젝트 루트에 `.vscode/settings.json` 생성:

```json
{
  "github.copilot.chat.codeGeneration.instructions": [
    "You are an expert data analysis assistant using the Hossam library.",
    "Always follow these rules:",
    "1. Use categories parameter when loading data: hs_util.load_data('data', categories=['col1', 'col2'])",
    "2. Use module names: hs_stats.*, hs_plot.*, hs_prep.*",
    "3. Never use astype('category') manually"
  ]
}
```

### 3단계: Copilot Chat 사용

Copilot Chat 열기 (`Cmd+L`) 후:

```
CSV 파일을 로드해서 charges의 분포를 보여줄래?
```

Copilot이 자동으로:
- ✅ 정확한 함수명 사용
- ✅ categories 파라미터 적용
- ✅ 코드 생성 또는 실행

---

## 📚 주요 도구 목록

### hs_stats (통계)
| 함수 | 설명 |
|------|------|
| `missing_values()` | 결측치 분석 |
| `outlier_table()` | 이상치 경계값 |
| `category_table()` | 범주형 빈도 |
| `category_describe()` | 범주형 요약 |
| `describe()` | 확장 기술통계 |
| `normal_test()` | 정규성 검정 |
| `correlation()` | 상관계수 |

### hs_plot (시각화)
| 함수 | 설명 |
|------|------|
| `distribution_plot()` | KDE + 상자그림 |
| `countplot()` | 범주별 빈도 |
| `boxplot()` | 상자그림 |
| `histplot()` | 히스토그램 |
| `scatterplot()` | 산점도 |
| `heatmap()` | 상관계수 히트맵 |

### hs_prep (전처리)
| 함수 | 설명 |
|------|------|
| `standard_scaler()` | Z-Score 표준화 |
| `minmax_scaler()` | Min-Max 정규화 |
| `get_dummies()` | One-Hot 인코딩 |
| `replace_outliner()` | 이상치 처리 |

### hs_util (유틸리티)
| 함수 | 설명 |
|------|------|
| `load_data()` | 데이터 로드 |
| `load_info()` | 샘플 데이터셋 목록 |

### hs_gis (지리정보)
| 함수 | 설명 |
|------|------|
| `geocode()` | 주소 → 위경도 |
| `load_shape()` | Shapefile 로드 |

### hs_timeserise (시계열)
| 함수 | 설명 |
|------|------|
| `diff()` | 자동 차분 |
| `rolling()` | 이동평균 |

---

## 💡 코딩 스타일

### ✅ 권장

```python
from hossam import *

# 데이터 로드
df = hs_util.load_data("insurance", categories=["sex", "smoker", "region"])

# 통계
stats = hs_stats.describe(df, "charges")

# 전처리
scaled = hs_prep.standard_scaler(df, features=["age", "bmi"])

# 시각화
hs_plot.distribution_plot(df, "charges")
```

### ❌ 피해야 할 것

```python
# 비권장 1: 개별 import
from hossam.hs_stats import describe
from hossam.hs_plot import distribution_plot

# 비권장 2: 수동 카테고리 변환
df[cols] = df[cols].astype("category")

# 비권장 3: 모듈명 생략
describe(df, "charges")  # ❌
hs_stats.describe(df, "charges")  # ✅
```

---

## 🐍 Jupyter Notebook 사용

```python
# 셀 1: 라이브러리 import
from hossam import *

# 셀 2: 데이터 로드
df = hs_util.load_data("insurance", categories=["sex", "smoker", "region"])
df.info()

# 셀 3: 기술통계
stats = hs_stats.describe(df, "charges")
print(stats)

# 셀 4: 시각화
hs_plot.distribution_plot(df, "charges")

# 셀 5: 마크다운 셀 (분석 결과 정리)
# ## 📊 분석 결과
# - 평균: $13,270
# - 분포: 우측 꼬리 (왜도: 1.52)
# - 이상치: 139개 (10.4%)
```

---

## ⚡ 자주 하는 질문

### Q1: "load_data() 함수를 찾을 수 없다"

```python
# ❌ 잘못된 방법
df = load_data("insurance")

# ✅ 올바른 방법
from hossam import *
df = hs_util.load_data("insurance", categories=[...])
```

### Q2: "categories 파라미터가 뭐에요?"

명목형(범주형) 변수를 자동으로 인식하도록 지정하는 파라미터입니다:

```python
# insurance 데이터셋의 경우
df = hs_util.load_data(
    "insurance",
    categories=["sex", "smoker", "region"]  # 이 3개 컬럼이 범주형
)

# 지정하지 않으면 문자열로 인식됨
df = hs_util.load_data("insurance")  # ❌ sex, smoker, region이 object 타입
```

### Q3: "시각화 결과를 파일로 저장하고 싶어요"

```python
hs_plot.distribution_plot(df, "charges", save_path="./output.png")

hs_plot.scatterplot(
    df,
    xname="bmi",
    yname="charges",
    hue="smoker",
    save_path="./scatter.png"
)
```

### Q4: "코드만 생성하고 실행은 안 하고 싶어요"

GitHub Copilot Chat에서 요청할 때:

```
CSV를 로드하는 코드만 보여줄래? (실행 안 함)
```

또는 코드를 Copilot이 반환하면 직접 복사해서 사용하면 됩니다.

---

## 📖 더 알아보기

- **도구 목록 상세**: `hs_mcp_list_tools()` (Copilot Chat 도구)
- **사용 가이드**: `hs_mcp_usage_guide()` (Copilot Chat 도구)
- **API 문서**: `pip show hossam` 후 설치 위치의 `docs/` 폴더

---

## 🔗 관련 링크

- GitHub: https://github.com/leekh4232/hossam-py
- PyPI: https://pypi.org/project/hossam
- 문제 보고: https://github.com/leekh4232/hossam-py/issues
