# HossamPy API Reference

<a id="hs_classroom"></a>

## hs\_classroom

<a id="hs_classroom.cluster_students"></a>

### cluster\_students

```python
def cluster_students(df,
                     n_groups: int,
                     score_cols: list = None,
                     interest_col: str = None,
                     max_iter: int = 200,
                     score_metric: str = 'total') -> DataFrame
```

학생들을 균형잡힌 조로 편성하는 함수.

관심사 기반 1차 군집과 점수/인원 균형 조정을 통해 동질성 있고
균형잡힌 조를 구성합니다.

**Arguments**:

- `df` - 학생 정보를 담은 데이터프레임 또는 엑셀/CSV 파일 경로.
  데이터프레임의 경우: 반드시 '학생번호' 컬럼 포함.
  파일 경로의 경우: 자동으로 hs_load_data 함수를 사용하여 로드.
  interest_col이 지정된 경우 해당 컬럼 필수.
  score_cols이 지정된 경우 해당 컬럼들 필수.
- `n_groups` - 목표 조의 개수.
- `score_cols` - 성적 계산에 사용할 점수 컬럼명 리스트.
- `예` - ['과목1점수', '과목2점수', '과목3점수']
  None일 경우 점수 기반 균형 조정을 하지 않습니다. 기본값: None
- `interest_col` - 관심사 정보가 있는 컬럼명.
  None일 경우 관심사 기반 군집화를 하지 않습니다. 기본값: None
- `max_iter` - 균형 조정 최대 반복 횟수. 기본값: 200
- `score_metric` - 점수 기준 선택 ('total' 또는 'average').
  'total'이면 총점, 'average'이면 평균점수 기준. 기본값: 'total'
  

**Returns**:

  '조' 컬럼이 추가된 데이터프레임. 관심사와 점수로 균형잡힌 조 배치 완료.
  

**Raises**:

- `ValueError` - 필수 컬럼이 없거나 입력값이 유효하지 않은 경우.
  

**Examples**:

```python
df = read_csv('students.csv')

from hossam import *
result = hs_classroom.cluster_students(
            df=df,
            n_groups=5,
            score_cols=['국어', '영어', '수학'],
            interest_col='관심사')
```

<a id="hs_classroom.report_summary"></a>

### report\_summary

```python
def report_summary(df: DataFrame,
                   interest_col: str = None,
                   width: int = config.width,
                   height: int = config.height,
                   dpi: int = config.dpi) -> None
```

조 편성 결과의 요약 통계를 시각화합니다.

조별 인원 분포, 관심사 분포, 평균점수 분포를 나타냅니다.

**Arguments**:

- `df` _DataFrame_ - cluster_students 함수의 반환 결과 데이터프레임.
- `interest_col` _str_ - 관심사 컬럼명
- `width` _int_ - 그래프 넓이. 기본값: config.width
- `height` _int_ - 그래프 높이. 기본값: config.height
- `dpi` _int_ - 그래프 해상도. 기본값: config.dpi
  

**Examples**:

```python
from hossam import *
df_result = hs_classroom.cluster_students(df, n_groups=5, score_cols=['국어', '영어', '수학'])
hs_classroom.report_summary(df_result)
```

<a id="hs_classroom.report_kde"></a>

### report\_kde

```python
def report_kde(df: DataFrame,
               metric: str = 'average',
               width: int = config.width,
               height: int = config.height,
               dpi: int = config.dpi) -> None
```

조별 점수 분포를 KDE(Kernel Density Estimation)로 시각화합니다.

각 조의 점수 분포를 커널 밀도 추정으로 표시하고 평균 및 95% 신뢰구간을 나타냅니다.

**Arguments**:

- `df` - cluster_students 함수의 반환 결과 데이터프레임.
- `metric` - 점수 기준 선택 ('total' 또는 'average').
  'total'이면 총점, 'average'이면 평균점수. 기본값: 'average'
- `width` - 그래프 넓이. 기본값: config.width
- `height` - 그래프 높이. 기본값: config.height
- `dpi` - 그래프 해상도. 기본값: config.dpi
  

**Examples**:

```python
from hossam import *
df_result = hs_classroom.cluster_students(df, n_groups=5, score_cols=['국어', '영어', '수학'])
hs_classroom.report_kde(df_result, metric='average')
```

<a id="hs_classroom.group_summary"></a>

### group\_summary

```python
def group_summary(df: DataFrame, name_col: str = '학생이름') -> DataFrame
```

조별로 학생 목록과 평균 점수를 요약합니다.

**Arguments**:

- `df` - cluster_students 함수의 반환 결과 데이터프레임.
  '조' 컬럼이 필수로 포함되어야 함.
- `name_col` - 학생 이름이 들어있는 컬럼명. 기본값: '학생이름'
  

**Returns**:

  조별 요약 정보가 담긴 데이터프레임.
- `컬럼` - '조', '학생', '총점평균', '평균점수평균'
  

**Examples**:

```python
from hossam import *
df_result = hs_classroom.cluster_students(df, n_groups=5, score_cols=['국어', '영어', '수학'])
summary = hs_classroom.group_summary(df_result, name_col='이름')
print(summary)
```

<a id="hs_classroom.analyze_classroom"></a>

### analyze\_classroom

```python
def analyze_classroom(df,
                      n_groups: int,
                      score_cols: list = None,
                      interest_col: str = None,
                      max_iter: int = 200,
                      score_metric: str = 'average',
                      name_col: str = '학생이름',
                      show_summary: bool = True,
                      show_kde: bool = True) -> DataFrame
```

학생 조 편성부터 시각화까지 전체 프로세스를 일괄 실행합니다.

다음 순서로 실행됩니다:
1. cluster_students: 학생들을 균형잡힌 조로 편성
2. group_summary: 조별 학생 목록과 평균 점수 요약
3. report_summary: 조 편성 결과 요약 시각화 (선택적)
4. report_kde: 조별 점수 분포 KDE 시각화 (선택적)

**Arguments**:

- `df` - 학생 정보를 담은 데이터프레임 또는 파일 경로.
- `n_groups` - 목표 조의 개수.
- `score_cols` - 성적 계산에 사용할 점수 컬럼명 리스트. 기본값: None
- `interest_col` - 관심사 정보가 있는 컬럼명. 기본값: None
- `max_iter` - 균형 조정 최대 반복 횟수. 기본값: 200
- `score_metric` - 점수 기준 선택 ('total' 또는 'average'). 기본값: 'average'
- `name_col` - 학생 이름 컬럼명. 기본값: '학생이름'
- `show_summary` - 요약 시각화 표시 여부. 기본값: True
- `show_kde` - KDE 시각화 표시 여부. 기본값: True
  

**Returns**:

  조별 요약 정보 (group_summary의 결과).
  

**Examples**:

```python
from hossam import *
summary = hs_classroom.analyze_classroom(df='students.csv',
                                         n_groups=5,
                                         score_cols=['국어', '영어', '수학'],
                                         interest_col='관심사',
                                         name_col='이름')
print(summary)
```

<a id="hs_prep"></a>

## hs\_prep

<a id="hs_prep.standard_scaler"></a>

### standard\_scaler

```python
def standard_scaler(data: any,
                    yname: str | None = None,
                    save_path: str | None = None,
                    load_path: str | None = None) -> DataFrame
```

연속형 변수에 대해 Standard Scaling을 수행한다.

- DataFrame 입력 시: 비수치형/종속변수를 분리한 후 스케일링하고 다시 합칩니다.
- 배열 입력 시: 그대로 스케일링된 ndarray를 반환합니다.
- `load_path`가 주어지면 기존 스케일러를 재사용하고, `save_path`가 주어지면 학습된 스케일러를 저장합니다.

**Arguments**:

- `data` _DataFrame | ndarray_ - 스케일링할 데이터.
- `yname` _str | None_ - 종속변수 컬럼명. 분리하지 않으려면 None.
- `save_path` _str | None_ - 학습된 스케일러 저장 경로.
- `load_path` _str | None_ - 기존 스케일러 로드 경로.
  

**Returns**:

  DataFrame | ndarray: 스케일링된 데이터(입력 타입과 동일).
  

**Examples**:

```python
from hossam import *
std_df = hs_prep.standard_scaler(df, yname="y", save_path="std.pkl")
```

<a id="hs_prep.minmax_scaler"></a>

### minmax\_scaler

```python
def minmax_scaler(data: any,
                  yname: str | None = None,
                  save_path: str | None = None,
                  load_path: str | None = None) -> DataFrame
```

연속형 변수에 대해 MinMax Scaling을 수행한다.

DataFrame은 비수치/종속변수를 분리 후 스케일링하고 재결합하며, 배열 입력은 그대로 ndarray를 반환한다.
`load_path` 제공 시 기존 스케일러를 사용하고, `save_path` 제공 시 학습 스케일러를 저장한다.

**Arguments**:

- `data` _DataFrame | ndarray_ - 스케일링할 데이터.
- `yname` _str | None_ - 종속변수 컬럼명. 분리하지 않으려면 None.
- `save_path` _str | None_ - 학습된 스케일러 저장 경로.
- `load_path` _str | None_ - 기존 스케일러 로드 경로.
  

**Returns**:

  DataFrame | ndarray: 스케일링된 데이터(입력 타입과 동일).
  

**Examples**:

```python
from hossam import *
mm_df = hs_prep.minmax_scaler(df, yname="y")
```

<a id="hs_prep.set_category"></a>

### set\_category

```python
def set_category(data: DataFrame, *args: str) -> DataFrame
```

카테고리 데이터를 설정한다.

**Arguments**:

- `data` _DataFrame_ - 데이터프레임 객체
- `*args` _str_ - 컬럼명 목록
  

**Returns**:

- `DataFrame` - 카테고리 설정된 데이터프레임

<a id="hs_prep.unmelt"></a>

### unmelt

```python
def unmelt(data: DataFrame,
           id_vars: str = "class",
           value_vars: str = "values") -> DataFrame
```

두 개의 컬럼으로 구성된 데이터프레임에서 하나는 명목형, 나머지는 연속형일 경우
명목형 변수의 값에 따라 고유한 변수를 갖는 데이터프레임으로 변환한다.

각 그룹의 데이터 길이가 다를 경우 짧은 쪽에 NaN을 채워 동일한 길이로 맞춥니다.
이는 독립표본 t-검정(ttest_ind) 등의 분석을 위한 데이터 준비에 유용합니다.

**Arguments**:

- `data` _DataFrame_ - 데이터프레임
- `id_vars` _str, optional_ - 명목형 변수의 컬럼명. Defaults to 'class'.
- `value_vars` _str, optional_ - 연속형 변수의 컬럼명. Defaults to 'values'.
  

**Returns**:

- `DataFrame` - 변환된 데이터프레임 (각 그룹이 개별 컬럼으로 구성)
  

**Examples**:

```python
df = pd.DataFrame({
    'group': ['A', 'A', 'B', 'B', 'B'],
    'value': [1, 2, 3, 4, 5]
})

from hossam import *
result = hs_prep.unmelt(df, id_vars='group', value_vars='value')
# 결과: A 컬럼에는 [1, 2, NaN], B 컬럼에는 [3, 4, 5]
```

<a id="hs_prep.outlier_table"></a>

### outlier\_table

```python
def outlier_table(data: DataFrame, *fields: str) -> DataFrame
```

수치형 컬럼에 대한 사분위수 및 IQR 기반 이상치 경계를 계산한다.

전달된 `fields`가 없으면 데이터프레임의 모든 수치형 컬럼을 대상으로 한다.
결측치는 제외하고 사분위수를 계산한다.

**Arguments**:

- `data` _DataFrame_ - 분석할 데이터프레임.
- `*fields` _str_ - 대상 컬럼명(들). 생략 시 모든 수치형 컬럼 대상.
  

**Returns**:

- `DataFrame` - Q1, Q2(중앙값), Q3, IQR, 하한, 상한을 포함한 통계표.
  

**Examples**:

  from hossam import *
  hs_prep.outlier_table(df, "value")

<a id="hs_prep.replace_outliner"></a>

### replace\_outliner

```python
def replace_outliner(data: DataFrame,
                     method: str = "nan",
                     *fields: str) -> DataFrame
```

이상치 경계값을 넘어가는 데이터를 경계값으로 대체한다.

**Arguments**:

- `data` _DataFrame_ - 데이터프레임
- `method` _str_ - 대체 방법
  - nan: 결측치 대체
  - outline: 경계값 대체
  - mean: 평균 대체
  - most: 최빈값 대체
  - median: 중앙값 대체
- `*fields` _str_ - 컬럼명 목록
  

**Returns**:

- `DataFrame` - 이상치가 경계값으로 대체된 데이터 프레임

<a id="hs_prep.drop_outliner"></a>

### drop\_outliner

```python
def drop_outliner(data: DataFrame, *fields: str) -> DataFrame
```

이상치를 결측치로 변환한 후 모두 삭제한다.

**Arguments**:

- `data` _DataFrame_ - 데이터프레임
- `*fields` _str_ - 컬럼명 목록
  

**Returns**:

- `DataFrame` - 이상치가 삭제된 데이터프레임

<a id="hs_prep.get_dummies"></a>

### get\_dummies

```python
def get_dummies(data: DataFrame,
                *args: str,
                drop_first=True,
                dtype="int") -> DataFrame
```

명목형 변수를 더미 변수로 변환한다.

컬럼명을 지정하면 그 컬럼들만 더미 변수로 변환하고,
지정하지 않으면 숫자 타입이 아닌 모든 컬럼(문자열/명목형)을 자동으로 더미 변수로 변환한다.

**Arguments**:

- `data` _DataFrame_ - 데이터프레임
- `*args` _str_ - 변환할 컬럼명 목록. 지정하지 않으면 숫자형이 아닌 모든 컬럼 자동 선택.
- `drop_first` _bool, optional_ - 첫 번째 더미 변수 제거 여부. 기본값 True.
- `dtype` _str, optional_ - 더미 변수 데이터 타입. 기본값 "int".
  

**Returns**:

- `DataFrame` - 더미 변수로 변환된 데이터프레임
  

**Examples**:

```python
from hossam import *
# 전체 비숫자 컬럼 자동 변환
result = hs_prep.get_dummies(df)
# 특정 컬럼만 변환
result = hs_prep.get_dummies(df, 'cut', 'color', 'clarity')
# 옵션 지정
result = hs_prep.get_dummies(df, 'col1', drop_first=False, dtype='bool')
```

<a id="hs_prep.labelling"></a>

### labelling

```python
def labelling(data: DataFrame, *fields: str) -> DataFrame
```

명목형 변수를 라벨링한다.

**Arguments**:

- `data` _DataFrame_ - 데이터프레임
- `*fields` _str_ - 명목형 컬럼 목록
  

**Returns**:

- `DataFrame` - 라벨링된 데이터프레임

<a id="hs_prep.bin_continuous"></a>

### bin\_continuous

```python
def bin_continuous(data: DataFrame,
                   field: str,
                   method: str = "natural_breaks",
                   bins: int | list[float] | None = None,
                   labels: list[str] | None = None,
                   new_col: str | None = None,
                   is_log_transformed: bool = False,
                   apply_labels: bool = True) -> DataFrame
```

연속형 변수를 다양한 알고리즘으로 구간화해 명목형 파생변수를 추가한다.

지원 방법:
- "natural_breaks"(기본): Jenks 자연 구간화. jenkspy 미사용 시 quantile로 대체
기본 라벨: "X-Y" 형식 (예: "18-30", "30-40")
- "quantile"/"qcut"/"equal_freq": 분위수 기반 동빈도
기본 라벨: "X-Y" 형식
- "equal_width"/"uniform": 동일 간격
기본 라벨: "X-Y" 형식
- "std": 평균±표준편차를 경계로 4구간 생성
라벨: "low", "mid_low", "mid_high", "high"
- "lifecourse"/"life_stage": 생애주기 5단계
라벨: "아동", "청소년", "청년", "중년", "노년" (경계: 0, 13, 19, 40, 65)
- "age_decade": 10대 단위 연령대
라벨: "아동", "10대", "20대", "30대", "40대", "50대", "60대 이상"
- "health_band"/"policy_band": 의료비 위험도 기반 연령대
라벨: "18-29", "30-39", "40-49", "50-64", "65+"
- 커스텀 구간: bins에 경계 리스트 전달 (예: [0, 30, 50, 100])

**Arguments**:

- `data` _DataFrame_ - 입력 데이터프레임
- `field` _str_ - 구간화할 연속형 변수명
- `method` _str_ - 구간화 알고리즘 키워드 (기본값: "natural_breaks")
  bins (int|list[float]|None):
  - int: 생성할 구간 개수 (quantile, equal_width, natural_breaks에서 사용)
  - list: 경계값 리스트 (커스텀 구간화)
  - None: 기본값 사용 (quantile/equal_width는 4~5, natural_breaks는 5)
- `labels` _list[str]|None_ - 구간 레이블 목록
  - None: method별 기본 라벨 자동 생성
  - list: 사용자 정의 라벨 (구간 개수와 일치해야 함)
- `new_col` _str|None_ - 생성할 컬럼명
  - None: f"{field}_bin" 사용 (예: "age_bin")
- `is_log_transformed` _bool_ - 대상 컬럼이 로그 변환되어 있는지 여부
  - True: 지정된 컬럼을 역변환(exp)한 후 구간화
  - False: 원래 값 그대로 구간화 (기본값)
- `apply_labels` _bool_ - 구간에 숫자 인덱스를 적용할지 여부
  - True: 숫자 인덱스 사용 (0, 1, 2, 3, ...) (기본값)
  - False: 문자 라벨 적용 (예: "18~30", "아동")
  

**Returns**:

- `DataFrame` - 원본에 구간화된 명목형 컬럼이 추가된 데이터프레임
  

**Examples**:

```python
from hossam import *

# 동일 간격으로 5개 구간 생성 (숫자 인덱스):
df = pd.DataFrame({'age': [20, 35, 50, 65]})
result = hs_prep.bin_continuous(df, 'age', method='equal_width', bins=5)
print(result['age_bin'])  # 0, 1, 2, ... (숫자 인덱스)

# 문자 레이블 사용:
result = hs_prep.bin_continuous(df, 'age', method='equal_width', bins=5, apply_labels=False)
print(result['age_bin'])  # 20~30, 30~40, ... (문자 레이블)

# 생애주기 기반 구간화:
result = hs_prep.bin_continuous(df, 'age', method='lifecourse')
print(result['age_bin'])  # 0, 1, 2, 3, 4 (숫자 인덱스)

# 생애주기 문자 레이블:
result = hs_prep.bin_continuous(df, 'age', method='lifecourse', apply_labels=False)
print(result['age_bin'])  # 아동, 청소년, 청년, 중년, 노년

# 의료비 위험도 기반 연령대 (health_band):
result = hs_prep.bin_continuous(df, 'age', method='health_band', apply_labels=False)
print(result['age_bin'])  # 18-29, 30-39, 40-49, 50-64, 65+

# 로그 변환된 컬럼 역변환 후 구간화:
df_log = pd.DataFrame({'charges_log': [np.log(1000), np.log(5000), np.log(50000)]})
result = hs_prep.bin_continuous(df_log, 'charges_log', method='equal_width', is_log_transformed=True)
print(result['charges_log_bin'])  # 0, 1, 2 (숫자 인덱스)
```

<a id="hs_prep.log_transform"></a>

### log\_transform

```python
def log_transform(data: DataFrame, *fields: str) -> DataFrame
```

수치형 변수에 대해 로그 변환을 수행한다.

자연로그(ln)를 사용하여 변환하며, 0 또는 음수 값이 있을 경우
최소값을 기준으로 보정(shift)을 적용한다.

**Arguments**:

- `data` _DataFrame_ - 변환할 데이터프레임.
- `*fields` _str_ - 변환할 컬럼명 목록. 지정하지 않으면 모든 수치형 컬럼을 처리.
  

**Returns**:

- `DataFrame` - 로그 변환된 데이터프레임.
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame
df = DataFrame({'x': [1, 10, 100], 'y': [2, 20, 200], 'z': ['a', 'b', 'c']})

# 전체 수치형 컬럼에 대한 로그 변환:
result = hs_prep.log_transform(df)
print(result)

# 특정 컬럼만 변환:
result = hs_prep.log_transform(df, 'x', 'y')
print(result)
```
  

**Notes**:

  - 수치형이 아닌 컬럼은 자동으로 제외됩니다.
  - 0 또는 음수 값이 있는 경우 자동으로 보정됩니다.
  - 변환 공식: log(x + shift), 여기서 shift = 1 - min(x) (min(x) <= 0인 경우)

<a id="hs_prep.add_interaction"></a>

### add\_interaction

```python
def add_interaction(data: DataFrame,
                    pairs: list[tuple[str, str]] | None = None) -> DataFrame
```

데이터프레임에 상호작용(interaction) 항을 추가한다.

수치형 및 명목형 변수 간의 상호작용 항을 생성하여 데이터프레임에 추가한다.
- `수치형 * 수치형`: 두 변수의 곱셈 (col1*col2)
- `수치형 * 명목형`: 명목형의 각 카테고리별 수치형 변수 생성 (col1*col2_category)
- `명목형 * 명목형`: 두 명목형을 결합한 새 명목형 변수 생성 (col1_col2)

**Arguments**:

- `data` _DataFrame_ - 원본 데이터프레임.
- `pairs` _list[tuple[str, str]], optional_ - 직접 지정할 교호작용 쌍의 리스트.
- `예` - [("age", "gender"), ("color", "cut")]
  None이면 모든 수치형 컬럼의 2-way 상호작용을 생성.
  

**Returns**:

- `DataFrame` - 상호작용 항이 추가된 새 데이터프레임.
  

**Examples**:

```python
from hossam import *
from padas import DataFrame

# 수치형 변수들의 상호작용:
df = DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6]})
result = hs_prep.add_interaction(df)
print(result.columns)  # x1, x2, x1*x2

# 수치형과 명목형의 상호작용:
df = DataFrame({'age': [20, 30, 40], 'gender': ['M', 'F', 'M']})
result = hs_prep.add_interaction(df, pairs=[('age', 'gender')])
print(result.columns)  # age, gender, age*gender_M, age*gender_F

# 명목형끼리의 상호작용:
df = DataFrame({'color': ['R', 'G', 'B'], 'cut': ['A', 'B', 'A']})
result = hs_prep.add_interaction(df, pairs=[('color', 'cut')])
print(result.columns)  # color, cut, color_cut
```

<a id="hs_stats"></a>

## hs\_stats

<a id="hs_stats.missing_values"></a>

### missing\_values

```python
def missing_values(data: DataFrame, *fields: str)
```

데이터프레임의 결측치 정보를 컬럼 단위로 반환한다.

각 컬럼의 결측치 수와 전체 대비 비율을 계산하여 데이터프레임으로 반환한다.

**Arguments**:

- `data` _DataFrame_ - 분석 대상 데이터프레임.
- `*fields` _str_ - 분석할 컬럼명 목록. 지정하지 않으면 모든 컬럼을 처리.
  

**Returns**:

- `DataFrame` - 각 컬럼별 결측치 정보를 포함한 데이터프레임.
  인덱스는 FIELD(컬럼명)이며, 다음 컬럼을 포함:
  
  - missing_count (int): 결측치의 수
  - missing_rate (float): 전체 행에서 결측치의 비율(%)
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame

# 전체 컬럼에 대한 결측치 확인:
df = DataFrame({'x': [1, 2, None, 4], 'y': [10, None, None, 40]})
result = hs_stats.missing_values(df)
print(result)

# 특정 컬럼만 분석:
result = hs_stats.missing_values(df, 'x', 'y')
print(result)
```

<a id="hs_stats.outlier_table"></a>

### outlier\_table

```python
def outlier_table(data: DataFrame, *fields: str)
```

데이터프레임의 사분위수와 이상치 경계값, 왜도를 구한다.

Tukey의 방법을 사용하여 각 숫자형 컬럼에 대한 사분위수(Q1, Q2, Q3)와
이상치 판단을 위한 하한(DOWN)과 상한(UP) 경계값을 계산한다.
함수 호출 전 상자그림을 통해 이상치가 확인된 필드에 대해서만 처리하는 것이 좋다.

**Arguments**:

- `data` _DataFrame_ - 분석 대상 데이터프레임.
- `*fields` _str_ - 분석할 컬럼명 목록. 지정하지 않으면 모든 숫자형 컬럼을 처리.
  

**Returns**:

- `DataFrame` - 각 필드별 사분위수 및 이상치 경계값을 포함한 데이터프레임.
  인덱스는 FIELD(컬럼명)이며, 다음 컬럼을 포함:
  
  - q1 (float): 제1사분위수 (25th percentile)
  - q2 (float): 제2사분위수 (중앙값, 50th percentile)
  - q3 (float): 제3사분위수 (75th percentile)
  - iqr (float): 사분위 범위 (q3 - q1)
  - up (float): 이상치 상한 경계값 (q3 + 1.5 * iqr)
  - down (float): 이상치 하한 경계값 (q1 - 1.5 * iqr)
  - min (float): 최소값
  - max (float): 최대값
  - skew (float): 왜도
  - outlier_count (int): 이상치 개수
  - outlier_rate (float): 이상치 비율(%)
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame

df = DataFrame({'x': [1, 2, 3, 100], 'y': [10, 20, 30, 40]})

# 전체 숫자형 컬럼에 대한 이상치 경계 확인:
result = hs_stats.outlier_table(df)
print(result)

# 특정 컬럼만 분석:
result = hs_stats.outlier_table(df, 'x', 'y')
print(result[['q1', 'q3', 'up', 'down']])
```
  

**Notes**:

  - DOWN 미만이거나 UP 초과인 값은 이상치(outlier)로 간주됩니다.
  - 숫자형이 아닌 컬럼은 자동으로 제외됩니다.
  - Tukey의 1.5 * IQR 규칙을 사용합니다 (상자그림의 표준 방법).

<a id="hs_stats.describe"></a>

### describe

```python
def describe(data: DataFrame, *fields: str, columns: list | None = None)
```

데이터프레임의 연속형 변수의 단위 및 현실성을 평가하기 위해 확장된 기술통계량을 반환한다.

각 연속형(숫자형) 컬럼의 기술통계량(describe)을 구하고, 이에 사분위수 범위(IQR),
이상치 경계값(UP, DOWN), 왜도(skew), 이상치 개수 및 비율, 분포 특성, 로그변환 필요성을
추가하여 반환한다.

**Arguments**:

- `data` _DataFrame_ - 분석 대상 데이터프레임.
- `*fields` _str_ - 분석할 컬럼명 목록. 지정하지 않으면 모든 숫자형 컬럼을 처리.
- `columns` _list, optional_ - 반환할 통계량 컬럼 목록. None이면 모든 통계량 반환.
  

**Returns**:

- `DataFrame` - 각 필드별 확장된 기술통계량을 포함한 데이터프레임.
  행은 다음과 같은 통계량을 포함:
  
  - count (float): 비결측치의 수
  - mean (float): 평균값
  - std (float): 표준편차
  - min (float): 최소값
  - 25% (float): 제1사분위수 (Q1)
  - 50% (float): 제2사분위수 (중앙값)
  - 75% (float): 제3사분위수 (Q3)
  - max (float): 최대값
  - iqr (float): 사분위 범위 (Q3 - Q1)
  - up (float): 이상치 상한 경계값 (Q3 + 1.5 * IQR)
  - down (float): 이상치 하한 경계값 (Q1 - 1.5 * IQR)
  - skew (float): 왜도
  - outlier_count (int): 이상치 개수
  - outlier_rate (float): 이상치 비율(%)
  - dist (str): 분포 특성 ("극단 우측 꼬리", "거의 대칭" 등)
  - log_need (str): 로그변환 필요성 ("높음", "중간", "낮음")
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame

df = DataFrame({
    'x': [1, 2, 3, 4, 5, 100],
    'y': [10, 20, 30, 40, 50, 60],
    'z': ['a', 'b', 'c', 'd', 'e', 'f']
})

# 전체 숫자형 컬럼에 대한 확장된 기술통계:
result = hs_stats.describe(df)
print(result)

# 특정 컬럼만 분석:
result = hs_stats.describe(df, 'x', 'y')
print(result)
```
  

**Notes**:

  - 숫자형이 아닌 컬럼은 자동으로 제외됩니다.
  - 결과는 필드(컬럼)가 행으로, 통계량이 열로 구성됩니다.
  - Tukey의 1.5 * IQR 규칙을 사용하여 이상치를 판정합니다.
  - 분포 특성은 왜도 값으로 판정합니다.
  - 로그변환 필요성은 왜도의 절댓값 크기로 판정합니다.

<a id="hs_stats.category_describe"></a>

### category\_describe

```python
def category_describe(data: DataFrame, *fields: str)
```

데이터프레임의 명목형(범주형) 변수에 대한 분포 편향을 요약한다.

각 명목형 컬럼의 최다 범주와 최소 범주의 정보를 요약하여 데이터프레임으로 반환한다.

**Arguments**:

- `data` _DataFrame_ - 분석 대상 데이터프레임.
- `*fields` _str_ - 분석할 컬럼명 목록. 지정하지 않으면 모든 명목형 컬럼을 처리.
  

**Returns**:

  tuple[DataFrame, DataFrame]: 각 컬럼별 최다/최소 범주 정보를 포함한 데이터프레임과
  각 범주별 빈도/비율 정보를 포함한 데이터프레임을 튜플로 반환.
  다음 컬럼을 포함:
  
  - 변수 (str): 컬럼명
  - 최다_범주: 가장 빈도가 높은 범주값
  - 최다_비율(%) (float): 최다 범주의 비율
  - 최소_범주: 가장 빈도가 낮은 범주값
  - 최소_비율(%) (float): 최소 범주의 비율
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame

df = DataFrame({
    'cut': ['Ideal', 'Premium', 'Good', 'Ideal', 'Premium'],
    'color': ['E', 'F', 'G', 'E', 'F'],
    'price': [100, 200, 150, 300, 120]
})

# 전체 명목형 컬럼에 대한 요약:
result, summary = hs_stats.category_describe(df)

# 특정 컬럼만 분석:
result, summary = hs_stats.category_describe(df, 'cut', 'color')
```
  

**Notes**:

  - 숫자형 컬럼은 자동으로 제외됩니다.
  - NaN 값도 하나의 범주로 포함됩니다.

<a id="hs_stats.normal_test"></a>

### normal\_test

```python
def normal_test(data: DataFrame,
                columns: list | str | None = None,
                method: str = "n") -> DataFrame
```

지정된 컬럼(또는 모든 수치형 컬럼)에 대해 정규성 검정을 수행하고 결과를 DataFrame으로 반환한다.

정규성 검정의 귀무가설은 "데이터가 정규분포를 따른다"이므로, p-value > 0.05일 때
귀무가설을 기각하지 않으며 정규성을 충족한다고 해석한다.

**Arguments**:

- `data` _DataFrame_ - 검정 대상 데이터를 포함한 데이터프레임.
- `columns` _list | str | None, optional_ - 검정 대상 컬럼명.
  - None 또는 빈 리스트: 모든 수치형 컬럼에 대해 검정 수행.
  - 컬럼명 리스트: 지정된 컬럼에 대해서만 검정 수행.
  - 콤마로 구분된 문자열: "A, B, C" 형식으로 컬럼명 지정 가능.
  기본값은 None.
- `method` _str, optional_ - 정규성 검정 방법.
  - "n": D'Agostino and Pearson's Omnibus test (표본 크기 20 이상 권장)
  - "s": Shapiro-Wilk test (표본 크기 5000 이하 권장)
  기본값은 "n".
  

**Returns**:

- `DataFrame` - 각 컬럼의 검정 결과를 담은 데이터프레임. 다음 컬럼 포함:
  - method (str): 사용된 검정 방법
  - column (str): 컬럼명
  - statistic (float): 검정 통계량
  - p-value (float): 유의확률
  - is_normal (bool): 정규성 충족 여부 (p-value > 0.05)
  

**Raises**:

- `ValueError` - 메서드가 "n" 또는 "s"가 아닐 경우.
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame
import numpy as np

df = DataFrame({
    'x': np.random.normal(0, 1, 100),
    'y': np.random.exponential(2, 100)
})

# 모든 수치형 컬럼 검정
result = hs_stats.normal_test(df, method='n')

# 특정 컬럼만 검정 (리스트)
result = hs_stats.normal_test(df, columns=['x'], method='n')

# 특정 컬럼만 검정 (문자열)
result = hs_stats.normal_test(df, columns='x, y', method='n')
```

<a id="hs_stats.equal_var_test"></a>

### equal\_var\_test

```python
def equal_var_test(data: DataFrame,
                   columns: list | str | None = None,
                   normal_dist: bool | None = None) -> DataFrame
```

수치형 컬럼들의 분산이 같은지 검정하고 결과를 DataFrame으로 반환한다.

등분산성 검정의 귀무가설은 "모든 그룹의 분산이 같다"이므로, p-value > 0.05일 때
귀무가설을 기각하지 않으며 등분산성을 충족한다고 해석한다.

**Arguments**:

- `data` _DataFrame_ - 검정 대상 데이터를 포함한 데이터프레임.
- `columns` _list | str | None, optional_ - 검정 대상 컬럼명.
  - None 또는 빈 리스트: 모든 수치형 컬럼에 대해 검정 수행.
  - 컬럼명 리스트: 지정된 컬럼에 대해서만 검정 수행.
  - 콤마로 구분된 문자열: "A, B, C" 형식으로 컬럼명 지정 가능.
  기본값은 None.
- `normal_dist` _bool | None, optional_ - 등분산성 검정 방법.
  - True: Bartlett 검정 (데이터가 정규분포를 따를 때, 모든 표본이 같은 크기일 때 권장)
  - False: Levene 검정 (정규분포를 따르지 않을 때 더 강건함)
  - None: normal_test()를 이용하여 자동으로 정규성을 판별 후 적절한 검정 방법 선택.
  모든 컬럼이 정규분포를 따르면 Bartlett, 하나라도 따르지 않으면 Levene 사용.
  기본값은 None.
  

**Returns**:

- `DataFrame` - 검정 결과를 담은 데이터프레임. 다음 컬럼 포함:
  - method (str): 사용된 검정 방법 (Bartlett 또는 Levene)
  - statistic (float): 검정 통계량
  - p-value (float): 유의확률
  - is_equal_var (bool): 등분산성 충족 여부 (p-value > 0.05)
  - n_columns (int): 검정에 사용된 컬럼 수
  - columns (str): 검정에 포함된 컬럼명 (쉼표로 구분)
  - normality_checked (bool): normal_dist가 None이었는지 여부 (자동 판별 사용 여부)
  

**Raises**:

- `ValueError` - 수치형 컬럼이 2개 미만일 경우 (검정에 최소 2개 필요).
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame
import numpy as np

df = DataFrame({
    'x': np.random.normal(0, 1, 100),
    'y': np.random.normal(0, 1, 100),
    'z': np.random.normal(0, 2, 100)
})

# 모든 수치형 컬럼 자동 판별
result = hs_stats.equal_var_test(df)

# 특정 컬럼만 검정 (리스트)
result = hs_stats.equal_var_test(df, columns=['x', 'y'])

# 특정 컬럼만 검정 (문자열)
result = hs_stats.equal_var_test(df, columns='x, y')

# 명시적 지정
result = hs_stats.equal_var_test(df, normal_dist=True)
```

<a id="hs_stats.ttest_1samp"></a>

### ttest\_1samp

```python
def ttest_1samp(data, mean_value: float = 0.0) -> DataFrame
```

연속형 데이터에 대해 일표본 t-검정을 수행하고 결과를 반환한다.

일표본 t-검정은 표본 평균이 특정 값(mean_value)과 같은지를 검정한다.
귀무가설(H0): 모집단 평균 = mean_value
대립가설(H1): alternative에 따라 달라짐 (!=, <, >)

**Arguments**:

- `data` _array-like_ - 검정 대상 연속형 데이터 (리스트, Series, ndarray 등).
- `mean_value` _float, optional_ - 귀무가설의 기준값(비교 대상 평균값).
  기본값은 0.0.
  

**Returns**:

- `DataFrame` - 검정 결과를 담은 데이터프레임. 다음 컬럼 포함:
  - alternative (str): 대립가설 방향 (two-sided, less, greater)
  - statistic (float): t-통계량
  - p-value (float): 유의확률
  - H0 (bool): 귀무가설 채택 여부 (p-value > 0.05)
  - H1 (bool): 대립가설 채택 여부 (p-value <= 0.05)
  - interpretation (str): 검정 결과 해석 문자열
  

**Examples**:

```python
from hossam import *
from pandas import Series
import numpy as np

# 리스트 데이터로 검정
data = [5.1, 4.9, 5.3, 5.0, 4.8]
result = hs_stats.ttest_1samp(data, mean_value=5.0)

# Series 데이터로 검정
s = Series(np.random.normal(5, 1, 100))
result = hs_stats.ttest_1samp(s, mean_value=5)
```

<a id="hs_stats.ttest_ind"></a>

### ttest\_ind

```python
def ttest_ind(x, y, equal_var: bool | None = None) -> DataFrame
```

두 독립 집단의 평균 차이를 검정한다 (독립표본 t-검정 또는 Welch's t-test).

독립표본 t-검정은 두 독립된 집단의 평균이 같은지를 검정한다.
귀무가설(H0): μ1 = μ2 (두 집단의 평균이 같다)

**Arguments**:

- `x` _array-like_ - 첫 번째 집단의 연속형 데이터 (리스트, Series, ndarray 등).
- `y` _array-like_ - 두 번째 집단의 연속형 데이터 (리스트, Series, ndarray 등).
- `equal_var` _bool | None, optional_ - 등분산성 가정 여부.
  - True: 독립표본 t-검정 (등분산 가정)
  - False: Welch's t-test (등분산 가정하지 않음, 더 강건함)
  - None: equal_var_test()로 자동 판별
  기본값은 None.
  

**Returns**:

- `DataFrame` - 검정 결과를 담은 데이터프레임. 다음 컬럼 포함:
  - test (str): 사용된 검정 방법
  - alternative (str): 대립가설 방향
  - statistic (float): t-통계량
  - p-value (float): 유의확률
  - H0 (bool): 귀무가설 채택 여부
  - H1 (bool): 대립가설 채택 여부
  - interpretation (str): 검정 결과 해석
  

**Examples**:

```python
from hossam import *
from pandas import Series, DataFrame
import numpy as np

# 리스트로 검정
group1 = [5.1, 4.9, 5.3, 5.0, 4.8]
group2 = [5.5, 5.7, 5.4, 5.6, 5.8]
result = hs_stats.ttest_ind(group1, group2)

# Series로 검정
s1 = Series(np.random.normal(5, 1, 100))
s2 = Series(np.random.normal(5.5, 1, 100))
result = hs_stats.ttest_ind(s1, s2, equal_var=False)
```

<a id="hs_stats.ttest_rel"></a>

### ttest\_rel

```python
def ttest_rel(x, y, parametric: bool | None = None) -> DataFrame
```

대응표본 t-검정 또는 Wilcoxon signed-rank test를 수행한다.

대응표본 t-검정은 동일 개체에서 측정된 두 시점의 평균 차이를 검정한다.
귀무가설(H0): 두 시점의 평균 차이가 0이다.

**Arguments**:

- `x` _array-like_ - 첫 번째 측정값의 연속형 데이터 (리스트, Series, ndarray 등).
- `y` _array-like_ - 두 번째 측정값의 연속형 데이터 (리스트, Series, ndarray 등).
- `parametric` _bool | None, optional_ - 정규성 가정 여부.
  - True: 대응표본 t-검정 (차이의 정규분포 가정)
  - False: Wilcoxon signed-rank test (비모수 검정, 더 강건함)
  - None: 차이의 정규성을 자동으로 검정하여 판별
  기본값은 None.
  

**Returns**:

- `DataFrame` - 검정 결과를 담은 데이터프레임. 다음 컬럼 포함:
  - test (str): 사용된 검정 방법
  - alternative (str): 대립가설 방향
  - statistic (float): 검정 통계량
  - p-value (float): 유의확률
  - H0 (bool): 귀무가설 채택 여부
  - H1 (bool): 대립가설 채택 여부
  - interpretation (str): 검정 결과 해석
  

**Examples**:

```python
from hossam import *
from pandas import Series
import numpy as np

# 리스트로 검정
before = [5.1, 4.9, 5.3, 5.0, 4.8]
after = [5.5, 5.2, 5.7, 5.3, 5.1]
result = hs_stats.ttest_rel(before, after)

# Series로 검정
s1 = Series(np.random.normal(5, 1, 100))
s2 = Series(np.random.normal(5.3, 1, 100))
result = hs_stats.ttest_rel(s1, s2, parametric=False)
```

<a id="hs_stats.vif_filter"></a>

### vif\_filter

```python
def vif_filter(data: DataFrame,
               yname: str = None,
               ignore: list | None = None,
               threshold: float = 10.0,
               verbose: bool = False) -> DataFrame
```

독립변수 간 다중공선성을 검사하여 VIF가 threshold 이상인 변수를 반복적으로 제거한다.

**Arguments**:

- `data` _DataFrame_ - 데이터프레임
- `yname` _str, optional_ - 종속변수 컬럼명. Defaults to None.
- `ignore` _list | None, optional_ - 제외할 컬럼 목록. Defaults to None.
- `threshold` _float, optional_ - VIF 임계값. Defaults to 10.0.
- `verbose` _bool, optional_ - True일 경우 각 단계의 VIF를 출력한다. Defaults to False.
  

**Returns**:

- `DataFrame` - VIF가 threshold 이하인 변수만 남은 데이터프레임 (원본 컬럼 순서 유지)
  

**Examples**:

```python
# 기본 사용 예
from hossam import *
filtered = hs_stats.vif_filter(df, yname="target", ignore=["id"], threshold=10.0)
```

<a id="hs_stats.trend"></a>

### trend

```python
def trend(x: any,
          y: any,
          degree: int = 1,
          value_count: int = 100) -> Tuple[np.ndarray, np.ndarray]
```

x, y 데이터에 대한 추세선을 구한다.

**Arguments**:

- `x` __type__ - 산점도 그래프에 대한 x 데이터
- `y` __type__ - 산점도 그래프에 대한 y 데이터
- `degree` _int, optional_ - 추세선 방정식의 차수. Defaults to 1.
- `value_count` _int, optional_ - x 데이터의 범위 안에서 간격 수. Defaults to 100.
  

**Returns**:

- `tuple` - (v_trend, t_trend)
  

**Examples**:

```python
# 2차 다항 회귀 추세선
from hossam import *
vx, vy = hs_stats.trend(x, y, degree=2, value_count=200)
print(len(vx), len(vy)) # 200, 200
```

<a id="hs_stats.ols_report"></a>

### ols\_report

```python
def ols_report(fit, data, full=False, alpha=0.05)
```

선형회귀 적합 결과를 요약 리포트로 변환한다.

**Arguments**:

- `fit` - statsmodels OLS 등 선형회귀 결과 객체 (`fit.summary()`를 지원해야 함).
- `data` - 종속변수와 독립변수를 모두 포함한 DataFrame.
- `full` - True이면 6개 값 반환, False이면 회귀계수 테이블(rdf)만 반환. 기본값 True.
- `alpha` - 유의수준. 기본값 0.05.
  

**Returns**:

- `tuple` - full=True일 때 다음 요소를 포함한다.
  - 성능 지표 표 (`pdf`, DataFrame): R, R², Adj. R², F, p-value, Durbin-Watson.
  - 회귀계수 표 (`rdf`, DataFrame): 변수별 B, 표준오차, Beta, t, p-value, significant, 공차, VIF.
  - 적합도 요약 (`result_report`, str): R, R², F, p-value, Durbin-Watson 등 핵심 지표 문자열.
  - 모형 보고 문장 (`model_report`, str): F-검정 유의성에 기반한 서술형 문장.
  - 변수별 보고 리스트 (`variable_reports`, list[str]): 각 예측변수에 대한 서술형 문장.
  - 회귀식 문자열 (`equation_text`, str): 상수항과 계수를 포함한 회귀식 표현.
  
  full=False일 때:
  - 회귀계수 표 (`rdf`, DataFrame)
  

**Examples**:

```python
from hossam import *

df = hs_util.load_data("some_data.csv")
fit = hs_stats.ols(df, yname="target")

# 전체 리포트
pdf, rdf, result_report, model_report, variable_reports, eq = hs_stats.ols_report(fit, data, full=True)

# 간단한 버전 (성능지표, 회귀계수 테이블만)
pdf, rdf = hs_stats.ols_report(fit, data)
```

<a id="hs_stats.ols"></a>

### ols

```python
def ols(df: DataFrame, yname: str, report=False)
```

선형회귀분석을 수행하고 적합 결과를 반환한다.

OLS(Ordinary Least Squares) 선형회귀분석을 실시한다.
필요시 상세한 통계 보고서를 함께 제공한다.

**Arguments**:

- `df` _DataFrame_ - 종속변수와 독립변수를 모두 포함한 데이터프레임.
- `yname` _str_ - 종속변수 컬럼명.
- `report` - 리포트 모드 설정. 다음 값 중 하나:
  - False (기본값): 리포트 미사용. fit 객체만 반환.
  - 1 또는 'summary': 요약 리포트 반환 (full=False).
  - 2 또는 'full': 풀 리포트 반환 (full=True).
  - True: 풀 리포트 반환 (2와 동일).
  

**Returns**:

- `statsmodels.regression.linear_model.RegressionResultsWrapper` - report=False일 때.
  선형회귀 적합 결과 객체. fit.summary()로 상세 결과 확인 가능.
  
- `tuple` _6개_ - report=1 또는 'summary'일 때.
  (fit, rdf, result_report, model_report, variable_reports, equation_text) 형태로 (pdf 제외).
  
- `tuple` _7개_ - report=2, 'full' 또는 True일 때.
  (fit, pdf, rdf, result_report, model_report, variable_reports, equation_text) 형태로:
  - fit: 선형회귀 적합 결과 객체
  - pdf: 성능 지표 표 (DataFrame): R, R², F, p-value, Durbin-Watson
  - rdf: 회귀계수 표 (DataFrame)
  - result_report: 적합도 요약 (str)
  - model_report: 모형 보고 문장 (str)
  - variable_reports: 변수별 보고 문장 리스트 (list[str])
  - equation_text: 회귀식 문자열 (str)
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame
import numpy as np

df = DataFrame({
    'target': np.random.normal(100, 10, 100),
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100)
})

# 적합 결과만 반환
fit = hs_stats.ols(df, 'target')

# 요약 리포트 반환
fit, pdf, rdf = hs_stats.ols(df, 'target', report=1)

# 풀 리포트 반환
fit, pdf, rdf, result_report, model_report, var_reports, eq = hs_stats.ols(df, 'target', report=2)
```

<a id="hs_stats.logit_report"></a>

### logit\_report

```python
def logit_report(fit, data, threshold=0.5, full=False, alpha=0.05)
```

로지스틱 회귀 적합 결과를 상세 리포트로 변환한다.

**Arguments**:

- `fit` - statsmodels Logit 결과 객체 (`fit.summary()`와 예측 확률을 지원해야 함).
- `data` - 종속변수와 독립변수를 모두 포함한 DataFrame.
- `threshold` - 예측 확률을 이진 분류로 변환할 임계값. 기본값 0.5.
- `full` - True이면 6개 값 반환, False이면 주요 2개(cdf, rdf)만 반환. 기본값 False.
- `alpha` - 유의수준. 기본값 0.05.
  

**Returns**:

- `tuple` - full=True일 때 다음 요소를 포함한다.
  - 성능 지표 표 (`cdf`, DataFrame): McFadden Pseudo R², Accuracy, Precision, Recall, FPR, TNR, AUC, F1.
  - 회귀계수 표 (`rdf`, DataFrame): B, 표준오차, z, p-value, significant, OR, 95% CI, VIF 등.
  - 적합도 및 예측 성능 요약 (`result_report`, str): Pseudo R², LLR χ², p-value, Accuracy, AUC.
  - 모형 보고 문장 (`model_report`, str): LLR p-value에 기반한 서술형 문장.
  - 변수별 보고 리스트 (`variable_reports`, list[str]): 각 예측변수의 오즈비 해석 문장.
  - 혼동행렬 (`cm`, ndarray): 예측 결과와 실제값의 혼동행렬 [[TN, FP], [FN, TP]].
  
  full=False일 때:
  - 성능 지표 표 (`cdf`, DataFrame)
  - 회귀계수 표 (`rdf`, DataFrame)
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame
import numpy as np

df = DataFrame({
    'target': np.random.binomial(1, 0.5, 100),
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100)
})

# 로지스틱 회귀 적합
fit = hs_stats.logit(df, yname="target")

# 전체 리포트
cdf, rdf, result_report, model_report, variable_reports, cm = hs_stats.logit_report(fit, df, full=True)

# 간단한 버전 (주요 테이블만)
cdf, rdf = hs_stats.logit_report(fit, df)
```

<a id="hs_stats.logit"></a>

### logit

```python
def logit(df: DataFrame, yname: str, report=False)
```

로지스틱 회귀분석을 수행하고 적합 결과를 반환한다.

종속변수가 이항(binary) 형태일 때 로지스틱 회귀분석을 실시한다.
필요시 상세한 통계 보고서를 함께 제공한다.

**Arguments**:

- `df` _DataFrame_ - 종속변수와 독립변수를 모두 포함한 데이터프레임.
- `yname` _str_ - 종속변수 컬럼명. 이항 변수여야 한다.
- `report` - 리포트 모드 설정. 다음 값 중 하나:
  - False (기본값): 리포트 미사용. fit 객체만 반환.
  - 1 또는 'summary': 요약 리포트 반환 (full=False).
  - 2 또는 'full': 풀 리포트 반환 (full=True).
  - True: 풀 리포트 반환 (2와 동일).
  

**Returns**:

- `statsmodels.genmod.generalized_linear_model.BinomialResults` - report=False일 때.
  로지스틱 회귀 적합 결과 객체. fit.summary()로 상세 결과 확인 가능.
  
- `tuple` _4개_ - report=1 또는 'summary'일 때.
  (fit, rdf, result_report, variable_reports) 형태로 (cdf 제외).
  
- `tuple` _6개_ - report=2, 'full' 또는 True일 때.
  (fit, cdf, rdf, result_report, model_report, variable_reports) 형태로:
  - fit: 로지스틱 회귀 적합 결과 객체
  - cdf: 성능 지표 표 (DataFrame)
  - rdf: 회귀계수 표 (DataFrame)
  - result_report: 적합도 및 예측 성능 요약 (str)
  - model_report: 모형 보고 문장 (str)
  - variable_reports: 변수별 보고 문장 리스트 (list[str])
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame
import numpy as np

df = DataFrame({
    'target': np.random.binomial(1, 0.5, 100),
    'x1': np.random.normal(0, 1, 100),
    'x2': np.random.normal(0, 1, 100)
})

# 적합 결과만 반환
fit = hs_stats.logit(df, 'target')

# 요약 리포트 반환
fit, rdf, result_report, var_reports = hs_stats.logit(df, 'target', report='summary')

# 풀 리포트 반환
fit, cdf, rdf, result_report, model_report, var_reports = hs_stats.logit(df, 'target', report='full')
```

<a id="hs_stats.ols_linearity_test"></a>

### ols\_linearity\_test

```python
def ols_linearity_test(fit, power: int = 2, alpha: float = 0.05) -> DataFrame
```

회귀모형의 선형성을 Ramsey RESET 검정으로 평가한다.

적합된 회귀모형에 대해 Ramsey RESET(Regression Specification Error Test) 검정을 수행하여
모형의 선형성 가정이 타당한지를 검정한다. 귀무가설은 '모형이 선형이다'이다.

**Arguments**:

- `fit` - 회귀 모형 객체 (statsmodels의 RegressionResultsWrapper).
  OLS 또는 WLS 모형이어야 한다.
- `power` _int, optional_ - RESET 검정에 사용할 거듭제곱 수. 기본값 2.
  power=2일 때 예측값의 제곱항이 추가됨.
  power가 클수록 더 높은 차수의 비선형성을 감지.
- `alpha` _float, optional_ - 유의수준. 기본값 0.05.
  

**Returns**:

- `DataFrame` - 선형성 검정 결과를 포함한 데이터프레임.
  - 검정통계량: F-statistic
  - p-value: 검정의 p값
  - 유의성: alpha 기준 결과 (bool)
  - 해석: 선형성 판정 (문자열)
  

**Examples**:

```python
from hossam import *
fit = hs_stats.logit(df, 'target')
result = hs_stats.ols_linearity_test(fit)
```
  

**Notes**:

  - p-value > alpha: 선형성 가정을 만족 (귀무가설 채택)
  - p-value <= alpha: 선형성 가정 위반 가능 (귀무가설 기각)

<a id="hs_stats.ols_normality_test"></a>

### ols\_normality\_test

```python
def ols_normality_test(fit, alpha: float = 0.05) -> DataFrame
```

회귀모형 잔차의 정규성을 검정한다.

회귀모형의 잔차가 정규분포를 따르는지 Shapiro-Wilk 검정과 Jarque-Bera 검정으로 평가한다.
정규성 가정은 회귀분석의 추론(신뢰구간, 가설검정)이 타당하기 위한 중요한 가정이다.

**Arguments**:

- `fit` - 회귀 모형 객체 (statsmodels의 RegressionResultsWrapper).
- `alpha` _float, optional_ - 유의수준. 기본값 0.05.
  

**Returns**:

- `DataFrame` - 정규성 검정 결과를 포함한 데이터프레임.
  - 검정명: 사용된 검정 방법
  - 검정통계량: 검정 통계량 값
  - p-value: 검정의 p값
  - 유의수준: 설정된 유의수준
  - 정규성_위반: alpha 기준 결과 (bool)
  - 해석: 정규성 판정 (문자열)
  

**Examples**:

```python
from hossam import *
fit = hs_stats.logit(df, 'target')
result = hs_stats.ols_normality_test(fit)
```
  

**Notes**:

  - Shapiro-Wilk: 샘플 크기가 작을 때 (< 5000) 강력한 검정
  - Jarque-Bera: 왜도(skewness)와 첨도(kurtosis) 기반 검정
  - p-value > alpha: 정규성 가정 만족 (귀무가설 채택)
  - p-value <= alpha: 정규성 가정 위반 (귀무가설 기각)

<a id="hs_stats.ols_variance_test"></a>

### ols\_variance\_test

```python
def ols_variance_test(fit, alpha: float = 0.05) -> DataFrame
```

회귀모형의 등분산성 가정을 검정한다.

잔차의 분산이 예측값의 수준에 관계없이 일정한지 Breusch-Pagan 검정과 White 검정으로 평가한다.
등분산성 가정은 회귀분석의 추론(표준오차, 검정통계량)이 정확하기 위한 중요한 가정이다.

**Arguments**:

- `fit` - 회귀 모형 객체 (statsmodels의 RegressionResultsWrapper).
- `alpha` _float, optional_ - 유의수준. 기본값 0.05.
  

**Returns**:

- `DataFrame` - 등분산성 검정 결과를 포함한 데이터프레임.
  - 검정명: 사용된 검정 방법
  - 검정통계량: 검정 통계량 값 (LM 또는 F)
  - p-value: 검정의 p값
  - 유의수준: 설정된 유의수준
  - 등분산성_위반: alpha 기준 결과 (bool)
  - 해석: 등분산성 판정 (문자열)
  

**Examples**:

```python
from hossam import *
fit = hs_stats.logit(df, 'target')
result = hs_stats.ols_variance_test(fit)
```
  

**Notes**:

  - Breusch-Pagan: 잔차 제곱과 독립변수의 선형관계 검정
  - White: 잔차 제곱과 독립변수 및 그 제곱, 교차항의 관계 검정
  - p-value > alpha: 등분산성 가정 만족 (귀무가설 채택)
  - p-value <= alpha: 이분산성 존재 (귀무가설 기각)

<a id="hs_stats.ols_independence_test"></a>

### ols\_independence\_test

```python
def ols_independence_test(fit, alpha: float = 0.05) -> DataFrame
```

회귀모형의 독립성 가정(자기상관 없음)을 검정한다.

Durbin-Watson 검정을 사용하여 잔차의 1차 자기상관 여부를 검정한다.
시계열 데이터나 순서가 있는 데이터에서 주로 사용된다.

**Arguments**:

- `fit` - statsmodels 회귀분석 결과 객체 (RegressionResultsWrapper).
- `alpha` _float, optional_ - 유의수준. 기본값은 0.05.
  

**Returns**:

- `DataFrame` - 검정 결과 데이터프레임.
  - 검정: 검정 방법명
  - 검정통계량(DW): Durbin-Watson 통계량 (0~4 범위, 2에 가까울수록 자기상관 없음)
  - 독립성_위반: 자기상관 의심 여부 (True/False)
  - 해석: 검정 결과 해석
  

**Examples**:

```python
from hossam import *
fit = hs_stats.logit(df, 'target')
result = hs_stats.ols_independence_test(fit)
```
  

**Notes**:

  - Durbin-Watson 통계량 해석:
  * 2에 가까우면: 자기상관 없음 (독립성 만족)
  * 0에 가까우면: 양의 자기상관 (독립성 위반)
  * 4에 가까우면: 음의 자기상관 (독립성 위반)
  - 일반적으로 1.5~2.5 범위를 자기상관 없음으로 판단
  - 시계열 데이터나 관측치에 순서가 있는 경우 중요한 검정

<a id="hs_stats.corr_pairwise"></a>

### corr\_pairwise

```python
def corr_pairwise(data: DataFrame,
                  fields: list[str] | None = None,
                  alpha: float = 0.05,
                  z_thresh: float = 3.0,
                  min_n: int = 8,
                  linearity_power: tuple[int, ...] = (2, ),
                  p_adjust: str = "none") -> tuple[DataFrame, DataFrame]
```

각 변수 쌍에 대해 선형성·이상치 여부를 점검한 뒤 Pearson/Spearman을 자동 선택해 상관을 요약한다.

절차:
1) z-score 기준(|z|>z_thresh)으로 각 변수의 이상치 존재 여부를 파악
2) 단순회귀 y~x에 대해 Ramsey RESET(linearity_power)로 선형성 검정 (모든 p>alpha → 선형성 충족)
3) 선형성 충족이고 양쪽 변수에서 |z|>z_thresh 이상치가 없으면 Pearson, 그 외엔 Spearman 선택
4) 상관계수/유의확률, 유의성 여부, 강도(strong/medium/weak/no correlation) 기록
5) 선택적으로 다중비교 보정(p_adjust="fdr_bh" 등) 적용하여 pval_adj와 significant_adj 추가

**Arguments**:

- `data` _DataFrame_ - 분석 대상 데이터프레임.
- `fields` _list[str]|None_ - 분석할 숫자형 컬럼 이름 리스트. None이면 모든 숫자형 컬럼 사용. 기본값 None.
- `alpha` _float, optional_ - 유의수준. 기본 0.05.
- `z_thresh` _float, optional_ - 이상치 판단 임계값(|z| 기준). 기본 3.0.
- `min_n` _int, optional_ - 쌍별 최소 표본 크기. 미만이면 계산 생략. 기본 8.
- `linearity_power` _tuple[int,...], optional_ - RESET 검정에서 포함할 차수 집합. 기본 (2,).
- `p_adjust` _str, optional_ - 다중비교 보정 방법. "none" 또는 statsmodels.multipletests 지원값 중 하나(e.g., "fdr_bh"). 기본 "none".
  

**Returns**:

  tuple[DataFrame, DataFrame]: 두 개의 데이터프레임을 반환.
  [0] result_df: 각 변수쌍별 결과 테이블. 컬럼:
  var_a, var_b, n, linearity(bool), outlier_flag(bool), chosen('pearson'|'spearman'),
  corr, pval, significant(bool), strength(str), (보정 사용 시) pval_adj, significant_adj
  [1] corr_matrix: 상관계수 행렬 (행과 열에 변수명, 값에 상관계수)
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame

df = DataFrame({'x1': [1,2,3,4,5], 'x2': [2,4,5,4,6], 'x3': [10,20,25,24,30]})
# 전체 숫자형 컬럼에 대해 상관분석
result_df, corr_matrix = hs_stats.corr_pairwise(df)
# 특정 컬럼만 분석
result_df, corr_matrix = hs_stats.corr_pairwise(df, fields=['x1', 'x2'])
```

<a id="hs_stats.oneway_anova"></a>

### oneway\_anova

```python
def oneway_anova(
        data: DataFrame,
        dv: str,
        between: str,
        alpha: float = 0.05) -> tuple[DataFrame, str, DataFrame | None, str]
```

일원분산분석(One-way ANOVA)을 일괄 처리한다.

정규성 및 등분산성 검정을 자동으로 수행한 후,
그 결과에 따라 적절한 ANOVA 방식을 선택하여 분산분석을 수행한다.
ANOVA 결과가 유의하면 자동으로 사후검정을 실시한다.

분석 흐름:
1. 정규성 검정 (각 그룹별로 normaltest 수행)
2. 등분산성 검정 (정규성 만족 시 Bartlett, 불만족 시 Levene)
3. ANOVA 수행 (등분산 만족 시 parametric ANOVA, 불만족 시 Welch's ANOVA)
4. ANOVA p-value ≤ alpha 일 때 사후검정 (등분산 만족 시 Tukey HSD, 불만족 시 Games-Howell)

**Arguments**:

- `data` _DataFrame_ - 분석 대상 데이터프레임. 종속변수와 그룹 변수를 포함해야 함.
- `dv` _str_ - 종속변수(Dependent Variable) 컬럼명.
- `between` _str_ - 그룹 구분 변수 컬럼명.
- `alpha` _float, optional_ - 유의수준. 기본값 0.05.
  

**Returns**:

  tuple:
  - anova_df (DataFrame): ANOVA 또는 Welch 결과 테이블(Source, ddof1, ddof2, F, p-unc, np2 등 포함).
  - anova_report (str): 정규성/등분산 여부와 F, p값, 효과크기를 요약한 보고 문장.
  - posthoc_df (DataFrame|None): 사후검정 결과(Tukey HSD 또는 Games-Howell). ANOVA가 유의할 때만 생성.
  - posthoc_report (str): 사후검정 유무와 유의한 쌍 정보를 요약한 보고 문장.
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame

df = DataFrame({
    'score': [5.1, 4.9, 5.3, 5.0, 4.8, 5.5, 5.2, 5.7, 5.3, 5.1],
    'group': ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']
})

anova_df, anova_report, posthoc_df, posthoc_report = hs_stats.oneway_anova(df, dv='score', between='group')

# 사후검정결과는 ANOVA가 유의할 때만 생성됨
if posthoc_df is not None:
    print(posthoc_report)
    print(posthoc_df.head())
```
  

**Raises**:

- `ValueError` - dv 또는 between 컬럼이 데이터프레임에 없을 경우.

<a id="hs_stats.twoway_anova"></a>

### twoway\_anova

```python
def twoway_anova(
        data: DataFrame,
        dv: str,
        factor_a: str,
        factor_b: str,
        alpha: float = 0.05) -> tuple[DataFrame, str, DataFrame | None, str]
```

두 범주형 요인에 대한 이원분산분석을 수행하고 해석용 보고문을 반환한다.

분석 흐름:
1) 각 셀(요인 조합)별 정규성 검정
2) 전체 셀을 대상으로 등분산성 검정 (정규성 충족 시 Bartlett, 불충족 시 Levene)
3) 두 요인 및 교호작용을 포함한 2원 ANOVA 수행
4) 유의한 요인에 대해 Tukey HSD 사후검정(요인별) 실행

**Arguments**:

- `data` _DataFrame_ - 종속변수와 두 개의 범주형 요인을 포함한 데이터프레임.
- `dv` _str_ - 종속변수 컬럼명.
- `factor_a` _str_ - 첫 번째 요인 컬럼명.
- `factor_b` _str_ - 두 번째 요인 컬럼명.
- `alpha` _float, optional_ - 유의수준. 기본 0.05.
  

**Returns**:

  tuple:
  - anova_df (DataFrame): 2원 ANOVA 결과(각 요인과 상호작용의 F, p, η²p 포함).
  - anova_report (str): 두 요인 및 상호작용의 유의성/가정 충족 여부를 요약한 문장.
  - posthoc_df (DataFrame|None): 유의한 요인에 대한 Tukey 사후검정 결과(요인명, A, B, p 포함). 없으면 None.
  - posthoc_report (str): 사후검정 유무 및 유의 쌍 요약 문장.
  

**Raises**:

- `ValueError` - 입력 컬럼이 데이터프레임에 없을 때.

<a id="hs_stats.predict"></a>

### predict

```python
def predict(fit, data: DataFrame | Series) -> DataFrame | Series | float
```

회귀 또는 로지스틱 모형을 이용하여 예측값을 생성한다.

statsmodels의 RegressionResultsWrapper(선형회귀) 또는
BinaryResultsWrapper(로지스틱 회귀) 객체를 받아 데이터에 대한
예측값을 생성하고 반환한다.

모형 학습 시 상수항이 포함되었다면, 예측 데이터에도 자동으로
상수항을 추가하여 차원을 맞춘다.

로지스틱 회귀의 경우 예측 확률과 함께 분류 해석을 포함한다.

**Arguments**:

- `fit` - 학습된 회귀 모형 객체.
  - statsmodels.regression.linear_model.RegressionResultsWrapper (선형회귀)
  - statsmodels.discrete.discrete_model.BinaryResultsWrapper (로지스틱 회귀)
- `data` _DataFrame|Series_ - 예측에 사용할 설명변수.
  - DataFrame: 여러 개의 관측치
  - Series: 단일 관측치 또는 변수 하나
  원본 모형 학습 시 사용한 특성과 동일해야 함.
  (상수항 제외, 자동으로 추가됨)
  

**Returns**:

- `DataFrame|Series|float` - 예측값.
  - DataFrame 입력:
  - 선형회귀: 예측값 컬럼을 포함한 DataFrame
  - 로지스틱: 확률, 분류, 해석 컬럼을 포함한 DataFrame
  - Series 입력: 단일 예측값 (float)
  

**Raises**:

- `ValueError` - fit 객체가 지원되지 않는 타입인 경우.
- `Exception` - 데이터와 모형의 특성 불일치로 인한 predict 실패.
  

**Examples**:

```python
from hossam import *

df = hs_util.load_data("some_data.csv")
fit1 = hs_stats.ols(df, yname="target")

pred = hs_stats.predict(fit1, df_new[['x1', 'x2']])  # DataFrame 반환

# 로지스틱 회귀 (상수항 자동 추가)
fit2 = hs_stats.logit(df, yname="target")
pred_prob = hs_stats.predict(fit2, df_new[['x1', 'x2']])  # DataFrame 반환 (해석 포함)
```

<a id="hs_stats.corr_effect_size"></a>

### corr\_effect\_size

```python
def corr_effect_size(data: DataFrame,
                     dv: str,
                     *fields: str,
                     alpha: float = 0.05) -> DataFrame
```

종속변수와의 편상관계수 및 효과크기를 계산한다.

각 독립변수와 종속변수 간의 상관계수를 계산하되, 정규성과 선형성을 검사하여
Pearson 또는 Spearman 상관계수를 적절히 선택한다.
Cohen's d (효과크기)를 계산하여 상관 강도를 정량화한다.

**Arguments**:

- `data` _DataFrame_ - 분석 대상 데이터프레임.
- `dv` _str_ - 종속변수 컬럼 이름.
- `*fields` _str_ - 독립변수 컬럼 이름들. 지정하지 않으면 수치형 컬럼 중 dv 제외 모두 사용.
- `alpha` _float, optional_ - 유의수준. 기본 0.05.
  

**Returns**:

- `DataFrame` - 다음 컬럼을 포함한 데이터프레임:
  - Variable (str): 독립변수 이름
  - Correlation (float): 상관계수 (Pearson 또는 Spearman)
  - Corr_Type (str): 선택된 상관계수 종류 ('Pearson' 또는 'Spearman')
  - P-value (float): 상관계수의 유의확률
  - Cohens_d (float): 표준화된 효과크기
  - Effect_Size (str): 효과크기 분류 ('Large', 'Medium', 'Small', 'Negligible')
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame

df = DataFrame({'age': [20, 30, 40, 50],
           'bmi': [22, 25, 28, 30],
           'charges': [1000, 2000, 3000, 4000]})

result = hs_stats.corr_effect_size(df, 'charges', 'age', 'bmi')
```

<a id="hs_plot"></a>

## hs\_plot

<a id="hs_plot.get_default_ax"></a>

### get\_default\_ax

```python
def get_default_ax(width: int = config.width,
                   height: int = config.height,
                   rows: int = 1,
                   cols: int = 1,
                   dpi: int = config.dpi,
                   flatten: bool = False,
                   ws: int | None = None,
                   hs: int | None = None,
                   title: str = None)
```

기본 크기의 Figure와 Axes를 생성한다.

**Arguments**:

- `width` _int_ - 가로 픽셀 크기.
- `height` _int_ - 세로 픽셀 크기.
- `rows` _int_ - 서브플롯 행 개수.
- `cols` _int_ - 서브플롯 열 개수.
- `dpi` _int_ - 해상도(DPI).
- `flatten` _bool_ - Axes 배열을 1차원 리스트로 평탄화할지 여부.
- `ws` _int|None_ - 서브플롯 가로 간격(`wspace`). rows/cols가 1보다 클 때만 적용.
- `hs` _int|None_ - 서브플롯 세로 간격(`hspace`). rows/cols가 1보다 클 때만 적용.
- `title` _str|None_ - Figure 제목.
  

**Returns**:

  tuple[Figure, Axes]: 생성된 matplotlib Figure와 Axes 객체.

<a id="hs_plot.create_figure"></a>

### create\_figure

```python
def create_figure(width: int = config.width,
                  height: int = config.height,
                  rows: int = 1,
                  cols: int = 1,
                  dpi: int = config.dpi,
                  flatten: bool = False,
                  ws: int | None = None,
                  hs: int | None = None,
                  title: str = None)
```

기본 크기의 Figure와 Axes를 생성한다. get_default_ax의 래퍼 함수.

**Arguments**:

- `width` _int_ - 가로 픽셀 크기.
- `height` _int_ - 세로 픽셀 크기.
- `rows` _int_ - 서브플롯 행 개수.
- `cols` _int_ - 서브플롯 열 개수.
- `dpi` _int_ - 해상도(DPI).
- `flatten` _bool_ - Axes 배열을 1차원 리스트로 평탄화할지 여부.
- `ws` _int|None_ - 서브플롯 가로 간격(`wspace`). rows/cols가 1보다 클 때만 적용.
- `hs` _int|None_ - 서브플롯 세로 간격(`hspace`). rows/cols가 1보다 클 때만 적용.
- `title` _str_ - Figure 제목.
  

**Returns**:

  tuple[Figure, Axes]: 생성된 matplotlib Figure와 Axes 객체.

<a id="hs_plot.finalize_plot"></a>

### finalize\_plot

```python
def finalize_plot(ax: Axes,
                  callback: any = None,
                  outparams: bool = False,
                  save_path: str = None,
                  grid: bool = True,
                  title: str = None) -> None
```

공통 후처리를 수행한다: 콜백 실행, 레이아웃 정리, 필요 시 표시/종료.

**Arguments**:

- `ax` _Axes|ndarray|list_ - 대상 Axes (단일 Axes 또는 subplots 배열).
- `callback` _Callable|None_ - 추가 설정을 위한 사용자 콜백.
- `outparams` _bool_ - 내부에서 생성한 Figure인 경우 True.
- `save_path` _str|None_ - 이미지 저장 경로. None이 아니면 해당 경로로 저장.
- `grid` _bool_ - 그리드 표시 여부. 기본값은 True입니다.
- `title` _str|None_ - 그래프 제목.

**Returns**:

  None

<a id="hs_plot.show_figure"></a>

### show\_figure

```python
def show_figure(ax: Axes,
                callback: any = None,
                outparams: bool = False,
                save_path: str = None,
                grid: bool = True,
                title: str = None) -> None
```

공통 후처리를 수행한다: 콜백 실행, 레이아웃 정리, 필요 시 표시/종료.
finalize_plot의 래퍼 함수.

**Arguments**:

- `ax` _Axes|ndarray|list_ - 대상 Axes (단일 Axes 또는 subplots 배열).
- `callback` _Callable|None_ - 추가 설정을 위한 사용자 콜백.
- `outparams` _bool_ - 내부에서 생성한 Figure인 경우 True.
- `save_path` _str|None_ - 이미지 저장 경로. None이 아니면 해당 경로로 저장.
- `grid` _bool_ - 그리드 표시 여부. 기본값은 True입니다.
- `title` _str|None_ - 그래프 제목.
  

**Returns**:

  None

<a id="hs_plot.lineplot"></a>

### lineplot

```python
def lineplot(df: DataFrame,
             xname: str = None,
             yname: str = None,
             hue: str = None,
             title: str | None = None,
             marker: str = None,
             palette: str = None,
             width: int = config.width,
             height: int = config.height,
             linewidth: float = config.line_width,
             dpi: int = config.dpi,
             save_path: str = None,
             callback: any = None,
             ax: Axes = None,
             **params) -> None
```

선 그래프를 그린다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str|None_ - x축 컬럼명.
- `yname` _str|None_ - y축 컬럼명.
- `hue` _str|None_ - 범주 구분 컬럼명.
- `title` _str|None_ - 그래프 제목.
- `marker` _str|None_ - 마커 모양.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 해상도.
- `save_path` _str|None_ - 이미지 저장 경로. None이면 화면에 표시.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn lineplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.boxplot"></a>

### boxplot

```python
def boxplot(df: DataFrame,
            xname: str = None,
            yname: str = None,
            title: str | None = None,
            orient: str = "v",
            palette: str = None,
            width: int = config.width,
            height: int = config.height,
            linewidth: float = config.line_width,
            dpi: int = config.dpi,
            save_path: str = None,
            callback: any = None,
            ax: Axes = None,
            **params) -> None
```

상자그림(boxplot)을 그린다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str|None_ - x축 범주 컬럼명.
- `yname` _str|None_ - y축 값 컬럼명.
- `title` _str|None_ - 그래프 제목.
- `orient` _str_ - 'v' 또는 'h' 방향.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `save_path` _str|None_ - 이미지 저장 경로. None이면 화면에 표시.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn boxplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.kdeplot"></a>

### kdeplot

```python
def kdeplot(df: DataFrame,
            xname: str = None,
            yname: str = None,
            hue: str = None,
            title: str | None = None,
            palette: str = None,
            fill: bool = False,
            fill_alpha: float = config.fill_alpha,
            linewidth: float = config.line_width,
            quartile_split: bool = False,
            width: int = config.width,
            height: int = config.height,
            dpi: int = config.dpi,
            save_path: str = None,
            callback: any = None,
            ax: Axes = None,
            **params) -> None
```

커널 밀도 추정(KDE) 그래프를 그린다.

quartile_split=True일 때는 1차원 KDE(xname 지정, yname 없음)를
사분위수 구간(Q1~Q4)으로 나누어 4개의 서브플롯에 그린다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str|None_ - x축 컬럼명.
- `yname` _str|None_ - y축 컬럼명.
- `hue` _str|None_ - 범주 컬럼명.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 팔레트 이름.
- `fill` _bool_ - 면적 채우기 여부.
- `fill_alpha` _float_ - 채움 투명도.
- `quartile_split` _bool_ - True면 1D KDE를 사분위수별 서브플롯으로 분할.
- `linewidth` _float_ - 선 굵기.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn kdeplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.histplot"></a>

### histplot

```python
def histplot(df: DataFrame,
             xname: str,
             hue: str | None = None,
             title: str | None = None,
             bins: int | None = None,
             kde: bool = True,
             palette: str = None,
             width: int = config.width,
             height: int = config.height,
             linewidth: float = config.line_width,
             dpi: int = config.dpi,
             save_path: str = None,
             callback: any = None,
             ax: Axes = None,
             **params) -> None
```

히스토그램을 그리고 필요 시 KDE를 함께 표시한다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str_ - 히스토그램 대상 컬럼명.
- `hue` _str|None_ - 범주 컬럼명.
- `title` _str|None_ - 그래프 제목.
- `bins` _int|sequence|None_ - 구간 수 또는 경계.
- `kde` _bool_ - KDE 표시 여부.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn histplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.stackplot"></a>

### stackplot

```python
def stackplot(df: DataFrame,
              xname: str,
              hue: str,
              title: str | None = None,
              palette: str = None,
              width: int = config.width,
              height: int = config.height,
              linewidth: float = 0.25,
              dpi: int = config.dpi,
              save_path: str = None,
              callback: any = None,
              ax: Axes = None,
              **params) -> None
```

클래스 비율을 100% 누적 막대로 표현한다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str_ - x축 기준 컬럼.
- `hue` _str_ - 클래스 컬럼.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn histplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.scatterplot"></a>

### scatterplot

```python
def scatterplot(df: DataFrame,
                xname: str,
                yname: str,
                hue=None,
                title: str | None = None,
                palette: str = None,
                width: int = config.width,
                height: int = config.height,
                linewidth: float = config.line_width,
                dpi: int = config.dpi,
                save_path: str = None,
                callback: any = None,
                ax: Axes = None,
                **params) -> None
```

산점도를 그린다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str_ - x축 컬럼.
- `yname` _str_ - y축 컬럼.
- `hue` _str|None_ - 범주 컬럼.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn scatterplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.regplot"></a>

### regplot

```python
def regplot(df: DataFrame,
            xname: str,
            yname: str,
            title: str | None = None,
            palette: str = None,
            width: int = config.width,
            height: int = config.height,
            linewidth: float = config.line_width,
            dpi: int = config.dpi,
            save_path: str = None,
            callback: any = None,
            ax: Axes = None,
            **params) -> None
```

단순 회귀선이 포함된 산점도를 그린다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str_ - 독립변수 컬럼.
- `yname` _str_ - 종속변수 컬럼.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 선/점 색상.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn regplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.lmplot"></a>

### lmplot

```python
def lmplot(df: DataFrame,
           xname: str,
           yname: str,
           hue=None,
           title: str | None = None,
           palette: str = None,
           width: int = config.width,
           height: int = config.height,
           linewidth: float = config.line_width,
           dpi: int = config.dpi,
           save_path: str = None,
           **params) -> None
```

seaborn lmplot으로 선형 모델 시각화를 수행한다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str_ - 독립변수 컬럼.
- `yname` _str_ - 종속변수 컬럼.
- `hue` _str|None_ - 범주 컬럼.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `**params` - seaborn lmplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.pairplot"></a>

### pairplot

```python
def pairplot(df: DataFrame,
             xnames=None,
             title: str | None = None,
             diag_kind: str = "kde",
             hue=None,
             palette: str = None,
             width: int = config.height,
             height: int = config.height,
             linewidth: float = config.line_width,
             dpi: int = config.dpi,
             save_path: str = None,
             **params) -> None
```

연속형 변수의 숫자형 컬럼 쌍에 대한 관계를 그린다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xnames` _str|list|None_ - 대상 컬럼명.
  - None: 모든 연속형(숫자형) 데이터에 대해 처리.
  - str: 해당 컬럼에 대해서만 처리.
  - list: 주어진 컬럼들에 대해서만 처리.
  기본값은 None.
- `title` _str|None_ - 그래프 제목.
- `diag_kind` _str_ - 대각선 플롯 종류('kde' 등).
- `hue` _str|None_ - 범주 컬럼.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 기본 크기 및 해상도(컬럼 수에 비례해 확대됨).
- `**params` - seaborn pairplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.countplot"></a>

### countplot

```python
def countplot(df: DataFrame,
              xname: str,
              hue=None,
              title: str | None = None,
              palette: str = None,
              order: int = 1,
              width: int = config.width,
              height: int = config.height,
              linewidth: float = config.line_width,
              dpi: int = config.dpi,
              save_path: str = None,
              callback: any = None,
              ax: Axes = None,
              **params) -> None
```

범주 빈도 막대그래프를 그린다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str_ - 범주 컬럼.
- `hue` _str|None_ - 보조 범주 컬럼.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 팔레트 이름.
- `order` _int_ - 숫자형일 때 정렬 방식(1: 값 기준, 기타: 빈도 기준).
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn countplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.barplot"></a>

### barplot

```python
def barplot(df: DataFrame,
            xname: str,
            yname: str,
            hue=None,
            title: str | None = None,
            palette: str = None,
            width: int = config.width,
            height: int = config.height,
            linewidth: float = config.line_width,
            dpi: int = config.dpi,
            save_path: str = None,
            callback: any = None,
            ax: Axes = None,
            **params) -> None
```

막대그래프를 그린다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str_ - 범주 컬럼.
- `yname` _str_ - 값 컬럼.
- `hue` _str|None_ - 보조 범주 컬럼.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn barplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.boxenplot"></a>

### boxenplot

```python
def boxenplot(df: DataFrame,
              xname: str,
              yname: str,
              hue=None,
              title: str | None = None,
              palette: str = None,
              width: int = config.width,
              height: int = config.height,
              linewidth: float = config.line_width,
              dpi: int = config.dpi,
              save_path: str = None,
              callback: any = None,
              ax: Axes = None,
              **params) -> None
```

박스앤 위스커 확장(boxen) 플롯을 그린다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str_ - 범주 컬럼.
- `yname` _str_ - 값 컬럼.
- `hue` _str|None_ - 보조 범주 컬럼.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn boxenplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.violinplot"></a>

### violinplot

```python
def violinplot(df: DataFrame,
               xname: str,
               yname: str,
               hue=None,
               title: str | None = None,
               palette: str = None,
               width: int = config.width,
               height: int = config.height,
               linewidth: float = config.line_width,
               dpi: int = config.dpi,
               save_path: str = None,
               callback: any = None,
               ax: Axes = None,
               **params) -> None
```

바이올린 플롯을 그린다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str_ - 범주 컬럼.
- `yname` _str_ - 값 컬럼.
- `hue` _str|None_ - 보조 범주 컬럼.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn violinplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.pointplot"></a>

### pointplot

```python
def pointplot(df: DataFrame,
              xname: str,
              yname: str,
              hue=None,
              title: str | None = None,
              palette: str = None,
              width: int = config.width,
              height: int = config.height,
              linewidth: float = config.line_width,
              dpi: int = config.dpi,
              save_path: str = None,
              callback: any = None,
              ax: Axes = None,
              **params) -> None
```

포인트 플롯을 그린다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str_ - 범주 컬럼.
- `yname` _str_ - 값 컬럼.
- `hue` _str|None_ - 보조 범주 컬럼.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn pointplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.jointplot"></a>

### jointplot

```python
def jointplot(df: DataFrame,
              xname: str,
              yname: str,
              hue=None,
              title: str | None = None,
              palette: str = None,
              width: int = config.width,
              height: int = config.height,
              linewidth: float = config.line_width,
              dpi: int = config.dpi,
              save_path: str = None,
              **params) -> None
```

공동 분포(joint) 플롯을 그린다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `xname` _str_ - x축 컬럼.
- `yname` _str_ - y축 컬럼.
- `hue` _str|None_ - 범주 컬럼.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `**params` - seaborn jointplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.heatmap"></a>

### heatmap

```python
def heatmap(data: DataFrame,
            title: str | None = None,
            palette: str = None,
            width: int | None = None,
            height: int | None = None,
            linewidth: float = 0.25,
            dpi: int = config.dpi,
            save_path: str = None,
            callback: any = None,
            ax: Axes = None,
            **params) -> None
```

히트맵을 그린다(값 주석 포함).

**Arguments**:

- `data` _DataFrame_ - 행렬 형태 데이터.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 컬러맵 이름.
- `width` _int|None_ - 캔버스 가로 픽셀. None이면 자동 계산.
- `height` _int|None_ - 캔버스 세로 픽셀. None이면 자동 계산.
- `linewidth` _float_ - 격자 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn heatmap 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.convex_hull"></a>

### convex\_hull

```python
def convex_hull(data: DataFrame,
                xname: str,
                yname: str,
                hue: str,
                title: str | None = None,
                palette: str = None,
                width: int = config.width,
                height: int = config.height,
                linewidth: float = config.line_width,
                dpi: int = config.dpi,
                save_path: str = None,
                callback: any = None,
                ax: Axes = None,
                **params)
```

클러스터별 볼록 껍질(convex hull)과 산점도를 그린다.

**Arguments**:

- `data` _DataFrame_ - 시각화할 데이터.
- `xname` _str_ - x축 컬럼.
- `yname` _str_ - y축 컬럼.
- `hue` _str_ - 클러스터/범주 컬럼.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn scatterplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.kde_confidence_interval"></a>

### kde\_confidence\_interval

```python
def kde_confidence_interval(data: DataFrame,
                            xnames=None,
                            title: str | None = None,
                            clevel=0.95,
                            width: int = config.width,
                            height: int = config.height,
                            linewidth: float = config.line_width,
                            fill: bool = False,
                            dpi: int = config.dpi,
                            save_path: str = None,
                            callback: any = None,
                            ax: Axes = None) -> None
```

각 숫자 컬럼에 대해 KDE와 t-분포 기반 신뢰구간을 그린다.

**Arguments**:

- `data` _DataFrame_ - 시각화할 데이터.
- `xnames` _str|list|None_ - 대상 컬럼명.
  - None: 모든 연속형 데이터에 대해 처리.
  - str: 해당 컬럼에 대해서만 처리.
  - list: 주어진 컬럼들에 대해서만 처리.
  기본값은 None.
- `title` _str|None_ - 그래프 제목.
- `clevel` _float_ - 신뢰수준(0~1).
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `fill` _bool_ - KDE 채우기 여부.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
  

**Returns**:

  None

<a id="hs_plot.pvalue1_anotation"></a>

### pvalue1\_anotation

```python
def pvalue1_anotation(data: DataFrame,
                      target: str,
                      hue: str,
                      title: str | None = None,
                      pairs: list = None,
                      test: str = "t-test_ind",
                      text_format: str = "star",
                      loc: str = "outside",
                      width: int = config.width,
                      height: int = config.height,
                      linewidth: float = config.line_width,
                      dpi: int = config.dpi,
                      save_path: str = None,
                      callback: any = None,
                      ax: Axes = None,
                      **params) -> None
```

statannotations를 이용해 상자그림에 p-value 주석을 추가한다.

**Arguments**:

- `data` _DataFrame_ - 시각화할 데이터.
- `target` _str_ - 값 컬럼명.
- `hue` _str_ - 그룹 컬럼명.
- `title` _str|None_ - 그래프 제목.
- `pairs` _list|None_ - 비교할 (group_a, group_b) 튜플 목록. None이면 hue 컬럼의 모든 고유값 조합을 자동 생성.
- `test` _str_ - 적용할 통계 검정 이름.
- `text_format` _str_ - 주석 형식('star' 등).
- `loc` _str_ - 주석 위치.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn boxplot 추가 인자.
  

**Returns**:

  None

<a id="hs_plot.ols_residplot"></a>

### ols\_residplot

```python
def ols_residplot(fit,
                  title: str | None = None,
                  lowess: bool = False,
                  mse: bool = False,
                  width: int = config.width,
                  height: int = config.height,
                  linewidth: float = config.line_width,
                  dpi: int = config.dpi,
                  save_path: str = None,
                  callback: any = None,
                  ax: Axes = None,
                  **params) -> None
```

잔차도를 그린다(선택적으로 MSE 범위와 LOWESS 포함).

회귀모형의 선형성을 시각적으로 평가하기 위한 그래프를 생성한다.
점들이 무작위로 흩어져 있으면 선형성 가정이 만족되며,
특정 패턴이 보이면 비선형 관계가 존재할 가능성을 시사한다.

**Arguments**:

- `fit` - 회귀 모형 객체 (statsmodels의 RegressionResultsWrapper).
  fit.resid와 fit.fittedvalues를 통해 잔차와 적합값을 추출한다.
- `title` _str|None_ - 그래프 제목.
- `lowess` _bool_ - LOWESS 스무딩 적용 여부.
- `mse` _bool_ - √MSE, 2√MSE, 3√MSE 대역선과 비율 표시 여부.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `save_path` _str|None_ - 저장 경로.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - seaborn residplot 추가 인자.
  

**Returns**:

  None
  

**Examples**:

```python
from hossam import *
fit = hs_stats.ols(data, yname='target', report=False)
residplot(fit, lowess=True, mse=True)
```

<a id="hs_plot.ols_qqplot"></a>

### ols\_qqplot

```python
def ols_qqplot(fit,
               title: str | None = None,
               line: str = 's',
               width: int = config.width,
               height: int = config.height,
               linewidth: float = config.line_width,
               dpi: int = config.dpi,
               save_path: str = None,
               callback: any = None,
               ax: Axes = None,
               **params) -> None
```

표준화된 잔차의 정규성 확인을 위한 QQ 플롯을 그린다.

statsmodels의 qqplot 함수를 사용하여 최적화된 Q-Q plot을 생성한다.
이론적 분위수와 표본 분위수를 비교하여 잔차의 정규성을 시각적으로 평가한다.

**Arguments**:

- `fit` - 회귀 모형 객체 (statsmodels의 RegressionResultsWrapper 등).
  fit.resid 속성을 통해 잔차를 추출하여 정규성을 확인한다.
- `title` _str|None_ - 그래프 제목.
- `line` _str_ - 참조선의 유형. 기본값 's' (standardized).
  - 's': 표본의 표준편차와 평균을 기반으로 조정된 선 (권장)
  - 'r': 실제 점들에 대한 회귀선 (데이터 추세 반영)
  - 'q': 1사분위수와 3사분위수를 통과하는 선
  - '45': 45도 대각선 (이론적 정규분포)
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `save_path` _str|None_ - 저장 경로.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - statsmodels qqplot 추가 인자.
  

**Returns**:

  None
  

**Examples**:

```python
from hossam import *
# 선형회귀 모형 적합
fit = hs_stats.ols(data, yname='target', report=False)
# 표준화된 선 (권장)
qqplot(fit)
# 회귀선 (데이터 추세 반영)
qqplot(fit, line='r')
# 45도 대각선 (전통적 방식)
qqplot(fit, line='45')
```

<a id="hs_plot.distribution_by_class"></a>

### distribution\_by\_class

```python
def distribution_by_class(data: DataFrame,
                          title: str | None = None,
                          xnames: list = None,
                          hue: str = None,
                          type: str = "kde",
                          bins: any = 5,
                          palette: str = None,
                          fill: bool = False,
                          width: int = config.width,
                          height: int = config.height,
                          linewidth: float = config.line_width,
                          dpi: int = config.dpi,
                          save_path: str = None,
                          callback: any = None) -> None
```

클래스별로 각 숫자형 특징의 분포를 KDE 또는 히스토그램으로 그린다.

**Arguments**:

- `data` _DataFrame_ - 시각화할 데이터.
- `xnames` _list|None_ - 대상 컬럼 목록(None이면 전 컬럼).
- `hue` _str|None_ - 클래스 컬럼.
- `title` _str|None_ - 그래프 제목.
- `type` _str_ - 'kde' | 'hist' | 'histkde'.
- `bins` _int|sequence|None_ - 히스토그램 구간.
- `palette` _str|None_ - 팔레트 이름.
- `fill` _bool_ - KDE 채움 여부.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
  

**Returns**:

  None

<a id="hs_plot.scatter_by_class"></a>

### scatter\_by\_class

```python
def scatter_by_class(data: DataFrame,
                     yname: str,
                     group: list | None = None,
                     hue: str | None = None,
                     title: str | None = None,
                     palette: str | None = None,
                     outline: bool = False,
                     width: int = config.width,
                     height: int = config.height,
                     linewidth: float = config.line_width,
                     dpi: int = config.dpi,
                     save_path: str = None,
                     callback: any = None) -> None
```

종속변수(y)와 각 연속형 독립변수(x) 간 산점도/볼록껍질을 그린다.

**Arguments**:

- `data` _DataFrame_ - 시각화할 데이터.
- `yname` _str_ - 종속변수 컬럼명(필수).
- `group` _list|None_ - x 컬럼 목록 또는 [[x, y], ...] 형태. None이면 자동 생성.
- `hue` _str|None_ - 클래스 컬럼.
- `title` _str|None_ - 그래프 제목.
- `palette` _str|None_ - 팔레트 이름.
- `outline` _bool_ - 볼록 껍질을 표시할지 여부.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
  

**Returns**:

  None

<a id="hs_plot.categorical_target_distribution"></a>

### categorical\_target\_distribution

```python
def categorical_target_distribution(data: DataFrame,
                                    yname: str,
                                    hue: list | str | None = None,
                                    title: str | None = None,
                                    kind: str = "box",
                                    kde_fill: bool = True,
                                    palette: str | None = None,
                                    width: int = config.width,
                                    height: int = config.height,
                                    linewidth: float = config.line_width,
                                    dpi: int = config.dpi,
                                    cols: int = 2,
                                    save_path: str = None,
                                    callback: any = None) -> None
```

명목형 변수별로 종속변수 분포 차이를 시각화한다.

**Arguments**:

- `data` _DataFrame_ - 시각화할 데이터.
- `yname` _str_ - 종속변수 컬럼명(연속형 추천).
- `hue` _list|str|None_ - 명목형 독립변수 목록. None이면 자동 탐지.
- `title` _str|None_ - 그래프 제목.
- `kind` _str_ - 'box', 'violin', 'kde'.
- `kde_fill` _bool_ - kind='kde'일 때 영역 채우기 여부.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 개별 서브플롯 가로 픽셀.
- `height` _int_ - 개별 서브플롯 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 해상도.
- `cols` _int_ - 서브플롯 열 수.
- `callback` _Callable|None_ - Axes 후처리 콜백.
  

**Returns**:

  None

<a id="hs_plot.roc_curve_plot"></a>

### roc\_curve\_plot

```python
def roc_curve_plot(fit,
                   y: np.ndarray | pd.Series = None,
                   X: pd.DataFrame | np.ndarray = None,
                   title: str | None = None,
                   width: int = config.height,
                   height: int = config.height,
                   linewidth: float = config.line_width,
                   dpi: int = config.dpi,
                   save_path: str = None,
                   callback: any = None,
                   ax: Axes = None) -> None
```

로지스틱 회귀 적합 결과의 ROC 곡선을 시각화한다.

**Arguments**:

- `fit` - statsmodels Logit 결과 객체 (`fit.predict()`로 예측 확률을 계산 가능해야 함).
- `y` _array-like|None_ - 외부 데이터의 실제 레이블. 제공 시 이를 실제값으로 사용.
- `X` _array-like|None_ - 외부 데이터의 설계행렬(독립변수). 제공 시 해당 데이터로 예측 확률 계산.
- `title` _str|None_ - 그래프 제목.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes. None이면 새로 생성.
  

**Notes**:

  - 실제값: `y`가 주어지면 이를 사용, 없으면 `fit.model.endog`를 사용합니다.
  - 예측 확률: `X`가 주어지면 `fit.predict(X)`를 사용, 없으면 `fit.predict(fit.model.exog)`를 사용합니다.
  

**Returns**:

  None

<a id="hs_plot.confusion_matrix_plot"></a>

### confusion\_matrix\_plot

```python
def confusion_matrix_plot(fit,
                          title: str | None = None,
                          threshold: float = 0.5,
                          width: int = config.width,
                          height: int = config.height,
                          dpi: int = config.dpi,
                          save_path: str = None,
                          callback: any = None,
                          ax: Axes = None) -> None
```

로지스틱 회귀 적합 결과의 혼동행렬을 시각화한다.

**Arguments**:

- `fit` - statsmodels Logit 결과 객체 (`fit.predict()`로 예측 확률을 계산 가능해야 함).
- `title` _str|None_ - 그래프 제목.
- `threshold` _float_ - 예측 확률을 이진 분류로 변환할 임계값. 기본값 0.5.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `dpi` _int_ - 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes. None이면 새로 생성.
  

**Returns**:

  None

<a id="hs_plot.radarplot"></a>

### radarplot

```python
def radarplot(df: DataFrame,
              columns: list = None,
              hue: str = None,
              title: str | None = None,
              normalize: bool = True,
              fill: bool = True,
              fill_alpha: float = 0.25,
              palette: str = None,
              width: int = config.width,
              height: int = config.height,
              linewidth: float = config.line_width,
              dpi: int = config.dpi,
              save_path: str = None,
              callback: any = None,
              ax: Axes = None,
              **params) -> None
```

레이더 차트(방사형 차트)를 그린다.

**Arguments**:

- `df` _DataFrame_ - 시각화할 데이터.
- `columns` _list|None_ - 레이더 차트에 표시할 컬럼 목록. None이면 모든 숫자형 컬럼 사용.
- `hue` _str|None_ - 집단 구분 컬럼. None이면 각 행을 개별 객체로 표시.
- `title` _str|None_ - 그래프 제목.
- `normalize` _bool_ - 0-1 범위로 정규화 여부. 기본값 True.
- `fill` _bool_ - 영역 채우기 여부.
- `fill_alpha` _float_ - 채움 투명도.
- `palette` _str|None_ - 팔레트 이름.
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 해상도.
- `callback` _Callable|None_ - Axes 후처리 콜백.
- `ax` _Axes|None_ - 외부에서 전달한 Axes.
- `**params` - 추가 플롯 옵션.
  

**Returns**:

  None

<a id="hs_plot.distribution_plot"></a>

### distribution\_plot

```python
def distribution_plot(data: DataFrame,
                      column: str,
                      title: str | None = None,
                      clevel: float = 0.95,
                      orient: str = "h",
                      hue: str | None = None,
                      kind: str = "boxplot",
                      width: int = config.width,
                      height: int = config.height,
                      linewidth: float = config.line_width,
                      dpi: int = config.dpi,
                      save_path: str = None,
                      callback: any = None) -> None
```

연속형 데이터의 분포를 KDE와 Boxplot으로 시각화한다.

1행 2열의 서브플롯을 생성하여:
- 왼쪽: KDE with 신뢰구간
- 오른쪽: Boxplot

**Arguments**:

- `data` _DataFrame_ - 시각화할 데이터.
- `column` _str_ - 분석할 컬럼명.
- `title` _str|None_ - 그래프 제목.
- `clevel` _float_ - KDE 신뢰수준 (0~1). 기본값 0.95.
- `orient` _str_ - Boxplot 방향 ('v' 또는 'h'). 기본값 'h'.
- `hue` _str|None_ - 명목형 컬럼명. 지정하면 각 범주별로 행을 늘려 KDE와 boxplot을 그림.
- `kind` _str_ - 두 번째 그래프의 유형 (boxplot, hist). 기본값 "boxplot".
- `width` _int_ - 캔버스 가로 픽셀.
- `height` _int_ - 캔버스 세로 픽셀.
- `linewidth` _float_ - 선 굵기.
- `dpi` _int_ - 그림 크기 및 해상도.
- `save_path` _str|None_ - 저장 경로.
- `callback` _Callable|None_ - Axes 후처리 콜백.
  

**Returns**:

  None

<a id="hs_util"></a>

## hs\_util

<a id="hs_util.my_packages"></a>

### my\_packages

```python
def my_packages()
```

현재 파이썬 인터프리터에 설치된 모든 패키지의 이름과 버전을
패키지 이름순으로 정렬하여 pandas DataFrame으로 반환합니다.

**Returns**:

- `pd.DataFrame` - columns=['name', 'version']

<a id="hs_util.make_normalize_values"></a>

### make\_normalize\_values

```python
def make_normalize_values(mean: float,
                          std: float,
                          size: int = 100,
                          round: int = 2) -> np.ndarray
```

정규분포를 따르는 데이터를 생성한다.

**Arguments**:

- `mean` _float_ - 평균
- `std` _float_ - 표준편차
- `size` _int, optional_ - 데이터 크기. Defaults to 100.
- `round` _int, optional_ - 소수점 반올림 자리수. Defaults to 2.
  

**Returns**:

- `np.ndarray` - 정규분포를 따르는 데이터
  

**Examples**:

```python
from hossam import *
x = hs.util.make_normalize_values(mean=0.0, std=1.0, size=100)
```

<a id="hs_util.make_normalize_data"></a>

### make\_normalize\_data

```python
def make_normalize_data(means: list | None = None,
                        stds: list | None = None,
                        sizes: list | None = None,
                        rounds: int = 2) -> DataFrame
```

정규분포를 따르는 데이터프레임을 생성한다.

**Arguments**:

- `means` _list, optional_ - 평균 목록. Defaults to [0, 0, 0].
- `stds` _list, optional_ - 표준편차 목록. Defaults to [1, 1, 1].
- `sizes` _list, optional_ - 데이터 크기 목록. Defaults to [100, 100, 100].
- `rounds` _int, optional_ - 반올림 자리수. Defaults to 2.
  

**Returns**:

- `DataFrame` - 정규분포를 따르는 데이터프레임

<a id="hs_util.pretty_table"></a>

### pretty\_table

```python
def pretty_table(data: DataFrame,
                 tablefmt="simple",
                 headers: str = "keys") -> None
```

`tabulate`를 사용해 DataFrame을 단순 표 형태로 출력한다.

**Arguments**:

- `data` _DataFrame_ - 출력할 데이터프레임
- `tablefmt` _str, optional_ - `tabulate` 테이블 포맷. Defaults to "simple".
- `headers` _str | list, optional_ - 헤더 지정 방식. Defaults to "keys".
  

**Returns**:

  None
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame
hs_util.pretty_table(DataFrame({"a":[1,2],"b":[3,4]}))
```

<a id="hs_util.load_data"></a>

### load\_data

```python
def load_data(key: str,
              index_col: str | None = None,
              timeindex: bool = False,
              info: bool = True,
              categories: list | None = None,
              local: str | None = None) -> DataFrame
```

데이터 키를 통해 데이터를 로드한 뒤 기본 전처리/출력을 수행한다.

**Arguments**:

- `key` _str_ - 데이터 키 (metadata.json에 정의된 데이터 식별자)
- `index_col` _str, optional_ - 인덱스로 설정할 컬럼명. Defaults to None.
- `timeindex` _bool, optional_ - True일 경우 인덱스를 시계열(DatetimeIndex)로 설정한다. Defaults to False.
- `info` _bool, optional_ - True일 경우 데이터 정보(head, tail, 기술통계, 카테고리 정보)를 출력한다. Defaults to True.
- `categories` _list, optional_ - 카테고리 dtype으로 설정할 컬럼명 목록. Defaults to None.
- `local` _str, optional_ - 원격 데이터 대신 로컬 메타데이터 경로를 사용한다. Defaults to None.
  

**Returns**:

- `DataFrame` - 전처리(인덱스 설정, 카테고리 변환)가 완료된 데이터프레임
  

**Examples**:

```python
from hossam import *
df = hs_util.load_data("AD_SALES", index_col=None, timeindex=False, info=False)
```

<a id="hs_timeserise"></a>

## hs\_timeserise

<a id="hs_timeserise.diff"></a>

### diff

```python
def diff(data: DataFrame,
         yname: str,
         plot: bool = True,
         max_diff: int = None,
         figsize: tuple = (10, 5),
         dpi: int = 100) -> None
```

시계열 데이터의 정상성을 검정하고 차분을 통해 정상성을 확보한다.

ADF(Augmented Dickey-Fuller) 검정을 사용하여 시계열 데이터의 정상성을 확인한다.
정상성을 만족하지 않으면(p-value > 0.05) 차분을 반복 수행하여 정상 시계열로 변환한다.
ARIMA 모델링 전 필수적인 전처리 과정이다.

**Arguments**:

- `data` _DataFrame_ - 시계열 데이터프레임. 인덱스가 datetime 형식이어야 한다.
- `yname` _str_ - 정상성 검정 및 차분을 수행할 대상 컬럼명.
- `plot` _bool, optional_ - 각 차분 단계마다 시계열 그래프를 표시할지 여부.
  기본값은 True.
- `max_diff` _int, optional_ - 최대 차분 횟수 제한. None이면 정상성을 만족할 때까지 반복.
  과도한 차분을 방지하기 위해 설정 권장. 기본값은 None.
- `figsize` _tuple, optional_ - 그래프 크기 (width, height). 기본값은 (10, 5).
- `dpi` _int, optional_ - 그래프 해상도. 기본값은 100.
  

**Returns**:

- `DataFrame` - 정상성을 만족하는 차분된 데이터프레임.
  

**Notes**:

  - ADF 검정의 귀무가설: 시계열이 단위근(unit root)을 가진다 (비정상).
  - p-value ≤ 0.05일 때 귀무가설 기각 → 정상 시계열.
  - 일반적으로 1~2차 차분으로 충분하며, 과도한 차분은 모델 성능을 저하시킬 수 있다.
  - 각 반복마다 ADF 검정 통계량, p-value, 기각값을 출력한다.
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame, date_range

# 기본 사용 (정상성 만족까지 자동 차분):
df = DataFrame({'value': [100, 102, 105, 110, 120]},
                  index=date_range('2020-01', periods=5, freq='M'))
stationary_df = hs_timeseries.diff(df, 'value')

# 최대 2차 차분으로 제한:
stationary_df = hs_timeseries.diff(df, 'value', max_diff=2)

# 그래프 없이 실행:
stationary_df = hs_timeseries.diff(df, 'value', plot=False)
```

<a id="hs_timeserise.rolling"></a>

### rolling

```python
def rolling(data: Series,
            window: int,
            plot: bool = True,
            figsize: tuple = (10, 5),
            dpi: int = 100) -> Series
```

단순 이동평균(Simple Moving Average, SMA)을 계산한다.

지정된 윈도우(기간) 내 데이터의 산술평균을 계산하여 시계열의 추세를 파악한다.
노이즈를 제거하고 장기 추세를 시각화하는 데 유용하다.

**Arguments**:

- `data` _Series_ - 시계열 데이터. 인덱스가 datetime 형식이어야 한다.
- `window` _int_ - 이동평균 계산 윈도우 크기 (기간).
- `예` - window=7이면 최근 7개 데이터의 평균을 계산.
- `plot` _bool, optional_ - 이동평균 그래프를 표시할지 여부. 기본값은 True.
- `figsize` _tuple, optional_ - 그래프 크기 (width, height). 기본값은 (10, 5).
- `dpi` _int, optional_ - 그래프 해상도. 기본값은 100.
  

**Returns**:

- `Series` - 이동평균이 계산된 시계열 데이터.
  처음 (window-1)개 값은 NaN으로 표시된다.
  

**Notes**:

  - 단순 이동평균은 모든 과거 데이터에 동일한 가중치를 부여한다.
  - 윈도우가 클수록 더 부드러운 곡선이 되지만 최신 변화에 덜 민감해진다.
  - 계절성 파악을 위해서는 계절 주기와 동일한 윈도우 사용을 권장한다.
  

**Examples**:

```python
from hossam import *
from pandas import Series, date_range

# 7일 이동평균 계산:
data = Series([10, 12, 13, 15, 14, 16, 18],
                index=date_range('2020-01-01', periods=7))
ma7 = hs_timeseries.rolling(data, window=7)

# 30일 이동평균, 그래프 없이:
ma30 = hs_timeseries.rolling(data, window=30, plot=False)
```

<a id="hs_timeserise.ewm"></a>

### ewm

```python
def ewm(data: Series,
        span: int,
        plot: bool = True,
        figsize: tuple = (10, 5),
        dpi: int = 100) -> Series
```

지수가중이동평균(Exponential Weighted Moving Average, EWMA)을 계산한다.

최근 데이터에 더 높은 가중치를 부여하는 이동평균으로, 단순이동평균보다
최신 변화에 민감하게 반응한다. 주가 분석, 수요 예측 등에 널리 사용된다.

**Arguments**:

- `data` _Series_ - 시계열 데이터. 인덱스가 datetime 형식이어야 한다.
- `span` _int_ - 지수가중이동평균 계산 기간 (span).
  α(smoothing factor) = 2 / (span + 1)로 계산된다.
  span이 클수록 과거 데이터의 영향이 천천히 감소한다.
- `plot` _bool, optional_ - EWMA 그래프를 표시할지 여부. 기본값은 True.
- `figsize` _tuple, optional_ - 그래프 크기 (width, height). 기본값은 (10, 5).
- `dpi` _int, optional_ - 그래프 해상도. 기본값은 100.
  

**Returns**:

- `Series` - 지수가중이동평균이 계산된 시계열 데이터.
  

**Notes**:

  - EWMA는 최근 관측값에 지수적으로 감소하는 가중치를 부여한다.
  - 단순이동평균과 달리 모든 과거 데이터를 사용하되 시간에 따라 가중치가 감소한다.
  - span=12는 대략 12개 기간의 정보를 반영하는 것으로 해석할 수 있다.
  - α = 2/(span+1) 공식으로 smoothing factor가 결정된다.
  

**Examples**:

```python
from hossam import ewm
from pandas import Series, date_range

# 12기간 지수가중이동평균:
data = Series([10, 12, 13, 15, 14, 16, 18],
                 index=date_range('2020-01-01', periods=7))
ewma = hs_timeseries.ewm(data, span=12)

# 단기 추세 파악 (span=5):
ewma_short = hs_timeseries.ewm(data, span=5, plot=False)
```

<a id="hs_timeserise.seasonal_decompose"></a>

### seasonal\_decompose

```python
def seasonal_decompose(data: Series,
                       model: str = "additive",
                       plot: bool = True,
                       figsize: tuple = (10, 5),
                       dpi: int = 100)
```

시계열을 추세(Trend), 계절성(Seasonal), 잔차(Residual) 성분으로 분해한다.

classical decomposition 기법을 사용하여 시계열을 구조적 성분으로 분해한다.
계절성 패턴 파악, 추세 분석, 이상치 탐지 등에 유용하다.

**Arguments**:

- `data` _Series_ - 시계열 데이터. 인덱스가 datetime 형식이며 일정한 주기를 가져야 한다.
- `model` _str, optional_ - 분해 모델 유형.
  - "additive": 가법 모델 (Y = T + S + R)
  계절성의 크기가 일정할 때 사용.
  - "multiplicative": 승법 모델 (Y = T × S × R)
  계절성의 크기가 추세에 비례하여 변할 때 사용.
  기본값은 "additive".
- `plot` _bool, optional_ - 분해된 4개 성분(원본, 추세, 계절, 잔차)의
  그래프를 표시할지 여부. 기본값은 True.
- `figsize` _tuple, optional_ - 각 그래프의 기본 크기 (width, height).
  실제 출력은 높이가 4배로 조정된다. 기본값은 (10, 5).
- `dpi` _int, optional_ - 그래프 해상도. 기본값은 100.
  

**Returns**:

- `DataFrame` - 분해된 성분을 포함한 데이터프레임. 다음 컬럼 포함:
  - original: 원본 시계열
  - trend: 추세 성분 (장기적 방향성)
  - seasonal: 계절 성분 (주기적 패턴)
  - resid: 잔차 성분 (설명되지 않는 불규칙 변동)
  

**Raises**:

- `ValueError` - model이 "additive" 또는 "multiplicative"가 아닐 경우.
  

**Notes**:

  - 가법 모델: 계절 변동폭이 일정한 경우 (예: 매년 여름 +10도).
  - 승법 모델: 계절 변동폭이 추세에 비례하는 경우 (예: 매년 여름 +20%).
  - 데이터에 0 또는 음수가 있으면 승법 모델 사용 불가.
  - 주기(period)는 데이터의 빈도(frequency)에서 자동 추론된다.
  

**Examples**:

```python
from hossam import *
from pandas import Series, date_range

# 월별 데이터 가법 분해:
data = Series([100, 120, 110, 130, 150, 140],
                 index=date_range('2020-01', periods=6, freq='M'))
components = hs_timeseries.seasonal_decompose(data, model='additive')

# 승법 모델 사용:
components = hs_timeseries.seasonal_decompose(data, model='multiplicative', plot=False)
print(components[['trend', 'seasonal']].head())
```

<a id="hs_timeserise.train_test_split"></a>

### train\_test\_split

```python
def train_test_split(data: DataFrame, test_size: float = 0.2) -> tuple
```

시계열 데이터를 시간 순서를 유지하며 학습/테스트 세트로 분할한다.

일반적인 random split과 달리 시간 순서를 엄격히 유지하여 분할한다.
미래 데이터가 과거 예측에 사용되는 data leakage를 방지한다.

**Arguments**:

- `data` _DataFrame_ - 시계열 데이터프레임.
  인덱스가 시간 순서대로 정렬되어 있어야 한다.
- `test_size` _float, optional_ - 테스트 데이터 비율 (0~1).
- `예` - 0.2는 전체 데이터의 마지막 20%를 테스트 세트로 사용.
  기본값은 0.2.
  

**Returns**:

- `tuple` - (train, test) 형태의 튜플.
  - train (DataFrame): 학습 데이터 (앞부분)
  - test (DataFrame): 테스트 데이터 (뒷부분)
  

**Notes**:

  - 시계열 데이터는 랜덤 분할이 아닌 순차 분할을 사용해야 한다.
  - 학습 데이터는 항상 테스트 데이터보다 시간적으로 앞선다.
  - Cross-validation이 필요한 경우 TimeSeriesSplit 사용을 권장한다.
  - 일반적으로 test_size는 0.1~0.3 범위를 사용한다.
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame, date_range

# 80:20 분할 (기본):
df = DataFrame({'value': range(100)}, index=date_range('2020-01-01', periods=100))
train, test = hs_timeseries.train_test_split(df)
print(len(train), len(test))  # 80, 20

# 70:30 분할:
train, test = hs_timeseries.train_test_split(df, test_size=0.3)
print(len(train), len(test))  # 70, 30
```

<a id="hs_timeserise.acf_plot"></a>

### acf\_plot

```python
def acf_plot(data: Series,
             figsize: tuple = (10, 5),
             dpi: int = 100,
             callback: any = None)
```

자기상관함수(ACF, Autocorrelation Function) 그래프를 시각화한다.

시계열 데이터와 시차(lag)를 둔 자기 자신과의 상관계수를 계산하여
시계열의 자기상관 구조를 파악한다. ARIMA 모델의 MA(q) 차수 결정에 사용된다.

**Arguments**:

- `data` _Series_ - 시계열 데이터. 정상성을 만족하는 것이 권장된다.
- `figsize` _tuple, optional_ - 그래프 크기 (width, height). 기본값은 (10, 5).
- `dpi` _int, optional_ - 그래프 해상도. 기본값은 100.
- `callback` _Callable, optional_ - 추가 그래프 설정을 위한 콜백 함수.
  함수는 ax(Axes) 객체를 인자로 받아야 한다. 기본값은 None.
  

**Notes**:

  - ACF는 y(t)와 y(t-k) 간의 상관계수를 lag k에 대해 표시한다.
  - 파란색 영역(신뢰구간)을 벗어나는 lag는 통계적으로 유의미한 자기상관을 나타낸다.
  - MA(q) 모델: ACF가 lag q 이후 급격히 0으로 수렴한다.
  - AR(p) 모델: ACF가 점진적으로 감소한다.
  - 계절성이 있으면 계절 주기마다 높은 ACF 값이 나타난다.
  

**Examples**:

```python
from hossam import *
from pandas import Series, date_range

# 기본 ACF 플롯:
data = Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 index=date_range('2020-01-01', periods=10))
hs_timeseries.acf_plot(data)

# 콜백으로 제목 추가:
hs_timeseries.acf_plot(data, callback=lambda ax: ax.set_title('My ACF Plot'))
```

<a id="hs_timeserise.pacf_plot"></a>

### pacf\_plot

```python
def pacf_plot(data: Series,
              figsize: tuple = (10, 5),
              dpi: int = 100,
              callback: any = None)
```

편자기상관함수(PACF, Partial Autocorrelation Function) 그래프를 시각화한다.

중간 lag의 영향을 제거한 순수한 자기상관을 계산하여 표시한다.
ARIMA 모델의 AR(p) 차수 결정에 사용된다.

**Arguments**:

- `data` _Series_ - 시계열 데이터. 정상성을 만족하는 것이 권장된다.
- `figsize` _tuple, optional_ - 그래프 크기 (width, height). 기본값은 (10, 5).
- `dpi` _int, optional_ - 그래프 해상도. 기본값은 100.
- `callback` _Callable, optional_ - 추가 그래프 설정을 위한 콜백 함수.
  함수는 ax(Axes) 객체를 인자로 받아야 한다. 기본값은 None.
  

**Notes**:

  - PACF는 중간 시차의 영향을 제거한 y(t)와 y(t-k) 간의 상관계수이다.
  - ACF와 달리 간접적 영향을 배제하고 직접적 관계만 측정한다.
  - AR(p) 모델: PACF가 lag p 이후 급격히 0으로 수렴한다.
  - MA(q) 모델: PACF가 점진적으로 감소한다.
  - 파란색 영역(신뢰구간)을 벗어나는 lag가 AR 항의 개수를 나타낸다.
  

**Examples**:

```python
from hossam import *
from pandas import Series, date_range

# 기본 PACF 플롯:
data = Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 index=date_range('2020-01-01', periods=10))
hs_timeseries.pacf_plot(data)

# 콜백으로 커스터마이징:
hs_timeseries.pacf_plot(data, callback=lambda ax: ax.set_ylabel('Partial Correlation'))
```

<a id="hs_timeserise.acf_pacf_plot"></a>

### acf\_pacf\_plot

```python
def acf_pacf_plot(data: Series,
                  figsize: tuple = (10, 5),
                  dpi: int = 100,
                  callback: any = None)
```

ACF와 PACF 그래프를 동시에 시각화하여 ARIMA 차수를 결정한다.

ACF와 PACF를 함께 분석하여 ARIMA(p,d,q) 모델의 p(AR 차수)와 q(MA 차수)를
경험적으로 선택할 수 있다.

**Arguments**:

- `data` _Series_ - 시계열 데이터. 정상성을 만족하는 것이 권장된다.
- `figsize` _tuple, optional_ - 각 그래프의 기본 크기 (width, height).
  실제 출력은 높이가 2배로 조정된다. 기본값은 (10, 5).
- `dpi` _int, optional_ - 그래프 해상도. 기본값은 100.
- `callback` _Callable, optional_ - 추가 그래프 설정을 위한 콜백 함수.
  함수는 ax1, ax2(두 개의 Axes) 객체를 인자로 받아야 한다. 기본값은 None.
  

**Notes**:

  ARIMA 차수 선택 가이드:
  
  - AR(p) 모델: ACF는 점진 감소, PACF는 lag p 이후 절단
  - MA(q) 모델: ACF는 lag q 이후 절단, PACF는 점진 감소
  - ARMA(p,q) 모델: 둘 다 점진 감소
  
  실전에서는 auto_arima를 사용한 자동 선택도 권장된다.
  

**Examples**:

```python
from hossam import *
from pandas import Series, date_range

# ARIMA 모델링 전 차수 탐색:
data = Series([10, 12, 13, 15, 14, 16, 18, 20],
                 index=date_range('2020-01-01', periods=8))

# 1차 차분 후 ACF/PACF 플롯:
stationary = hs_timeseries.diff(data, 'value')
hs_timeseries.acf_pacf_plot(stationary)

<a id="hs_timeserise.arima"></a>

### arima

```python
def arima(train: Series,
      test: Series,
      auto: bool = False,
      p: int = 3,
      d: int = 3,
      q: int = 3,
      s: int = None,
      periods: int = 0,
      figsize: tuple = (15, 5),
      dpi: int = 100) -> ARIMA
```

ARIMA 또는 SARIMA 모델을 학습하고 예측 결과를 시각화한다.

ARIMA(p,d,q) 또는 SARIMA(p,d,q)(P,D,Q,s) 모델을 구축하여 시계열 예측을 수행한다.
auto=True로 설정하면 pmdarima의 auto_arima로 최적 하이퍼파라미터를 자동 탐색한다.

**Arguments**:

- `train` _Series_ - 학습용 시계열 데이터. 정상성을 만족해야 한다.
- `test` _Series_ - 테스트용 시계열 데이터. 모델 평가에 사용된다.
- `auto` _bool, optional_ - auto_arima로 최적 (p,d,q) 자동 탐색 여부.
  - False: 수동 지정한 p,d,q 사용
  - True: AIC 기반 그리드 서치로 최적 모델 탐색
  기본값은 False.
- `p` _int, optional_ - AR(AutoRegressive) 차수. 과거 값의 영향을 모델링.
  PACF 그래프를 참고하여 결정. auto=True일 때 max_p로 사용. 기본값은 3.
- `d` _int, optional_ - 차분(Differencing) 차수. 비정상 데이터를 정상화.
  diff() 결과를 참고하여 결정. 기본값은 3.
- `q` _int, optional_ - MA(Moving Average) 차수. 과거 오차의 영향을 모델링.
  ACF 그래프를 참고하여 결정. auto=True일 때 max_q로 사용. 기본값은 3.
- `s` _int, optional_ - 계절 주기(Seasonality). None이면 비계절 ARIMA.
- `예` - 월별 데이터는 s=12, 주별 데이터는 s=52.
  설정 시 SARIMA(p,d,q)(P,D,Q,s) 모델 사용. 기본값은 None.
- `periods` _int, optional_ - test 기간 이후 추가 예측 기간 수.
  0이면 test 기간까지만 예측. 기본값은 0.
- `figsize` _tuple, optional_ - 그래프 크기 (width, height). 기본값은 (15, 5).
- `dpi` _int, optional_ - 그래프 해상도. 기본값은 100.
  

**Returns**:

  ARIMA | AutoARIMA: 학습된 ARIMA 모델 객체.
  auto=False일 때 statsmodels.ARIMA,
  auto=True일 때 pmdarima.AutoARIMA 반환.
  

**Notes**:

  - ARIMA(p,d,q): p=AR차수, d=차분차수, q=MA차수
  - SARIMA(p,d,q)(P,D,Q,s): 계절 성분 추가 (P,D,Q,s)
  - 모델 선택: ACF/PACF 패턴 분석 또는 auto_arima 사용
  - 데이터는 반드시 정상성을 만족해야 하므로 diff() 먼저 수행 권장
  - auto=True는 시간이 오래 걸릴 수 있으나 최적 모델을 찾아줌
  

**Examples**:

```python
from hossam import *
from pandas import Series, date_range

# 수동으로 ARIMA(2,1,2) 모델 생성:
data = Series([100, 102, 105, 110, 115, 120, 125, 130],
                 index=date_range('2020-01', periods=8, freq='M'))
train, test = hs_timeseries.train_test_split(data, test_size=0.25)
model = hs_timeseries.arima(train, test, p=2, d=1, q=2)

# auto_arima로 최적 모델 탐색:
model = hs_timeseries.arima(train, test, auto=True)

# 계절성 모델 SARIMA(1,1,1)(1,1,1,12):
model = hs_timeseries.arima(train, test, p=1, d=1, q=1, s=12)
```

<a id="hs_timeserise.prophet"></a>

### prophet

```python
def prophet(train: DataFrame,
            test: DataFrame = None,
            periods: int = 0,
            freq: str = "D",
            report: bool = True,
            print_forecast: bool = False,
            figsize=(20, 8),
            dpi: int = 200,
            callback: any = None,
            **params) -> DataFrame
```

Facebook Prophet 모델을 학습하고 최적 모델을 반환한다.

Facebook(Meta)의 Prophet 라이브러리를 사용하여 시계열 예측 모델을 구축한다.
추세, 계절성, 휴일 효과를 자동으로 고려하며, 하이퍼파라미터 그리드 서치를 지원한다.

**Arguments**:

- `train` _DataFrame_ - 학습 데이터. 'ds'(날짜)와 'y'(값) 컬럼 필수.
  - ds: datetime 형식의 날짜/시간 컬럼
  - y: 예측 대상 수치형 컬럼
- `test` _DataFrame, optional_ - 테스트 데이터. 동일한 형식(ds, y).
  제공시 모델 성능 평가(MAE, MSE, RMSE)를 수행. 기본값은 None.
- `periods` _int, optional_ - test 기간 이후 추가로 예측할 기간 수.
  기본값은 0.
- `freq` _str, optional_ - 예측 빈도.
  - 'D': 일별 (Daily)
  - 'M': 월별 (Monthly)
  - 'Y': 연별 (Yearly)
  - 'H': 시간별 (Hourly)
  기본값은 'D'.
- `report` _bool, optional_ - 예측 결과 시각화 및 성분 분해 그래프 표시 여부.
  기본값은 True.
- `print_forecast` _bool, optional_ - 예측 결과 테이블 출력 여부.
  기본값은 False.
- `figsize` _tuple, optional_ - 그래프 크기 (width, height). 기본값은 (20, 8).
- `dpi` _int, optional_ - 그래프 해상도. 기본값은 200.
- `callback` _Callable, optional_ - Prophet 모델 객체를 커스터마이징하기 위한 콜백 함수.
  함수는 model 객체를 인자로 받아 add_regressor, add_seasonality 등 추가 설정 가능.
  기본값은 None.
- `**params` - Prophet 하이퍼파라미터 그리드 서치용 딕셔너리.
  주요 파라미터:
  - changepoint_prior_scale: 추세 변화점 유연성 (0.001~0.5)
  - seasonality_prior_scale: 계절성 강도 (0.01~10)
  - seasonality_mode: 'additive' 또는 'multiplicative'
  - changepoint_range: 변화점 탐지 범위 (0~1)
- `예` - changepoint_prior_scale=[0.001, 0.01, 0.1]
  

**Returns**:

- `tuple` - (best_model, best_params, best_score, best_forecast, best_pred)
  - best_model (Prophet): RMSE 기준 최적 모델
  - best_params (dict): 최적 하이퍼파라미터
  - best_score (float): 최소 RMSE 값
  - best_forecast (DataFrame): 전체 예측 결과 (ds, yhat, yhat_lower, yhat_upper 등)
  - best_pred (DataFrame): test 기간 예측값 (ds, yhat)
  

**Notes**:

  - Prophet은 결측치와 계절성을 자동 처리하므로 ARIMA보다 전처리가 적다.
  - changepoint_prior_scale이 클수록 더 유연하게 피팅 (과적합 주의).
  - 휴일 효과는 add_country_holidays() 또는 holidays 파라미터로 추가.
  - 외부 회귀변수는 callback에서 add_regressor()로 추가 가능.
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame, date_range

# 기본 사용 (단일 모델):
train = DataFrame({
            'ds': date_range('2020-01-01', periods=100),
            'y': range(100)
        })
model, params, score, forecast, pred = hs_timeseries.prophet(train)

하이퍼파라미터 그리드 서치:

model, params, score, forecast, pred = hs_timeseries.prophet(
        train,
        changepoint_prior_scale=[0.001, 0.01, 0.1],
        seasonality_prior_scale=[0.01, 0.1, 1.0],
        seasonality_mode=['additive', 'multiplicative']
)

# 휴일 효과 추가:
def add_holidays(m):
    m.add_country_holidays(country_name='KR')

model, _, _, _, _ = hs_timeseries.prophet(train, callback=add_holidays)
```

<a id="hs_timeserise.prophet_report"></a>

### prophet\_report

```python
def prophet_report(model: Prophet,
                   forecast: DataFrame,
                   pred: DataFrame,
                   test: DataFrame = None,
                   print_forecast: bool = False,
                   figsize: tuple = (20, 8),
                   dpi: int = 100) -> DataFrame
```

Prophet 모델의 예측 결과와 성분 분해를 시각화하고 성능을 평가한다.

학습된 Prophet 모델의 예측 결과, 변화점(changepoints), 신뢰구간을 시각화하고,
추세, 계절성 등 성분을 분해하여 표시한다. test 데이터가 있으면 성능 지표를 계산한다.

**Arguments**:

- `model` _Prophet_ - 학습된 Prophet 모델 객체.
- `forecast` _DataFrame_ - model.predict()의 반환값. 전체 예측 결과.
  다음 컬럼 포함: ds, yhat, yhat_lower, yhat_upper, trend, seasonal 등.
- `pred` _DataFrame_ - test 기간의 예측값. ds와 yhat 컬럼 포함.
- `test` _DataFrame, optional_ - 테스트 데이터. ds와 y 컬럼 필수.
  제공시 실제값과 비교하여 MAE, MSE, RMSE를 계산. 기본값은 None.
- `print_forecast` _bool, optional_ - 예측 결과 테이블 전체를 출력할지 여부.
  기본값은 False.
- `figsize` _tuple, optional_ - 그래프 크기 (width, height). 기본값은 (20, 8).
- `dpi` _int, optional_ - 그래프 해상도. 기본값은 100.
  

**Returns**:

- `None` - 출력만 수행하고 반환값 없음.
  

**Notes**:

  - 첨번째 그래프: 예측 결과 + 신뢰구간 + 변화점
  - 두번째 그래프: 성분 분해 (trend, weekly, yearly 등)
  - test 데이터가 있으면 붉은 점과 선으로 실제값 표시
  - 변화점은 모델이 추세 변화를 감지한 시점을 수직선으로 표시
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame, date_range

# 기본 리포트 출력:
model, _, _, forecast, pred = hs_timeseries.prophet(train)
hs_timeseries.prophet_report(model, forecast, pred)

# test 데이터와 함께 성능 평가:
hs_timeseries.prophet_report(model, forecast, pred, test=test)

# 예측 테이블 출력:
hs_timeseries.prophet_report(model, forecast, pred, print_forecast=True)
```

<a id="hs_timeserise.get_weekend_df"></a>

### get\_weekend\_df

```python
def get_weekend_df(start: any, end: any = None) -> DataFrame
```

주말 날짜를 포함하는 휴일 데이터프레임을 생성한다.

Prophet 모델의 holidays 파라미터에 사용할 수 있는 형식의 주말 휴일
데이터프레임을 생성한다. 토요일과 일요일을 'holiday'로 표시한다.

**Arguments**:

- `start` _datetime | str_ - 시작일.
  datetime 객체 또는 문자열 형식 ("YYYY-MM-DD") 가능.
- `예` - "2021-01-01" 또는 datetime(2021, 1, 1)
- `end` _datetime | str, optional_ - 종료일.
  지정하지 않으면 현재 날짜까지. 기본값은 None.
  

**Returns**:

- `DataFrame` - 주말 휴일 데이터프레임. 다음 컬럼 포함:
  - ds (datetime): 토요일 또는 일요일 날짜
  - holiday (str): 'holiday' 문자열 (고정값)
  

**Notes**:

  - Prophet의 holidays 파라미터는 'ds'와 'holiday' 컬럼이 필요하다.
  - 주말뿐 아니라 다른 휴일도 추가하려면 이 함수 결과와 병합(concat)하여 사용.
  - Prophet은 add_country_holidays() 메서드로 국가별 공휴일도 지원한다.
  - 토요일(Saturday), 일요일(Sunday)을 자동 탐지하여 추출한다.
  

**Examples**:

```python
from hossam import *
from pandas import DataFrame, date_range

# 2020년 전체 주말 생성:
weekends = hs_timeseries.get_weekend_df('2020-01-01', '2020-12-31')
print(len(weekends))  # 104 (52주 × 2일)

# 현재까지의 주말:
weekends = hs_timeseries.get_weekend_df('2023-01-01')
print(weekends.head())

# Prophet 모델에 주말 효과 추가:
weekends = hs_timeseries.get_weekend_df('2020-01-01', '2025-12-31')
model = hs_timeseries.prophet(train, holidays=weekends)
```

<a id="hs_gis"></a>

## hs\_gis

<a id="hs_gis.geocode"></a>

### geocode

```python
def geocode(df: DataFrame, addr: str, key: str) -> DataFrame
```

주소 컬럼을 일괄 지오코딩하여 위도/경도 컬럼을 추가합니다.

**Arguments**:

- `df` - 입력 `DataFrame`.
- `addr` - 주소가 들어있는 컬럼명.
- `key` - VWorld API 키.
  

**Returns**:

  위도(`latitude`), 경도(`longitude`) 컬럼이 추가된 `DataFrame`.
  

**Raises**:

- `Exception` - 지오코딩 과정에서 발생한 예외를 전파합니다.
  

**Examples**:

```python
from hossam import *
result = hs_gis.geocode(df, addr="address", key="YOUR_VWORLD_KEY")
set(["latitude","longitude"]).issubset(result.columns)
# True
```

<a id="hs_gis.load_shape"></a>

### load\_shape

```python
def load_shape(path: str, info: bool = True) -> GeoDataFrame
```

Shapefile을 읽어 `GeoDataFrame`으로 로드합니다.

**Arguments**:

- `path` - 읽을 Shapefile(.shp) 경로.
- `info` - True면 데이터 프리뷰와 통계를 출력.
  

**Returns**:

  로드된 `GeoDataFrame`.
  

**Raises**:

- `FileNotFoundError` - 파일이 존재하지 않는 경우.
  

**Examples**:

```python
from hossam import *
gdf = hs_gis.load_shape("path/to/file.shp", info=False)
```

<a id="hs_gis.save_shape"></a>

### save\_shape

```python
def save_shape(gdf: GeoDataFrame | DataFrame,
               path: str,
               crs: str | None = None,
               lat_col: str = "latitude",
               lon_col: str = "longitude") -> None
```

전처리된 데이터(GeoDataFrame 또는 DataFrame)를 Shapefile 또는 GeoPackage로 저장합니다.

- GeoDataFrame 입력:
- CRS가 있으면 그대로 유지합니다.
- CRS가 없으면 `crs`(기본 WGS84)를 지정합니다.
- DataFrame 입력:
- 오직 이 경우에만 `lat_col`, `lon_col`을 사용해 포인트 지오메트리를 생성합니다.
- 좌표가 유효하지 않은 행은 제외되며, 유효한 좌표가 하나도 없으면 예외를 발생시킵니다.

파일 형식:
- .shp: ESRI Shapefile (필드명 10자 제한, ASCII 권장)
- .gpkg: GeoPackage (필드명 제약 없음, 한글 가능)
- 확장자 없으면 .shp로 저장

**Arguments**:

- `gdf` - 저장할 `GeoDataFrame` 또는 `DataFrame`.
- `path` - 저장 경로(.shp 또는 .gpkg, 확장자 없으면 .shp 자동 추가).
- `crs` - 좌표계 문자열(e.g., "EPSG:4326"). 미지정 시 WGS84.
- `lat_col` - DataFrame 입력 시 위도 컬럼명.
- `lon_col` - DataFrame 입력 시 경도 컬럼명.
  

**Returns**:

- `None` - 파일을 저장하고 반환값이 없습니다.
  

**Raises**:

- `TypeError` - 입력 타입이 잘못된 경우.
- `ValueError` - 경로가 잘못되었거나 CRS가 유효하지 않은 경우,
  또는 DataFrame에서 유효 좌표가 하나도 없는 경우.

<a id="data_loader"></a>

## data\_loader

<a id="data_loader.load_info"></a>

### load\_info

```python
def load_info(search: str = None, local: str = None) -> DataFrame
```

메타데이터에서 사용 가능한 데이터셋 정보를 로드한다.

**Arguments**:

- `search` _str, optional_ - 이름 필터 문자열. 포함하는 항목만 반환.
- `local` _str, optional_ - 로컬 메타데이터 경로. None이면 원격(BASE_URL) 사용.
  

**Returns**:

- `DataFrame` - name, chapter, desc, url 컬럼을 갖는 테이블
  

**Examples**:

```python
from hossam import *
info = load_info()
list(info.columns) #['name', 'chapter', 'desc', 'url']
```

<a id="data_loader.load_data"></a>

### load\_data

```python
def load_data(key: str, local: str = None) -> Optional[DataFrame]
```

키로 지정된 데이터셋을 로드한다.

**Arguments**:

- `key` _str_ - 메타데이터에 정의된 데이터 식별자(파일명 또는 별칭)
- `local` _str, optional_ - 로컬 메타데이터 경로. None이면 원격(BASE_URL) 사용.
  

**Returns**:

  DataFrame | None: 성공 시 데이터프레임, 실패 시 None
  

**Examples**:

```python
from hossam import *
df = load_data('AD_SALES')  # 메타데이터에 해당 키가 있어야 함
```

<a id="__init__"></a>

## \_\_init\_\_
