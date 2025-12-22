# hossam

Hossam Data Loader

이 라이브러리는 아이티윌에서 진행중인 머신러닝 데이터 분석 수업에서 사용되는 샘플 데이터를 로드하는 기능을 제공하는 라이브러리 입니다.

이광호 강사의 수업에서 활용되기는 것을 목적으로 합니다.

This library provides functionality for loading sample datasets used in the Machine Learning Data Analysis course conducted at ITWILL.
It is intended to be utilized in the lectures of instructor Kwangho Lee.

추가적으로 데이터 분석에서 사용할 수 있는 몇 가지 유틸리티 기능을 계획중입니다.

## Usage:

### 샘플 데이터 가져오기 (load sample data)

```python
from hossam import load_data
df = load_data('AD_SALES')
```

### 데이터 목록 보기 (view data list)

```python
from hossam import load_info

# search keyword paramter (default=None --> All List)
df = load_info(search="keyword")
```


License: MIT
