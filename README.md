# 🎓 Hossam Data Helper

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.5.16-green.svg)](https://pypi.org/project/hossam/)
[![Documentation](https://img.shields.io/badge/docs-py.hossam.kr-blue.svg)](https://py.hossam.kr)

**Hossam**은 데이터 분석, 시각화, 통계 처리를 위한 종합 헬퍼 라이브러리입니다.

아이티윌(ITWILL)에서 진행 중인 머신러닝 및 데이터 분석 수업을 위해 개발되었으며, 이광호 강사의 강의에서 활용됩니다.

## ✨ 주요 특징

- 📊 **풍부한 시각화**: 25+ 시각화 함수 (Seaborn/Matplotlib 기반) — 분포·관계·회귀 진단·분류/군집 평가 시각화
- 🔧 **데이터 품질 점검**: 자료형 변환, 중복·결측·이상치 탐지, 대화형 자동 점검
- 📦 **샘플 데이터**: 학습용 데이터셋 즉시 로드
- 🎓 **수업 운영 도구**: 수강생 그룹 편성 및 분석 리포트
- 📈 **교육용 최적화**: 데이터 분석 교육에 특화된 설계

## 📦 설치

```bash
pip install --upgrade hossam
```

**요구사항**: Python 3.11 ~ 3.13

## 🧩 모듈 구성

`hossam` 패키지는 다음 4개 모듈로 구성됩니다. 모듈별 상세 함수 설명은 [API 문서](https://py.hossam.kr)를 참고하세요.

| 모듈 | 설명 | 대표 기능 |
| --- | --- | --- |
| **`my_util`** | 데이터 로딩·유틸리티 | `load_data`(학습용 데이터셋 로드), `load_info`(데이터셋 설명), `make_normalize_data`(정규화), `pretty_table`(표 출력) |
| **`my_plot`** | 시각화 (25+ 함수) | `histplot`·`boxplot`·`violinplot`·`scatterplot`·`lmplot`·`heatmap` 등 분포/관계 그래프, `ols_residplot`·`roc_curve_plot`·`confusion_matrix_plot`·`silhouette_plot`·`pca_plot` 등 분석 진단 그래프 |
| **`my_qtcheck`** | 데이터 품질 점검 | `set_type`(자료형 변환), `check_duplicates`(중복), `check_missing_values`(결측), `numerical_summary`/`categorical_summary`(기술통계), `auto_qtcheck`(대화형 자동 점검) |
| **`my_classroom`** | 수업 운영 도구 | `cluster_students`(수강생 그룹 편성), `group_summary`·`analyze_classroom`(그룹 분석 리포트) |

```python
from hossam import load_data
from hossam import my_plot, my_qtcheck

df = load_data("diamonds")        # 학습용 데이터셋 로드
my_qtcheck.auto_qtcheck(df)       # 대화형 데이터 품질 점검
my_plot.boxplot(data=df, x="price")
```

## 📚 전체 문서

**완전한 API 문서와 가이드는 [py.hossam.kr](https://py.hossam.kr)에서 확인하세요.**
## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자유롭게 사용, 수정, 배포할 수 있습니다.

## 🔗 링크

- **문서**: [py.hossam.kr](https://py.hossam.kr)
- **PyPI**: [pypi.org/project/hossam](https://pypi.org/project/hossam/)
- **강사**: 이광호 (ITWILL 머신러닝 및 데이터 분석)
