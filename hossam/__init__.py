# -*- coding: utf-8 -*-
from pathlib import Path as _Path

# `from hossam import *` 사용 시 함께 제공되는 편의 모듈 (기존 동작 유지)
import pandas as pd
from matplotlib import pyplot as plt

# -------------------------------------
# 전역 상수
# -------------------------------------
# 무작위성이 개입하는 모든 기능(PCA, 군집, 데이터 분할 등)의 재현성을 위한 랜덤시드.
# 하위 모듈에서 `from . import RANDOM_STATE` 로 참조하므로 모듈 임포트보다 먼저 정의한다.
RANDOM_STATE = 3217

# -------------------------------------
# 내보낼 모듈 임포트
# -------------------------------------
from . import my_qtcheck    # 데이터 품질 점검 관련 함수 모듈
from . import my_plot       # 시각화 관련 함수 모듈
from . import my_stats      # 통계 분석 관련 함수 모듈
from . import my_prep       # 데이터 전처리 관련 함수 모듈
from . import my_ols        # 선형회귀 관련 함수 모듈
from . import my_logit      # 로지스틱 회귀 관련 함수 모듈
from . import my_ts         # 시계열 분석 관련 함수 모듈
from . import my_cluster    # 군집 분석 관련 함수 모듈
from . import code_checker  # 제출 코드를 원본 모듈과 대조하는 모듈
from . import make_docs     # 소스코드로 API 레퍼런스 문서를 생성하는 모듈
from . import my_pipeline
from . import my_util
from .make_docs import make_api_docs
from .code_checker import diff
from .my_util import load_info
from .my_util import load_data

# -------------------------------------
# 초기화
# 패키지 이름/안내 메시지는 _config.py, 공통 로직은 _bootstrap.py 에 있다.
# (한글 폰트 등록, 그래프/pandas 출력 옵션, 버전 확인, 안내 메시지 출력)
# -------------------------------------
from ._bootstrap import check_pypi_latest
from ._bootstrap import init as _init

__version__ = _init(_Path(__file__).resolve().parent)
