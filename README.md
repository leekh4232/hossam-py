---
title: 🎓 Hossam Data Helper
---

# 🎓 Hossam Data Helper

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.3.19-green.svg)](https://pypi.org/project/hossam/)
[![Documentation](https://img.shields.io/badge/docs-py.hossam.kr-blue.svg)](https://py.hossam.kr)

**Hossam**은 데이터 분석, 시각화, 통계 처리를 위한 종합 헬퍼 라이브러리입니다.

아이티윌(ITWILL)에서 진행 중인 머신러닝 및 데이터 분석 수업을 위해 개발되었으며, 이광호 강사의 강의에서 활용됩니다.

## ✨ 주요 특징

- 📊 **풍부한 시각화**: 25+ 시각화 함수 (Seaborn/Matplotlib 기반)
- 🎯 **통계 분석**: 회귀, 분류, 시계열 분석 도구
- 🤖 **머신 러닝**: 예측, 분류, 군집 학습 모델 구축 및 성능 평가
- 📦 **샘플 데이터**: 학습용 데이터셋 즉시 로드
- 🔧 **데이터 전처리**: 결측치 처리, 이상치 탐지, 스케일링
- 📈 **교육용 최적화**: 데이터 분석 교육에 특화된 설계


## 📦 설치

```bash
pip install hossam
```

**요구사항**: Python 3.13.9 이상

## 📚 전체 문서

**완전한 API 문서와 가이드는 [py.hossam.kr](https://py.hossam.kr)에서 확인하세요.**

### 주요 모듈

- **hs_plot**: 25+ 시각화 함수 (선 그래프, 산점도, 히스토그램, 박스플롯, 히트맵 등)
- **hs_stats**: 회귀/분류 분석, 교차검증, 정규성 검정, 상관분석 등
- **hs_prep**: 결측치 처리, 이상치 탐지, 스케일링, 인코딩 등의 데이터 전처리 기능
- **hs_timeserise**: 시계열 분석 기능 지원
- **hs_gis**: GIS 데이터 로드 및 시각화 (대한민국 지도 지원)
- **hs_util**: 예쁜 테이블 출력, 그리드 서치 등
- **hs_cluster**: 군집분석, PCA 등 (작업중)
- **hs_ml**: 예측, 분류 분석 (예정)

자세한 사용법은 [API 문서](https://py.hossam.kr/api/hossam/)를 참고하세요.


## 📄 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 자유롭게 사용, 수정, 배포할 수 있습니다.

## 🔗 링크

- **문서**: [py.hossam.kr](https://py.hossam.kr)
- **PyPI**: [pypi.org/project/hossam](https://pypi.org/project/hossam/)
- **강사**: 이광호 (ITWILL 머신러닝 및 데이터 분석)

---

{% include 'HOSSAM_API.md' %}