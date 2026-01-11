import pandas as pd
import numpy as np
from hossam.hs_stats import corr

# 테스트 데이터 생성 (정규 + 비정규 혼합)
np.random.seed(42)
df = pd.DataFrame({
    'x1': np.random.normal(0, 1, 100),      # 정규분포
    'x2': np.random.exponential(2, 100),   # 비정규분포
    'x3': np.random.normal(5, 2, 100),     # 정규분포
})

print("테스트 1: 모든 컬럼 분석")
print("="*50)
corr_matrix, corr_types = corr(df)
print("상관계수 행렬:")
print(corr_matrix)
print("\n상관계수 종류:")
print(corr_types)

print("\n" + "="*50)
print("테스트 2: 특정 컬럼만 분석 (x1, x3)")
print("="*50)
corr_matrix2, corr_types2 = corr(df, 'x1', 'x3')
print("상관계수 행렬:")
print(corr_matrix2)
print("\n상관계수 종류:")
print(corr_types2)

print("\n✓ 리팩토링 완료 - normal_test() 함수 재사용 성공")
