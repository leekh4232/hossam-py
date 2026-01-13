# -*- coding: utf-8 -*-
# ===================================================================
# 상호작용(Interaction) 항 생성 함수
# ===================================================================
import numpy as np
from pandas import DataFrame
from itertools import combinations


# ===================================================================
# 변수 간의 상호작용 항을 추가한 데이터프레임을 반환한다
# ===================================================================
def interaction(*fields: str):
    """변수 간의 상호작용(interaction) 항을 생성하는 데코레이터 함수.

    사용 방법: 원본 데이터프레임에 대해 호출하면 상호작용 컬럼이 추가된 새 데이터프레임을 반환한다.

    Args:
        *fields (str): 상호작용할 변수들의 컬럼명. 2개 이상의 컬럼을 지정하면 모든 조합의 상호작용을 생성.
                      지정하지 않으면 모든 수치형 컬럼의 모든 2-way 상호작용을 생성.

    Returns:
        function: 데이터프레임을 입력받아 상호작용 항이 추가된 데이터프레임을 반환하는 함수.

    Examples:
        >>> from hossam.hs_prep import interaction
        >>> import pandas as pd
        >>> df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6], 'x3': [7, 8, 9]})

        # 특정 변수들의 상호작용만 생성
        >>> result = interaction('x1', 'x2')(df)
        >>> print(result.columns)  # x1, x2, x3, x1*x2

        # 모든 2-way 상호작용 생성
        >>> result = interaction()(df)
        >>> print(result.columns)  # x1, x2, x3, x1*x2, x1*x3, x2*x3
    """
    def wrapper(data: DataFrame) -> DataFrame:
        df = data.copy()

        # fields가 비어있으면 모든 수치형 컬럼의 2-way 상호작용 생성
        if not fields:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cols_to_interact = list(combinations(numeric_cols, 2))
        else:
            # fields가 지정된 경우 모든 가능한 조합 생성
            field_list = [f for f in fields if f in df.columns]
            if len(field_list) < 2:
                return df
            cols_to_interact = list(combinations(field_list, 2))

        # 상호작용 항 생성
        for col1, col2 in cols_to_interact:
            # 두 컬럼이 모두 수치형인지 확인
            if not (pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2])):
                continue

            interaction_col_name = f"{col1}*{col2}"
            df[interaction_col_name] = df[col1] * df[col2]

        return df

    return wrapper


# ===================================================================
# 직접 상호작용 항을 추가하는 함수 (데코레이터 없이 직접 사용)
# ===================================================================
def add_interaction(data: DataFrame, *fields: str) -> DataFrame:
    """데이터프레임에 상호작용 항을 추가한다.

    특정 변수 쌍 또는 모든 수치형 변수 간의 상호작용 항을 생성하여 데이터프레임에 추가한다.

    Args:
        data (DataFrame): 원본 데이터프레임.
        *fields (str): 상호작용할 변수들의 컬럼명 목록.
                      지정하지 않으면 모든 수치형 컬럼의 모든 2-way 상호작용을 생성.
                      지정된 경우, 지정된 컬럼들의 모든 조합에 대해 상호작용을 생성.

    Returns:
        DataFrame: 상호작용 항이 추가된 새 데이터프레임.

    Examples:
        >>> from hossam.hs_prep import add_interaction
        >>> import pandas as pd
        >>> df = pd.DataFrame({'x1': [1, 2, 3], 'x2': [4, 5, 6], 'x3': [7, 8, 9]})

        # 특정 변수들의 상호작용만 추가
        >>> result = add_interaction(df, 'x1', 'x2')
        >>> print(result.columns)  # x1, x2, x3, x1*x2
        >>> print(result['x1*x2'].tolist())  # [4, 10, 18]

        # 모든 2-way 상호작용 추가
        >>> result = add_interaction(df)
        >>> print(result.columns)  # x1, x2, x3, x1*x2, x1*x3, x2*x3
    """
    df = data.copy()

    # fields가 비어있으면 모든 수치형 컬럼의 2-way 상호작용 생성
    if not fields:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_interact = list(combinations(numeric_cols, 2))
    else:
        # fields가 지정된 경우 모든 가능한 조합 생성
        field_list = [f for f in fields if f in df.columns]
        if len(field_list) < 2:
            return df
        cols_to_interact = list(combinations(field_list, 2))

    # 상호작용 항 생성
    for col1, col2 in cols_to_interact:
        # 두 컬럼이 모두 수치형인지 확인
        if not (pd.api.types.is_numeric_dtype(df[col1]) and pd.api.types.is_numeric_dtype(df[col2])):
            continue

        interaction_col_name = f"{col1}*{col2}"
        df[interaction_col_name] = df[col1] * df[col2]

    return df
