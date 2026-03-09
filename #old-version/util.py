# -*- coding: utf-8 -*-
# -------------------------------------------------------------
import sys
import re
import requests
import contractions
import numpy as np
import nltk
import joblib
from nltk.corpus import stopwords as stw
from typing import Literal
from datetime import datetime as dt
from PIL import Image, ImageEnhance

# -------------------------------------------------------------
import joblib

# -------------------------------------------------------------
from pandas import DataFrame, Series, read_excel, read_csv, get_dummies, DatetimeIndex
from pandas import read_sql, read_sql_table

# -------------------------------------------------------------
from matplotlib import pyplot as plt
from tabulate import tabulate

# -------------------------------------------------------------
from scipy.stats import normaltest

# -------------------------------------------------------------
from pca import pca

# -------------------------------------------------------------
from statsmodels.stats.outliers_influence import variance_inflation_factor

# -------------------------------------------------------------
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# -------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

# -------------------------------------------------------------
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------------------------------------
# 형태소 분석 엔진 -> Okt
# from konlpy.tag import Okt

# 형태소 분석 엔진 -> Mecab
from konlpy.tag import Mecab

# -------------------------------------------------------------
from .core import *

# -------------------------------------------------------------
import pymysql
from sqlalchemy import create_engine


# -------------------------------------------------------------

def my_pca(
    data: DataFrame,
    n_components: int | float = 0.95,
    standardize: bool = True,
    plot: bool = True,
    figsize: tuple = (15, 7),
    dpi: int = 100,
) -> DataFrame:
    """PCA를 수행하여 차원을 축소한다.

    Args:
        data (DataFrame): 데이터프레임
        n_components (int, optional): 축소할 차원 수. Defaults to 2.
        standardize (bool, optional): True일 경우 표준화를 수행한다. Defaults to False.

    Returns:
        DataFrame: PCA를 수행한 데이터프레임
    """
    if standardize:
        df = my_standard_scaler(data)
    else:
        df = data.copy()

    model = pca(n_components=n_components, random_state=get_random_state())
    result = model.fit_transform(X=df)

    my_pretty_table(result["loadings"])
    my_pretty_table(result["topfeat"])

    if plot:
        fig, ax = model.biplot(figsize=figsize, fontsize=12, dpi=dpi)
        ax.set_title(ax.get_title(), fontsize=14)
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
        plt.show()
        plt.close()

        fig, ax = model.plot(figsize=figsize)
        fig.set_dpi(dpi)
        ax.set_title(ax.get_title(), fontsize=14)
        ax.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax.set_ylabel(ax.get_ylabel(), fontsize=12)

        labels = ax.get_xticklabels()
        pc_labels = [f"PC{i+1}" for i in range(len(labels))]
        ax.set_xticklabels(pc_labels, fontsize=11, rotation=0)

        ax.set_yticklabels(ax.get_yticklabels(), fontsize=11)
        plt.show()
        plt.close()

        plt.rcParams["font.family"] = (
            "AppleGothic" if sys.platform == "darwin" else "Malgun Gothic"
        )

    return result["PC"]


# -------------------------------------------------------------

def tune_image(
    img: Image,
    mode: Literal["RGB", "L"] = "RGB",
    size: tuple = None,
    color: float = None,
    contrast: int = None,
    brightness: float = None,
    sharpness: float = None,
) -> Image:
    """이미지를 튜닝한다.

    Args:
        img (Image): 이미지 객체
        mode (Literal['RGB', 'L'], optional): 이미지 색상/흑백 모드
        size (tuple, optional): 이미지 크기. Defaults to None.
        color (float, optional): 이미지의 색상 균형을 조정한다. 0 부터 1 사이의 실수값으로 이미지의 색상을 조절 한다. 0 에 가까울 수록 색이 빠진 흑백에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 색이 더해진다. Defaults to None.
        contrast (int, optional): 이미지의 대비를 조정한다.  0에 가까울 수록 대비가 없는 회색 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 대비가 강해진다. Defaults to None.
        brightness (float, optional): 이미지의 밝기를 조정한다.  0에 가까울 수록 그냥 검정 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 밝기가 강해진다. Defaults to None.
        sharpness (float, optional): 이미지의 선명도를 조정한다. 0 에 가까울 수록 이미지는 흐릿한 이미지에 가깝게 되고 1 이 원본 값이고 1이 넘어가면 원본에 비해 선명도가 강해진다. Defaults to None.

    Returns:
        Image: 튜닝된 이미지
    """
    if mode:
        img = img.convert(mode=mode)

    if size:
        w = size[0] if size[0] > 0 else 0
        h = size[1] if size[1] > 0 else 0
        img = img.resize(size=(w, h))

    if color:
        if color < 0:
            color = 0
        img = ImageEnhance.Color(image=img).enhance(factor=color)

    if contrast:
        img = ImageEnhance.Contrast(image=img).enhance(
            factor=contrast if contrast > 0 else 0
        )

    if brightness:
        img = ImageEnhance.Brightness(image=img).enhance(
            factor=brightness if brightness > 0 else 0
        )

    if sharpness:
        img = ImageEnhance.Sharpness(image=img).enhance(
            factor=sharpness if sharpness > 0 else 0
        )

    img.array = np.array(img)

    return img


# -------------------------------------------------------------

def load_image(
    path: str,
    mode: Literal["RGB", "L"] = None,
    size: tuple = None,
    color: float = None,
    contrast: int = None,
    brightness: float = None,
    sharpness: float = None,
) -> Image:
    """이미지 파일을 로드한다. 필요한 경우 로드한 이미지에 대해 튜닝을 수행한다. 최종 로드된 이미지에 대한 배열 데이터를 array 속성에 저장한다.

    Args:
        path (str): 이미지 파일 경로
        mode (Literal['RGB', 'color', 'L', 'gray'], optional): 이미지 색상/흑백 모드
        size (tuple, optional): 이미지 크기. Defaults to None.
        color (float, optional): 이미지의 색상 균형을 조정한다. 0 부터 1 사이의 실수값으로 이미지의 색상을 조절 한다. 0 에 가까울 수록 색이 빠진 흑백에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 색이 더해진다. Defaults to None.
        contrast (int, optional): 이미지의 대비를 조정한다.  0에 가까울 수록 대비가 없는 회색 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 대비가 강해진다. Defaults to None.
        brightness (float, optional): 이미지의 밝기를 조정한다.  0에 가까울 수록 그냥 검정 이미지에 가깝게 되고 1 이 원본 값이되고 1이 넘어가면 밝기가 강해진다. Defaults to None.
        sharpness (float, optional): 이미지의 선명도를 조정한다. 0 에 가까울 수록 이미지는 흐릿한 이미지에 가깝게 되고 1 이 원본 값이고 1이 넘어가면 원본에 비해 선명도가 강해진다. Defaults to None.

    Returns:
        Image: 로드된 이미지
    """
    img = Image.open(fp=path)
    img = tune_image(
        img=img,
        mode=mode,
        size=size,
        color=color,
        contrast=contrast,
        brightness=brightness,
        sharpness=sharpness,
    )

    return img


# -------------------------------------------------------------

def my_stopwords(lang: str = "ko") -> list:
    stopwords = None

    if lang == "ko":
        session = requests.Session()
        session.headers.update(
            {
                "Referer": "",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }
        )

        try:
            r = session.get("https://data.hossam.kr/tmdata/stopwords-ko.txt")

            # HTTP 상태값이 200이 아닌 경우는 에러로 간주한다.
            if r.status_code != 200:
                msg = "[%d Error] %s 에러가 발생함" % (r.status_code, r.reason)
                raise Exception(msg)

            r.encoding = "utf-8"
            stopwords = r.text.split("\n")
        except Exception as e:
            print(e)

    elif lang == "en":
        nltk.download("stopwords")
        stopwords = list(stw.words("english"))

    return stopwords


# -------------------------------------------------------------

def my_text_morph(
    source: str, mode: str = "nouns", stopwords: list = None, dicpath: str = None
) -> list:
    """Mecab을 사용하여 텍스트를 형태소 분석한다.

    Args:
        source (str): 텍스트
        mode (str, optional): 분석 모드. Defaults to 'nouns'.

    Returns:
        list: 형태소 분석 결과
    """
    desc = None

    if dicpath is not None:
        mecab = Mecab(dicpath=dicpath)
    else:
        mecab = Mecab()

    if mode == "nouns":
        desc = mecab.nouns(phrase=source)
    elif mode == "morphs":
        desc = mecab.morphs(phrase=source)
    elif mode == "pos":
        desc = mecab.pos(phrase=source)
    else:
        desc = mecab.nouns(phrase=source)

    if stopwords:
        desc = [w for w in desc if w not in stopwords]

    return desc


# -------------------------------------------------------------

def my_tokenizer(
    source: any, num_words: int = None, oov_token: str = "<OOV>", stopwords: list = None
):
    if type(source) == str:
        source = my_text_morph(source=source, stopwords=stopwords)

    if num_words is None:
        tokenizer = Tokenizer(oov_token=oov_token)
    else:
        tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)

    tokenizer.fit_on_texts(source)

    return tokenizer


# -------------------------------------------------------------

def my_text_preprocessing(
    source: str,
    rm_abbr: bool = True,
    rm_email: bool = True,
    rm_html: bool = True,
    rm_url: bool = True,
    rm_num: bool = True,
    rm_special: bool = True,
    stopwords: list = None,
) -> str:
    """영문 텍스트를 전처리한다.

    Args:
        source (str): 텍스트
        rm_abbr (bool, optional): 약어 제거. Defaults to True.
        rm_email (bool, optional): 이메일 주소 제거. Defaults to True.
        rm_html (bool, optional): HTML 태그 제거. Defaults to True.
        rm_url (bool, optional): URL 주소 제거. Defaults to True.
        rm_num (bool, optional): 숫자 제거. Defaults to True.
        rm_special (bool, optional): 특수문자 제거. Defaults to True.
        stopwords (list, optional): 불용어 목록. Defaults to None.

    Returns:
        str: 전처리된 텍스트
    """
    # print(source)

    if stopwords is not None:
        source = " ".join([w for w in source.split() if w not in stopwords])
        # print(source)

    if rm_abbr:
        source = contractions.fix(source)
        # print(source)

    if rm_email:
        source = re.sub(
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "", source
        )
        # print(source)

    if rm_html:
        source = re.sub(r"<[^>]*>", "", source)
        # print(source)

    if rm_url:
        source = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            source,
        )
        # print(source)

    if rm_num:
        source = re.sub(r"\b[0-9]+\b", "", source)
        # print(source)

    if rm_special:
        x = re.sub(r"[^\w ]+", "", source)
        source = " ".join(x.split())
        # print(source)

    return source


# -------------------------------------------------------------

def my_text_data_preprocessing(
    data: DataFrame,
    fields: list = None,
    rm_abbr: bool = True,
    rm_email: bool = True,
    rm_html: bool = True,
    rm_url: bool = True,
    rm_num: bool = True,
    rm_special: bool = True,
    rm_stopwords: bool = True,
    stopwords: list = None,
) -> DataFrame:
    if not fields:
        fields = data.columns

    if type(fields) == str:
        fields = [fields]

    df = data.copy()

    for f in fields:
        df[f] = df[f].apply(
            lambda x: my_text_preprocessing(
                source=x,
                rm_abbr=rm_abbr,
                rm_email=rm_email,
                rm_html=rm_html,
                rm_url=rm_url,
                rm_num=rm_num,
                rm_special=rm_special,
                stopwords=stopwords,
            )
        )

    return df


# -------------------------------------------------------------

def my_token_process(
    data: any,
    xname: str = None,
    yname: str = None,
    threshold: int = 10,
    num_words: int = None,
    max_word_count: int = None,
) -> DataFrame:
    # 훈련, 종속변수 분리
    x = None
    y = None

    if xname is not None:
        x = data[xname]
    else:
        x = data

    if yname is not None:
        y = data[yname]

    # 토큰화
    tokenizer = my_tokenizer(source=x)

    # 전체 단어의 수
    total_cnt = len(tokenizer.word_index)

    # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트할 값
    rare_cnt = 0

    # 훈련 데이터의 전체 단어 빈도수 총 합
    total_freq = 0

    # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합
    rare_freq = 0

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    # --> [('one', 50324), ('reviewers', 500), ('mentioned', 1026), ('watching', 8909), ('oz', 256)]
    # --> key = 'one', value = 50324
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if value < threshold:
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    print("단어 집합(vocabulary)의 크기 :", total_cnt)
    print("등장 빈도가 %s번 미만인 희귀 단어의 수: %s" % (threshold, rare_cnt))
    print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt) * 100)
    print(
        "전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq) * 100
    )

    # 자주 등장하는 단어 집합의 크기 구하기 -> 이 값이 첫 번째 학습층의 input 수가 된다.
    vocab_size = total_cnt - rare_cnt + 1
    print("단어 집합의 크기 :", vocab_size)

    # 최종 토큰화
    if num_words is None:
        num_words = vocab_size

    tokenizer2 = my_tokenizer(x, num_words=num_words)
    token_set = tokenizer2.texts_to_sequences(x)

    # 토큰화 결과 길이가 0인 항목의 index 찾기
    drop_target_index = []

    for i, v in enumerate(token_set):
        if len(v) < 1:
            drop_target_index.append(i)

    token_set2 = np.asarray(token_set, dtype="object")

    # 토큰 결과에서 해당 위치의 항목들을 삭제한다.
    fill_token_set = np.delete(token_set2, drop_target_index, axis=0)

    # 종속변수와 원래의 독립변수에서도 같은 위치의 항목들을 삭제해야 한다.
    future_set = np.delete(x, drop_target_index, axis=0)
    print("독립변수(텍스트) 데이터 수: ", len(fill_token_set))

    if y is not None:
        label_set = np.delete(y, drop_target_index, axis=0)
        print("종속변수(레이블) 데이터 수: ", len(label_set))

    # 문장별 단어 수 계산
    word_counts = []

    for s in fill_token_set:
        word_counts.append(len(s))

    if max_word_count is None:
        max_word_count = max(word_counts)

    pad_token_set = pad_sequences(fill_token_set, maxlen=max_word_count)
    pad_token_set_arr = [np.array(x, dtype="int") for x in pad_token_set]

    datadic = {}

    if y is not None:
        datadic[yname] = label_set

    if xname is not None:
        xname = "text"

    datadic[xname] = future_set
    datadic["count"] = word_counts
    datadic["token"] = fill_token_set
    datadic["pad_token"] = pad_token_set_arr

    df = DataFrame(data=datadic)

    return df, pad_token_set, vocab_size


def my_save_model(model, path = None):

    if path is None:
        path = "%s.pkl" % model.__class__.__name__

    try:
        joblib.dump(model, path)
        print(f"{path}에 모델이 저장되었습니다.")
    except Exception as e:
        print(f"모델 저장 중 오류가 발생했습니다. ({e})")

def my_load_model(path):
    try:
        model = joblib.load(path)
        print(f"{path}에서 모델을 로드했습니다.")
        return model
    except Exception as e:
        print(f"모델 로드 중 오류가 발생했습니다. ({e})")
        return None


def my_feature_importance(estimator: any, rate: float = 1) -> DataFrame:
    if hasattr(estimator, "get_booster"):
        feature_important = estimator.get_booster().get_score(
            importance_type="weight"
        )
        ikeys = list(feature_important.keys())
        ivalues = list(feature_important.values())
    elif hasattr(estimator, "booster_"):
        ikeys = estimator.booster_.feature_name()
        ivalues = list(estimator.booster_.feature_importance())

    if ikeys is not None and ivalues is not None:
        data = DataFrame(data=ivalues, index=ikeys, columns=["score"]).sort_values(
            by="score", ascending=False
        )

        data["rate"] = data["score"] / data["score"].sum()
        data["cumsum"] = data["rate"].cumsum()

        if rate < 1:
            data = data[data["cumsum"] <= rate]

        return data