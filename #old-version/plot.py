# -*- coding: utf-8 -*-
# -------------------------------------------------------------
import os
import sys
import graphviz
import numpy as np
import seaborn as sb
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes

from math import sqrt
from pandas import DataFrame, Series, concat

from scipy.stats import t
from scipy.spatial import ConvexHull
from scipy.stats import zscore, probplot

from statannotations.Annotator import Annotator

from sklearn.metrics import (
    mean_squared_error,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from IPython import display
from IPython.display import Image

from .core import get_random_state, get_n_jobs
from xgboost import plot_importance as xgb_plot_importance
from lightgbm import plot_importance as lgb_plot_importance

try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False

__FONT_NAME__ = "Malgun Gothic"

if sys.platform == "darwin":
    __FONT_NAME__ = "AppleGothic"
elif sys.platform in ["linux", "linux2"]:
    __FONT_NAME__ = "NanumGothic"
elif IN_COLAB:
    __FONT_NAME__ = "NanumGothic"

plt.rcParams["font.family"] = __FONT_NAME__
plt.rcParams["font.size"] = 10
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["axes.unicode_minus"] = False

import pandas as pd

if pd.__version__ > "2.0.0":
    pd.DataFrame.iteritems = pd.DataFrame.items



def my_lineplot(
    df: DataFrame,
    xname: str = None,
    yname: str = None,
    hue: str = None,
    marker: str = None,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    sb.lineplot(
        data=df,
        x=xname,
        y=yname,
        hue=hue,
        marker=marker,
        palette=palette,
        ax=ax,
        **params,
    )
    ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_boxplot(
    df: DataFrame,
    xname: str = None,
    yname: str = None,
    orient: str = "v",
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """데이터프레임 내의 모든 컬럼에 대해 상자그림을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str, optional): x축에 사용할 컬럼명. Defaults to None.
        yname (str, optional): y축에 사용할 컬럼명. Defaults to None.
        orient (str, optional): 그래프의 방향. 'v' 또는 'h'. Defaults to 'v'.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    if xname != None and yname != None:
        sb.boxplot(
            data=df, x=xname, y=yname, orient=orient, palette=palette, ax=ax, **params
        )
    else:

        sb.boxplot(data=df, orient=orient, palette=palette, ax=ax, **params)

    ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_kdeplot(
    df: DataFrame,
    xname: str = None,
    yname: str = None,
    hue: str = None,
    palette: str = None,
    fill: bool = False,
    fill_alpha: float = 0.3,
    linewidth: float = 1,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """데이터프레임 내의 컬럼에 대해 커널밀도추정을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str, optional): x축에 사용할 컬럼명. Defaults to None.
        yname (str, optional): y축에 사용할 컬럼명. Defaults to None.
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        fill (bool, optional): 채우기 여부. Defaults to False.
        fill_alpha (float, optional): 채우기의 투명도. Defaults to 0.3.
        linewidth (float, optional): 선의 두께. Defaults to 0.5.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    if fill:
        sb.kdeplot(
            data=df,
            x=xname,
            y=yname,
            hue=hue,
            palette=palette,
            fill=fill,
            alpha=fill_alpha,
            linewidth=linewidth,
            ax=ax,
            **params,
        )
    else:
        sb.kdeplot(
            data=df,
            x=xname,
            y=yname,
            hue=hue,
            palette=palette,
            fill=fill,
            linewidth=linewidth,
            ax=ax,
            **params,
        )

    # plt.grid()
    ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_histplot(
    df: DataFrame,
    xname: str,
    hue=None,
    bins=None,
    kde: bool = True,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """데이터프레임 내의 컬럼에 대해 히스토그램을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        bins (optional): 히스토그램의 구간 수 혹은 리스트. Defaults to None.
        kde (bool, optional): 커널밀도추정을 함께 출력할지 여부. Defaults to True.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    if bins:
        sb.histplot(
            data=df,
            x=xname,
            hue=hue,
            kde=kde,
            bins=bins,
            palette=palette,
            ax=ax,
            **params,
        )
    else:
        sb.histplot(
            data=df, x=xname, hue=hue, kde=kde, palette=palette, ax=ax, **params
        )

    # plt.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_stackplot(
    df: DataFrame,
    xname: str,
    hue: str,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """hue로 구분되는 막대 그래프를 비율로 표시한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        kde (bool, optional): 커널밀도추정을 함께 출력할지 여부. Defaults to True.
        multiple (str, optional): hue가 있을 경우 전체의 비율을 어떻게 표시할지 여부. Deafults to layer
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    df2 = df[[xname, hue]]
    df2[xname] = df2[xname].astype(str)

    sb.histplot(
        data=df2,
        x=xname,
        hue=hue,
        palette=palette,
        linewidth=0.5,
        stat="probability",  # 전체에서의 비율로 그리기
        multiple="fill",  # 전체를 100%로 그리기
        shrink=0.8,  # 막대의 폭
        ax=ax,
        **params,
    )

    # 그래프의 x축 항목 수 만큼 반복
    for p in ax.patches:
        # 각 막대의 위치, 넓이, 높이
        left, bottom, width, height = p.get_bbox().bounds
        # 막대의 중앙에 글자 표시하기
        ax.annotate(
            "%0.1f%%" % (height * 100),
            xy=(left + width / 2, bottom + height / 2),
            ha="center",
            va="center",
        )

    # plt.grid()

    if str(df[xname].dtype) in ["int", "int32", "int64", "float", "float32", "float64"]:
        xticks = list(df[xname].unique())
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_scatterplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """데이터프레임 내의 두 컬럼에 대해 산점도를 그려서 관계를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    sb.scatterplot(data=df, x=xname, y=yname, hue=hue, palette=palette, ax=ax, **params)
    ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_regplot(
    df: DataFrame,
    xname: str,
    yname: str,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """데이터프레임 내의 컬럼에 대해 회귀선을 포함한 산점도를 그려서 관계를 확인한다.
    색상 팔레트는 지원하지 않는다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    sb.regplot(data=df, x=xname, y=yname, color=palette, ax=ax, **params)
    ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_lmplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    **params,
) -> None:
    """데이터프레임 내의 컬럼에 대해 회귀선을 포함한 산점도를 그려서 관계를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
    """
    g = sb.lmplot(data=df, x=xname, y=yname, hue=hue, palette=palette, **params)
    g.fig.set_figwidth(figsize[0])
    g.fig.set_figheight(figsize[1])
    g.fig.set_dpi(dpi)
    plt.grid()

    plt.tight_layout()
    plt.show()
    plt.close()



def my_pairplot(
    df: DataFrame,
    diag_kind: str = "kde",
    hue=None,
    palette: str = None,
    figsize: tuple = (3.5, 2.7),
    dpi: int = 100,
    **params,
) -> None:
    """데이터프레임 내의 모든 컬럼에 대해 쌍별 관계를 시각화한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
    """
    w = figsize[0] * len(df.columns)
    h = figsize[1] * len(df.columns)

    g = sb.pairplot(data=df, hue=hue, diag_kind=diag_kind, palette=palette, **params)
    g.fig.set_size_inches(w=w, h=h)
    g.fig.set_dpi(val=dpi)
    g.map_lower(func=sb.kdeplot, fill=True, alpha=0.3)
    g.map_upper(func=sb.scatterplot)
    plt.tight_layout()
    plt.show()
    plt.close()



def my_countplot(
    df: DataFrame,
    xname: str,
    hue=None,
    palette: str = None,
    order: int = 1,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """데이터프레임 내의 컬럼에 대해 카운트플롯을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        order (str, optional): 정렬 방식. '1=값의 이름순' 또는 '2=수량 많은 순'. Defaults to "1".
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False
    sort = None
    if str(df[xname].dtype) in ["int", "int32", "int64", "float", "float32", "float64"]:
        if order == 1:
            sort = sorted(list(df[xname].unique()))
        else:
            sort = sorted(list(df[xname].value_counts().index))

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    sb.countplot(
        data=df, x=xname, hue=hue, palette=palette, order=sort, ax=ax, **params
    )
    ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_barplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """데이터프레임 내의 컬럼에 대해 바플롯을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    sb.barplot(data=df, x=xname, y=yname, hue=hue, palette=palette, ax=ax, **params)
    ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_boxenplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """데이터프레임 내의 컬럼에 대해 박슨플롯을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    sb.boxenplot(data=df, x=xname, y=yname, hue=hue, palette=palette, ax=ax, **params)
    ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_violinplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """데이터프레임 내의 컬럼에 대해 바이올린플롯(상자그림+커널밀도)을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    sb.violinplot(data=df, x=xname, y=yname, hue=hue, palette=palette, ax=ax, **params)
    ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_pointplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """데이터프레임 내의 컬럼에 대해 포인트플롯을 그려서 분포를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    sb.pointplot(data=df, x=xname, y=yname, hue=hue, palette=palette, ax=ax, **params)
    ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_jointplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    **params,
) -> None:
    """데이터프레임 내의 컬럼에 대해 산점도와 히스토그램을 함께 그려서 관계를 확인한다.

    Args:
        df (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str, optional): 색상을 구분할 기준이 되는 컬럼명. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    g = sb.jointplot(data=df, x=xname, y=yname, hue=hue, palette=palette, **params)
    g.fig.set_figwidth(figsize[0])
    g.fig.set_figheight(figsize[1])
    g.fig.set_dpi(dpi)
    plt.tight_layout()
    plt.show()
    plt.close()



def my_heatmap(
    data: DataFrame,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """데이터프레임 내의 컬럼에 대해 히트맵을 그려서 관계를 확인한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        palette (str, optional): 칼라맵. Defaults to None.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    sb.heatmap(data, annot=True, cmap=palette, fmt=".2f", ax=ax, **params)

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_convex_hull(
    data: DataFrame,
    xname: str,
    yname: str,
    hue: str,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
):
    """데이터프레임 내의 컬럼에 대해 외곽선을 그려서 군집을 확인한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        xname (str): x축에 사용할 컬럼명
        yname (str): y축에 사용할 컬럼명
        hue (str): 색상을 구분할 기준이 되는 컬럼명
        palette (str, optional): 칼라맵. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    # 군집별 값의 종류별로 반복 수행
    for c in data[hue].unique():
        if c == -1:
            continue

        # 한 종류만 필터링한 결과에서 두 변수만 선택
        df_c = data.loc[data[hue] == c, [xname, yname]]

        try:
            # 외각선 좌표 계산
            hull = ConvexHull(df_c)

            # 마지막 좌표 이후에 첫 번째 좌표를 연결
            points = np.append(hull.vertices, hull.vertices[0])

            ax.plot(
                df_c.iloc[points, 0], df_c.iloc[points, 1], linewidth=1, linestyle=":"
            )
            ax.fill(df_c.iloc[points, 0], df_c.iloc[points, 1], alpha=0.1)
        except:
            pass

    sb.scatterplot(
        data=data, x=xname, y=yname, hue=hue, palette=palette, ax=ax, **params
    )
    ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_kde_confidence_interval(
    data: DataFrame,
    clevel=0.95,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
) -> None:
    """커널밀도추정을 이용하여 신뢰구간을 그려서 분포를 확인한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        clevel (float, optional): 신뢰수준. Defaults to 0.95.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    # 데이터 프레임의 컬럼이 여러 개인 경우 처리
    for c in data.columns:
        column = data[c].dropna()
        # print(column)
        max = column.max()  # 최대값
        dof = len(column) - 1  # 자유도
        sample_mean = column.mean()  # 표본평균
        sample_std = column.std(ddof=1)  # 표본표준편차
        sample_std_error = sample_std / sqrt(len(column))  # 표본표준오차
        # print(max, dof, sample_mean, sample_std, sample_std_error)

        # 신뢰구간
        cmin, cmax = t.interval(clevel, dof, loc=sample_mean, scale=sample_std_error)
        # print(cmin, cmax)

        # 현재 컬럼에 대한 커널밀도추정
        sb.kdeplot(data=column, ax=ax)

        # 그래프 축의 범위
        xmin, xmax, ymin, ymax = plt.axis()

        # 신뢰구간 그리기
        ax.plot([cmin, cmin], [ymin, ymax], linestyle=":")
        ax.plot([cmax, cmax], [ymin, ymax], linestyle=":")
        # print([cmin, cmax])
        ax.fill_between([cmin, cmax], y1=ymin, y2=ymax, alpha=0.1)

        # 평균 그리기
        ax.plot([sample_mean, sample_mean], [0, ymax], linestyle="--", linewidth=2)

        ax.text(
            x=(cmax - cmin) / 2 + cmin,
            y=ymax,
            s="[%s] %0.1f ~ %0.1f" % (column.name, cmin, cmax),
            horizontalalignment="center",
            verticalalignment="bottom",
            fontdict={"size": 10, "color": "red"},
        )

    ax.set_ylim([ymin, ymax * 1.1])
    ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_pvalue1_anotation(
    data: DataFrame,
    target: str,
    hue: str,
    pairs: list,
    test: str = "t-test_ind",
    text_format: str = "star",
    loc: str = "outside",
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """데이터프레임 내의 컬럼에 대해 상자그림을 그리고 p-value를 함께 출력한다.

    Args:
        data (DataFrame): 데이터프레임 객체
        target (str): 종속변수에 대한 컬럼명
        hue (str): 명목형 변수에 대한 컬럼명
        pairs (list, optional): 비교할 그룹의 목록. 명목형 변수에 포함된 값 중에서 비교 대상을 [("A","B")] 형태로 선정한다.
        test (str, optional): 검정방법. Defaults to "t-test_ind".
            - t-test_ind(독립,등분산), t-test_welch(독립,이분산)
            - t-test_paired(대응,등분산), Mann-Whitney(대응,이분산), Mann-Whitney-gt, Mann-Whitney-ls
            - Levene(분산분석), Wilcoxon, Kruskal
        text_format (str, optional): 출력형식(full, simple, star). Defaults to "star".
        loc (str, optional): 출력위치(inside, outside). Defaults to "outside".
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    sb.boxplot(data=data, x=hue, y=target, ax=ax, **params)

    annotator = Annotator(ax, data=data, x=hue, y=target, pairs=pairs)
    annotator.configure(test=test, text_format=text_format, loc=loc)
    annotator.apply_and_annotate()

    sb.despine()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_resid_histplot(
    y: np.ndarray,
    y_pred: np.ndarray,
    bins=None,
    kde: bool = True,
    palette: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """예측값과 잔차를 히스토그램으로 출력한다.

    Args:
        y (np.ndarray): 종속변수에 대한 관측치
        y_pred (np.ndarray): 종속변수에 대한 예측치
        bins (_type_, optional): 히스토그램의 구간 수 혹은 리스트. Defaults to None.
        kde (bool, optional): 커널밀도추정을 함께 출력할지 여부. Defaults to True.
        palette (str, optional): 색상 팔레트. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 4).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    resid = y - y_pred
    resid_df = DataFrame({"resid": resid}).reset_index(drop=True)
    my_histplot(
        df=resid_df,
        xname="resid",
        bins=bins,
        kde=kde,
        figsize=figsize,
        dpi=dpi,
        palette=palette,
        callback=callback,
        ax=ax,
        **params,
    )



def my_residplot(
    y,
    y_pred,
    lowess: bool = False,
    mse: bool = False,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """예측값과 잔차를 그래프로 출력한다.

    Args:
        y (_type_): 종속변수에 대한 관측치
        y_pred (_type_): 종속변수에 대한 예측치
        lowess (bool, optional): 로우에스티메이션을 사용할지 여부. Defaults to False.
        mse (bool, optional): MSE를 통한 잔차의 분포를 함께 출력할지 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False
    resid = y - y_pred

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    sb.residplot(
        x=y_pred,
        y=resid,
        lowess=lowess,
        line_kws={"color": "red", "linewidth": 1},
        scatter_kws={"edgecolor": "white", "alpha": 0.7},
        ax=ax,
        **params,
    )

    if mse:
        mse = mean_squared_error(y, y_pred)
        mse_sq = np.sqrt(mse)

        r1 = resid[(resid > -mse_sq) & (resid < mse_sq)].count() / resid.count() * 100
        r2 = (
            resid[(resid > -2 * mse_sq) & (resid < 2 * mse_sq)].count()
            / resid.count()
            * 100
        )
        r3 = (
            resid[(resid > -3 * mse_sq) & (resid < 3 * mse_sq)].count()
            / resid.count()
            * 100
        )

        mse_r = [r1, r2, r3]

        for i, c in enumerate(["red", "green", "black"]):
            ax.axhline(mse_sq * (i + 1), color=c, linestyle="--", linewidth=0.5)
            ax.axhline(mse_sq * (-(i + 1)), color=c, linestyle="--", linewidth=0.5)

        # 현재 표시되는 그래프의 x축 범위를 가져온다.
        xmin, xmax = ax.get_xlim()

        target = [68, 95, 99]
        for i, c in enumerate(["red", "green", "black"]):
            ax.text(
                s=f"{i+1}"
                r"${}\sqrt{MSE}$ = %.2f%% (%.2f%%)" % (mse_r[i], mse_r[i] - target[i]),
                x=xmax + 0.2,
                y=(i + 1) * mse_sq,
                color=c,
            )
            ax.text(
                s=f"-{i+1}"
                r"${}\sqrt{MSE}$ = %.2f%% (%.2f%%)" % (mse_r[i], mse_r[i] - target[i]),
                x=xmax + 0.2,
                y=-(i + 1) * mse_sq,
                color=c,
            )
    else:
        ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_qqplot(
    y_pred,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """QQ플롯을 출력한다.

    Args:
        y_pred (Series): 종속변수에 대한 예측치
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    (x, y), _ = probplot(zscore(y_pred))
    k = (max(x) + 0.5).round()

    sb.scatterplot(x=x, y=y, ax=ax, **params)
    sb.lineplot(x=[-k, k], y=[-k, k], color="red", linestyle="--", ax=ax)
    ax.grid()

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_confusion_matrix(
    y: np.ndarray,
    y_pred: np.ndarray,
    cmap: str = None,
    figsize: tuple = (4, 3),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """혼동행렬을 출력한다.

    Args:
        y_true (np.ndarray): 실제값
        y_pred (np.ndarray): 예측값
        cmap (str, optional): 칼라맵. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    outparams = False

    if ax == None:
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        outparams = True

    # 이진분류인지 다항분류인지 구분
    if hasattr(y, "unique"):
        labels = sorted(list(y.unique()))
    else:
        labels = sorted(list(set(y)))

    is_binary = len(labels) == 2

    if is_binary:
        labels = ["Negative", "Positive"]
    else:
        labels = None

    # 다중 로지스틱을 살펴볼 때 함수 파라미터 설정이 변경될 수 있다.
    ConfusionMatrixDisplay.from_predictions(
        y,  # 관측치
        y_pred,  # 예측치
        display_labels=labels,
        cmap=cmap,
        text_kw={"fontsize": 24, "weight": "bold"},
        ax=ax,
        **params,
    )

    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams == True:
        plt.show()
        plt.close()



def my_learing_curve(
    estimator: any,
    data: DataFrame,
    yname: str = "target",
    scalling: bool = False,
    cv: int = 5,
    train_sizes: np.ndarray = np.array([0.1, 0.3, 0.5, 0.7, 1]),
    scoring: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    random_state: int = get_random_state(),
    callback: any = None,
) -> None:
    # if estimator.__class__.__name__ in [
    #     "XGBRegressor",
    #     "XGBClassifier",
    #     "LGBMRegressor",
    #     "LGBMClassifier",
    # ]:
    #     my_loss_curve(estimator=estimator, figsize=figsize, dpi=dpi, callback=callback)
    # else:
    my_ml_learing_curve(
        estimator=estimator,
        data=data,
        yname=yname,
        scalling=scalling,
        cv=cv,
        train_sizes=train_sizes,
        scoring=scoring,
        figsize=figsize,
        dpi=dpi,
        random_state=random_state,
        callback=callback,
    )



def my_ml_learing_curve(
    estimator: any,
    data: DataFrame,
    yname: str = "target",
    scalling: bool = False,
    cv: int = 5,
    train_sizes: np.ndarray = np.array([0.1, 0.3, 0.5, 0.7, 1]),
    scoring: str = None,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    random_state: int = get_random_state(),
    callback: any = None,
) -> None:
    """일반적인 머신러닝 알고리즘에 대한 학습곡선을 출력한다.

    Args:
        estimator (any): 학습모델 객체
        data (DataFrame): 독립변수
        yname (Series): 종속변수
        scaling (bool, optional): 스케일링 여부. Defaults to False.
        cv (int, optional): 교차검증의 수. Defaults to 10.
        train_sizes (np.ndarray, optional): 훈련 데이터의 비율. Defaults to np.linspace(0.1, 1.0, 10).
        scoring (str, optional): 평가지표. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        random_state (int, optional): 난수 시드값. Defaults to 123.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    if yname not in data.columns:
        raise Exception(f"\x1b[31m종속변수 {yname}가 존재하지 않습니다.\x1b[0m")

    x = data.drop(yname, axis=1)
    y = data[yname]
    w = 1
    cv = 3

    if scalling:
        scaler = StandardScaler()
        x = DataFrame(scaler.fit_transform(x), index=x.index, columns=x.columns)

    # 평가지표가 없는 경우
    if scoring == None:
        try:
            train_sizes, train_scores, test_scores = learning_curve(
                estimator=estimator,
                X=x,
                y=y,
                cv=cv,
                n_jobs=get_n_jobs(),
                train_sizes=train_sizes,
                random_state=random_state,
                # explore_incremental_learning=True,
            )

            ylabel = "Score"
        except Exception as e:
            print(
                f"\x1b[31m이 데이터는 학습곡선을 도출하는데 적합하지 않습니다.\n\n\x1b[0m"
            )
            return

    # 평가지표가 있는 경우
    else:
        scoring = scoring.lower()

        if scoring == "rmse":
            scoring = "neg_root_mean_squared_error"
            w = -1
        elif scoring == "mse":
            scoring = "neg_mean_squared_error"
            w = -1
        elif scoring == "r2":
            scoring = "r2"

        scoring_list = [
            "r2",
            "max_error",
            "matthews_corrcoef",
            "neg_median_absolute_error",
            "neg_mean_absolute_error",
            "neg_mean_absolute_percentage_error",
            "neg_mean_squared_error",
            "neg_mean_squared_log_error",
            "neg_root_mean_squared_error",
            "neg_root_mean_squared_log_error",
            "neg_mean_poisson_deviance",
            "neg_mean_gamma_deviance",
            "accuracy",
            "top_k_accuracy",
            "roc_auc",
            "roc_auc_ovr",
            "roc_auc_ovo",
            "roc_auc_ovr_weighted",
            "roc_auc_ovo_weighted",
            "balanced_accuracy",
            "average_precision",
            "neg_log_loss",
            "neg_brier_score",
            "positive_likelihood_ratio",
            "neg_negative_likelihood_ratio",
            "adjusted_rand_score",
            "rand_score",
            "homogeneity_score",
            "completeness_score",
            "v_measure_score",
            "mutual_info_score",
            "adjusted_mutual_info_score",
            "normalized_mutual_info_score",
            "fowlkes_mallows_score",
        ]

        if scoring not in scoring_list:
            raise Exception(f"\x1b[31m평가지표 {scoring}가 존재하지 않습니다.\x1b[0m")

        ylabel = scoring.upper()

        try:
            train_sizes, train_scores, test_scores = learning_curve(
                estimator=estimator,
                X=x,
                y=y,
                cv=cv,
                n_jobs=get_n_jobs(),
                train_sizes=train_sizes,
                scoring=scoring,
                random_state=get_random_state(),
                # explore_incremental_learning=True,
            )
        except Exception as e:
            print(
                f"\x1b[31m이 데이터는 학습곡선을 도출하는데 적합하지 않습니다.\x1b[0m\n\n"
            )
            return

    # train_scores = [[ -61003.39168011, -21392.74320181, -21392.74320181],
    #         [ -76013.80551592,  -32713.77746228,  -32713.77746228],
    #         [ -87385.64805928-100000,  -43309.22213511-100000,  -43309.22213511-100000],
    #         [-112835.37952499-200000,  -87366.06685146-200000,  -62626.18305325-200000],
    #         [-154305.06180818-300000, -135729.50446442-300000,  -79106.13103102-300000]]

    # test_scores = [[ -758010.29567053, -1993073.60277415, -7286529.65151857],
    #         [ -769563.79017967, -1127215.58336001, -5289498.25740984],
    #         [ -768868.04482785,  -554770.29942592, -3640333.34595124],
    #         [ -771288.32211202,  -485353.5457528,  -2679328.88540856],
    #         [ -872974.46103711,  -495561.41897267, -1113166.29049758]]

    # for i in range(len(test_scores)):
    #     for j in range(len(test_scores[i])):
    #         if i > 1:
    #             x = 0
    #             if i == 1:
    #                 x = 330000
    #             elif x == 2:
    #                 x = 300000
    #             elif x == 3:
    #                 x = 270000
    #             else:
    #                 x = 230000
    #             train_scores[i][j] = train_scores[i][j] - (x * i)
    #             test_scores[i][j] = test_scores[i][j] - (150000 * i)

    #         if i == 1:
    #             train_scores[i][j] = train_scores[i][j] - 320000



    train_mean = w * np.mean(train_scores, axis=1)
    train_std = w * np.std(train_scores, axis=1)
    test_mean = w * np.mean(test_scores, axis=1)
    test_std = w * np.std(test_scores, axis=1)

    plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.gca()

    # 훈련 데이터 수에 따른 훈련 데이터의 score 평균
    sb.lineplot(
        x=train_sizes,
        y=train_mean,
        marker="o",
        markersize=5,
        label="훈련 데이터",
        color="#ff2200",
        ax=ax,
    )
    ax.fill_between(
        x=train_sizes,
        y1=train_mean + train_std,
        y2=train_mean - train_std,
        alpha=0.15,
        color="#ff2200",
    )

    # 검증 데이터 수에 따른 검증 데이터의 score 평균
    sb.lineplot(
        x=train_sizes,
        y=test_mean,
        linestyle="--",
        marker="s",
        markersize=5,
        label="검증 데이터",
        color="#0066ff",
        ax=ax,
    )
    ax.fill_between(
        x=train_sizes,
        y1=test_mean + test_std,
        y2=test_mean - test_std,
        alpha=0.15,
        color="#0066ff",
    )

    ax.grid()
    ax.set_xlabel(xlabel="훈련 셋트 크기")
    ax.set_ylabel(ylabel=ylabel)
    ax.set_title(label="Learning Curve")
    ax.legend()

    if callback:
        callback(ax)

    plt.tight_layout()
    plt.show()
    plt.close()



def my_loss_curve(
    estimator: any,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
    ax: Axes = None,
) -> None:
    # 손실률 데이터 가져오기
    if hasattr(estimator, "evals_result"):
        results = estimator.evals_result()
    elif hasattr(estimator, "evals_result_"):
        results = estimator.evals_result_
    else:
        print("\x1b[31m손실률 데이터가 존재하지 않습니다.\x1b[0m")
        return

    # 손실률 데이터 정리
    keys = list(results.keys())
    score_df = DataFrame()

    for key in keys:
        score_keys = list(results[key].keys())

        df = DataFrame({"dataset": [key] * len(results[key][score_keys[0]])})

        for score_key in score_keys:
            k = score_key[score_key.rfind("_") + 1 :]
            df[k] = results[key][score_key]

        df.reset_index(drop=False, names="epoch", inplace=True)
        df["epoch"] = df["epoch"] + 1

        score_df = concat([score_df, df], axis=0, ignore_index=True)

    col = score_df.columns[2:]
    cols = len(col)
    figsize = (10 if cols == 1 else 7, 5)
    fig, ax = plt.subplots(
        nrows=1, ncols=cols, figsize=(figsize[0] * cols, figsize[1]), dpi=100
    )

    if cols == 1:
        ax = [ax]

    for i in range(cols):
        sb.lineplot(data=score_df, x="epoch", y=col[i], hue="dataset", ax=ax[i])
        ax[i].set_title(col[i])
        ax[i].set_xlabel("epoch")
        ax[i].set_ylabel(col[i])
        ax[i].grid(True)

    plt.show()
    plt.close()



def my_roc_curve_binary(
    estimator: any,
    x: DataFrame,
    y: Series,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    figsize: tuple = (6, 5),
    dpi: int = 100,
    callback: any = None,
) -> None:
    """이진 분류에 대한 ROC와 Precision-Recall 곡선을 출력한다.

    Args:
        estimator (any): 학습모델 객체
        x (DataFrame): 독립변수
        y (Series): 실제값
        hist (bool, optional): 히스토그램 출력 여부. Defaults to True.
        roc (bool, optional): ROC 곡선 출력 여부. Defaults to True.
        pr (bool, optional): Precision-Recall 곡선 출력 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 10).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    cols = 0

    if hist:
        cols += 1

    if roc:
        cols += 1

    if pr:
        cols += 1

    if cols == 0:
        return

    fig_index = 0

    figsize_ = (figsize[0] * cols, figsize[1])

    # 히스토그램을 위한 구간
    bins = [i / 20 for i in range(0, 21)]

    title = "{0} vs {1}".format(estimator.classes_[0], estimator.classes_[1])

    # 예측확률
    y_proba = estimator.predict_proba(x)[:, 1]

    # 비교 대상을 위한 데이터
    df_aux = DataFrame({"class": y, "prob": y_proba})

    fig, ax = plt.subplots(1, cols, figsize=figsize_, dpi=dpi)

    # ax[0] : histogram -------------------------
    if hist:
        sb.histplot(data=df_aux, x="prob", hue="class", bins=bins, ax=ax[fig_index])
        # ax[fig_index].legend()
        ax[fig_index].grid()
        fig_index += 1

    # ROC Curve
    if roc:
        fpr, tpr, thresholds = roc_curve(df_aux["class"], df_aux["prob"])
        sb.lineplot(
            x=fpr, y=tpr, color="red", linewidth=1, label="ROC Curve", ax=ax[fig_index]
        )
        ax[fig_index].fill_between(fpr, tpr, facecolor="blue", alpha=0.1)
        sb.lineplot(
            x=[0, 1],
            y=[0, 1],
            color="black",
            linestyle="--",
            linewidth=0.7,
            label="chance",
            ax=ax[fig_index],
        )
        ax[fig_index].set_xlabel("False Positive Rate")
        ax[fig_index].set_ylabel("True Positive Rate")
        ax[fig_index].set_xticks(
            np.round(np.arange(start=0, stop=1.1, step=0.1), decimals=2)
        )
        ax[fig_index].set_xlim([-0.01, 1.01])
        ax[fig_index].set_ylim([-0.01, 1.01])
        ax[fig_index].text(
            0.95,
            0.05,
            "AUC=%0.3f" % roc_auc_score(y_true=df_aux["class"], y_score=df_aux["prob"]),
            fontsize=16,
            ha="right",
            va="bottom",
        )
        ax[fig_index].legend()
        ax[fig_index].grid()
        fig_index += 1

    # Precision-Recall Curve
    if pr:
        precision, recall, thresholds = precision_recall_curve(
            y_true=df_aux["class"], probas_pred=df_aux["prob"]
        )
        y_mean = y.mean()

        sb.lineplot(
            x=recall,
            y=precision,
            label="Precision / Recall Curve",
            color="blue",
            linewidth=1,
            ax=ax[fig_index],
        )
        sb.lineplot(
            x=[0, 1],
            y=[y_mean, y_mean],
            color="black",
            linewidth=0.7,
            linestyle="--",
            label="chance",
            ax=ax[fig_index],
        )
        ax[fig_index].set_xlabel("Recall")
        ax[fig_index].set_ylabel("Precision")
        ax[fig_index].set_xticks(
            np.round(a=np.arange(start=0, stop=1.1, step=0.1), decimals=2)
        )
        ax[fig_index].set_xlim([-0.01, 1.01])
        ax[fig_index].set_ylim([y_mean - 0.05, 1.01])
        ax[fig_index].legend()
        ax[fig_index].grid()
        fig_index += 1

    if callback:
        callback(*ax)

    plt.tight_layout()
    plt.show()
    plt.close()



def my_roc_curve_multiclass_ovo(
    estimator: any,
    x: DataFrame,
    y: Series,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    figsize: tuple = (6, 5),
    dpi: int = 100,
) -> None:
    """다중 분류에 대한 ROC와 Precision-Recall 곡선을 출력한다.

    Args:
        estimator (any): 학습모델 객체
        x (DataFrame): 독립변수
        y (Series): 실제값
        hist (bool, optional): 히스토그램 출력 여부. Defaults to True.
        roc (bool, optional): ROC 곡선 출력 여부. Defaults to True.
        pr (bool, optional): Precision-Recall 곡선 출력 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 10).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
    """
    cols = 0

    if hist:
        cols += 1

    if roc:
        cols += 1

    if pr:
        cols += 1

    if cols == 0:
        return

    figsize_ = (figsize[0] * cols, figsize[1])

    # 추정확률
    y_proba = estimator.predict_proba(x)

    # 히스토그램을 위한 구간
    bins = [i / 20 for i in range(0, 21)]

    # 각 대조군 별로 auc값을 저장할 리스트
    auc_list = []

    # 모든 클래스 쌍의 조합 만들기
    class_combinations = []
    class_list = list(estimator.classes_)
    size = len(class_list)

    for i in range(0, size):
        j = (i + 1) % size
        comb_item = [class_list[i], class_list[j]]
        class_combinations.append(comb_item)

        comb_item = [class_list[j], class_list[i]]
        class_combinations.append(comb_item)

    # 대조군 단위로 반복
    for i in range(len(class_combinations)):
        fig_index = 0

        c1, c2 = class_combinations[i]
        # print(c1, c2)
        c1_index = class_list.index(c1)
        title = "{0} vs {1}".format(c1, c2)
        # print(title)

        # 비교 대상에 대한 데이터
        df_aux = DataFrame({"class": y, "prob": y_proba[:, c1_index]})

        # 현재 대조군에 맞는 항목만 필터링
        df_aux = df_aux[(df_aux["class"] == c1) | (df_aux["class"] == c2)]

        # 현재 대조군 데이터 중에서 맞춘 것은 1, 못맞춘 것은 0으로 재설정
        df_aux["class"] = [1 if y == c1 else 0 for y in df_aux["class"]]
        df_aux = df_aux.reset_index(drop=True)

        # my_pretty_table(df_aux.head(10))

        # 시각화 시작
        fig, ax = plt.subplots(1, cols, figsize=figsize_, dpi=dpi)
        plt.suptitle(title, fontsize=16)

        if cols < 2:
            ax = [ax]

        # ax[0] : histogram -------------------------
        if hist:
            sb.histplot(data=df_aux, x="prob", hue="class", bins=bins, ax=ax[fig_index])
            # ax[fig_index].legend()
            ax[fig_index].set_xlabel(f"P(x = {c1})")
            ax[fig_index].grid()
            fig_index += 1

        # ax[1] : ROC Curve -------------------------
        if roc:
            fpr, tpr, _ = roc_curve(df_aux["class"], df_aux["prob"])
            auc = roc_auc_score(df_aux["class"], df_aux["prob"])

            auc_list.append(auc)

            sb.lineplot(
                x=fpr,
                y=tpr,
                linewidth=1,
                label="ROC Curve",
                color="#0066ff",
                ci=0,
                ax=ax[fig_index],
            )
            sb.lineplot(
                x=[0, 1],
                y=[0, 1],
                linestyle="--",
                color="#005500",
                linewidth=0.5,
                label="chance level",
                ax=ax[fig_index],
            )
            ax[fig_index].set_xlabel("Fase Positive Rate")
            ax[fig_index].set_ylabel("True Positive Rate")
            ax[fig_index].set_xticks(np.round(np.arange(0, 1.1, 0.1), 2))
            ax[fig_index].set_xlim([-0.01, 1.01])
            ax[fig_index].set_ylim([-0.01, 1.01])
            ax[fig_index].grid()
            ax[fig_index].legend()
            ax[fig_index].text(
                0.95, 0.05, "AUC=%0.3f" % auc, fontsize=16, ha="right", va="bottom"
            )
            fig_index += 1

        # ax[2] : PR Curve -------------------------
        if pr:
            precision, recall, thresholds = precision_recall_curve(
                df_aux["class"], df_aux["prob"]
            )
            y_mean = df_aux["class"].mean()

            sb.lineplot(
                x=recall,
                y=precision,
                label="Precision / Recall Curve",
                color="blue",
                linewidth=1,
                ax=ax[fig_index],
            )
            sb.lineplot(
                x=[0, 1],
                y=[y_mean, y_mean],
                color="#005500",
                linewidth=0.5,
                linestyle="--",
                label="chance level",
                ax=ax[fig_index],
            )
            ax[fig_index].set_xlabel("Recall")
            ax[fig_index].set_ylabel("Precision")
            ax[fig_index].set_xticks(np.round(np.arange(0, 1.1, 0.1), 2))
            ax[fig_index].set_xlim([-0.01, 1.01])
            ax[fig_index].set_ylim([y_mean - 0.05, 1.01])
            ax[fig_index].legend()
            ax[fig_index].grid()
            fig_index += 1

        plt.tight_layout()
        plt.show()
        plt.close()

        # break

    if auc_list:
        print(f"mean roc auc ovo: {sum(auc_list)/len(auc_list):.4f}")



def my_roc_curve_multiclass_ovr(
    estimator: any,
    x: DataFrame,
    y: Series,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    figsize: tuple = (6, 5),
    dpi: int = 100,
) -> None:
    """다중 분류에 대한 ROC와 Precision-Recall 곡선을 출력한다.

    Args:
        estimator (any): 학습모델 객체
        x (DataFrame): 독립변수
        y (Series): 실제값
        hist (bool, optional): 히스토그램 출력 여부. Defaults to True.
        roc (bool, optional): ROC 곡선 출력 여부. Defaults to True.
        pr (bool, optional): Precision-Recall 곡선 출력 여부. Defaults to True.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 10).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
    """
    cols = 0

    if hist:
        cols += 1

    if roc:
        cols += 1

    if pr:
        cols += 1

    if cols == 0:
        return

    figsize_ = (figsize[0] * cols, figsize[1])

    # 추정확률
    y_proba = estimator.predict_proba(x)

    # 히스토그램을 위한 구간
    bins = [i / 20 for i in range(0, 21)]

    # 각 대조군 별로 auc값을 저장할 리스트
    auc_list = []

    # 모든 클래스
    class_list = list(estimator.classes_)
    size = len(class_list)

    # 대조군 단위로 반복
    for i in range(0, size):
        fig_index = 0

        c = class_list[i]
        title = "{0} vs Rest".format(c)

        # 비교 대상에 대한 데이터
        df_aux = DataFrame({"class": y, "prob": y_proba[:, i]})

        # class값에서 현재 c를 True에 해당하는 1로, 나머지를 False에 해당하는 0으로 변경
        df_aux["class"] = np.where(df_aux["class"] == c, 1, 0)

        # 인덱스 재설정
        df_aux = df_aux.reset_index(drop=True)

        # my_pretty_table(df_aux.head(10))

        # 시각화 시작
        fig, ax = plt.subplots(1, cols, figsize=figsize_, dpi=dpi)
        plt.suptitle(title, fontsize=16)

        if cols < 2:
            ax = [ax]

        # ax[0] : histogram -------------------------
        if hist:
            sb.histplot(data=df_aux, x="prob", hue="class", bins=bins, ax=ax[fig_index])
            # ax[fig_index].legend([f"{c}", "Rest"])
            ax[fig_index].set_xlabel(f"P(x = {c})")
            ax[fig_index].grid()
            fig_index += 1

        # ax[1] : ROC Curve -------------------------
        if roc:
            fpr, tpr, _ = roc_curve(df_aux["class"], df_aux["prob"])
            auc = roc_auc_score(df_aux["class"], df_aux["prob"])

            auc_list.append(auc)

            sb.lineplot(
                x=fpr,
                y=tpr,
                linewidth=1,
                label="ROC Curve",
                color="#0066ff",
                ci=0,
                ax=ax[fig_index],
            )
            sb.lineplot(
                x=[0, 1],
                y=[0, 1],
                linestyle="--",
                color="#005500",
                linewidth=0.5,
                label="chance",
                ax=ax[fig_index],
            )
            ax[fig_index].set_xlabel("Fase Positive Rate")
            ax[fig_index].set_ylabel("True Positive Rate")
            ax[fig_index].set_xticks(np.round(np.arange(0, 1.1, 0.1), 2))
            ax[fig_index].set_xlim([-0.01, 1.01])
            ax[fig_index].set_ylim([-0.01, 1.01])
            ax[fig_index].grid()
            ax[fig_index].text(
                0.95, 0.05, "AUC=%0.3f" % auc, fontsize=16, ha="right", va="bottom"
            )
            fig_index += 1

        # ax[2] : PR Curve -------------------------
        if roc:
            precision, recall, thresholds = precision_recall_curve(
                df_aux["class"], df_aux["prob"]
            )
            y_mean = df_aux["class"].mean()

            sb.lineplot(
                x=recall,
                y=precision,
                label="Precision / Recall Curve",
                color="blue",
                linewidth=1,
                ax=ax[fig_index],
            )
            sb.lineplot(
                x=[0, 1],
                y=[y_mean, y_mean],
                color="#005500",
                linewidth=0.5,
                linestyle="--",
                label="chance",
                ax=ax[fig_index],
            )
            ax[fig_index].set_xlabel("Recall")
            ax[fig_index].set_ylabel("Precision")
            ax[fig_index].set_xticks(np.round(np.arange(0, 1.1, 0.1), 2))
            ax[fig_index].set_xlim([-0.01, 1.01])
            ax[fig_index].set_ylim([y_mean - 0.05, 1.01])
            ax[fig_index].legend()
            ax[fig_index].grid()

        plt.tight_layout()
        plt.show()
        plt.close()

        # break

    if auc_list:
        print(f"mean roc auc ovo: {sum(auc_list)/len(auc_list):.4f}")



def my_roc_curve(
    estimator: any,
    x: DataFrame,
    y: Series,
    hist: bool = True,
    roc: bool = True,
    pr: bool = True,
    multiclass: str = None,
    figsize: tuple = (6, 5),
    dpi: int = 100,
    callback: any = None,
) -> None:
    """ROC와 Precision-Recall 곡선을 출력한다. 이진분류와 다중분류에 대해 모두 대응 가능한 통합 함수이다.

    Args:
        estimator (any): 학습모델 객체
        x (DataFrame): 독립변수
        y (Series): 실제값
        hist (bool, optional): 히스토그램 출력 여부. Defaults to True.
        roc (bool, optional): ROC 곡선 출력 여부. Defaults to True.
        pr (bool, optional): Precision-Recall 곡선 출력 여부. Defaults to True.
        multiclass (str, optional): 다중분류일 경우 ovo, ovr 중 선택. Defaults to None.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 10).
        dpi (int, optional): 그래프의 해상도. Defaults to 200.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    is_binary = len(estimator.classes_) == 2

    if is_binary:
        my_roc_curve_binary(estimator, x, y, hist, roc, pr, figsize, dpi, callback)
    else:
        if multiclass == "ovo":
            my_roc_curve_multiclass_ovo(estimator, x, y, hist, roc, pr, figsize, dpi)
        elif multiclass == "ovr":
            my_roc_curve_multiclass_ovr(estimator, x, y, hist, roc, pr, figsize, dpi)
        else:
            my_roc_curve_multiclass_ovo(estimator, x, y, hist, roc, pr, figsize, dpi)
            my_roc_curve_multiclass_ovr(estimator, x, y, hist, roc, pr, figsize, dpi)



def my_distribution_by_class(
    data: DataFrame,
    xnames: list = None,
    hue: str = None,
    type: str = "kde",
    bins: any = 5,
    palette: str = None,
    fill: bool = False,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
) -> None:
    """클래스별로 독립변수의 분포를 출력한다.

    Args:
        data (DataFrame): 독립변수
        xnames (list, optional): 독립변수의 이름. Defaults to None.
        hue (str, optional): 클래스별로 구분할 변수. Defaults to None.
        type (str, optional): 그래프 종류 (kde, hist, histkde). Defaults to "kde".
        bins (any, optional): 히스토그램의 구간 수. Defaults to 5.
        palette (str, optional): 칼라맵. Defaults to None.
        fill (bool, optional): kde 그래프의 채우기 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """
    if xnames == None:
        xnames = data.columns

    for i, v in enumerate(xnames):
        # 종속변수이거나 숫자형이 아닌 경우는 제외
        if v == hue or data[v].dtype not in [
            "int",
            "int32",
            "int64",
            "float",
            "float32",
            "float64",
        ]:
            continue

        if type == "kde":
            my_kdeplot(
                df=data,
                xname=v,
                hue=hue,
                palette=palette,
                fill=fill,
                figsize=figsize,
                dpi=dpi,
                callback=callback,
            )
        elif type == "hist":
            my_histplot(
                df=data,
                xname=v,
                hue=hue,
                bins=bins,
                kde=False,
                palette=palette,
                figsize=figsize,
                dpi=dpi,
                callback=callback,
            )
        elif type == "histkde":
            my_histplot(
                df=data,
                xname=v,
                hue=hue,
                bins=bins,
                kde=True,
                palette=palette,
                figsize=figsize,
                dpi=dpi,
                callback=callback,
            )



def my_scatter_by_class(
    data: DataFrame,
    group: list = None,
    hue: str = None,
    palette: str = None,
    outline: bool = False,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    callback: any = None,
) -> None:
    """클래스별로 독립변수의 산점도를 출력한다.

    Args:
        data (DataFrame): 독립변수
        group (list, optional): 독립변수의 조합. Defaults to None.
        hue (str, optional): 클래스별로 구분할 변수. Defaults to None.
        palette (str, optional): 칼라맵. Defaults to None.
        outline (bool, optional): 테두리 여부. Defaults to False.
        figsize (tuple, optional): 그래프의 크기. Defaults to (10, 5).
        dpi (int, optional): 그래프의 해상도. Defaults to 100.
        callback (any, optional): ax객체를 전달받아 추가적인 옵션을 처리할 수 있는 콜백함수. Defaults to None.
    """

    if group == None:
        group = []

        xnames = data.columns

        for i, v in enumerate(xnames):
            j = (i + 1) % len(xnames)

            if (
                v == hue
                or xnames[j] == hue
                or data[v].dtype
                not in [
                    "int",
                    "int32",
                    "int64",
                    "float",
                    "float32",
                    "float64",
                ]
            ):
                continue

            group.append([v, xnames[j]])

    if outline:
        for i, v in enumerate(group):
            my_convex_hull(data, v[0], v[1], hue, palette, figsize, dpi, callback)
    else:
        for i, v in enumerate(group):
            my_scatterplot(data, v[0], v[1], hue, palette, figsize, dpi, callback)



def my_tree(estimator: DecisionTreeClassifier) -> None:
    """의사결정나무를 출력한다.

    Args:
        estimator (DecisionTreeClassifier): 학습된 의사결정나무 객체
    """

    print("depth:", estimator.get_depth(), "leaves:", estimator.get_n_leaves())

    xnames = list(estimator.feature_names_in_)
    class_names = estimator.classes_
    class_names = [str(object=i) for i in class_names]

    export_graphviz(
        decision_tree=estimator,
        out_file="tree.dot",
        feature_names=xnames,
        class_names=class_names,
        rounded=True,  # 노드의 모서리를 둥글게
        filled=True,  # 노드의 색상을 다르게
        fontname=__FONT_NAME__,
    )

    with open("tree.dot") as f:
        dot = f.read()
        display(graphviz.Source(dot))

    os.remove("tree.dot")



def my_plot_importance(
    estimator: any,
    figsize: tuple = (10, 5),
    dpi: int = 100,
    max_num_features: int = None,
    ax: Axes = None,
) -> None:
    outparams = False

    if ax == None:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.gca()
        outparams = True

    if estimator.__class__.__name__ in ["LGBMClassifier", "LGBMRegressor"]:
        lgb_plot_importance(booster=estimator, height=0.6, ax=ax, max_num_features=max_num_features)
    elif estimator.__class__.__name__ in ["XGBClassifier", "XGBRegressor"]:
        xgb_plot_importance(booster=estimator, height=0.6, ax=ax, max_num_features=max_num_features)

    plt.tight_layout()

    if outparams == True:
        plt.show()
        plt.close()
