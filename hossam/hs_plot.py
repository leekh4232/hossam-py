# -*- coding: utf-8 -*-
from __future__ import annotations

# -------------------------------------------------------------
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from math import sqrt
from pandas import DataFrame
from . import hs_dpi, hs_fig_width, hs_fig_height

# -------------------------------------------------------------
from scipy.stats import t
from scipy.spatial import ConvexHull
from scipy.stats import zscore, probplot

# -------------------------------------------------------------
from statannotations.Annotator import Annotator

# -------------------------------------------------------------
from sklearn.metrics import (
    mean_squared_error,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    confusion_matrix
)

# -------------------------------------------------------------
if pd.__version__ > "2.0.0":
    pd.DataFrame.iteritems = pd.DataFrame.items


# -------------------------------------------------------------
def get_default_ax(width: int = hs_fig_width, height: int = hs_fig_height, rows: int = 1, cols: int = 1, dpi: int = hs_dpi, flatten: bool = False, ws: int | None = None, hs: int | None = None):
    """기본 크기의 Figure와 Axes를 생성한다.

    Args:
        width (int): 가로 픽셀 크기.
        height (int): 세로 픽셀 크기.
        rows (int): 서브플롯 행 개수.
        cols (int): 서브플롯 열 개수.
        dpi (int): 해상도(DPI).

    Returns:
        tuple[Figure, Axes]: 생성된 matplotlib Figure와 Axes 객체.
    """
    figsize = (width * cols / dpi, height * rows / dpi)
    fig, ax = plt.subplots(rows, cols, figsize=figsize, dpi=dpi)

    if (rows > 1 or cols > 1) and (ws != None and hs != None):
        fig.subplots_adjust(wspace=ws, hspace=hs)

    if flatten == True:
        ax = ax.flatten()

    return fig, ax


# -------------------------------------------------------------
def finalize_plot(ax: Axes, callback: any = None, outparams: bool = False) -> None:
    """공통 후처리를 수행한다: 콜백 실행, 레이아웃 정리, 필요 시 표시/종료.

    Args:
        ax (Axes): 대상 Axes.
        callback (Callable|None): 추가 설정을 위한 사용자 콜백.
        outparams (bool): 내부에서 생성한 Figure인 경우 True.

    Returns:
        None
    """
    if callback:
        callback(ax)

    plt.tight_layout()
    if outparams:
        plt.show()
        plt.close()


# -------------------------------------------------------------
def lineplot(
    df: DataFrame,
    xname: str = None,
    yname: str = None,
    hue: str = None,
    marker: str = None,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """선 그래프를 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str|None): x축 컬럼명.
        yname (str|None): y축 컬럼명.
        hue (str|None): 범주 구분 컬럼명.
        marker (str|None): 마커 모양.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn lineplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # hue가 있을 때만 palette 사용, 없으면 color 사용
    lineplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "hue": hue,
        "marker": marker,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        lineplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        lineplot_kwargs["color"] = sb.color_palette(palette)[0]

    lineplot_kwargs.update(params)

    sb.lineplot(**lineplot_kwargs)
    ax.grid(True, alpha=0.3)
    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def boxplot(
    df: DataFrame,
    xname: str = None,
    yname: str = None,
    orient: str = "v",
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """상자그림(boxplot)을 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str|None): x축 범주 컬럼명.
        yname (str|None): y축 값 컬럼명.
        orient (str): 'v' 또는 'h' 방향.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn boxplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    if xname is not None and yname is not None:
        boxplot_kwargs = {
            "data": df,
            "x": xname,
            "y": yname,
            "orient": orient,
            "ax": ax,
        }

        # hue 파라미터 확인 (params에 있을 수 있음)
        hue_value = params.get("hue", None)

        if hue_value is not None and palette is not None:
            boxplot_kwargs["palette"] = palette
        elif hue_value is None and palette is not None:
            boxplot_kwargs["color"] = sb.color_palette(palette)[0]

        boxplot_kwargs.update(params)
        sb.boxplot(**boxplot_kwargs)
    else:
        sb.boxplot(data=df, orient=orient, ax=ax, **params)

    ax.grid(True, alpha=0.3)
    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def kdeplot(
    df: DataFrame,
    xname: str = None,
    yname: str = None,
    hue: str = None,
    palette: str = None,
    fill: bool = False,
    fill_alpha: float = 0.3,
    linewidth: float = 1,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """커널 밀도 추정(KDE) 그래프를 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str|None): x축 컬럼명.
        yname (str|None): y축 컬럼명.
        hue (str|None): 범주 컬럼명.
        palette (str|None): 팔레트 이름.
        fill (bool): 면적 채우기 여부.
        fill_alpha (float): 채움 투명도.
        linewidth (float): 선 굵기.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn kdeplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # 기본 kwargs 설정
    kdeplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "hue": hue,
        "fill": fill,
        "ax": ax,
    }

    # fill이 True일 때 alpha 추가
    if fill:
        kdeplot_kwargs["alpha"] = fill_alpha

    # hue가 있을 때만 palette 추가
    if hue is not None and palette is not None:
        kdeplot_kwargs["palette"] = palette

    # yname이 없을 때만 linewidth 추가 (1D KDE에서만 사용)
    if yname is None:
        kdeplot_kwargs["linewidth"] = linewidth

    # 추가 params 병합
    kdeplot_kwargs.update(params)

    sb.kdeplot(**kdeplot_kwargs)

    # plt.grid()
    ax.grid(True, alpha=0.3)
    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def histplot(
    df: DataFrame,
    xname: str,
    hue=None,
    bins=None,
    kde: bool = True,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """히스토그램을 그리고 필요 시 KDE를 함께 표시한다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 히스토그램 대상 컬럼명.
        hue (str|None): 범주 컬럼명.
        bins (int|sequence|None): 구간 수 또는 경계.
        kde (bool): KDE 표시 여부.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn histplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    if bins:
        histplot_kwargs = {
            "data": df,
            "x": xname,
            "hue": hue,
            "kde": kde,
            "bins": bins,
            "ax": ax,
        }

        if hue is not None and palette is not None:
            histplot_kwargs["palette"] = palette
        elif hue is None and palette is not None:
            histplot_kwargs["color"] = sb.color_palette(palette)[0]

        histplot_kwargs.update(params)
        sb.histplot(**histplot_kwargs)
    else:
        histplot_kwargs = {
            "data": df,
            "x": xname,
            "hue": hue,
            "kde": kde,
            "ax": ax,
        }

        if hue is not None and palette is not None:
            histplot_kwargs["palette"] = palette
        elif hue is None and palette is not None:
            histplot_kwargs["color"] = sb.color_palette(palette)[0]

        histplot_kwargs.update(params)
        sb.histplot(**histplot_kwargs)

    ax.grid(True, alpha=0.3)
    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def stackplot(
    df: DataFrame,
    xname: str,
    hue: str,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """클래스 비율을 100% 누적 막대로 표현한다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): x축 기준 컬럼.
        hue (str): 클래스 컬럼.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn histplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    df2 = df[[xname, hue]].copy()
    df2[xname] = df2[xname].astype(str)

    # stackplot은 hue가 필수이므로 palette를 그대로 사용
    stackplot_kwargs = {
        "data": df2,
        "x": xname,
        "hue": hue,
        "linewidth": 0.5,
        "stat": "probability",  # 전체에서의 비율로 그리기
        "multiple": "fill",  # 전체를 100%로 그리기
        "shrink": 0.8,  # 막대의 폭
        "ax": ax,
    }

    if palette is not None:
        stackplot_kwargs["palette"] = palette

    stackplot_kwargs.update(params)

    sb.histplot(**stackplot_kwargs)

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

    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def scatterplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """산점도를 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): x축 컬럼.
        yname (str): y축 컬럼.
        hue (str|None): 범주 컬럼.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn scatterplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # hue가 있을 때만 palette 사용, 없으면 color 사용
    scatterplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "hue": hue,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        scatterplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        scatterplot_kwargs["color"] = sb.color_palette(palette)[0]

    scatterplot_kwargs.update(params)

    sb.scatterplot(**scatterplot_kwargs)
    ax.grid(True, alpha=0.3)

    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def regplot(
    df: DataFrame,
    xname: str,
    yname: str,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """단순 회귀선이 포함된 산점도를 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 독립변수 컬럼.
        yname (str): 종속변수 컬럼.
        palette (str|None): 선/점 색상.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn regplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # regplot은 hue를 지원하지 않으므로 palette를 color로 변환
    scatter_color = None
    if palette is not None:
        scatter_color = sb.color_palette(palette)[0]

    regplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "scatter_kws": {"color": scatter_color} if scatter_color else {},
        "line_kws": {
            "color": "red",
            "linestyle": "--",
            "linewidth": 2
        },
        "ax": ax,
    }

    regplot_kwargs.update(params)

    sb.regplot(**regplot_kwargs)
    ax.grid(True, alpha=0.3)

    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def lmplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    **params,
) -> None:
    """seaborn lmplot으로 선형 모델 시각화를 수행한다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 독립변수 컬럼.
        yname (str): 종속변수 컬럼.
        hue (str|None): 범주 컬럼.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        **params: seaborn lmplot 추가 인자.

    Returns:
        None
    """
    # hue가 있을 때만 palette 사용, 없으면 scatter_kws에 color 설정
    lmplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "hue": hue,
    }

    if hue is not None and palette is not None:
        lmplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        lmplot_kwargs["scatter_kws"] = {"color": sb.color_palette(palette)[0]}

    lmplot_kwargs.update(params)

    g = sb.lmplot(**lmplot_kwargs)
    g.fig.set_size_inches(width / dpi, height / dpi)
    g.fig.set_dpi(dpi)
    g.ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close()


# -------------------------------------------------------------
def pairplot(
    df: DataFrame,
    diag_kind: str = "kde",
    hue=None,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    **params,
) -> None:
    """모든 숫자형/지정 컬럼 쌍에 대한 관계를 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        diag_kind (str): 대각선 플롯 종류('kde' 등).
        hue (str|None): 범주 컬럼.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 기본 크기 및 해상도(컬럼 수에 비례해 확대됨).
        **params: seaborn pairplot 추가 인자.

    Returns:
        None
    """
    # hue가 있을 때만 palette 사용
    pairplot_kwargs = {
        "data": df,
        "hue": hue,
        "diag_kind": diag_kind,
    }

    if hue is not None and palette is not None:
        pairplot_kwargs["palette"] = palette
    # pairplot은 hue 없이 palette만 쓰는 경우가 드물어서 color로 변환 불필요

    pairplot_kwargs.update(params)

    g = sb.pairplot(**pairplot_kwargs)
    scale = len(df.columns)
    g.fig.set_size_inches(w=(width / dpi) * scale, h=(height / dpi) * scale)
    g.fig.set_dpi(dpi)
    g.map_lower(func=sb.kdeplot, fill=True, alpha=0.3)
    g.map_upper(func=sb.scatterplot)
    plt.tight_layout()
    plt.show()
    plt.close()


# -------------------------------------------------------------
def countplot(
    df: DataFrame,
    xname: str,
    hue=None,
    palette: str = None,
    order: int = 1,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """범주 빈도 막대그래프를 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 범주 컬럼.
        hue (str|None): 보조 범주 컬럼.
        palette (str|None): 팔레트 이름.
        order (int): 숫자형일 때 정렬 방식(1: 값 기준, 기타: 빈도 기준).
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn countplot 추가 인자.

    Returns:
        None
    """
    outparams = False
    sort = None
    if str(df[xname].dtype) in ["int", "int32", "int64", "float", "float32", "float64"]:
        if order == 1:
            sort = sorted(list(df[xname].unique()))
        else:
            sort = sorted(list(df[xname].value_counts().index))

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # hue가 있을 때만 palette 사용, 없으면 color 사용
    countplot_kwargs = {
        "data": df,
        "x": xname,
        "hue": hue,
        "order": sort,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        countplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        # palette의 첫 번째 색상을 color로 사용
        countplot_kwargs["color"] = sb.color_palette(palette)[0]

    countplot_kwargs.update(params)

    sb.countplot(**countplot_kwargs)
    ax.grid(True, alpha=0.3)

    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def barplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """막대그래프를 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 범주 컬럼.
        yname (str): 값 컬럼.
        hue (str|None): 보조 범주 컬럼.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn barplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # hue가 있을 때만 palette 사용, 없으면 color 사용
    barplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "hue": hue,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        barplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        barplot_kwargs["color"] = sb.color_palette(palette)[0]

    barplot_kwargs.update(params)

    sb.barplot(**barplot_kwargs)
    ax.grid(True, alpha=0.3)

    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def boxenplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """박스앤 위스커 확장(boxen) 플롯을 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 범주 컬럼.
        yname (str): 값 컬럼.
        hue (str|None): 보조 범주 컬럼.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn boxenplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # palette은 hue가 있을 때만 사용
    boxenplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "hue": hue,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        boxenplot_kwargs["palette"] = palette

    boxenplot_kwargs.update(params)

    sb.boxenplot(**boxenplot_kwargs)
    ax.grid(True, alpha=0.3)

    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def violinplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """바이올린 플롯을 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 범주 컬럼.
        yname (str): 값 컬럼.
        hue (str|None): 보조 범주 컬럼.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn violinplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # palette은 hue가 있을 때만 사용
    violinplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "hue": hue,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        violinplot_kwargs["palette"] = palette

    violinplot_kwargs.update(params)

    sb.violinplot(**violinplot_kwargs)
    ax.grid(True, alpha=0.3)

    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def pointplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """포인트 플롯을 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 범주 컬럼.
        yname (str): 값 컬럼.
        hue (str|None): 보조 범주 컬럼.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn pointplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # hue가 있을 때만 palette 사용, 없으면 color 사용
    pointplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "hue": hue,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        pointplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        pointplot_kwargs["color"] = sb.color_palette(palette)[0]

    pointplot_kwargs.update(params)

    sb.pointplot(**pointplot_kwargs)
    ax.grid(True, alpha=0.3)

    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def jointplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    **params,
) -> None:
    """공동 분포(joint) 플롯을 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): x축 컬럼.
        yname (str): y축 컬럼.
        hue (str|None): 범주 컬럼.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        **params: seaborn jointplot 추가 인자.

    Returns:
        None
    """
    # hue가 있을 때만 palette 사용
    jointplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "hue": hue,
    }

    if hue is not None and palette is not None:
        jointplot_kwargs["palette"] = palette
    # jointplot은 hue 없이 palette만 쓰는 경우가 드물어서 color로 변환 불필요

    jointplot_kwargs.update(params)

    g = sb.jointplot(**jointplot_kwargs)
    g.fig.set_size_inches(width / dpi, height / dpi)
    g.fig.set_dpi(dpi)

    # 중앙 및 주변 플롯에 grid 추가
    g.ax_joint.grid(True, alpha=0.3)
    g.ax_marg_x.grid(True, alpha=0.3)
    g.ax_marg_y.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close()


# -------------------------------------------------------------
def heatmap(
    data: DataFrame,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """히트맵을 그린다(값 주석 포함).

    Args:
        data (DataFrame): 행렬 형태 데이터.
        palette (str|None): 컬러맵 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn heatmap 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # heatmap은 hue를 지원하지 않으므로 cmap에 palette 사용
    sb.heatmap(data, annot=True, cmap=palette, fmt=".2f", ax=ax, **params)

    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def convex_hull(
    data: DataFrame,
    xname: str,
    yname: str,
    hue: str,
    palette: str = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
):
    """클러스터별 볼록 껍질(convex hull)과 산점도를 그린다.

    Args:
        data (DataFrame): 시각화할 데이터.
        xname (str): x축 컬럼.
        yname (str): y축 컬럼.
        hue (str): 클러스터/범주 컬럼.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn scatterplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
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

    # convex_hull은 hue가 필수이므로 palette를 그대로 사용
    sb.scatterplot(
        data=data, x=xname, y=yname, hue=hue, palette=palette, ax=ax, **params
    )
    ax.grid(True, alpha=0.3)
    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def kde_confidence_interval(
    data: DataFrame,
    clevel=0.95,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
) -> None:
    """각 숫자 컬럼에 대해 KDE와 t-분포 기반 신뢰구간을 그린다.

    Args:
        data (DataFrame): 시각화할 데이터.
        clevel (float): 신뢰수준(0~1).
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.

    Returns:
        None
    """
    outparams = False
    y_min, y_max = None, None

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # 데이터 프레임의 컬럼이 여러 개인 경우 처리
    for c in data.columns:
        column = data[c].dropna()
        if len(column) < 2:
            continue
        # print(column)
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
        y_min = ymin if y_min is None else min(y_min, ymin)
        y_max = ymax if y_max is None else max(y_max, ymax)

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
            fontdict={"size": 7, "color": "red"},
        )

    if y_min is not None and y_max is not None:
        ax.set_ylim([y_min, y_max * 1.1])
    ax.grid(True, alpha=0.3)
    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def pvalue1_anotation(
    data: DataFrame,
    target: str,
    hue: str,
    pairs: list,
    test: str = "t-test_ind",
    text_format: str = "star",
    loc: str = "outside",
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params
) -> None:
    """statannotations를 이용해 상자그림에 p-value 주석을 추가한다.

    Args:
        data (DataFrame): 시각화할 데이터.
        target (str): 값 컬럼명.
        hue (str): 그룹 컬럼명.
        pairs (list): 비교할 (group_a, group_b) 튜플 목록.
        test (str): 적용할 통계 검정 이름.
        text_format (str): 주석 형식('star' 등).
        loc (str): 주석 위치.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # params에서 palette 추출 (있으면)
    palette_value = params.pop("palette", None)

    # boxplot kwargs 구성
    boxplot_kwargs = {
        "data": data,
        "x": hue,
        "y": target,
        "ax": ax,
    }

    # palette가 있으면 추가 (hue는 x에 이미 할당됨)
    if palette_value is not None:
        boxplot_kwargs["palette"] = palette_value

    boxplot_kwargs.update(params)

    sb.boxplot(**boxplot_kwargs)
    annotator = Annotator(ax, data=data, x=hue, y=target, pairs=pairs)
    annotator.configure(test=test, text_format=text_format, loc=loc)
    annotator.apply_and_annotate()

    sb.despine()

    ax.grid(True, alpha=0.3)
    finalize_plot(ax, callback, outparams)



# -------------------------------------------------------------
def residplot(
    y,
    y_pred,
    lowess: bool = False,
    mse: bool = False,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """잔차 대 예측치 산점도를 그린다(선택적으로 MSE 범위와 LOWESS 포함).

    Args:
        y (array-like): 실제 값.
        y_pred (array-like): 예측 값.
        lowess (bool): LOWESS 스무딩 적용 여부.
        mse (bool): √MSE, 2√MSE, 3√MSE 대역선과 비율 표시 여부.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn residplot 추가 인자.

    Returns:
        None
    """
    outparams = False
    resid = y - y_pred

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
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
        mse_val = mean_squared_error(y, y_pred)
        mse_sq = np.sqrt(mse_val)

        r1 = resid[(resid > -mse_sq) & (resid < mse_sq)].size / resid.size * 100
        r2 = (
            resid[(resid > -2 * mse_sq) & (resid < 2 * mse_sq)].size
            / resid.size
            * 100
        )
        r3 = (
            resid[(resid > -3 * mse_sq) & (resid < 3 * mse_sq)].size
            / resid.size
            * 100
        )

        mse_r = [r1, r2, r3]

        xmin, xmax = ax.get_xlim()

        # 구간별 반투명 색상 채우기 (안쪽부터 바깥쪽으로, 진한 색에서 연한 색으로)
        colors = ["red", "green", "blue"]
        alphas = [0.15, 0.10, 0.05]  # 안쪽이 더 진하게

        # 3σ 영역 (가장 바깥쪽, 가장 연함)
        ax.axhspan(-3 * mse_sq, 3 * mse_sq, facecolor=colors[2], alpha=alphas[2], zorder=0)
        # 2σ 영역 (중간)
        ax.axhspan(-2 * mse_sq, 2 * mse_sq, facecolor=colors[1], alpha=alphas[1], zorder=1)
        # 1σ 영역 (가장 안쪽, 가장 진함)
        ax.axhspan(-mse_sq, mse_sq, facecolor=colors[0], alpha=alphas[0], zorder=2)

        # 경계선 그리기
        for i, c in enumerate(["red", "green", "blue"]):
            ax.axhline(mse_sq * (i + 1), color=c, linestyle="--", linewidth=0.5)
            ax.axhline(mse_sq * (-(i + 1)), color=c, linestyle="--", linewidth=0.5)

        target = [68, 95, 99.7]
        for i, c in enumerate(["red", "green", "blue"]):
            ax.text(
                s=f"{i+1} sqrt(MSE) = {mse_r[i]:.2f}% ({mse_r[i] - target[i]:.2f}%)",
                x=xmax + 0.2,
                y=(i + 1) * mse_sq,
                color=c,
            )
            ax.text(
                s=f"-{i+1} sqrt(MSE) = {mse_r[i]:.2f}% ({mse_r[i] - target[i]:.2f}%)",
                x=xmax + 0.2,
                y=-(i + 1) * mse_sq,
                color=c,
            )
    ax.grid(True, alpha=0.3)
    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def qqplot(
    y_pred,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
    **params,
) -> None:
    """표준화된 예측값의 정규성 확인을 위한 QQ 플롯을 그린다.

    Args:
        y_pred (array-like): 예측 값.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn scatterplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # probplot은 내부적으로 정규화를 수행하므로 zscore를 미리 적용하면 안됨
    (x, y), _ = probplot(y_pred)
    k = (max(x) + 0.5).round()

    sb.scatterplot(x=x, y=y, ax=ax, **params)
    sb.lineplot(x=[-k, k], y=[-k, k], color="red", linestyle="--", ax=ax)
    ax.grid(True, alpha=0.3)
    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def distribution_by_class(
    data: DataFrame,
    xnames: list = None,
    hue: str = None,
    type: str = "kde",
    bins: any = 5,
    palette: str = None,
    fill: bool = False,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
) -> None:
    """클래스별로 각 숫자형 특징의 분포를 KDE 또는 히스토그램으로 그린다.

    Args:
        data (DataFrame): 시각화할 데이터.
        xnames (list|None): 대상 컬럼 목록(None이면 전 컬럼).
        hue (str|None): 클래스 컬럼.
        type (str): 'kde' | 'hist' | 'histkde'.
        bins (int|sequence|None): 히스토그램 구간.
        palette (str|None): 팔레트 이름.
        fill (bool): KDE 채움 여부.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.

    Returns:
        None
    """
    if xnames is None:
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
            kdeplot(
                df=data,
                xname=v,
                hue=hue,
                palette=palette,
                fill=fill,
                width=width,
                height=height,
                dpi=dpi,
                callback=callback,
            )
        elif type == "hist":
            histplot(
                df=data,
                xname=v,
                hue=hue,
                bins=bins,
                kde=False,
                palette=palette,
                width=width,
                height=height,
                dpi=dpi,
                callback=callback,
            )
        elif type == "histkde":
            histplot(
                df=data,
                xname=v,
                hue=hue,
                bins=bins,
                kde=True,
                palette=palette,
                width=width,
                height=height,
                dpi=dpi,
                callback=callback,
            )


# -------------------------------------------------------------
def scatter_by_class(
    data: DataFrame,
    group: list = None,
    hue: str = None,
    palette: str = None,
    outline: bool = False,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
) -> None:
    """클래스별로 특징 쌍을 산점도 또는 볼록 껍질로 시각화한다.

    Args:
        data (DataFrame): 시각화할 데이터.
        group (list|None): [[x, y], ...] 형태의 축 쌍 목록(None이면 자동 생성).
        hue (str|None): 클래스 컬럼.
        palette (str|None): 팔레트 이름.
        outline (bool): 볼록 껍질을 표시할지 여부.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 그림 크기 및 해상도.
        callback (Callable|None): Axes 후처리 콜백.

    Returns:
        None
    """

    if group is None:
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
            convex_hull(data, v[0], v[1], hue, palette, width, height, dpi, callback)
    else:
        for i, v in enumerate(group):
            scatterplot(data, v[0], v[1], hue, palette, width, height, dpi, callback)


# -------------------------------------------------------------
def roc_curve(
    fit,
    y: np.ndarray | pd.Series = None,
    X: pd.DataFrame | np.ndarray = None,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
) -> None:
    """로지스틱 회귀 적합 결과의 ROC 곡선을 시각화한다.

    Args:
        fit: statsmodels Logit 결과 객체 (`fit.predict()`로 예측 확률을 계산 가능해야 함).
        y (array-like|None): 외부 데이터의 실제 레이블. 제공 시 이를 실제값으로 사용.
        X (array-like|None): 외부 데이터의 설계행렬(독립변수). 제공 시 해당 데이터로 예측 확률 계산.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes. None이면 새로 생성.

    Notes:
        - 실제값: `y`가 주어지면 이를 사용, 없으면 `fit.model.endog`를 사용합니다.
        - 예측 확률: `X`가 주어지면 `fit.predict(X)`를 사용, 없으면 `fit.predict(fit.model.exog)`를 사용합니다.

    Returns:
        None
    """
    outparams = False
    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # 실제값(y_true) 결정
    if y is not None:
        y_true = np.asarray(y)
    else:
        # 학습 데이터의 종속변수 사용
        y_true = np.asarray(fit.model.endog)

    # 예측 확률 결정
    if X is not None:
        y_pred_proba = np.asarray(fit.predict(X))
    else:
        y_pred_proba = np.asarray(fit.predict(fit.model.exog))

    # ROC 곡선 계산
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # ROC 곡선 그리기
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('위양성율 (False Positive Rate)', fontsize=12)
    ax.set_ylabel('재현율 (True Positive Rate)', fontsize=12)
    ax.set_title('ROC 곡선', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    finalize_plot(ax, callback, outparams)


# -------------------------------------------------------------
def confusion_matrix_plot(
    fit,
    threshold: float = 0.5,
    width: int = hs_fig_width,
    height: int = hs_fig_height,
    dpi: int = hs_dpi,
    callback: any = None,
    ax: Axes = None,
) -> None:
    """로지스틱 회귀 적합 결과의 혼동행렬을 시각화한다.

    Args:
        fit: statsmodels Logit 결과 객체 (`fit.predict()`로 예측 확률을 계산 가능해야 함).
        threshold (float): 예측 확률을 이진 분류로 변환할 임계값. 기본값 0.5.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        dpi (int): 해상도.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes. None이면 새로 생성.

    Returns:
        None
    """
    outparams = False
    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1, dpi)
        outparams = True

    # 학습 데이터 기반 실제값/예측 확률 결정
    y_true = np.asarray(fit.model.endog)
    y_pred_proba = np.asarray(fit.predict(fit.model.exog))
    y_pred = (y_pred_proba >= threshold).astype(int)

    # 혼동행렬 계산
    cm = confusion_matrix(y_true, y_pred)

    # 혼동행렬 시각화
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['음성', '양성'])
    # 가독성을 위해 텍스트 크기/굵기 조정
    disp.plot(ax=ax, cmap='Blues', values_format='d', text_kw={"fontsize": 24, "weight": "bold"})

    ax.set_title(f'혼동행렬 (임계값: {threshold})', fontsize=14, fontweight='bold')

    finalize_plot(ax, callback, outparams)
