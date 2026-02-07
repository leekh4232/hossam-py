# -*- coding: utf-8 -*-
from __future__ import annotations
from types import SimpleNamespace
from typing import Callable, Literal
from itertools import combinations

# ===================================================================
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.figure import Figure  # type: ignore
from matplotlib.pyplot import Axes  # type: ignore
from pandas import Index, Series, DataFrame
from math import sqrt
from pandas import DataFrame

# ===================================================================
from scipy.stats import t
from scipy.spatial import ConvexHull
from scipy.cluster.hierarchy import dendrogram as sch_dendrogram
from statsmodels.graphics.gofplots import qqplot as sm_qqplot
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

# ===================================================================
from statannotations.Annotator import Annotator

# ===================================================================
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import (
    mean_squared_error,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    confusion_matrix,
    silhouette_score,
    silhouette_samples,
)

from .hs_util import is_2d

# ===================================================================
config = SimpleNamespace(
    dpi=200,
    width=1200,
    height=640,
    font_size=7,
    text_size=6,
    title_font_size=16,
    title_pad=12,
    label_font_size=12,
    font_weight="normal",
    frame_width=0.7,
    line_width=2,
    grid_alpha=0.3,
    grid_width=0.7,
    fill_alpha=0.3,
)


# ===================================================================
# 기본 크기가 설정된 Figure와 Axes를 생성한다
# ===================================================================
def get_default_ax(
    width: int = config.width,
    height: int = config.height,
    rows: int = 1,
    cols: int = 1,
    flatten: bool = False,
    ws: int | None = None,
    hs: int | None = None,
    title: str | None = None,
):
    """기본 크기의 Figure와 Axes를 생성한다.

    Args:
        width (int): 가로 픽셀 크기.
        height (int): 세로 픽셀 크기.
        rows (int): 서브플롯 행 개수.
        cols (int): 서브플롯 열 개수.
        flatten (bool): Axes 배열을 1차원 리스트로 평탄화할지 여부.
        ws (int|None): 서브플롯 가로 간격(`wspace`). rows/cols가 1보다 클 때만 적용.
        hs (int|None): 서브플롯 세로 간격(`hspace`). rows/cols가 1보다 클 때만 적용.
        title (str|None): Figure 제목.

    Returns:
        tuple[Figure, Axes]: 생성된 matplotlib Figure와 Axes 객체.
    """
    figsize = (width * cols / 100, height * rows / 100)
    #print(f"📐 Figure 크기: {figsize[0]:.2f} x {figsize[1]:.2f} 인치 (DPI: {dpi})")
    fig, ax = plt.subplots(rows, cols, figsize=figsize, dpi=config.dpi)

    # ax가 배열 (subplots)인지 단일 Axes인지 확인
    is_array = isinstance(ax, (np.ndarray, list))

    if is_array and (ws != None and hs != None):
        fig.subplots_adjust(wspace=ws, hspace=hs)

    if title and is_array:
        fig.suptitle(title, fontsize=config.font_size * 1.5, fontweight="bold", pad=15)

    if flatten == True:
        # 단일 Axes인 경우 리스트로 변환
        if rows == 1 and cols == 1:
            ax = [ax]
        else:
            ax = ax.flatten()

    # 테두리 굵기 설정
    if flatten and isinstance(ax, list):
        for a in ax:
            for spine in a.spines.values(): # type: ignore
                spine.set_linewidth(config.frame_width)
    elif isinstance(ax, np.ndarray):
        for a in ax.flat:
            for spine in a.spines.values(): # type: ignore
                spine.set_linewidth(config.frame_width)
    else:
        for spine in ax.spines.values():  # type: ignore
            spine.set_linewidth(config.frame_width)

    return fig, ax


# ===================================================================
# 기본 크기가 설정된 Figure와 Axes를 생성한다
# ===================================================================
def create_figure(
    width: int = config.width,
    height: int = config.height,
    rows: int = 1,
    cols: int = 1,
    flatten: bool = False,
    ws: int | None = None,
    hs: int | None = None,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """기본 크기의 Figure와 Axes를 생성한다. get_default_ax의 래퍼 함수.

    Args:
        width (int): 가로 픽셀 크기.
        height (int): 세로 픽셀 크기.
        rows (int): 서브플롯 행 개수.
        cols (int): 서브플롯 열 개수.
        flatten (bool): Axes 배열을 1차원 리스트로 평탄화할지 여부.
        ws (int|None): 서브플롯 가로 간격(`wspace`). rows/cols가 1보다 클 때만 적용.
        hs (int|None): 서브플롯 세로 간격(`hspace`). rows/cols가 1보다 클 때만 적용.
        title (str|None): Figure 제목.

    Returns:
        tuple[Figure, Axes]: 생성된 matplotlib Figure와 Axes 객체.
    """
    fig, ax = get_default_ax(width, height, rows, cols, flatten, ws, hs, title)
    return fig, ax  # type: ignore


# ===================================================================
# 그래프의 그리드, 레이아웃을 정리하고 필요 시 저장 또는 표시한다
# ===================================================================
def finalize_plot(
    ax: Axes | np.ndarray | list,
    callback: Callable | None = None,
    outparams: bool = False,
    save_path: str | None = None,
    grid: bool = True,
    title: str | None = None,
) -> None:
    """공통 후처리를 수행한다: 콜백 실행, 레이아웃 정리, 필요 시 표시/종료.

    Args:
        ax (Axes|np.ndarray): 대상 Axes (단일 Axes 또는 subplots 배열).
        callback (Callable|None): 추가 설정을 위한 사용자 콜백.
        outparams (bool): 내부에서 생성한 Figure인 경우 True.
        save_path (str|None): 이미지 저장 경로. None이 아니면 해당 경로로 저장.
        grid (bool): 그리드 표시 여부. 기본값은 True입니다.
        title (str|None): 그래프 제목.
    Returns:
        None
    """
    # ax가 배열 (subplots)인지 단일 Axes인지 확인
    is_array = isinstance(ax, (np.ndarray, list))

    # callback 실행
    if callback:
        if is_array:
            for a in ax.flat if isinstance(ax, np.ndarray) else ax:
                callback(a)
        else:
            callback(ax)

    # grid 설정
    if grid:
        if is_array:
            for a in ax.flat if isinstance(ax, np.ndarray) else ax:
                a.grid(True, alpha=config.grid_alpha, linewidth=config.grid_width)
        else:
            ax.grid(True, alpha=config.grid_alpha, linewidth=config.grid_width)

    plt.tight_layout()

    if title and not is_array:
        ax.set_title(title, fontsize=config.title_font_size, pad=config.title_pad)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    if outparams:
        plt.show()
        plt.close()


# ===================================================================
# 그래프의 그리드, 레이아웃을 정리하고 필요 시 저장 또는 표시한다
# ===================================================================
def show_figure(
    ax: Axes | np.ndarray,
    callback: Callable | None = None,
    outparams: bool = False,
    save_path: str | None = None,
    grid: bool = True,
    title: str | None = None,
) -> None:
    """공통 후처리를 수행한다: 콜백 실행, 레이아웃 정리, 필요 시 표시/종료.
    finalize_plot의 래퍼 함수.

    Args:
        ax (Axes|np.ndarray): 대상 Axes (단일 Axes 또는 subplots 배열).
        callback (Callable|None): 추가 설정을 위한 사용자 콜백.
        outparams (bool): 내부에서 생성한 Figure인 경우 True.
        save_path (str|None): 이미지 저장 경로. None이 아니면 해당 경로로 저장.
        grid (bool): 그리드 표시 여부. 기본값은 True입니다.
        title (str|None): 그래프 제목.

    Returns:
        None
    """
    finalize_plot(ax, callback, outparams, save_path, grid, title)


# ===================================================================
# 선 그래프를 그린다
# ===================================================================
def lineplot(
    df: DataFrame | None = None,
    xname: str | Series | np.ndarray | list | None = None,
    yname: str | Series | np.ndarray | list | None = None,
    hue: str | None = None,
    title: str | None = None,
    marker: str | None = None,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """선 그래프를 그린다.

    Args:
        df (DataFrame | None): 시각화할 데이터.
        xname (str | Series | np.ndarray | list | None): x축 컬럼명 혹은 x축 값 시퀀스.
        yname (str | Series | np.ndarray | list | None): y축 컬럼명 혹은 y축 값 시퀀스.
        hue (str | None): 범주 구분 컬럼명.
        title (str | None): 그래프 제목.
        marker (str | None): 마커 모양.
        palette (str | None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        save_path (str|None): 이미지 저장 경로. None이면 화면에 표시.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn lineplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
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

    sb.lineplot(**lineplot_kwargs, linewidth=linewidth)
    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 상자그림(boxplot)을 그린다
# ===================================================================
def boxplot(
    df: DataFrame | None = None,
    xname: str | None = None,
    yname: str | None = None,
    title: str | None = None,
    orient: str = "v",
    stat_test: str | None = None,
    stat_pairs: list[tuple] | None = None,
    stat_text_format: str = "star",
    stat_loc: str = "inside",
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """상자그림(boxplot)을 그린다.

    Args:
        df (DataFrame|None): 시각화할 데이터.
        xname (str|None): x축 범주 컬럼명.
        yname (str|None): y축 값 컬럼명.
        title (str|None): 그래프 제목.
        orient (str): 'v' 또는 'h' 방향.
        stat_test (str|None): 통계 검정 방법. None이면 검정 안함. xname과 yname이 모두 지정되어야 함.
        stat_pairs (list[tuple]|None): 통계 검정할 그룹 쌍 목록.
        stat_text_format (str): 통계 결과 표시 형식.
        stat_loc (str): 통계 결과 위치.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        save_path (str|None): 이미지 저장 경로. None이면 화면에 표시.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn boxplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    if xname is not None or yname is not None:
        if xname is not None and yname is None:
            orient = "h"
        elif xname is None and yname is not None:
            orient = "v"


        boxplot_kwargs = {
            "data": df,
            "x": xname,
            "y": yname,
            "orient": orient,
            "ax": ax,
            "linewidth": linewidth,
        }

        # hue 파라미터 확인 (params에 있을 수 있음)
        hue_value = params.get("hue", None)

        if hue_value is not None and palette is not None:
            boxplot_kwargs["palette"] = palette
        elif hue_value is None and palette is not None:
            boxplot_kwargs["color"] = sb.color_palette(palette)[0]

        boxplot_kwargs.update(params)
        sb.boxplot(**boxplot_kwargs)

        # 통계 검정 추가
        if stat_test is not None:
            if stat_pairs is None:
                stat_pairs = [df[xname].dropna().unique().tolist()] # type: ignore

            annotator = Annotator(
                ax, data=df, x=xname, y=yname, pairs=stat_pairs, orient=orient
            )
            annotator.configure(
                test=stat_test, text_format=stat_text_format, loc=stat_loc
            )
            annotator.apply_and_annotate()
    else:
        sb.boxplot(data=df, orient=orient, ax=ax, linewidth=linewidth, **params)  # type: ignore

    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 상자그림에 p-value 주석을 추가한다
# ===================================================================
def pvalue1_anotation(
    data: DataFrame,
    target: str,
    hue: str,
    title: str | None = None,
    pairs: list | None = None,
    test: str = "t-test_ind",
    text_format: str = "star",
    loc: str = "outside",
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """
    boxplot의 wrapper 함수로, 상자그림에 p-value 주석을 추가한다.
    """
    boxplot(
        data,
        xname=hue,
        yname=target,
        title=title,
        orient="v",
        stat_test=test,
        stat_pairs=pairs,
        stat_text_format=text_format,
        stat_loc=loc,
        palette=None,
        width=width,
        height=height,
        linewidth=linewidth,
        save_path=save_path,
        callback=callback,
        ax=ax,
        **params,
    )


# ===================================================================
# 커널 밀도 추정(KDE) 그래프를 그린다
# ===================================================================
def kdeplot(
    df: DataFrame,
    xname: str | None = None,
    yname: str | None = None,
    hue: str | None = None,
    title: str | None = None,
    palette: str | None = None,
    fill: bool = False,
    fill_alpha: float = config.fill_alpha,
    linewidth: float = config.line_width,
    quartile_split: bool = False,
    width: int = config.width,
    height: int = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """커널 밀도 추정(KDE) 그래프를 그린다.

    quartile_split=True일 때는 1차원 KDE(xname 지정, yname 없음)를
    사분위수 구간(Q1~Q4)으로 나누어 4개의 서브플롯에 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str|None): x축 컬럼명.
        yname (str|None): y축 컬럼명.
        hue (str|None): 범주 컬럼명.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        fill (bool): 면적 채우기 여부.
        fill_alpha (float): 채움 투명도.
        quartile_split (bool): True면 1D KDE를 사분위수별 서브플롯으로 분할.
        linewidth (float): 선 굵기.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn kdeplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    # 사분위수 분할 전용 처리 (1D KDE만 지원)
    if quartile_split:
        if yname is not None:
            raise ValueError(
                "quartile_split은 1차원 KDE(xname)에서만 사용할 수 있습니다."
            )

        series = df[xname].dropna()
        if series.empty:
            return

        q = series.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values
        bounds = list(zip(q[:-1], q[1:]))  # [(Q0,Q1),(Q1,Q2),(Q2,Q3),(Q3,Q4)]

        fig, axes = get_default_ax(width, height, len(bounds), 1, flatten=True)
        outparams = True

        for idx, (lo, hi) in enumerate(bounds):
            subset = series[(series >= lo) & (series <= hi)]
            if subset.empty:
                continue

            # hue를 지원하려면 원본 데이터에서 해당 인덱스로 슬라이싱
            cols = [xname]
            if hue is not None and hue in df.columns:
                cols.append(hue)
            df_quartile = df.loc[subset.index, cols].copy()

            kdeplot_kwargs = {
                "data": df_quartile,
                "x": xname,
                "fill": fill,
                "ax": axes[idx],
            }

            if hue is not None and hue in df_quartile.columns:
                kdeplot_kwargs["hue"] = hue
            if fill:
                kdeplot_kwargs["alpha"] = fill_alpha
            if hue is not None and palette is not None:
                kdeplot_kwargs["palette"] = palette
            kdeplot_kwargs["linewidth"] = linewidth
            kdeplot_kwargs.update(params)

            sb.kdeplot(**kdeplot_kwargs)
            axes[idx].set_title(f"Q{idx+1}: [{lo:.3g}, {hi:.3g}]", fontsize=config.title_font_size, pad=config.title_pad) # type: ignore
            axes[idx].grid(True, alpha=config.grid_alpha, linewidth=config.grid_width) # type: ignore

        finalize_plot(axes[0], callback, outparams, save_path, True, title)
        return

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
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

    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 히스토그램을 그린다
# ===================================================================
def histplot(
    df: DataFrame,
    xname: str,
    hue: str | None = None,
    title: str | None = None,
    bins: int | None = None,
    kde: bool = True,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """히스토그램을 그리고 필요 시 KDE를 함께 표시한다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 히스토그램 대상 컬럼명.
        hue (str|None): 범주 컬럼명.
        title (str|None): 그래프 제목.
        bins (int|sequence|None): 구간 수 또는 경계.
        kde (bool): KDE 표시 여부.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn histplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    if bins:
        histplot_kwargs = {
            "data": df,
            "x": xname,
            "hue": hue,
            "kde": kde,
            "bins": bins,
            "linewidth": linewidth,
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
            "linewidth": linewidth,
            "ax": ax,
        }

        if hue is not None and palette is not None:
            histplot_kwargs["palette"] = palette
        elif hue is None and palette is not None:
            histplot_kwargs["color"] = sb.color_palette(palette)[0]

        histplot_kwargs.update(params)
        sb.histplot(**histplot_kwargs)

    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 범주별 비율을 100% 누적 막대그래프로 나타낸다
# ===================================================================
def stackplot(
    df: DataFrame,
    xname: str,
    hue: str,
    title: str | None = None,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = 0.25,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """클래스 비율을 100% 누적 막대로 표현한다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): x축 기준 컬럼.
        hue (str): 클래스 컬럼.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn histplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    df2 = df[[xname, hue]].copy()
    df2[xname] = df2[xname].astype(str)

    # stackplot은 hue가 필수이므로 palette를 그대로 사용
    stackplot_kwargs = {
        "data": df2,
        "x": xname,
        "hue": hue,
        "linewidth": linewidth,
        "stat": "probability",  # 전체에서의 비율로 그리기
        "multiple": "fill",  # 전체를 100%로 그리기
        "shrink": 0.8,  # 막대의 폭
        "linewidth": linewidth,
        "ax": ax,
    }

    if palette is not None:
        stackplot_kwargs["palette"] = palette

    stackplot_kwargs.update(params)

    sb.histplot(**stackplot_kwargs)

    # 그래프의 x축 항목 수 만큼 반복
    for p in ax.patches:  # type: ignore
        # 각 막대의 위치, 넓이, 높이
        left, bottom, width, height = p.get_bbox().bounds  # type: ignore
        # 막대의 중앙에 글자 표시하기
        ax.annotate(  # type: ignore
            "%0.1f%%" % (height * 100),
            xy=(left + width / 2, bottom + height / 2),
            ha="center",
            va="center",
        )

    if str(df[xname].dtype) in ["int", "int32", "int64", "float", "float32", "float64"]:
        xticks = list(df[xname].unique())
        ax.set_xticks(xticks)  # type: ignore
        ax.set_xticklabels(xticks)  # type: ignore

    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 산점도를 그린다
# ===================================================================
def scatterplot(
    df: DataFrame | None,
    xname: str | Index,
    yname: str | Index,
    hue=None,
    vector: str | None = None,
    outline: bool = False,
    title: str | None = None,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """산점도를 그린다.

    Args:
        df (DataFrame | None): 시각화할 데이터.
        xname (str | Index): x축 컬럼.
        yname (str | Index): y축 컬럼.
        hue (str|None): 범주 컬럼.
        vector (str|None): 벡터 종류 컬럼.
        outline (bool): 점 외곽선 표시 여부.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn scatterplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    if outline and hue is not None:
        # 군집별 값의 종류별로 반복 수행
        for c in df[hue].unique():  # type: ignore
            if c == -1:
                continue

            # 한 종류만 필터링한 결과에서 두 변수만 선택
            df_c = df.loc[df[hue] == c, [xname, yname]] # type: ignore

            try:
                # 외각선 좌표 계산
                hull = ConvexHull(df_c)

                # 마지막 좌표 이후에 첫 번째 좌표를 연결
                points = np.append(hull.vertices, hull.vertices[0])

                ax.plot(  # type: ignore
                    df_c.iloc[points, 0],
                    df_c.iloc[points, 1],
                    linewidth=linewidth,
                    linestyle=":",
                )
                ax.fill(df_c.iloc[points, 0], df_c.iloc[points, 1], alpha=0.1)  # type: ignore
            except:
                pass

    # hue가 있을 때만 palette 사용, 없으면 color 사용
    scatterplot_kwargs = {
        "x": xname,
        "y": yname,
        "hue": hue,
        "linewidth": linewidth,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        scatterplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        scatterplot_kwargs["color"] = sb.color_palette(palette)[0]

    scatterplot_kwargs.update(params)

    # 백터 종류 구분 필드가 전달되지 않은 경우에는 원본 데이터를 그대로 사용
    if vector is None:
        sb.scatterplot(data=df, **scatterplot_kwargs)
    else:
        # 핵심벡터
        scatterplot_kwargs["edgecolor"] = "#ffffff"
        sb.scatterplot(data=df[df[vector] == "core"], **scatterplot_kwargs) # type: ignore

        # 외곽백터
        scatterplot_kwargs["edgecolor"] = "#000000"
        scatterplot_kwargs["s"] = 25
        scatterplot_kwargs["marker"] = "^"
        scatterplot_kwargs["linewidth"] = 0.8
        sb.scatterplot(data=df[df[vector] == "border"], **scatterplot_kwargs) # type: ignore
 
        # 노이즈벡터
        scatterplot_kwargs["edgecolor"] = None
        scatterplot_kwargs["s"] = 25
        scatterplot_kwargs["marker"] = "x"
        scatterplot_kwargs["linewidth"] = 2
        scatterplot_kwargs["color"] = "#ff0000"
        scatterplot_kwargs["hue"] = None
        sb.scatterplot(data=df[df[vector] == "noise"], **scatterplot_kwargs)    # type: ignore

    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 회귀선이 포함된 산점도를 그린다
# ===================================================================
def regplot(
    df: DataFrame,
    xname: str,
    yname: str,
    title: str | None = None,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """단순 회귀선이 포함된 산점도를 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 독립변수 컬럼.
        yname (str): 종속변수 컬럼.
        title (str|None): 그래프 제목.
        palette (str|None): 선/점 색상.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn regplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    # regplot은 hue를 지원하지 않으므로 palette를 color로 변환
    scatter_color = None
    if palette is not None:
        scatter_color = sb.color_palette(palette)[0]

    regplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "scatter_kws": {
            "s": 20,
            "linewidths": 0.5,
            "edgecolor": "w",
            "color": scatter_color,
        },
        "line_kws": {"color": "red", "linestyle": "--", "linewidth": linewidth},
        "ax": ax,
    }

    regplot_kwargs.update(params)

    sb.regplot(**regplot_kwargs)

    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 범주별 회귀선이 표시된 선형 모델 그래프를 그린다
# ===================================================================
def lmplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    title: str | None = None,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    **params,
) -> None:
    """seaborn lmplot으로 선형 모델 시각화를 수행한다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 독립변수 컬럼.
        yname (str): 종속변수 컬럼.
        hue (str|None): 범주 컬럼.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
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
    g.fig.set_size_inches(width / config.dpi, height / config.dpi)
    g.fig.set_dpi(config.dpi)

    # 회귀선에 linewidth 적용
    for ax in g.axes.flat:
        for line in ax.get_lines():
            if line.get_marker() == "o":  # 산점도는 건너뛰기
                continue
            line.set_linewidth(linewidth)

    g.fig.grid(True, alpha=config.grid_alpha, linewidth=config.grid_width)  # type: ignore

    if title:
        g.fig.suptitle(title, fontsize=config.font_size * 1.5, fontweight="bold")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    plt.close()


# ===================================================================
# 연속형 변수들의 차속 관계 그래프 매트릭스를 그린다
# ===================================================================
def pairplot(
    df: DataFrame,
    xnames=None,
    title: str | None = None,
    diag_kind: str = "kde",
    hue=None,
    palette: str | None = None,
    width: int = config.height,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    **params,
) -> None:
    """연속형 변수의 숫자형 컬럼 쌍에 대한 관계를 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xnames (str|list|None): 대상 컬럼명.
            - None: 모든 연속형(숫자형) 데이터에 대해 처리.
            - str: 해당 컬럼에 대해서만 처리.
            - list: 주어진 컬럼들에 대해서만 처리.
            기본값은 None.
        title (str|None): 그래프 제목.
        diag_kind (str): 대각선 플롯 종류('kde' 등).
        hue (str|None): 범주 컬럼.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        dpi (int): 기본 크기 및 해상도(컬럼 수에 비례해 확대됨).
        **params: seaborn pairplot 추가 인자.

    Returns:
        None
    """
    # xnames 파라미터 처리 (연속형 변수만, 명목형 제외)
    if xnames is None:
        # 모든 연속형(숫자형) 컬럼 선택 (명목형/카테고리 제외)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_cols = [col for col in numeric_cols if df[col].dtype.name != "category"]
    elif isinstance(xnames, str):
        # 문자열: 해당 컬럼만
        target_cols = [xnames]
    elif isinstance(xnames, list):
        # 리스트: 주어진 컬럼들
        target_cols = xnames
    else:
        # 기본값으로 연속형 컬럼
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        target_cols = [col for col in numeric_cols if df[col].dtype.name != "category"]

    # hue 컬럼이 있으면 target_cols에 포함시키기 (pairplot 자체에서 필요)
    if hue is not None and hue not in target_cols:
        target_cols = target_cols + [hue]

    # target_cols를 포함하는 부분 데이터프레임 생성
    df_filtered = df[target_cols].copy()

    # hue가 있을 때만 palette 사용
    pairplot_kwargs = {
        "data": df_filtered,
        "hue": hue,
        "diag_kind": diag_kind,
    }

    if hue is not None and palette is not None:
        pairplot_kwargs["palette"] = palette
    # pairplot은 hue 없이 palette만 쓰는 경우가 드물어서 color로 변환 불필요

    pairplot_kwargs.update(params)

    g = sb.pairplot(**pairplot_kwargs)
    scale = len(target_cols)
    g.fig.set_size_inches(w=(width / config.dpi) * scale, h=(height / config.dpi) * scale)
    g.fig.set_dpi(config.dpi)

    if title:
        g.fig.suptitle(title, fontsize=config.font_size * 1.5, fontweight="bold")

    g.map_lower(
        func=sb.kdeplot, fill=True, alpha=config.fill_alpha
    )
    g.map_upper(func=sb.scatterplot)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    plt.close()


# ===================================================================
# 범주 빠른도 막대그래프를 그린다
# ===================================================================
def countplot(
    df: DataFrame,
    xname: str,
    hue=None,
    title: str | None = None,
    palette: str | None = None,
    order: int = 1,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """범주 빈도 막대그래프를 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 범주 컬럼.
        hue (str|None): 보조 범주 컬럼.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        order (int): 숫자형일 때 정렬 방식(1: 값 기준, 기타: 빈도 기준).
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
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
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    # hue가 있을 때만 palette 사용, 없으면 color 사용
    countplot_kwargs = {
        "data": df,
        "x": xname,
        "hue": hue,
        "order": sort,
        "linewidth": linewidth,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        countplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        # palette의 첫 번째 색상을 color로 사용
        countplot_kwargs["color"] = sb.color_palette(palette)[0]

    countplot_kwargs.update(params)

    sb.countplot(**countplot_kwargs)

    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 막대그래프를 그린다
# ===================================================================
def barplot(
    df: DataFrame,
    xname: str | Index,
    yname: str | Index,
    hue=None,
    title: str | None = None,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """막대그래프를 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str | Index): 범주 컬럼.
        yname (str | Index): 값 컬럼.
        hue (str|None): 보조 범주 컬럼.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn barplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    # hue가 있을 때만 palette 사용, 없으면 color 사용
    barplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "hue": hue,
        "linewidth": linewidth,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        barplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        barplot_kwargs["color"] = sb.color_palette(palette)[0]

    barplot_kwargs.update(params)

    sb.barplot(**barplot_kwargs)
    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# boxen 플롯을 그린다
# ===================================================================
def boxenplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    title: str | None = None,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """박스앤 위스커 확장(boxen) 플롯을 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 범주 컬럼.
        yname (str): 값 컬럼.
        hue (str|None): 보조 범주 컬럼.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn boxenplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    # palette은 hue가 있을 때만 사용
    boxenplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "hue": hue,
        "linewidth": linewidth,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        boxenplot_kwargs["palette"] = palette

    boxenplot_kwargs.update(params)

    sb.boxenplot(**boxenplot_kwargs)
    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 바이올린 플롯을 그린다
# ===================================================================
def violinplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    title: str | None = None,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """바이올린 플롯을 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 범주 컬럼.
        yname (str): 값 컬럼.
        hue (str|None): 보조 범주 컬럼.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn violinplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    # palette은 hue가 있을 때만 사용
    violinplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "hue": hue,
        "linewidth": linewidth,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        violinplot_kwargs["palette"] = palette

    violinplot_kwargs.update(params)
    sb.violinplot(**violinplot_kwargs)
    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 포인트 플롯을 그린다
# ===================================================================
def pointplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    title: str | None = None,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """포인트 플롯을 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): 범주 컬럼.
        yname (str): 값 컬럼.
        hue (str|None): 보조 범주 컬럼.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn pointplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    # hue가 있을 때만 palette 사용, 없으면 color 사용
    pointplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "hue": hue,
        "linewidth": linewidth,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        pointplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        pointplot_kwargs["color"] = sb.color_palette(palette)[0]

    pointplot_kwargs.update(params)
    sb.pointplot(**pointplot_kwargs)
    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 공동 분포(joint) 플롯을 그린다
# ===================================================================
def jointplot(
    df: DataFrame,
    xname: str,
    yname: str,
    hue=None,
    title: str | None = None,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    **params,
) -> None:
    """공동 분포(joint) 플롯을 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        xname (str): x축 컬럼.
        yname (str): y축 컬럼.
        hue (str|None): 범주 컬럼.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        **params: seaborn jointplot 추가 인자.

    Returns:
        None
    """
    # hue가 있을 때만 palette 사용
    jointplot_kwargs = {
        "data": df,
        "x": xname,
        "y": yname,
        "linewidth": linewidth,
        "hue": hue,
    }

    if hue is not None and palette is not None:
        jointplot_kwargs["palette"] = palette
    # jointplot은 hue 없이 palette만 쓰는 경우가 드물어서 color로 변환 불필요

    jointplot_kwargs.update(params)

    g = sb.jointplot(**jointplot_kwargs)
    g.fig.set_size_inches(width / config.dpi, height / config.dpi)
    g.fig.set_dpi(config.dpi)

    if title:
        g.fig.suptitle(title, fontsize=config.font_size * 1.5, fontweight="bold")

    # 중앙 및 주변 플롯에 grid 추가
    g.ax_joint.grid(True, alpha=config.grid_alpha, linewidth=config.grid_width)
    g.ax_marg_x.grid(True, alpha=config.grid_alpha, linewidth=config.grid_width)
    g.ax_marg_y.grid(True, alpha=config.grid_alpha, linewidth=config.grid_width)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    plt.close()


# ===================================================================
# 히트린띄 그린다
# ===================================================================
def heatmap(
    data: DataFrame,
    title: str | None = None,
    palette: str | None = None,
    width: int | None = None,
    height: int | None = None,
    linewidth: float = 0.25,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """히트맵을 그린다(값 주석 포함).

    Args:
        data (DataFrame): 행렬 형태 데이터.
        title (str|None): 그래프 제목.
        palette (str|None): 컬러맵 이름.
        width (int|None): 캔버스 가로 픽셀. None이면 자동 계산.
        height (int|None): 캔버스 세로 픽셀. None이면 자동 계산.
        linewidth (float): 격자 선 굵기.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn heatmap 추가 인자.

    Returns:
        None
    """
    outparams = False

    if width == None or height == None:
        width = (config.font_size * config.dpi / 72) * 4.5 * len(data.columns)
        height = width * 0.8  # type: ignore

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    heatmatp_kwargs = {
        "data": data,
        "annot": True,
        "cmap": palette,
        "fmt": ".2f",
        "ax": ax,
        "linewidths": linewidth,
        "annot_kws": {"size": 10},
    }

    heatmatp_kwargs.update(params)

    # heatmap은 hue를 지원하지 않으므로 cmap에 palette 사용
    sb.heatmap(**heatmatp_kwargs)

    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# KDE와 신뢰구간을 나타낸 그래프를 그린다
# ===================================================================
def kde_confidence_interval(
    data: DataFrame,
    xnames=None,
    title: str | None = None,
    clevel=0.95,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    fill: bool = False,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
) -> None:
    """각 숫자 컬럼에 대해 KDE와 t-분포 기반 신뢰구간을 그린다.

    Args:
        data (DataFrame): 시각화할 데이터.
        xnames (str|list|None): 대상 컬럼명.
            - None: 모든 연속형 데이터에 대해 처리.
            - str: 해당 컬럼에 대해서만 처리.
            - list: 주어진 컬럼들에 대해서만 처리.
            기본값은 None.
        title (str|None): 그래프 제목.
        clevel (float): 신뢰수준(0~1).
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        fill (bool): KDE 채우기 여부.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.

    Returns:
        None
    """
    outparams = False

    # xnames 파라미터 처리
    if xnames is None:
        # 모든 연속형(숫자형) 컬럼 선택
        target_cols = list(data.select_dtypes(include=[np.number]).columns)
    elif isinstance(xnames, str):
        # 문자열: 해당 컬럼만
        target_cols = [xnames]
    elif isinstance(xnames, list):
        # 리스트: 주어진 컬럼들
        target_cols = xnames
    else:
        # 기본값으로 전체 컬럼
        target_cols = list(data.columns)

    # 외부에서 ax를 전달하지 않은 경우 서브플롯 생성
    if ax is None:
        n_cols = len(target_cols)
        fig, axes = get_default_ax(width, height, n_cols, 1, flatten=True)
        outparams = True
    else:
        # 외부에서 ax를 전달한 경우 (시뮬레이션용)
        axes = [ax]
        outparams = False

    # 데이터 프레임의 컬럼별로 개별 서브플롯에 처리
    for idx, c in enumerate(target_cols):
        if idx >= len(axes):
            break

        current_ax = axes[idx]
        column = data[c].dropna()

        if len(column) < 2:
            continue

        dof = len(column) - 1  # 자유도
        sample_mean = column.mean()  # 표본평균
        sample_std = column.std(ddof=1)  # 표본표준편차
        sample_std_error = sample_std / sqrt(len(column))  # 표본표준오차

        # 신뢰구간
        cmin, cmax = t.interval(clevel, dof, loc=sample_mean, scale=sample_std_error)

        # 현재 컬럼에 대한 커널밀도추정
        sb.kdeplot(data=column, linewidth=linewidth, ax=current_ax, fill=fill, alpha=config.fill_alpha)  # type: ignore

        # 그래프 축의 범위
        xmin, xmax, ymin, ymax = current_ax.get_position().bounds  # type: ignore
        ymin_val, ymax_val = 0, current_ax.get_ylim()[1]    # type: ignore

        # 신뢰구간 그리기
        current_ax.plot(    # type: ignore
            [cmin, cmin], [ymin_val, ymax_val], linestyle=":", linewidth=linewidth * 0.5
        )
        current_ax.plot(    # type: ignore
            [cmax, cmax], [ymin_val, ymax_val], linestyle=":", linewidth=linewidth * 0.5
        )
        current_ax.fill_between(    # type: ignore
            [cmin, cmax], y1=ymin_val, y2=ymax_val, alpha=config.fill_alpha
        )

        # 평균 그리기
        current_ax.plot(    # type: ignore
            [sample_mean, sample_mean],
            [0, ymax_val],
            linestyle="--",
            linewidth=linewidth,
        )

        current_ax.text(    # type: ignore
            x=(cmax - cmin) / 2 + cmin,
            y=ymax_val,
            s="[%s] %0.1f ~ %0.1f" % (column.name, cmin, cmax),
            horizontalalignment="center",
            verticalalignment="bottom",
            fontdict={"color": "red"},
        )

        current_ax.grid(True, alpha=config.grid_alpha, linewidth=config.grid_width) # type: ignore

    finalize_plot(axes[0] if isinstance(axes, list) and len(axes) > 0 else ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 잔차도 (선형회귀의 선형성 검정)
# ===================================================================
def ols_residplot(
    fit,
    title: str | None = None,
    lowess: bool = False,
    mse: bool = False,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """잔차도를 그린다(선택적으로 MSE 범위와 LOWESS 포함).

    회귀모형의 선형성을 시각적으로 평가하기 위한 그래프를 생성한다.
    점들이 무작위로 흩어져 있으면 선형성 가정이 만족되며,
    특정 패턴이 보이면 비선형 관계가 존재할 가능성을 시사한다.

    Args:
        fit: 회귀 모형 객체 (statsmodels의 RegressionResultsWrapper).
             fit.resid와 fit.fittedvalues를 통해 잔차와 적합값을 추출한다.
        title (str|None): 그래프 제목.
        lowess (bool): LOWESS 스무딩 적용 여부.
        mse (bool): √MSE, 2√MSE, 3√MSE 대역선과 비율 표시 여부.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        save_path (str|None): 저장 경로.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn residplot 추가 인자.

    Returns:
        None

    Examples:
        ```python
        from hossam import *
        fit = hs_stats.ols(data, yname='target', report=False)
        residplot(fit, lowess=True, mse=True)
        ```
    """
    outparams = False

    # fit 객체에서 잔차와 적합값 추출
    resid = fit.resid
    y_pred = fit.fittedvalues
    y = y_pred + resid  # 실제값 = 적합값 + 잔차

    if ax is None:
        fig, ax = get_default_ax(width + 150 if mse else width, height, 1, 1)  # type: ignore
        outparams = True

    sb.residplot(
        x=y_pred,
        y=resid,
        lowess=True,  # 잔차의 추세선 표시
        line_kws={"color": "red", "linewidth": linewidth * 0.7},  # 추세선 스타일
        scatter_kws={"edgecolor": "white", "alpha": config.alpha},
        **params
    )

    if mse:
        mse_val = mean_squared_error(y, y_pred)
        mse_sq = np.sqrt(mse_val)

        r1 = resid[(resid > -mse_sq) & (resid < mse_sq)].size / resid.size * 100
        r2 = resid[(resid > -2 * mse_sq) & (resid < 2 * mse_sq)].size / resid.size * 100
        r3 = resid[(resid > -3 * mse_sq) & (resid < 3 * mse_sq)].size / resid.size * 100

        mse_r = [r1, r2, r3]

        xmin, xmax = ax.get_xlim()  # type: ignore

        # 구간별 반투명 색상 채우기 (안쪽부터 바깥쪽으로, 진한 색에서 연한 색으로)
        colors = ["red", "green", "blue"]
        alphas = [0.15, 0.10, 0.05]  # 안쪽이 더 진하게

        # 3σ 영역 (가장 바깥쪽, 가장 연함)
        ax.axhspan(-3 * mse_sq, 3 * mse_sq, facecolor=colors[2], alpha=alphas[2], zorder=0)  # type: ignore
        # 2σ 영역 (중간)
        ax.axhspan(-2 * mse_sq, 2 * mse_sq, facecolor=colors[1], alpha=alphas[1], zorder=1)  # type: ignore
        # 1σ 영역 (가장 안쪽, 가장 진함)
        ax.axhspan(-mse_sq, mse_sq, facecolor=colors[0], alpha=alphas[0], zorder=2)  # type: ignore

        # 경계선 그리기
        for i, c in enumerate(["red", "green", "blue"]):
            ax.axhline(mse_sq * (i + 1), color=c, linestyle="--", linewidth=linewidth / 2)  # type: ignore
            ax.axhline(mse_sq * (-(i + 1)), color=c, linestyle="--", linewidth=linewidth / 2)  # type: ignore

        target = [68, 95, 99.7]
        for i, c in enumerate(["red", "green", "blue"]):
            ax.text(  # type: ignore
                s=f"{i+1} sqrt(MSE) = {mse_r[i]:.2f}% ({mse_r[i] - target[i]:.2f}%)",
                x=xmax + 0.05,
                y=(i + 1) * mse_sq,
                color=c,
            )
            ax.text(  # type: ignore
                s=f"-{i+1} sqrt(MSE) = {mse_r[i]:.2f}% ({mse_r[i] - target[i]:.2f}%)",
                x=xmax + 0.05,
                y=-(i + 1) * mse_sq,
                color=c,
            )

    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# Q-Q Plot (선형회귀의 정규성 검정)
# ===================================================================
def ols_qqplot(
    fit,
    title: str | None = None,
    line: str = "s",
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """표준화된 잔차의 정규성 확인을 위한 QQ 플롯을 그린다.

    statsmodels의 qqplot 함수를 사용하여 최적화된 Q-Q plot을 생성한다.
    이론적 분위수와 표본 분위수를 비교하여 잔차의 정규성을 시각적으로 평가한다.

    Args:
        fit: 회귀 모형 객체 (statsmodels의 RegressionResultsWrapper 등).
             fit.resid 속성을 통해 잔차를 추출하여 정규성을 확인한다.
        title (str|None): 그래프 제목.
        line (str): 참조선의 유형. 기본값 's' (standardized).
                    - 's': 표본의 표준편차와 평균을 기반으로 조정된 선 (권장)
                    - 'r': 실제 점들에 대한 회귀선 (데이터 추세 반영)
                    - 'q': 1사분위수와 3사분위수를 통과하는 선
                    - '45': 45도 대각선 (이론적 정규분포)
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        save_path (str|None): 저장 경로.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: statsmodels qqplot 추가 인자.

    Returns:
        None

    Examples:
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
    """
    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    # fit 객체에서 잔차(residuals) 추출
    residuals = fit.resid

    # markersize 기본값 설정 (기존 크기의 2/3)
    if "markersize" not in params:
        params["markersize"] = 2

    # statsmodels의 qqplot 사용 (더 전문적이고 최적화된 구현)
    # line 옵션으로 다양한 참조선 지원
    sm_qqplot(residuals, line=line, ax=ax, **params)

    # 점의 스타일 개선: 연한 내부, 진한 테두리
    for collection in ax.collections:  # type: ignore
        # PathCollection (scatter plot의 점들)
        collection.set_facecolor("#4A90E2")  # 연한 파란색 내부
        collection.set_edgecolor("#1E3A8A")  # 진한 파란색 테두리
        collection.set_linewidth(0.8)  # 테두리 굵기
        collection.set_alpha(0.7)  # 약간의 투명도

    # 선 굵기 조정
    for line in ax.get_lines():  # type: ignore
        line.set_linewidth(linewidth)  # type: ignore

    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
#
# ===================================================================
def distribution_by_class(
    data: DataFrame,
    title: str | None = None,
    xnames: list | None = None,
    hue: str | None = None,
    type: str = "kde",
    bins: list[int] | int = 5,
    palette: str | None = None,
    fill: bool = False,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
) -> None:
    """클래스별로 각 숫자형 특징의 분포를 KDE 또는 히스토그램으로 그린다.

    Args:
        data (DataFrame): 시각화할 데이터.
        xnames (list|None): 대상 컬럼 목록(None이면 전 컬럼).
        hue (str|None): 클래스 컬럼.
        title (str|None): 그래프 제목.
        type (str): 'kde' | 'hist' | 'histkde'.
        bins (int|sequence|None): 히스토그램 구간.
        palette (str|None): 팔레트 이름.
        fill (bool): KDE 채움 여부.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        callback (Callable|None): Axes 후처리 콜백.

    Returns:
        None
    """
    if xnames is None:
        xnames = data.columns  # type: ignore

    for i, v in enumerate(xnames):  # type: ignore
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
                linewidth=linewidth,
                callback=callback,
                save_path=save_path,
            )
        elif type == "hist":
            histplot(
                df=data,
                xname=v,
                hue=hue,
                bins=bins,  # type: ignore
                kde=False,
                palette=palette,
                width=width,
                height=height,
                linewidth=linewidth,
                callback=callback,
                save_path=save_path,
            )
        elif type == "histkde":
            histplot(
                df=data,
                xname=v,
                hue=hue,
                bins=bins,  # type: ignore
                kde=True,
                palette=palette,
                width=width,
                height=height,
                linewidth=linewidth,
                callback=callback,
                save_path=save_path,
            )


# ===================================================================
#
# ===================================================================
def scatter_by_class(
    data: DataFrame,
    yname: str,
    group: list | None = None,
    hue: str | None = None,
    title: str | None = None,
    palette: str | None = None,
    outline: bool = False,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
) -> None:
    """종속변수(y)와 각 연속형 독립변수(x) 간 산점도/볼록껍질을 그린다.

    Args:
        data (DataFrame): 시각화할 데이터.
        yname (str): 종속변수 컬럼명(필수).
        group (list|None): x 컬럼 목록 또는 [[x, y], ...] 형태. None이면 자동 생성.
        hue (str|None): 클래스 컬럼.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        outline (bool): 볼록 껍질을 표시할지 여부.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        callback (Callable|None): Axes 후처리 콜백.

    Returns:
        None
    """

    # 자동 생성: yname 제외, hue 제외, 연속형만
    if group is None:
        group = []

        numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
        xnames = [
            col
            for col in numeric_cols
            if col not in [yname, hue]
            and data[col].dtype.name not in ["category", "bool", "boolean"]
        ]

        for v in xnames:
            group.append([v, yname])
    else:
        # 사용자가 지정한 경우: 문자열 리스트면 yname과 페어링, 이미 페어면 그대로 사용
        processed = []
        for item in group:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                processed.append(list(item))
            else:
                processed.append([item, yname])
        group = processed

    for v in group:
        scatterplot(data=data, xname=v[0], yname=v[1], outline=outline, hue=hue, palette=palette, width=width, height=height, linewidth=linewidth, dpi=dpi, callback=callback, save_path=save_path)  # type: ignore


# ===================================================================
#
# ===================================================================
def categorical_target_distribution(
    data: DataFrame,
    yname: str,
    hue: list | str | None = None,
    title: str | None = None,
    kind: str = "box",
    kde_fill: bool = True,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    cols: int = 2,
    save_path: str | None = None,
    callback: Callable | None = None,
) -> None:
    """명목형 변수별로 종속변수 분포 차이를 시각화한다.

    Args:
        data (DataFrame): 시각화할 데이터.
        yname (str): 종속변수 컬럼명(연속형 추천).
        hue (list|str|None): 명목형 독립변수 목록. None이면 자동 탐지.
        title (str|None): 그래프 제목.
        kind (str): 'box', 'violin', 'kde'.
        kde_fill (bool): kind='kde'일 때 영역 채우기 여부.
        palette (str|None): 팔레트 이름.
        width (int): 개별 서브플롯 가로 픽셀.
        height (int): 개별 서브플롯 세로 픽셀.
        linewidth (float): 선 굵기.
        cols (int): 서브플롯 열 수.
        callback (Callable|None): Axes 후처리 콜백.

    Returns:
        None
    """

    # 명목형 컬럼 후보: object, category, bool
    if hue is None:
        cat_cols = data.select_dtypes(
            include=["object", "category", "bool", "boolean"]
        ).columns
        target_cols = [c for c in cat_cols if c != yname]
    elif isinstance(hue, str):
        target_cols = [hue]
    else:
        target_cols = list(hue)

    if len(target_cols) == 0:
        return

    n_plots = len(target_cols)
    rows = (n_plots + cols - 1) // cols

    fig, axes = get_default_ax(width, height, rows, cols, dpi, flatten=True) # type: ignore
    outparams = True

    for idx, col in enumerate(target_cols):
        if idx >= len(axes):
            break

        ax = axes[idx]
        plot_kwargs = {
            "data": data.dropna(subset=[col, yname]),
            "ax": ax,
        }

        if kind == "violin":
            plot_kwargs.update({"x": col, "y": yname, "palette": palette})
            sb.violinplot(**plot_kwargs, linewidth=linewidth)
        elif kind == "kde":
            plot_kwargs.update(
                {
                    "x": yname,
                    "hue": col,
                    "palette": palette,
                    "fill": kde_fill,
                    "common_norm": False,
                    "linewidth": linewidth,
                }
            )
            sb.kdeplot(**plot_kwargs)
        else:  # box
            plot_kwargs.update({"x": col, "y": yname, "hue": col, "palette": palette})
            sb.boxplot(**plot_kwargs, linewidth=linewidth)

        ax.set_title(f"{col} vs {yname}", fontsize=config.title_font_size, pad=config.title_pad)  # type: ignore

    # 불필요한 빈 축 숨기기
    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False) # type: ignore

    finalize_plot(axes[0], callback, outparams, save_path, True, title)


# ===================================================================
# ROC 커브를 시각화 한다.
# ===================================================================
def roc_curve_plot(
    fit,
    y: np.ndarray | Series | None = None,
    X: DataFrame | np.ndarray | None = None,
    title: str | None = None,
    width: int = config.height,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
) -> None:
    """로지스틱 회귀 적합 결과의 ROC 곡선을 시각화한다.

    Args:
        fit: statsmodels Logit 결과 객체 (`fit.predict()`로 예측 확률을 계산 가능해야 함).
        y (array-like|None): 외부 데이터의 실제 레이블. 제공 시 이를 실제값으로 사용.
        X (array-like|None): 외부 데이터의 설계행렬(독립변수). 제공 시 해당 데이터로 예측 확률 계산.
        title (str|None): 그래프 제목.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
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
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
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
    ax.plot(fpr, tpr, color="darkorange", lw=linewidth, label=f"ROC curve (AUC = {roc_auc:.4f})")  # type: ignore
    ax.plot([0, 1], [0, 1], color="navy", lw=linewidth, linestyle="--", label="Random Classifier")  # type: ignore

    ax.set_xlim([0.0, 1.0])  # type: ignore
    ax.set_ylim([0.0, 1.05])  # type: ignore
    ax.set_xlabel("위양성율 (False Positive Rate)", fontsize=config.label_font_size)  # type: ignore
    ax.set_ylabel("재현율 (True Positive Rate)", fontsize=config.label_font_size)  # type: ignore
    ax.set_title("ROC 곡선", fontsize=config.title_font_size, pad=config.title_pad)  # type: ignore
    ax.legend(loc="lower right", fontsize=config.label_font_size)  # type: ignore
    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 혼동행렬 시각화
# ===================================================================
def confusion_matrix_plot(
    fit,
    title: str | None = None,
    threshold: float = 0.5,
    width: int = config.width,
    height: int = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
) -> None:
    """로지스틱 회귀 적합 결과의 혼동행렬을 시각화한다.

    Args:
        fit: statsmodels Logit 결과 객체 (`fit.predict()`로 예측 확률을 계산 가능해야 함).
        title (str|None): 그래프 제목.
        threshold (float): 예측 확률을 이진 분류로 변환할 임계값. 기본값 0.5.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes. None이면 새로 생성.

    Returns:
        None
    """
    outparams = False
    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    # 학습 데이터 기반 실제값/예측 확률 결정
    y_true = np.asarray(fit.model.endog)
    y_pred_proba = np.asarray(fit.predict(fit.model.exog))
    y_pred = (y_pred_proba >= threshold).astype(int)

    # 혼동행렬 계산
    cm = confusion_matrix(y_true, y_pred)

    # 혼동행렬 시각화
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["음성", "양성"])
    # 가독성을 위해 텍스트 크기/굵기 조정
    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        text_kw={"fontsize": 16, "weight": "bold"},
    )

    ax.set_title(f"혼동행렬 (임계값: {threshold})", fontsize=config.title_font_size, pad=config.title_pad)  # type: ignore

    finalize_plot(ax, callback, outparams, save_path, False, title)  # type: ignore


# ===================================================================
# 레이더 차트(방사형 차트)
# ===================================================================
def radarplot(
    df: DataFrame,
    columns: list | None = None,
    hue: str | None = None,
    title: str | None = None,
    normalize: bool = True,
    fill: bool = True,
    fill_alpha: float = 0.25,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params,
) -> None:
    """레이더 차트(방사형 차트)를 그린다.

    Args:
        df (DataFrame): 시각화할 데이터.
        columns (list|None): 레이더 차트에 표시할 컬럼 목록. None이면 모든 숫자형 컬럼 사용.
        hue (str|None): 집단 구분 컬럼. None이면 각 행을 개별 객체로 표시.
        title (str|None): 그래프 제목.
        normalize (bool): 0-1 범위로 정규화 여부. 기본값 True.
        fill (bool): 영역 채우기 여부.
        fill_alpha (float): 채움 투명도.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: 추가 플롯 옵션.

    Returns:
        None
    """
    outparams = False

    # 컬럼 선택
    if columns is None:
        # 숫자형 컬럼만 선택 (hue 제외)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if hue is not None and hue in numeric_cols:
            numeric_cols.remove(hue)
        columns = numeric_cols

    if len(columns) == 0:
        raise ValueError("레이더 차트에 표시할 숫자형 컬럼이 없습니다.")

    # 데이터 준비
    if hue is not None:
        # 집단별 평균 계산
        plot_data = df.groupby(hue)[columns].mean()
        labels = plot_data.index.tolist()
    else:
        # 각 행을 개별 객체로 사용
        plot_data = df[columns].copy()
        if plot_data.index.name:
            labels = plot_data.index.tolist()
        else:
            labels = [f"Row {i}" for i in range(len(plot_data))]

    # 정규화
    if normalize:
        for col in columns:
            min_val = plot_data[col].min()
            max_val = plot_data[col].max()
            if max_val - min_val > 0:
                plot_data[col] = (plot_data[col] - min_val) / (max_val - min_val)
            else:
                plot_data[col] = 0.5

    # Axes 생성 (polar projection)
    if ax is None:
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=config.dpi)
        ax = fig.add_subplot(111, projection="polar")
        outparams = True

    # 각도 계산
    num_vars = len(columns)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # 닫힌 도형을 만들기 위해 첫 번째 각도 추가

    # 색상 팔레트 설정
    if palette is not None:
        colors = sb.color_palette(palette, len(labels))
    else:
        colors = sb.color_palette("husl", len(labels))

    # 각 집단/객체별로 플롯
    for idx, (label_name, row) in enumerate(plot_data.iterrows()):
        values = row.tolist()
        values += values[:1]  # 닫힌 도형을 만들기 위해 첫 번째 값 추가

        color = colors[idx]

        # 선 그리기
        ax.plot(
            angles,
            values,
            "o-",
            linewidth=linewidth,
            label=str(label_name),
            color=color,
            **params,
        )

        # 영역 채우기
        if fill:
            ax.fill(angles, values, alpha=fill_alpha, color=color)

    # 축 레이블 설정
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(columns)

    # y축 범위 설정
    if normalize:
        ax.set_ylim(0, 1)

    # 범례
    if len(labels) <= 10:  # 너무 많으면 범례 생략
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # 제목
    if hue is not None:
        title = title if title else f"Radar Chart by {hue}"
    else:
        title = title if title else "Radar Chart"

    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 연속형 데이터 분포 시각화 (KDE + Boxplot)
# ===================================================================
def distribution_plot(
    data: DataFrame,
    column: str | list[str],
    clevel: float = 0.95,
    orient: str = "h",
    hue: str | None = None,
    kind: str = "boxplot",
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
) -> None:
    """연속형 데이터의 분포를 KDE와 Boxplot으로 시각화한다.

    1행 2열의 서브플롯을 생성하여:
    - 왼쪽: KDE with 신뢰구간
    - 오른쪽: Boxplot

    Args:
        data (DataFrame): 시각화할 데이터.
        column (str): 분석할 컬럼명.
        clevel (float): KDE 신뢰수준 (0~1). 기본값 0.95.
        orient (str): Boxplot 방향 ('v' 또는 'h'). 기본값 'h'.
        hue (str|None): 명목형 컬럼명. 지정하면 각 범주별로 행을 늘려 KDE와 boxplot을 그림.
        kind (str): 두 번째 그래프의 유형 (boxplot, hist). 기본값 "boxplot".
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        save_path (str|None): 저장 경로.
        callback (Callable|None): Axes 후처리 콜백.

    Returns:
        None
    """
    if isinstance(column, str):
        column = [column]

    for c in column:
        title = f"Distribution Plot of {c}"

        if hue is None:
            # 1행 2열 서브플롯 생성
            fig, axes = get_default_ax(
                width, height, rows=1, cols=2, title=title
            )

            kde_confidence_interval(
                data=data,
                xnames=c,
                clevel=clevel,
                linewidth=linewidth,
                ax=axes[0], # type: ignore
            )

            if kind == "hist":
                histplot(df=data, xname=c, linewidth=linewidth, ax=axes[1])  # type: ignore
            else:
                boxplot(
                    df=data[column], linewidth=linewidth, ax=axes[1]  # type: ignore
                )

            fig.suptitle(title, fontsize=14, y=1.02)
        else:
            if hue not in data.columns:
                raise ValueError(f"hue column '{hue}' not found in DataFrame")

            categories = list(Series(data[hue].dropna().unique()).sort_values())
            n_cat = len(categories) if categories else 1

            fig, axes = get_default_ax(
                width, height, rows=n_cat, cols=2, title=title
            )
            axes_2d = np.atleast_2d(axes)

            for idx, cat in enumerate(categories):
                subset = data[data[hue] == cat]
                left_ax, right_ax = axes_2d[idx, 0], axes_2d[idx, 1]

                kde_confidence_interval(
                    data=subset,
                    xnames=c,
                    clevel=clevel,
                    linewidth=linewidth,
                    ax=left_ax,
                )
                left_ax.set_title(f"{hue} = {cat}", fontsize=config.title_font_size, pad=config.title_pad)  # type: ignore

                if kind == "hist":
                    histplot(
                        df=subset,
                        xname=c,
                        linewidth=linewidth,
                        ax=right_ax,
                    )
                else:
                    boxplot(
                        df=subset[c], linewidth=linewidth, ax=right_ax  # type: ignore
                    )

            fig.suptitle(f"{title} by {hue}", fontsize=14, y=1.02)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, bbox_inches="tight")
                plt.close()
            else:
                plt.show()


def silhouette_plot(
    estimator: KMeans | AgglomerativeClustering,
    data: DataFrame,
    title: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
) -> None:
    """
    군집분석 결과의 실루엣 플롯을 시각화함.

    Args:
        estimator (KMeans | AgglomerativeClustering): 학습된 KMeans 또는 AgglomerativeClustering 군집 모델 객체.
        data (DataFrame): 군집분석에 사용된 입력 데이터 (n_samples, n_features).
        title (str, optional): 플롯 제목. None이면 자동 생성.
        width (int, optional): 플롯 가로 크기 (inch 단위).
        height (int, optional): 플롯 세로 크기 (inch 단위).
        linewidth (float, optional): 기준선 등 선 두께.
        save_path (str, optional): 저장 경로 지정 시 파일로 저장.
        callback (Callable, optional): 추가 커스텀 콜백 함수.
        ax (Axes, optional): 기존 matplotlib Axes 객체. None이면 새로 생성.

    Returns:
        None

    Note:
        - 각 군집별 실루엣 계수 분포를 막대그래프로 시각화
        - 군집 품질(응집도/분리도) 평가에 활용
        - 붉은색 세로선은 전체 평균 실루엣 스코어를 의미
    """

    outparams = False
    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    sil_avg = silhouette_score(X=data, labels=estimator.labels_)
    sil_values = silhouette_samples(X=data, labels=estimator.labels_)

    y_lower = 10

    # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현.
    n_clusters: int = 0
    if hasattr(estimator, "n_clusters") and estimator.n_clusters is not None:   # type: ignore
        n_clusters = estimator.n_clusters  # type: ignore
    elif hasattr(estimator, "n_clusters_") and estimator.n_clusters_ is not None:    # type: ignore
        n_clusters = estimator.n_clusters_  # type: ignore
    else:
        n_clusters = len(np.unique(estimator.labels_))  # type: ignore

    for i in range(n_clusters):  # type: ignore
        ith_cluster_sil_values = sil_values[estimator.labels_ == i]  # type: ignore
        ith_cluster_sil_values.sort()  # type: ignore

        size_cluster_i = ith_cluster_sil_values.shape[0]  # type: ignore
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(  # type: ignore
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_sil_values,  # type: ignore
            alpha=0.7,
        )
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))  # type: ignore
        y_lower = y_upper + 10

    ax.axvline(x=sil_avg, color="red", linestyle="--", linewidth=linewidth)  # type: ignore

    ax.set_xlabel("The silhouette coefficient values")  # type: ignore
    ax.set_ylabel("Cluster label")  # type: ignore
    ax.set_xlim([-0.1, 1])  # type: ignore
    ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])  # type: ignore
    ax.set_yticks([])  # type: ignore
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # type: ignore

    if title is None:
        title = "Number of Cluster : " + str(n_clusters) + ", Silhouette Score :" + str(round(sil_avg, 3))  # type: ignore

    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# 군집분석 결과 시각화
# ===================================================================
def cluster_plot(
    estimator: KMeans | AgglomerativeClustering | None = None,
    data: DataFrame | None = None,
    xname: str | None = None,
    yname: str | None = None,
    hue: str | None = None,
    vector: str | None = None,
    title: str | None = None,
    palette: str | None = None,
    outline: bool = True,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    ax: Axes | None = None,
) -> None:
    """
    2차원 공간에서 군집분석 결과를 산점도로 시각화함.

    Args:
        estimator (KMeans): 학습된 KMeans 군집 모델 객체.
        data (DataFrame): 군집분석에 사용된 입력 데이터 (n_samples, n_features).
        xname (str, optional): x축에 사용할 컬럼명. None이면 첫 번째 컬럼 사용.
        yname (str, optional): y축에 사용할 컬럼명. None이면 두 번째 컬럼 사용.
        hue (str, optional): 군집 구분에 사용할 컬럼명. None이면 'cluster' 자동 생성.
        vector (str, optional): 벡터 종류를 의미하는 컬럼명. None이면 사용 안함.
        title (str, optional): 플롯 제목. None이면 기본값 사용.
        palette (str, optional): 색상 팔레트.
        outline (bool, optional): 외곽선 표시 여부.
        width (int, optional): 플롯 가로 크기 (inch 단위).
        height (int, optional): 플롯 세로 크기 (inch 단위).
        linewidth (float, optional): 중심점 등 선 두께.
        save_path (str, optional): 저장 경로 지정 시 파일로 저장.
        ax (Axes, optional): 기존 matplotlib Axes 객체. None이면 새로 생성.

    Returns:
        None

    Example:
        ```python
        cluster_plot(estimator, data, xname='Sepal.Length', yname='Sepal.Width')
        ```

    Note:
        - 각 군집별 산점도와 중심점(빨간색 원/숫자) 표시
        - 2차원 특성 공간에서 군집 분포와 분리도 시각화
    """
    outparams = False
    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True

    df = data.copy() if data is not None else None  # type: ignore

    if not hue:
        df["cluster"] = estimator.labels_  # type: ignore
        hue = "cluster"

    if xname is None:
        xname = df.columns[0]  # type: ignore

    if yname is None:
        yname = df.columns[1]  # type: ignore

    xindex = df.columns.get_loc(xname)  # type: ignore
    yindex = df.columns.get_loc(yname)  # type: ignore

    def callback(ax: Axes) -> None:
        ax.set_xlabel("Feature space for the " + xname)
        ax.set_ylabel("Feature space for the " + yname)

        if hasattr(estimator, "cluster_centers_"):
            # 클러스터 중심점 표시
            centers = estimator.cluster_centers_  # type: ignore
            ax.scatter(  # type: ignore
                centers[:, xindex],
                centers[:, yindex],
                marker="o",  # type: ignore
                color="white",
                alpha=1,
                s=200,
                edgecolor="r",
                linewidth=linewidth,  # type: ignore
            )

            for i, c in enumerate(centers):
                ax.scatter(
                    c[xindex], c[yindex], marker="$%d$" % i, alpha=1, s=50, edgecolor="k"  # type: ignore
                )

    scatterplot(
        df=df,          # type: ignore
        xname=xname,
        yname=yname,
        hue=hue,
        vector=vector,
        title="The visualization of the clustered data." if title is None else title,
        outline=outline,
        palette=palette,
        width=width,
        height=height,
        linewidth=linewidth,
        save_path=save_path,
        callback=callback,
        ax=ax,
    )


# ===================================================================
# 군집분석 결과의 실루엣 플롯과 군집 산점도를 한 화면에 함께 시각화
# ===================================================================
def visualize_silhouette(
    estimator: KMeans | AgglomerativeClustering,
    data: DataFrame,
    xname: str | None = None,
    yname: str | None = None,
    title: str | None = None,
    palette: str | None = None,
    outline: bool = True,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
) -> None:
    """
    군집분석 결과의 실루엣 플롯과 군집 산점도를 한 화면에 함께 시각화함.

    수업에서 사용한 visualize_silhouette 함수와 동일한 기능을 수행함.

    Args:
        estimator (KMeans | AgglomerativeClustering): 학습된 KMeans 또는 AgglomerativeClustering 군집 모델 객체.
        data (DataFrame): 군집분석에 사용된 입력 데이터 (n_samples, n_features).
        xname (str, optional): 산점도 x축에 사용할 컬럼명. None이면 첫 번째 컬럼 사용.
        yname (str, optional): 산점도 y축에 사용할 컬럼명. None이면 두 번째 컬럼 사용.
        title (str, optional): 플롯 제목. None이면 기본값 사용.
        palette (str, optional): 색상 팔레트.
        outline (bool, optional): 산점도 외곽선 표시 여부.
        width (int, optional): 플롯 가로 크기 (inch 단위).
        height (int, optional): 플롯 세로 크기 (inch 단위).
        linewidth (float, optional): 기준선 등 선 두께.
        save_path (str, optional): 저장 경로 지정 시 파일로 저장.

    Returns:
        None

    Note:
        - 실루엣 플롯(왼쪽)과 2차원 군집 산점도(오른쪽)를 동시에 확인 가능
        - 군집 품질과 분포를 한눈에 비교·분석할 때 유용
    """
    fig, ax = get_default_ax(rows=1, cols=2, width=width, height=height, title=title)

    silhouette_plot(
        estimator=estimator,
        data=data,
        ax=ax[0],  # type: ignore
        linewidth=linewidth,
        width=width,
        height=height
    )

    cluster_plot(
        estimator=estimator,
        data=data,
        xname=xname,
        yname=yname,
        ax=ax[1],  # type: ignore
        outline=outline,
        palette=palette,
        width=width,
        height=height
    )

    finalize_plot(ax)



# ===================================================================
# 덴드로그램 시각화
# ===================================================================
def dandrogram(
    estimator: AgglomerativeClustering,
    p: int = 30,
    count_sort: Literal["ascending", "descending", False] = "ascending",
    title: str | None = None,
    width: int = config.width,
    height: int = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None
) -> None:
    """덴드로그램 시각화

    Args:
        estimator (AgglomerativeClustering): 학습된 AgglomerativeClustering 군집 모델 객체.
        p (int): 덴드로그램에서 표시할 마지막 병합된 군집 수. 기본값 30.
        count_sort (str): 'ascending' 또는 'descending'으로 병합 순서 정렬.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        save_path (str|None): 저장 경로.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes. None이면 새로 생성.

    Returns:
        None
    """
    # 덴드로그램을 그리기 위해 linkage 행렬 생성
    counts = np.zeros(estimator.children_.shape[0]) # type: ignore
    n_samples = len(estimator.labels_)

    for i, merge in enumerate(estimator.children_): # type: ignore
        current_count = 0
        for child_idx in merge:  # type: ignore
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

        linkage_matrix = np.column_stack(
            [estimator.children_, estimator.distances_, counts]
        ).astype(float)

    outparams = False

    if ax is None:
        fig, ax = get_default_ax(width, height, 1, 1)  # type: ignore
        outparams = True


    sch_dendrogram(
        linkage_matrix,
        ax=ax,
        p=p,
        truncate_mode="lastp" if p > 0 else None,
        leaf_rotation=0,
        leaf_font_size=8,
        count_sort=count_sort,
        color_threshold=None,
        above_threshold_color="grey",
    )

    finalize_plot(ax, callback, outparams, save_path, True, title)  # type: ignore


# ===================================================================
# PCA 분석 결과에 대한 biplot 시각화
# ===================================================================
def pca_plot(
    estimator: PCA,
    data: DataFrame,
    yname: str | None = None,
    fields: list | tuple | list[list] | tuple[list] | list[tuple] | tuple[tuple] | None = None,
    hue: str | None = None,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    linewidth: float = config.line_width,
    save_path: str | None = None,
    callback: Callable | None = None,
) -> None:
    """
    PCA 분석 결과에 대한 biplot 시각화

    Args:
        estimator (PCA): 학습된 PCA 객체.
        data (DataFrame): PCA에 사용된 원본 데이터.
        yname (str | None): 종속변수 컬럼명.
        fields (list | tuple | list[list] | tuple[list] | list[tuple] | tuple[tuple] | None): 시각화할 독립변수 목록. None이면 자동 탐지.
        hue (str|None): 집단 구분 컬럼명.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        linewidth (float): 선 굵기.
        save_path (str|None): 저장 경로.
        callback (Callable|None): Axes 후처리 콜백.

    Returns:
        None
    """
    df = data.copy()
    df_columns = df.columns.tolist()

    # 종속변수가 지정되었다면 해당 컬럼 추출
    yfield = None
    if yname is not None and yname in data.columns:
        yfield = df[[yname]].copy()
        df = df.drop(columns=[yname])

    # PCA 변환 수행
    #display(df)
    score = estimator.transform(df)
    #print(score)

    # 추정기로부터 PCA 결과 데이터 프레임 생성
    pca_df = DataFrame(
        data=score,
        columns=[f"PC{i+1}" for i in range(estimator.n_components_)],
    )
    #display(pca_df)

    # 종속변수 컬럼 추가
    if yfield is not None:
        pca_df[yname] = yfield

    # 모든 컬럼명에 대한 조합 생성
    if fields is None:
        feature_cols = pca_df.columns.tolist()
        if yname is not None and yname in feature_cols:
            feature_cols.remove(yname)
        fields = list(combinations(feature_cols, 2))

    if not is_2d(fields):
        fields = [fields]   # type: ignore

    components = estimator.components_

    x_index: int = 0
    y_index: int = 0

    def __callable(ax) -> None:
        for i in range(n):
            ax.arrow(
                0,
                0,
                components[x_index, i],
                components[y_index, i],
                color="r",
                head_width=0.007,
                head_length=0.007,
                linewidth=linewidth * 0.75,
                alpha=0.75,
            )
            ax.text(
                components[x_index, i] * 1.15,
                components[y_index, i] * 1.15,
                f"{df_columns[i]} ({components[x_index, i]:.2f})",
                color="b",
                ha="center",
                va="center",
            )

        if callback is not None:
            callback(ax)

    for field_group in fields:  # type: ignore
        x_index = int(pca_df.columns.get_loc(field_group[0]))   # type: ignore
        y_index = int(pca_df.columns.get_loc(field_group[1]))   # type: ignore

        xs = score[:, x_index]
        ys = score[:, y_index]
        n = score.shape[1]

        scalex = 1.0 / (xs.max() - xs.min())
        scaley = 1.0 / (ys.max() - ys.min())

        title = "PCA Biplot"
        if field_group is not None:
            title += " - " + ", ".join(field_group)

        tdf = DataFrame({
            field_group[0]: xs * scalex,
            field_group[1]: ys * scaley,
        })

        scatterplot(
            df=tdf,
            xname=field_group[0],
            yname=field_group[1],
            hue=pca_df[hue] if hue is not None else None,
            outline=False,
            palette=palette,
            width=width,
            height=height,
            linewidth=linewidth,
            save_path=save_path,
            title=title,
            callback=__callable,
        )
    