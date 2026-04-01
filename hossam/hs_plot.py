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
import matplotlib.patches as patches
import matplotlib as mpl
from pandas import Index, Series, DataFrame, pivot_table
from math import sqrt

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
DEFAULT_DPI = 200

config = SimpleNamespace(
    dpi=DEFAULT_DPI,
    width=1280,
    height=640,
    font_size=10,
    text_font_size=14,
    title_font_weight=500,
    title_font_size=24,
    title_pad=15,
    xlabel_fontsize=16,
    xlabel_fontweight=400,
    xlabel_pad=5,
    ylabel_fontsize=16,
    ylabel_fontweight=400,
    ylabel_pad=5,
    label_font_size=14,
    font_weight="normal",
    frame_width=1,
    line_width=2,
    grid_alpha=0.5,
    grid_width=1,
    fill_alpha=0.3,
    ws=0.1,
    hs=0.2,
)


# ===================================================================
# 전역 설정 객체의 DPI 및 폰트 크기를 설정한다.
# ===================================================================
def set_dpi(dpi: int = DEFAULT_DPI) -> None:
    """
    전역 설정 객체의 DPI 및 폰트 크기를 설정한다.

    Args:
        dpi (int): 설정할 DPI 값.

    Returns:
        None
    """
    config.dpi = dpi

    if dpi > 200:
        config.font_size = config.font_size * (dpi * 0.0011 + 0.7)
        config.text_font_size = config.text_font_size * (dpi * 0.0011 + 0.7)
        config.title_font_size = config.title_font_size * (dpi * 0.0011 + 0.7)
        config.title_pad = config.title_pad * (dpi * 0.0011 + 0.7)
        config.label_font_size = config.label_font_size * (dpi * 0.0011 + 0.7)
    elif dpi > 100:
        config.font_size = config.font_size * (dpi * 0.0012 + 0.75)
        config.text_font_size = config.text_font_size * (dpi * 0.0012 + 0.75)
        config.title_font_size = config.title_font_size * (dpi * 0.0012 + 0.75)
        config.title_pad = config.title_pad * (dpi * 0.0012 + 0.75)
        config.label_font_size = config.label_font_size * (dpi * 0.0012 + 0.75)
    else:
        config.font_size = 10
        config.text_font_size = 8
        config.title_font_size = 18
        config.title_pad = 15
        config.label_font_size = 14

# 초기 DPI 설정
set_dpi(DEFAULT_DPI)

# ===================================================================
# 기본 크기가 설정된 Figure와 Axes를 생성한다
# ===================================================================
def init(
    width: int = config.width,
    height: int = config.height,
    rows: int = 1,
    cols: int = 1,
    flatten: bool = False,
    ws: int | None = config.ws,
    hs: int | None = config.hs,
    title: str | None = None,
    xlabel: str | None = None,
    xlabel_fontsize: int = config.xlabel_fontsize,
    xlabel_fontweight: str = config.xlabel_fontweight,
    xlabel_pad: int = config.xlabel_pad,
    ylabel: str | None = None,
    ylabel_fontsize: int = config.ylabel_fontsize,
    ylabel_fontweight: str = config.ylabel_fontweight,
    ylabel_pad: int = config.ylabel_pad,
    grid: bool = True):
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
        xlabel (str|None): x축 레이블.
        xlabel_fontsize (int): x축 레이블 폰트 크기.
        xlabel_fontweight (str): x축 레이블 폰트 두께.
        xlabel_pad (int): x축 레이블 패드.
        ylabel (str|None): y축 레이블.
        ylabel_fontsize (int): y축 레이블 폰트 크기.
        ylabel_fontweight (str): y축 레이블 폰트 두께.
        ylabel_pad (int): y축 레이블 패드.
        grid (bool): 생성된 Axes에 그리드를 표시할지 여부.

    Returns:
        tuple[Figure, Axes]: 생성된 matplotlib Figure와 Axes 객체.
    """
    figsize = (width * cols / 100, height * rows / 100)
    fig, ax = plt.subplots(rows, cols, figsize=figsize, dpi=config.dpi)

    if rows == 1 and cols == 1:
        if title:
            ax.set_title(title, fontsize=config.title_font_size, fontweight=config.title_font_weight, pad=config.title_pad) # type: ignore

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=xlabel_fontsize, fontweight=config.xlabel_fontweight, labelpad=xlabel_pad) # type: ignore

        if ylabel:
            ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontweight=config.ylabel_fontweight, labelpad=ylabel_pad) # type: ignore

        # 그리드 설정
        ax.grid(grid, alpha=config.grid_alpha, linewidth=config.grid_width) # type: ignore

        # 테두리 굵기 설정
        for spine in ax.spines.values(): # type: ignore
            spine.set_linewidth(config.frame_width)
        
        plt.tight_layout()

    else:
        if title:
            fig.suptitle(title, fontsize=config.title_font_size * (rows * cols / 2), fontweight=config.title_font_weight) # type: ignore

        # Grid, 테두리 굵기 설정
        for f in ax.flatten():
            f.grid(grid, alpha=config.grid_alpha, linewidth=config.grid_width) # type: ignore
            for spine in f.spines.values(): # type: ignore
                spine.set_linewidth(config.frame_width)
        
        plt.tight_layout()
        
        fig.subplots_adjust(wspace=ws, hspace=hs)

    if (rows > 1 or cols > 1) and flatten == True:
        ax = ax.flatten()

    return fig, ax


# ===================================================================
# 그래프의 그리드, 레이아웃을 정리하고 필요 시 저장 또는 표시한다
# ===================================================================
def show(
    save_path: str | None = None) -> None:
    """공통 후처리를 수행한다: 콜백 실행, 레이아웃 정리, 필요 시 표시/종료.

    Args:
        save_path (str|None): 이미지 저장 경로. None이 아니면 해당 경로로 저장.
    Returns:
        None
    """
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    plt.close()


# ===================================================================
# Matplotlib 컬러맵 팔레트 시각화
# ===================================================================
def colormaps(n_colors=8):
    """
    Matplotlib에서 제공하는 다양한 컬러맵을
    연속형 그라디언트가 아니라 8단계 팔레트 형태로 보여주는 함수.
    """

    cmaps = [
        ('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis'
        ]),
        ('Sequential', [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
        ]),
        ('Sequential (2)', [
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper'
        ]),
        ('Diverging', [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic'
        ]),
        ('Cyclic', [
            'twilight', 'twilight_shifted', 'hsv'
        ]),
        ('Qualitative', [
            'Pastel1', 'Pastel2', 'Paired', 'Accent',
            'Dark2', 'Set1', 'Set2', 'Set3',
            'tab10', 'tab20', 'tab20b', 'tab20c'
        ])
    ]

    def plot_color_blocks(cmap_category, cmap_list):
        nrows = len(cmap_list)
        figh = 0.5 + (nrows * 0.42)

        fig, axs = plt.subplots(nrows=nrows, figsize=(8, figh))

        # cmap이 1개뿐일 때 axs를 iterable로 맞춤
        if nrows == 1:
            axs = [axs]

        fig.subplots_adjust(
            top=1 - 0.35 / figh,
            bottom=0.05,
            left=0.25,
            right=0.98,
            hspace=0.45
        )

        axs[0].set_title(f"{cmap_category} colormaps", fontsize=14)

        for ax, cmap_name in zip(axs, cmap_list):
            cmap = mpl.colormaps[cmap_name]

            # 0~1 구간에서 n_colors개 샘플 추출
            sample_points = np.linspace(0, 1, n_colors)
            colors = cmap(sample_points)

            # 팔레트 블록 그리기
            for i, color in enumerate(colors):
                rect = patches.Rectangle(
                    (i, 0), 1, 1,
                    facecolor=color,
                    edgecolor='none'
                )
                ax.add_patch(rect)

            # 이름 표시
            ax.text(
                -0.02, 0.5, f"{cmap_name:>20}",
                va='center', ha='right',
                fontsize=10,
                transform=ax.transAxes
            )

            ax.set_xlim(0, n_colors)
            ax.set_ylim(0, 1)
            ax.set_axis_off()

        plt.show()

    for cmap_category, cmap_list in cmaps:
        plot_color_blocks(cmap_category, cmap_list)


# ===================================================================
# 선 그래프를 그린다
# ===================================================================
def lineplot(
    data: DataFrame | None = None,
    x: str | Series | np.ndarray | list | None = None,
    y: str | Series | np.ndarray | list | None = None,
    hue: str | None = None,
    linewidth: float = config.line_width,
    marker: str | None = None,
    markersize: int | None = None,
    markeredgewidth: int | None = None,
    markeredgecolor: str | None = None,
    markerfacecolor: str | None = None,
    #----- 공통 파라미터 ------
    title: str | None = None,
    xlabel: str | None = None,
    xlabel_fontsize: int = config.xlabel_fontsize,
    xlabel_fontweight: str = config.xlabel_fontweight,
    xlabel_pad: int = config.xlabel_pad,
    ylabel: str | None = None,
    ylabel_fontsize: int = config.ylabel_fontsize,
    ylabel_fontweight: str = config.ylabel_fontweight,
    ylabel_pad: int = config.ylabel_pad,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params) -> None:
    """선 그래프를 그린다.

    Args:
        data (DataFrame | None): 시각화할 데이터.
        x (str | Series | np.ndarray | list | None): x축 컬럼명 혹은 x축 값 시퀀스.
        y (str | Series | np.ndarray | list | None): y축 컬럼명 혹은 y축 값 시퀀스.
        hue (str | None): 범주 구분 컬럼명.
        linewidth (float): 선 굵기.
        marker (str | None): 마커 모양.
        markersize (int | None): 마커 크기.
        markeredgewidth (int | None): 마커 테두리 두께.
        markeredgecolor (str | None): 마커 테두리 색상.
        markerfacecolor (str | None): 마커 배경 색상.

    Common Args:
        title (str | None): 그래프 제목.
        xlabel (str|None): x축 레이블.
        xlabel_fontsize (int): x축 레이블 폰트 크기.
        xlabel_fontweight (str): x축 레이블 폰트 두께.
        xlabel_pad (int): x축 레이블 패드.
        ylabel (str|None): y축 레이블.
        ylabel_fontsize (int): y축 레이블 폰트 크기.
        ylabel_fontweight (str): y축 레이블 폰트 두께.
        ylabel_pad (int): y축 레이블 패드.
        palette (str | None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        save_path (str|None): 이미지 저장 경로.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn lineplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title, xlabel=xlabel, xlabel_fontsize=xlabel_fontsize, xlabel_fontweight=xlabel_fontweight, xlabel_pad=xlabel_pad, ylabel=ylabel, ylabel_fontsize=ylabel_fontsize, ylabel_fontweight=ylabel_fontweight, ylabel_pad=ylabel_pad)  # type: ignore
        outparams = True

    # hue가 있을 때만 palette 사용, 없으면 color 사용
    lineplot_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "marker": marker,
        "markersize": markersize,
        "markeredgewidth": markeredgewidth,
        "markeredgecolor": markeredgecolor,
        "markerfacecolor": markerfacecolor,
        "linewidth": linewidth,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        lineplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        lineplot_kwargs["color"] = sb.color_palette(palette)[0]

    lineplot_kwargs.update(params)

    sb.lineplot(**lineplot_kwargs)

    if callback is not None:
        callback(ax)

    if outparams:
        show(save_path)  # type: ignore


# ===================================================================
# 단변량 커널 밀도 추정(KDE) 그래프를 그린다
# ===================================================================
def kdeplot(
    data: DataFrame,
    x: str | None = None,
    meanline: bool = False,
    fill: bool = False,
    alpha: float = config.fill_alpha,
    linewidth: float = config.line_width,
    quartile_split: bool = False,
    #----- 공통 파라미터 ------
    title: str | None = None,
    xlabel: str | None = None,
    xlabel_fontsize: int = config.xlabel_fontsize,
    xlabel_fontweight: str = config.xlabel_fontweight,
    xlabel_pad: int = config.xlabel_pad,
    ylabel: str | None = None,
    ylabel_fontsize: int = config.ylabel_fontsize,
    ylabel_fontweight: str = config.ylabel_fontweight,
    ylabel_pad: int = config.ylabel_pad,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params) -> None:
    """단변량 커널 밀도 추정(KDE) 그래프를 그린다. 범주에 따른 구분은 지원하지 않는다.

    quartile_split=True일 때는 사분위수 구간(Q1~Q4)으로 나누어 4개의 서브플롯에 그린다.

    Args:
        data (DataFrame): 시각화할 데이터.
        x (str|None): x축 컬럼명.
        fill (bool): 면적 채우기 여부.
        alpha (float): 채움 투명도.
        quartile_split (bool): True면 1D KDE를 사분위수별 서브플롯으로 분할.
        linewidth (float): 선 굵기.

    Common Args:
        title (str|None): 그래프 제목.
        xlabel (str|None): x축 레이블.
        xlabel_fontsize (int): x축 레이블 폰트 크기.
        xlabel_fontweight (str): x축 레이블 폰트 두께.
        xlabel_pad (int): x축 레이블 패드.
        ylabel (str|None): y축 레이블.
        ylabel_fontsize (int): y축 레이블 폰트 크기.
        ylabel_fontweight (str): y축 레이블 폰트 두께.
        ylabel_pad (int): y축 레이블 패드.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        save_path (str|None): 이미지 저장 경로.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn kdeplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    # 사분위수 분할 전용 처리 (1D KDE만 지원)
    if quartile_split:
        series = data[x].dropna()
        if series.empty:
            raise ValueError(f"{x} 컬럼에 유효한 데이터가 없습니다.")

        q = series.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values
        bounds = list(zip(q[:-1], q[1:]))  # [(Q0,Q1),(Q1,Q2),(Q2,Q3),(Q3,Q4)]

        fig, axes = init(width=width, height=height, rows=2, cols=2, flatten=True, title=title, xlabel=xlabel, xlabel_fontsize=xlabel_fontsize, xlabel_fontweight=xlabel_fontweight, xlabel_pad=xlabel_pad, ylabel=ylabel, ylabel_fontsize=ylabel_fontsize, ylabel_fontweight=ylabel_fontweight, ylabel_pad=ylabel_pad)  # type: ignore
        outparams = True

        for idx, (lo, hi) in enumerate(bounds):
            subset = series[(series >= lo) & (series <= hi)]
            if subset.empty:
                continue

            cols = [x]
            data_quartile = data.loc[subset.index, cols].copy()

            kdeplot_kwargs = {
                "data": data_quartile,
                "x": x,
                "fill": fill,
                "linewidth": linewidth,
                "label": f"Q{idx+1} ({lo:.2f}~{hi:.2f})",
                "ax": axes[idx],
            }

            if fill:
                kdeplot_kwargs["alpha"] = alpha

            kdeplot_kwargs.update(params)
            sb.kdeplot(**kdeplot_kwargs)
            axes[idx].legend()  # type: ignore
            
            if meanline:
                mean_value = subset.mean()
                axes[idx].axvline(x=mean_value, color='red', linestyle='--', linewidth=linewidth * 0.5)  # type: ignore
                axes[idx].text(x=mean_value + 0.05, y=axes[idx].get_ylim()[1]*0.95, 
                                s=f'Mean: {mean_value:.2f}', color='red', fontsize=config.text_font_size)  # type: ignore

        show(save_path)
        return

    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title, xlabel=xlabel, ylabel=ylabel)  # type: ignore
        outparams = True

    # 기본 kwargs 설정
    kdeplot_kwargs = {
        "data": data,
        "x": x,
        "fill": fill,
        "linewidth": linewidth,
        "palette": palette,
        "ax": ax
    }

    if fill:
        kdeplot_kwargs["alpha"] = alpha

    # 커널밀도 추정 그래프 그리기
    sb.kdeplot(**kdeplot_kwargs)

    # 평균선 표시
    if meanline:
        mean_value = data[x].mean()
        ax.axvline(x=mean_value, color='red', linestyle='--', linewidth=linewidth * 0.5)  # type: ignore
        ax.text(x=mean_value + 0.05, y=ax.get_ylim()[1]*0.95, 
                s=f'Mean: {mean_value:.2f}', color='red', fontsize=config.text_font_size)

    if callback is not None:
        callback(ax)

    if outparams:
        show(save_path)  # type: ignore


# ===================================================================
# 히스토그램을 그린다
# ===================================================================
def histplot(
    data: DataFrame,
    x: str,
    bins: int | list | str = "auto",
    hue: str | None = None,
    linewidth: float = 1,
    kde: bool = False,
    #----- 공통 파라미터 ------
    title: str | None = None,
    xlabel: str | None = None,
    xlabel_fontsize: int = config.xlabel_fontsize,
    xlabel_fontweight: str = config.xlabel_fontweight,
    xlabel_pad: int = config.xlabel_pad,
    ylabel: str | None = None,
    ylabel_fontsize: int = config.ylabel_fontsize,
    ylabel_fontweight: str = config.ylabel_fontweight,
    ylabel_pad: int = config.ylabel_pad,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params) -> None:
    """히스토그램을 그리고 필요 시 KDE를 함께 표시한다.

    Args:
        data (DataFrame): 시각화할 데이터.
        x (str): 히스토그램 대상 컬럼명.
        bins (int|sequence|str): 구간 수 또는 경계.
        hue (str|None): 범주 컬럼명.
        linewidth (float): 선 굵기.
        kde (bool): KDE 표시 여부.

    Common Args:
        title (str|None): 그래프 제목.
        xlabel (str|None): x축 레이블.
        xlabel_fontsize (int): x축 레이블 폰트 크기.
        xlabel_fontweight (str): x축 레이블 폰트 두께.
        xlabel_pad (int): x축 레이블 패드.
        ylabel (str|None): y축 레이블.
        ylabel_fontsize (int): y축 레이블 폰트 크기.
        ylabel_fontweight (str): y축 레이블 폰트 두께.
        ylabel_pad (int): y축 레이블 패드.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        save_path (str|None): 이미지 저장 경로.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn histplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title, xlabel=xlabel, xlabel_fontsize=xlabel_fontsize, xlabel_fontweight=xlabel_fontweight, xlabel_pad=xlabel_pad, ylabel=ylabel, ylabel_fontsize=ylabel_fontsize, ylabel_fontweight=ylabel_fontweight, ylabel_pad=ylabel_pad)  # type: ignore
        outparams = True

    histplot_kwargs = {
        "data": data,
        "x": x,
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

    if callback is not None:
        callback(ax)

    if outparams:
        show(save_path)  # type: ignore


# ===================================================================
# 상자그림(boxplot)을 그린다
# ===================================================================
def boxplot(
    data: DataFrame | None = None,
    x: str | None = None,
    y: str | None = None,
    hue: str | None = None,
    orient: str = "v",
    stat_test: str | None = None,
    stat_pairs: list[tuple] | None = None,
    stat_text_format: str = "star",
    stat_loc: str = "inside",
    #----- 공통 파라미터 ------
    title: str | None = None,
    xlabel: str | None = None,
    xlabel_fontsize: int = config.xlabel_fontsize,
    xlabel_fontweight: str = config.xlabel_fontweight,
    xlabel_pad: int = config.xlabel_pad,
    ylabel: str | None = None,
    ylabel_fontsize: int = config.ylabel_fontsize,
    ylabel_fontweight: str = config.ylabel_fontweight,
    ylabel_pad: int = config.ylabel_pad,
    palette: str | None = None,
    width: int = config.width,
    height: int = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params) -> None:
    """
    상자그림(boxplot)을 그린다.

    Args:
        data (DataFrame|None): 시각화할 데이터.
        x (str|None): x축 범주 컬럼명.
        y (str|None): y축 값 컬럼명.
        hue (str|None): 범주 구분 컬럼명.
        orient (str): 'v' 또는 'h' 방향.
        stat_test (str|None): 통계 검정 방법. None이면 검정 안함. x과 y가 모두 지정되어야 함.
        stat_pairs (list[tuple]|None): 통계 검정할 그룹 쌍 목록.
        stat_text_format (str): 통계 결과 표시 형식.
        stat_loc (str): 통계 결과 위치.

    Common Args:
        title (str|None): 그래프 제목.
        xlabel (str|None): x축 레이블.
        xlabel_fontsize (int): x축 레이블 폰트 크기.
        xlabel_fontweight (str): x축 레이블 폰트 두께.
        xlabel_pad (int): x축 레이블 패드.
        ylabel (str|None): y축 레이블.
        ylabel_fontsize (int): y축 레이블 폰트 크기.
        ylabel_fontweight (str): y축 레이블 폰트 두께.
        ylabel_pad (int): y축 레이블 패드.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        save_path (str|None): 이미지 저장 경로. None이면 화면에 표시.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn boxplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title, xlabel=xlabel, xlabel_fontsize=xlabel_fontsize, xlabel_fontweight=xlabel_fontweight, xlabel_pad=xlabel_pad, ylabel=ylabel, ylabel_fontsize=ylabel_fontsize, ylabel_fontweight=ylabel_fontweight, ylabel_pad=ylabel_pad)  # type: ignore
        outparams = True

    if x is not None or y is not None:
        if x is not None and y is None:
            orient = "h"
        elif x is None and y is not None:
            orient = "v"

    boxplot_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "orient": orient,
        "hue": hue,
        "ax": ax
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
    # if stat_test is not None and x is not None and y is not None:
    #     if stat_pairs is None:
    #         stat_pairs = [data[x].dropna().unique().tolist()] # type: ignore

    #     annotator = Annotator(
    #         ax, data=data, x=x, y=y, pairs=stat_pairs, orient=orient
    #     )
    #     annotator.configure(
    #         test=stat_test, text_format=stat_text_format, loc=stat_loc
    #     )
    #     annotator.apply_and_annotate()

    if callback is not None:
        callback(ax)

    if outparams:
        show(save_path)  # type: ignore


# ===================================================================
# 바이올린 플롯(violinplot)을 그린다
# ===================================================================
def violinplot(
    data: DataFrame | None = None,
    x: str | None = None,
    y: str | None = None,
    hue: str | None = None,
    orient: str = "v",
    #----- 공통 파라미터 ------
    title: str | None = None,
    xlabel: str | None = None,
    xlabel_fontsize: int = config.xlabel_fontsize,
    xlabel_fontweight: str = config.xlabel_fontweight,
    xlabel_pad: int = config.xlabel_pad,
    ylabel: str | None = None,
    ylabel_fontsize: int = config.ylabel_fontsize,
    ylabel_fontweight: str = config.ylabel_fontweight,
    ylabel_pad: int = config.ylabel_pad,
    palette: str | None = None,
    width: int | None = config.width,
    height: int | None = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params) -> None:
    """
    바이올린 플롯(violinplot)을 그린다.

    Args:
        data (DataFrame|None): 시각화할 데이터.
        x (str|None): x축 범주 컬럼명.
        y (str|None): y축 값 컬럼명.
        hue (str|None): 범주 구분 컬럼명.
        orient (str): 'v' 또는 'h' 방향.

    Common Args:
        title (str|None): 그래프 제목.
        xlabel (str|None): x축 레이블.
        xlabel_fontsize (int): x축 레이블 폰트 크기.
        xlabel_fontweight (str): x축 레이블 폰트 두께.
        xlabel_pad (int): x축 레이블 패드.
        ylabel (str|None): y축 레이블.
        ylabel_fontsize (int): y축 레이블 폰트 크기.
        ylabel_fontweight (str): y축 레이블 폰트 두께.
        ylabel_pad (int): y축 레이블 패드.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        save_path (str|None): 이미지 저장 경로. None이면 화면에 표시.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn violinplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title, xlabel=xlabel, xlabel_fontsize=xlabel_fontsize, xlabel_fontweight=xlabel_fontweight, xlabel_pad=xlabel_pad, ylabel=ylabel, ylabel_fontsize=ylabel_fontsize, ylabel_fontweight=ylabel_fontweight, ylabel_pad=ylabel_pad)  # type: ignore
        outparams = True

    if x is not None or y is not None:
        if x is not None and y is None:
            orient = "h"
        elif x is None and y is not None:
            orient = "v"

    violinplot_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "orient": orient,
        "ax": ax
    }

    # hue 파라미터 확인 (params에 있을 수 있음)
    hue_value = params.get("hue", None)

    if hue_value is not None and palette is not None:
        violinplot_kwargs["palette"] = palette
    elif hue_value is None and palette is not None:
        violinplot_kwargs["color"] = sb.color_palette(palette)[0]

    violinplot_kwargs.update(params)
    sb.violinplot(**violinplot_kwargs)

    if callback is not None:
        callback(ax)

    if outparams:
        show(save_path)  # type: ignore


# ===================================================================
# 히트맵을 그린다
# ===================================================================
def heatmap(
    data: DataFrame,
    annot: bool = True,
    fmt: str = "0.2f",
    linewidth: float = 0.5,
    #----- 공통 파라미터 ------
    title: str | None = None,
    xlabel: str | None = None,
    xlabel_fontsize: int = config.xlabel_fontsize,
    xlabel_fontweight: str = config.xlabel_fontweight,
    xlabel_pad: int = config.xlabel_pad,
    ylabel: str | None = None,
    ylabel_fontsize: int = config.ylabel_fontsize,
    ylabel_fontweight: str = config.ylabel_fontweight,
    ylabel_pad: int = config.ylabel_pad,
    palette: str | None = None,
    width: int | None = config.width,
    height: int | None = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params) -> None:
    """
    히트맵을 그린다.

    Args:
        data (DataFrame|None): 시각화할 데이터.
        annot (bool): 셀에 값 표시 여부.
        fmt (str): 셀에 표시할 값 형식.
        linewidth (float): 셀 경계선 굵기.

    Common Args:
        title (str|None): 그래프 제목.
        xlabel (str|None): x축 레이블.
        xlabel_fontsize (int): x축 레이블 폰트 크기.
        xlabel_fontweight (str): x축 레이블 폰트 두께.
        xlabel_pad (int): x축 레이블 패드.
        ylabel (str|None): y축 레이블.
        ylabel_fontsize (int): y축 레이블 폰트 크기.
        ylabel_fontweight (str): y축 레이블 폰트 두께.
        ylabel_pad (int): y축 레이블 패드.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        save_path (str|None): 이미지 저장 경로. None이면 화면에 표시.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn violinplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if width == None or height == None:
        width = (config.font_size * config.dpi / 100) * 4.5 * len(data.columns)
        height = width * 0.6  # type: ignore

    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title, xlabel=xlabel, xlabel_fontsize=xlabel_fontsize, xlabel_fontweight=xlabel_fontweight, xlabel_pad=xlabel_pad, ylabel=ylabel, ylabel_fontsize=ylabel_fontsize, ylabel_fontweight=ylabel_fontweight, ylabel_pad=ylabel_pad)  # type: ignore
        outparams = True

    ax.grid(False)

    heatmatp_kwargs = { 
        "data": data,
        "annot": annot,
        "cmap": palette,
        "fmt": fmt,
        "ax": ax,
        "linewidths": linewidth,
        "annot_kws": {"size": config.font_size}  # 셀 안의 텍스트 크기,
    }

    heatmatp_kwargs.update(params)
    sb.heatmap(**heatmatp_kwargs)

    if callback is not None:
        callback(ax)

    if outparams:
        show(save_path)  # type: ignore


# ===================================================================
# 막대그래프를 그린다
# ===================================================================
def barplot(
    data: DataFrame,
    x: str | Index,
    y: str | Index,
    hue: str | None = None,
    estimator: Callable = np.mean,
    #----- 공통 파라미터 ------
    title: str | None = None,
    xlabel: str | None = None,
    xlabel_fontsize: int = config.xlabel_fontsize,
    xlabel_fontweight: str = config.xlabel_fontweight,
    xlabel_pad: int = config.xlabel_pad,
    ylabel: str | None = None,
    ylabel_fontsize: int = config.ylabel_fontsize,
    ylabel_fontweight: str = config.ylabel_fontweight,
    ylabel_pad: int = config.ylabel_pad,
    palette: str | None = None,
    width: int | None = config.width,
    height: int | None = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params) -> None:
    """
    막대그래프를 그린다.

    Args:
        data (DataFrame): 시각화할 데이터.
        x (str | Index): 범주 컬럼.
        y (str | Index): 값 컬럼.
        hue (str|None): 보조 범주 컬럼.
        estimator (Callable): 요약 함수.

    Common Args:
        title (str|None): 그래프 제목.
        xlabel (str|None): x축 레이블.
        xlabel_fontsize (int): x축 레이블 폰트 크기.
        xlabel_fontweight (str): x축 레이블 폰트 두께.
        xlabel_pad (int): x축 레이블 패드.
        ylabel (str|None): y축 레이블.
        ylabel_fontsize (int): y축 레이블 폰트 크기.
        ylabel_fontweight (str): y축 레이블 폰트 두께.
        ylabel_pad (int): y축 레이블 패드.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        save_path (str|None): 이미지 저장 경로. None이면 화면에 표시.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn violinplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title, xlabel=xlabel, xlabel_fontsize=xlabel_fontsize, xlabel_fontweight=xlabel_fontweight, xlabel_pad=xlabel_pad, ylabel=ylabel, ylabel_fontsize=ylabel_fontsize, ylabel_fontweight=ylabel_fontweight, ylabel_pad=ylabel_pad)  # type: ignore
        outparams = True

    # hue가 있을 때만 palette 사용, 없으면 color 사용
    barplot_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "estimator": estimator,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        barplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        barplot_kwargs["color"] = sb.color_palette(palette)[0]

    barplot_kwargs.update(params)

    sb.barplot(**barplot_kwargs)

    if callback is not None:
        callback(ax)

    if outparams:
        show(save_path)  # type: ignore


# ===================================================================
# 빈도그래프를 그린다
# ===================================================================
def countplot(
    data: DataFrame,
    x: str | Index,
    y: str | Index,
    hue: str | None = None,
    #----- 공통 파라미터 ------
    title: str | None = None,
    xlabel: str | None = None,
    xlabel_fontsize: int = config.xlabel_fontsize,
    xlabel_fontweight: str = config.xlabel_fontweight,
    xlabel_pad: int = config.xlabel_pad,
    ylabel: str | None = None,
    ylabel_fontsize: int = config.ylabel_fontsize,
    ylabel_fontweight: str = config.ylabel_fontweight,
    ylabel_pad: int = config.ylabel_pad,
    palette: str | None = None,
    width: int | None = config.width,
    height: int | None = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params) -> None:
    """
    # 빈도그래프를 그린다.

    Args:
        data (DataFrame): 시각화할 데이터.
        x (str | Index): 범주 컬럼.
        y (str | Index): 값 컬럼.
        hue (str|None): 보조 범주 컬럼.

    Common Args:
        title (str|None): 그래프 제목.
        xlabel (str|None): x축 레이블.
        xlabel_fontsize (int): x축 레이블 폰트 크기.
        xlabel_fontweight (str): x축 레이블 폰트 두께.
        xlabel_pad (int): x축 레이블 패드.
        ylabel (str|None): y축 레이블.
        ylabel_fontsize (int): y축 레이블 폰트 크기.
        ylabel_fontweight (str): y축 레이블 폰트 두께.
        ylabel_pad (int): y축 레이블 패드.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        save_path (str|None): 이미지 저장 경로. None이면 화면에 표시.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn violinplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title, xlabel=xlabel, xlabel_fontsize=xlabel_fontsize, xlabel_fontweight=xlabel_fontweight, xlabel_pad=xlabel_pad, ylabel=ylabel, ylabel_fontsize=ylabel_fontsize, ylabel_fontweight=ylabel_fontweight, ylabel_pad=ylabel_pad)  # type: ignore
        outparams = True

    # hue가 있을 때만 palette 사용, 없으면 color 사용
    barplot_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        barplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        barplot_kwargs["color"] = sb.color_palette(palette)[0]

    barplot_kwargs.update(params)

    sb.countplot(**barplot_kwargs)

    if callback is not None:
        callback(ax)

    if outparams:
        show(save_path)  # type: ignore


# ===================================================================
# 파이 그래프 혹은 도넛 그래프를 그린다
# ===================================================================
def pieplot(
    x: str | Index,
    labels: str | Index,
    autopct: str = "%0.1f%%",
    startangle: int = 90,
    counterclock: bool = False,
    explode: list[float] | None = None,
    donutchart: bool = False,
    wedge_width: float = 0.7,
    wedge_color: str | None = "#ffffff",
    wedge_linewidth: float = 3,
    #----- 공통 파라미터 ------
    title: str | None = None,
    xlabel: str | None = None,
    xlabel_fontsize: int = config.xlabel_fontsize,
    xlabel_fontweight: str = config.xlabel_fontweight,
    xlabel_pad: int = config.xlabel_pad,
    ylabel: str | None = None,
    ylabel_fontsize: int = config.ylabel_fontsize,
    ylabel_fontweight: str = config.ylabel_fontweight,
    ylabel_pad: int = config.ylabel_pad,
    palette: str | None = None,
    width: int | None = config.width,
    height: int | None = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params) -> None:
    """
    파이 그래프 혹은 도넛 그래프를 그린다

    Args:
        x (str | Index): 값 컬럼.
        labels (str | Index): 범주 컬럼.
        autopct (str): 조각 안에 표시할 값 형식.
        startangle (int): 시작 각도.
        counterclock (bool): 시계 반대 방향으로 그릴지 여부.
        explode (list[float]|None): 조각 간격.
        donutchart (bool): 도넛 차트 여부.
        wedge_width (float): 도넛 차트일 때 조각 너비 비율
        wedge_color (str|None): 도넛 차트일 때 조각 사이 경계선 색상.
        wedge_linewidth (float): 도넛 차트일 때 조각 사이 경계선 굵기.
    Common Args:
        title (str|None): 그래프 제목.
        xlabel (str|None): x축 레이블.
        xlabel_fontsize (int): x축 레이블 폰트 크기.
        xlabel_fontweight (str): x축 레이블 폰트 두께.
        xlabel_pad (int): x축 레이블 패드.
        ylabel (str|None): y축 레이블.
        ylabel_fontsize (int): y축 레이블 폰트 크기.
        ylabel_fontweight (str): y축 레이블 폰트 두께.
        ylabel_pad (int): y축 레이블 패드.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        save_path (str|None): 이미지 저장 경로. None이면 화면에 표시.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn violinplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title, xlabel=xlabel, xlabel_fontsize=xlabel_fontsize, xlabel_fontweight=xlabel_fontweight, xlabel_pad=xlabel_pad, ylabel=ylabel, ylabel_fontsize=ylabel_fontsize, ylabel_fontweight=ylabel_fontweight, ylabel_pad=ylabel_pad)  # type: ignore
        outparams = True

    barplot_kwargs = {
        "x": x,
        "labels": labels,
        "autopct": autopct,
        "startangle": startangle,
        "counterclock": counterclock,
    }

    if palette is not None:
        barplot_kwargs["colors"] = sb.color_palette("Set2", n_colors=len(labels))

    if explode is not None:
        barplot_kwargs["explode"] = explode

    if donutchart:
        barplot_kwargs["wedgeprops"] = {
            "width": wedge_width,
            "edgecolor": wedge_color,
            "linewidth": wedge_linewidth,
        }

    barplot_kwargs.update(params)

    ax.pie(**barplot_kwargs)

    if callback is not None:
        callback(ax)

    if outparams:
        show(save_path)  # type: ignore


# ===================================================================
# 누적 막대 그래프를 그린다.
# ===================================================================
def stackplot(
    data: DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    aggfunc: Callable = np.sum,
    ratio: bool = False,
    orient: str = "v",
    text: bool = True,
    text_color: str = "#ffffff",
    text_fontsize: int = config.text_font_size,
    text_format: str = "0.1f%%",
    #----- 공통 파라미터 ------
    title: str | None = None,
    xlabel: str | None = None,
    xlabel_fontsize: int = config.xlabel_fontsize,
    xlabel_fontweight: str = config.xlabel_fontweight,
    xlabel_pad: int = config.xlabel_pad,
    ylabel: str | None = None,
    ylabel_fontsize: int = config.ylabel_fontsize,
    ylabel_fontweight: str = config.ylabel_fontweight,
    ylabel_pad: int = config.ylabel_pad,
    palette: str | None = None,
    width: int | None = config.width,
    height: int | None = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    ax: Axes | None = None,
    **params) -> None:
    """
    누적 막대 그래프를 그린다.

    Args:
        data (DataFrame): 시각화할 데이터.
        x (str): x축 범주 컬럼명.
        y (str): y축 값 컬럼명.
        hue (str|None): 보조 범주 컬럼명.
        aggfunc (Callable): 집계 함수.
        ratio (bool): 누적 비율로 표시 여부.
        orient (str): 'v' 또는 'h' 방향.
        text (bool): 막대 안에 텍스트 표시 여부.
        text_color (str): 텍스트 색상.
        text_fontsize (int): 텍스트 폰트 크기.
        text_format (str): 텍스트 형식.

    Common Args:
        title (str|None): 그래프 제목.
        xlabel (str|None): x축 레이블.
        xlabel_fontsize (int): x축 레이블 폰트 크기.
        xlabel_fontweight (str): x축 레이블 폰트 두께.
        xlabel_pad (int): x축 레이블 패드.
        ylabel (str|None): y축 레이블.
        ylabel_fontsize (int): y축 레이블 폰트 크기.
        ylabel_fontweight (str): y축 레이블 폰트 두께.
        ylabel_pad (int): y축 레이블 패드.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        save_path (str|None): 이미지 저장 경로. None이면 화면에 표시.
        callback (Callable|None): Axes 후처리 콜백.
        ax (Axes|None): 외부에서 전달한 Axes.
        **params: seaborn violinplot 추가 인자.

    Returns:
        None
    """
    outparams = False

    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title, xlabel=xlabel, xlabel_fontsize=xlabel_fontsize, xlabel_fontweight=xlabel_fontweight, xlabel_pad=xlabel_pad, ylabel=ylabel, ylabel_fontsize=ylabel_fontsize, ylabel_fontweight=ylabel_fontweight, ylabel_pad=ylabel_pad)  # type: ignore
        outparams = True

    # 데이터 피벗팅
    df = pivot_table(data=data, index=x, values=y, columns=hue, aggfunc=aggfunc, fill_value=0)

    if ratio:
        text_format = "{:.1f}%"
        df['sum'] = df.sum(axis=1)

        for col in df.columns:
            df[col] = df[col] / df['sum'] * 100

        df.drop(columns='sum', inplace=True)

        if orient == 'v':
            ax.set_ylim(0, 100)
        else:
            ax.set_xlim(0, 100)
    else:
        text_format = "{:.1f}"

    color_list = None
    if palette is not None:
        color_list = sb.color_palette(palette, n_colors=len(df.columns))

    for i, col in enumerate(df.columns):
        color = None
        
        if color_list is not None:
            color = color_list[i]

        if orient == 'v':
            ax.bar(df.index, df[col], bottom=df.iloc[:, :i].sum(axis=1), color=color, label=col)
        else:
            ax.barh(df.index, df[col], left=df.iloc[:, :i].sum(axis=1), color=color, label=col)

        # 누적값 텍스트 표시
        if text:
            for j, val in enumerate(df[col]):
                if orient == 'v':
                    ax.text(x=j, y=df.iloc[j, :i].sum() + val / 2, s=text_format.format(val), ha='center', va='center', color=text_color, fontsize=text_fontsize)
                else:
                    ax.text(x=df.iloc[j, :i].sum() + val / 2, y=j, s=text_format.format(val), ha='center', va='center', color=text_color, fontsize=text_fontsize)

    ax.legend(bbox_to_anchor=(1, 1))

    if callback is not None:
        callback(ax)

    if outparams:
        show(save_path)  # type: ignore



#######################################################################




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
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
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

    show(save_path)  # type: ignore


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
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
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

    show(save_path)  # type: ignore


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

    show(save_path)  # type: ignore


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
        fig, axes = init(width=width, height=height, rows=n_cols, cols=1, flatten=True, title=title)  # type: ignore
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

    show(save_path)  # type: ignore


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
        fig, ax = init(width=width + 150 if mse else width, height=height, rows=1, cols=1, title=title)  # type: ignore
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

    show(save_path)  # type: ignore


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
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
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

    show(save_path)  # type: ignore



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
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
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
    show(save_path)  # type: ignore


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
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
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

    show(save_path)  # type: ignore



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
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
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

    ax.set_xlabel("The silhouette coefficient values", fontsize=config.label_font_size)  # type: ignore
    ax.set_ylabel("Cluster label", fontsize=config.label_font_size)  # type: ignore
    ax.set_xlim([-0.1, 1])  # type: ignore
    ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])  # type: ignore
    ax.set_yticks([])  # type: ignore
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])  # type: ignore

    if title is None:
        title = "Number of Cluster : " + str(n_clusters) + ", Silhouette Score :" + str(round(sil_avg, 3))  # type: ignore

    show(save_path)  # type: ignore


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
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
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
        ax.set_xlabel("Feature space for the " + xname, fontsize=config.label_font_size)
        ax.set_ylabel("Feature space for the " + yname, fontsize=config.label_font_size)

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
    fig, ax = init(rows=1, cols=2, width=width, height=height, title=title)

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

    show(save_path)  # type: ignore



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
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title)  # type: ignore
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

    show(save_path)  # type: ignore


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
