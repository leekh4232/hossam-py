# -*- coding: utf-8 -*-
from __future__ import annotations
from types import SimpleNamespace
from typing import Callable

# ===================================================================
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes  # type: ignore
import matplotlib.patches as patches
import matplotlib as mpl
from pandas import Index, Series, DataFrame, pivot_table

# ===================================================================
from scipy.spatial import ConvexHull

# ===================================================================
from .my_stats import ci

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
    scatter_edge_linewidth=1.5,
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
        config.line_width = config.line_width * (dpi * 0.0011 + 0.7)
        config.grid_width = config.grid_width * (dpi * 0.0011 + 0.7)
        config.scatter_edge_linewidth = config.scatter_edge_linewidth * (dpi * 0.0011 + 0.7)
    elif dpi > 100:
        config.font_size = config.font_size * (dpi * 0.0012 + 0.75)
        config.text_font_size = config.text_font_size * (dpi * 0.0012 + 0.75)
        config.title_font_size = config.title_font_size * (dpi * 0.0012 + 0.75)
        config.title_pad = config.title_pad * (dpi * 0.0012 + 0.75)
        config.label_font_size = config.label_font_size * (dpi * 0.0012 + 0.75)
        config.line_width = config.line_width * (dpi * 0.0012 + 0.75)
        config.grid_width = config.grid_width * (dpi * 0.0012 + 0.75)
        config.scatter_edge_linewidth = config.scatter_edge_linewidth * (dpi * 0.0012 + 0.75)
    else:
        config.font_size = 10
        config.text_font_size = 8
        config.title_font_size = 18
        config.title_pad = 15
        config.label_font_size = 14
        config.line_width = 2
        config.grid_width = 1
        config.scatter_edge_linewidth = 1.5

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
    flatten: bool = True, 
    twinx: bool =False,
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
        twinx (bool): twinx 서브플롯 생성 여부. True면 마지막 서브플롯에 twinx Axes 추가.
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

    else:
        if title:
            fig.suptitle(title, fontsize=config.title_font_size * (rows * cols / 2), fontweight=config.title_font_weight) # type: ignore

        # Grid, 테두리 굵기 설정
        for f in ax.flatten():
            f.grid(grid, alpha=config.grid_alpha, linewidth=config.grid_width) # type: ignore
            for spine in f.spines.values(): # type: ignore
                spine.set_linewidth(config.frame_width)
        
        fig.subplots_adjust(wspace=ws, hspace=hs)

        if flatten == True:
            ax = ax.flatten()

    if twinx:
        ax_right = ax.twinx()
        ax = (ax, ax_right)

    return fig, ax


# ===================================================================
# 그래프의 그리드, 레이아웃을 정리하고 필요 시 저장 또는 표시한다
# ===================================================================
def show(save_path: str | None = None) -> None:
    """공통 후처리를 수행한다: 콜백 실행, 레이아웃 정리, 필요 시 표시/종료.

    Args:
        save_path (str|None): 이미지 저장 경로. None이 아니면 해당 경로로 저장.
    Returns:
        None
    """
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.tight_layout()
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
    """
    선 그래프를 그린다.

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

def _kde_peaks(ax):
    """
    ax에 그려진 KDE 곡선들의 최고점 y값을 곡선별로 반환하는 보조 함수.
    fill=False면 곡선이 Line2D로, fill=True면 PolyCollection으로 그려지므로 둘 다 살핀다.

    주의: 반환되는 순서는 seaborn이 곡선을 '그린 순서'다. hue가 적용되면 seaborn은
    범주를 역순으로 그리므로(첫 범주가 맨 위), data[hue].unique() 순서와 맞추려면
    호출 측에서 리스트를 뒤집어야 한다.

    Args:
        ax: KDE 곡선이 그려진 Axes 객체.

    Returns:
        곡선별 최고점 y값 리스트 (그린 순서). 곡선을 찾지 못하면 빈 리스트.
    """
    peaks = []

    if ax.lines:
        # fill=False인 경우: 곡선이 Line2D로 그려진다 (범주당 한 개)
        for line in ax.lines:
            yd = line.get_ydata()
            if len(yd):
                peaks.append(float(np.nanmax(yd)))
    else:
        # fill=True인 경우: 곡선이 채워진 영역(PolyCollection)으로 그려진다
        for col in ax.collections:
            ymax = 0.0
            for path in col.get_paths():
                v = path.vertices
                if len(v):
                    ymax = max(ymax, float(np.nanmax(v[:, 1])))
            peaks.append(ymax)

    return peaks

def _draw_ci(ax, interval, color, ymax):
    """
    kdeplot에서 단일 신뢰구간(하한~상한)을 지정한 색상으로 그리는 보조 함수

    Args:
        ax: 그래프를 그릴 Axes 객체.
        interval: (신뢰구간 하한, 신뢰구간 상한) 튜플.
        color: 신뢰구간 선/텍스트/영역에 적용할 색상.
        ymax: 영역 채우기와 텍스트 위치 계산에 사용할 y축 상한.
    """
    cmin, cmax = interval

    # 신뢰구간 범위에 대한 세로 직선 그리기 (cmin ~ cmax)
    ax.axvline(cmin, linestyle=':', color=color, linewidth=0.5)
    ax.axvline(cmax, linestyle=':', color=color, linewidth=0.5)

    # 신뢰구간 범위에 대한 텍스트 추가
    ax.text(cmin, ymax * 0.9, f'{cmin:.2f}', color=color, fontsize=11, ha='right', va='center')
    ax.text(cmax, ymax * 0.9, f'{cmax:.2f}', color=color, fontsize=11, ha='left', va='center')

    # 신뢰구간 범위에 대한 영역 채우기 (cmin ~ cmax)
    ax.fill_between([cmin, cmax], 0, ymax, alpha=0.1, color=color)

def kdeplot(
    data: DataFrame,
    x: str | None = None,
    hue: str | None = None,
    meanline: bool = False,
    clevel: float = 0,
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
    """단변량 커널 밀도 추정(KDE) 그래프를 그린다. 평균선은 hue가 설정된 경우 지원하지 않는다.

    quartile_split=True일 때는 사분위수 구간(Q1~Q4)으로 나누어 4개의 서브플롯에 그린다.

    Args:
        data (DataFrame): 시각화할 데이터.
        x (str|None): x축 컬럼명.
        hue (str|None): 범주 컬럼명.
        meanline (bool): 평균선 표시 여부.
        clevel (float): 신뢰구간 수준 (0~100). 0이면 신뢰구간 표시 안 함.
        fill (bool): 면적 채우기 여부.
        alpha (float): 채움 투명도.
        linewidth (float): 선 굵기.
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

    if hue is None:
        palette = None  # hue가 지정되지 않았는데 palette가 지정된 경우, 무의미하므로 None으로 설정

    if ax is None:
        fig, ax = init(width=width, height=height, rows=1, cols=1, title=title, xlabel=xlabel, ylabel=ylabel)  # type: ignore
        outparams = True

    # 기본 kwargs 설정
    kdeplot_kwargs = {
        "data": data,
        "x": x,
        "hue": hue,
        "fill": fill,
        "linewidth": linewidth,
        "palette": palette,
        "ax": ax
    }

    if fill:
        kdeplot_kwargs["alpha"] = alpha

    # 커널밀도 추정 그래프 그리기
    sb.kdeplot(**kdeplot_kwargs)

    # 평균선 텍스트를 곡선 최고점에 맞추기 위해, 곡선만 그려진 시점에 최고점을 미리 구해둔다
    # (이후 clevel 블록이 axvline/fill_between으로 ax.lines, ax.collections를 오염시키기 전에 캡처)
    kde_peaks = _kde_peaks(ax) if meanline else None

    # 평균선 표시
    if meanline:
        ymin, ymax = ax.get_ylim()

        # 곡선 위에 둘 여백: peak에 비례시키지 않고 y축 높이에 비례한 고정값을 더한다.
        # → 곡선 높이와 무관하게 그래프마다(그리고 범주마다) 동일한 시각적 간격이 된다.
        gap = (ymax - ymin) * 0.02
        text_top = ymin  # 텍스트가 놓인 최고 높이 (영역 이탈 방지용으로 추적)

        if hue is None:
            # 그래프에 적용된 팔레트의 첫 번째 색상을 따른다 (팔레트가 없으면 기본 파란색)
            color = sb.color_palette(palette)[0] if palette else '#0066ff'

            # 텍스트 높이를 실제 KDE 곡선의 최고점 + 고정 여백에 맞춘다
            peak = max(kde_peaks) if kde_peaks else ymax

            mv = data[x].mean()
            ax.axvline(x=mv, color=color, linestyle='--',
                       linewidth=linewidth * 0.5)
            ty = peak + gap
            ax.text(x=mv + 0.05, y=ty, s=f'Mean: {mv:.2f}', color=color, fontsize=14, fontweight=500, ha='center', va='center')
            text_top = max(text_top, ty)
        else:
            # hue 범주별 평균선 표시 (kdeplot이 그린 라인의 색상과 일치시킴)
            categories = list(data[hue].unique())

            # 팔레트에서 범주의 수에 맞는 색상값 추출
            colors = sb.color_palette(palette, n_colors=len(categories))

            # 곡선별 최고점을 범주 순서에 맞춘다.
            # seaborn은 범주를 역순으로 그리므로 뒤집어야 unique() 순서와 정렬된다.
            # (곡선 수가 범주 수와 다르면 매칭이 어긋날 수 있으므로 전역 최고점으로 대체)
            if kde_peaks and len(kde_peaks) == len(categories):
                cat_peaks = list(reversed(kde_peaks))
            else:
                fallback = max(kde_peaks) if kde_peaks else ymax
                cat_peaks = [fallback] * len(categories)

            # 각 범주에 대해 평균선 표시 (텍스트는 해당 범주 곡선의 최고점 + 고정 여백에 위치)
            for i, cat in enumerate(categories):
                mv = data.loc[data[hue] == cat, x].mean()
                ax.axvline(x=mv, color=colors[i], linestyle='--', linewidth=linewidth * 0.5)
                ty = cat_peaks[i] + gap
                ax.text(x=mv + 0.05, y=ty, s=f'{cat} Mean: {mv:.2f}', color=colors[i], fontsize=14, fontweight=500, ha='center', va='center')
                text_top = max(text_top, ty)

        # 텍스트(글자 높이 포함)가 그래프 영역을 벗어나지 않도록 y축 상한을 확장한다
        ax.set_ylim(ymin, max(ymax, text_top + gap * 2))

    # 신뢰구간 표시 (신뢰수준이 0이 아닌 경우에만)
    if clevel:
        ymin, ymax = ax.get_ylim()

        if hue is None:
            # 그래프에 적용된 팔레트의 첫 번째 색상을 따른다 (팔레트가 없으면 기본 파란색)
            color = sb.color_palette(palette)[0] if palette else '#0066ff'

            # 전체 데이터에 대한 신뢰구간 표시
            _draw_ci(ax, ci(data, column=x, clevel=clevel), color, ymax)
        else:
            # hue 범주별로 신뢰구간 표시 (kdeplot이 그린 라인의 색상과 일치시킴)
            categories = list(data[hue].unique())

            # 팔레트에서 범주의 수에 맞는 색상값 추출
            colors = sb.color_palette(palette, n_colors=len(categories))

            # 각 범주에 대해 신뢰구간 표시
            for i, cat in enumerate(categories):
                cdata = data.loc[data[hue] == cat, x]
                _draw_ci(ax, ci(cdata, clevel=clevel), colors[i], ymax)

        ax.set_ylim(ymin, ymax)  # y축 범위 유지

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

    # 구간 산정
    if isinstance(bins, int):
        hist, bins = np.histogram(data[x], bins=bins)
        bins = np.round(bins, 1)
        ax.set_xticks(bins, bins)
    elif isinstance(bins, (list, np.ndarray)):
        ax.set_xticks(bins, bins)

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
    order: list | None = None,
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
        order (list|None): x축 범주 순서. None이면 데이터에 나타난 순서대로.
        stat_test (str|None): 통계 검정 방법. None이면 검정 안함. x과 y가 모두 지정되어야 함.
        stat_pairs (list[tuple]|None): 통계 검정할 그룹 쌍 목록.
        stat_text_format (str): 통계 결과 표시 형식.
        stat_loc (str): 통계 결과 위치.
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
    # elif hue_value is None and palette is not None:
    #     boxplot_kwargs["color"] = sb.color_palette(palette)[0]

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
    order: list | None = None,
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
        order (list|None): x축 범주 순서. None이면 데이터에 나타난 순서.
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
    kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "estimator": estimator,
        "order": order,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        kwargs["palette"] = palette
    elif hue is None and palette is not None:
        kwargs["color"] = sb.color_palette(palette)[0]

    kwargs.update(params)

    sb.barplot(**kwargs)

    if callback is not None:
        callback(ax)

    if outparams:
        show(save_path)  # type: ignore


# ===================================================================
# 빈도그래프를 그린다
# ===================================================================
def countplot(
    data: DataFrame,
    x: str | Index = None,
    y: str | Index = None,
    hue: str | None = None,
    order: list | None = None,
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
    빈도그래프를 그린다.

    Args:
        data (DataFrame): 시각화할 데이터.
        x (str | Index): 범주 컬럼.
        y (str | Index): 값 컬럼.
        hue (str|None): 보조 범주 컬럼.
        order (list|None): x축 범주 순서. None이면 데이터에 나타난 순서.
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
    kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "order": order,
        "ax": ax,
    }

    if hue is not None and palette is not None:
        kwargs["palette"] = palette
    elif hue is None and palette is not None:
        kwargs["color"] = sb.color_palette(palette)[0]

    kwargs.update(params)

    sb.countplot(**kwargs)

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

    kwargs = {
        "x": x,
        "labels": labels,
        "autopct": autopct,
        "startangle": startangle,
        "counterclock": counterclock,
    }

    if palette is not None:
        kwargs["colors"] = sb.color_palette("Set2", n_colors=len(labels))

    if explode is not None:
        kwargs["explode"] = explode

    if donutchart:
        kwargs["wedgeprops"] = {
            "width": wedge_width,
            "edgecolor": wedge_color,
            "linewidth": wedge_linewidth,
        }

    kwargs.update(params)

    ax.pie(**kwargs)

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
    orient: str = "v",
    ratio: bool = False,
    text: bool = True,
    text_color: str = "#ffffff",
    text_fontsize: int = config.text_font_size,
    text_format: str = None,
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
        orient (str): 'v' 또는 'h' 방향.
        ratio (bool): 누적 비율로 표시 여부.
        text (bool): 막대 안에 텍스트 표시 여부.
        text_color (str): 텍스트 색상.
        text_fontsize (int): 텍스트 폰트 크기.
        text_format (str): 텍스트 형식.
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

    # 데이터 피벗팅 (fill_value=0 --> 결측치를 0으로 채움)후 인덱스를 문자열 카테고리로 변환
    df = pivot_table(data=data, index=x, values=y, columns=hue, aggfunc=aggfunc, fill_value=0)
    df.index = df.index.astype("str").astype("category")

    # 누적값을 비율로 변환하는 경우
    if ratio:
        if text_format is None:                     # 텍스트 포멧이 없다면 강제 지정
            text_format = "{:.1f}%"
        
        df['sum'] = df.sum(axis=1)                  # 각 행의 합 계산하여 'sum' 열에 저장

        for col in df.columns:                      # 각 열에 대해 누적값을 비율로 변환
            df[col] = df[col] / df['sum'] * 100

        df.drop(columns='sum', inplace=True)        # 'sum' 열 제거

        if orient == 'v':                           # 그래프 방향에 따라 축 범위 설정
            ax.set_ylim(0, 100)
        else:
            ax.set_xlim(0, 100)
    else:
        if text_format is None:                     # 텍스트 포멧이 없다면 강제 지정
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
                if val == 0:  # 누적값이 0인 경우 텍스트 표시 안함
                    continue

                if orient == 'v':
                    ax.text(x=j, y=df.iloc[j, :i].sum() + val / 2, s=text_format.format(val), ha='center', va='center', color=text_color, fontsize=text_fontsize)
                else:
                    ax.text(x=df.iloc[j, :i].sum() + val / 2, y=j, s=text_format.format(val), ha='center', va='center', color=text_color, fontsize=text_fontsize)

    ax.legend(bbox_to_anchor=(1, 1))

    if callback is not None:
        callback(ax)

    if outparams:
        show(save_path)  # type: ignore


# ===================================================================
# 산점도 그래프를 그린다
# ===================================================================
def scatterplot(
    data: DataFrame,
    x: str | Index,
    y: str | Index,
    hue: str | None = None,
    marker: str = "o",
    color: str | None = None,
    size: int = 100,
    edgecolor: str = "#ffffff",
    linewidth: float = config.scatter_edge_linewidth,
    alpha: float = 1.0,
    outline: bool = True,
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
    산점도 그래프를 그린다.

    Args:
        data (DataFrame): 시각화할 데이터.
        x (str | Index): x축 값 컬럼명.
        y (str | Index): y축 값 컬럼명.
        hue (str|None): 범주 구분 컬럼명.
        marker (str): 점 모양.
        color (str|None): 점 색상. hue가 있을 때는 무시됨.
        size (int): 점 크기.
        edgecolor (str): 점 외곽선 색상.
        linewidth (float): 점 외곽선 굵기.
        alpha (float): 점 투명도.
        outline (bool): 점 외곽선 표시 여부.
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
    kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "marker": marker,
        "s": size,
        "edgecolor": edgecolor,
        "linewidth": linewidth,
        "alpha": alpha,
    }

    # 군집을 구분할 분류값이 없다면 palette 옵션이 무의미하므로 None으로 설정
    if hue == None:
        if color is None and palette is not None:
            kwargs["color"] = sb.color_palette(palette)[0]
        else:
            kwargs["color"] = color
    else:
        kwargs["palette"] = palette
        

    kwargs.update(params)

    sb.scatterplot(**kwargs)

    if outline and hue is not None:
        plot_hull(data=data, x=x, y=y, hue=hue, palette=palette, ax=ax)

    if callback is not None:
        callback(ax)

    if outparams:
        show(save_path)  # type: ignore


# ===================================================================
# ConvexHull을 이용하여 각 군집의 외곽선을 그리는 함수
# ===================================================================
def plot_hull(data: DataFrame, 
              x: str, 
              y: str, 
              hue: str, 
              palette: str, 
              ax: Axes) -> None:
    """
    ConvexHull을 이용하여 각 군집의 외곽선을 그리는 함수

    Args:
        data (DataFrame): 시각화할 데이터.
        x (str): x축 값 컬럼명.
        y (str): y축 값 컬럼명.
        hue (str): 범주 구분 컬럼명.
        palette (str): 팔레트 이름.
        ax (Axes): 외부에서 전달한 Axes.
    """

    # 데이터의 군집 종류 얻기
    classes = list(data[hue].unique())
    
    # 각 클래스에 대하여 반복 수행
    for i, v in enumerate(classes):
        # 현재 클래스에 해당하는 데이터 포인트 추출
        df_c = data.loc[data[hue] == v, [x, y]]

        # ConvexHull은 3개 이상의 점이 필요하므로, 데이터 포인트가 3개 미만인 경우 중단해야 함
        if len(df_c) < 3:
            continue

        hull = ConvexHull(df_c)
        points = np.append(hull.vertices, hull.vertices[0])

        # 현재 클래스에 적용될 색상값 생성
        color = sb.color_palette(palette)[i]

        # points를 index로 하는 데이터 포인트를 선과 면으로 표시
        ax.plot(df_c.iloc[points, 0], df_c.iloc[points, 1], linewidth=1, linestyle=":", color=color)
        ax.fill(df_c.iloc[points, 0], df_c.iloc[points, 1], alpha=0.1, color=color)




# ===================================================================
# 범주별 회귀선이 표시된 선형 모델 그래프를 그린다
# ===================================================================
def lmplot(
    data: DataFrame,
    x: str,
    y: str,
    hue=None,
    col: str | None = None,
    row: str | None = None,
    markers: str | list[str] = "o",
    scatter_edgecolor: str | None = "#ffffff",
    scatter_linewidths: float = 1,
    scatter_size: int = 50,
    scatter_alpha: float = 0.8,
    linestyle: str = "-",
    linecolor: str | None = None,
    linewidth: float = 2,
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
    **params,
) -> None:
    """seaborn lmplot으로 선형 모델 시각화를 수행한다.

    Args:
        data (DataFrame): 시각화할 데이터.
        x (str): 독립변수 컬럼.
        y (str): 종속변수 컬럼.
        hue (str|None): 범주 컬럼.
        col (str|None): 열 패싯 컬럼.
        row (str|None): 행 패싯 컬럼.
        markers (str|list[str]): 산점도 점 모양.
        scatter_edgecolor (str|None): 산점도 점 외곽선 색상.
        scatter_linewidths (float): 산점도 점 외곽선 굵기
        scatter_size (int): 산점도 점 크기.
        scatter_alpha (float): 산점도 점 투명도.
        linestyle (str): 회귀선 스타일.
        linecolor (str|None): 회귀선 색상. hue가 있을 때는 무시됨.
        linewidth (float): 회귀선 굵기.
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
        **params: seaborn lmplot 추가 인자.

    Returns:
        None
    """
    w = width / 100
    h = height / 100

    if not hue and palette:
        palette = None
        linecolor = None

    # hue가 있을 때만 palette 사용, 없으면 scatter_kws에 color 설정
    lmplot_kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "col": col,
        "row": row,
        "height": h,
        "aspect": w / h,
        "legend": False,
        "markers": markers,
        "scatter_kws": {
            "edgecolor": scatter_edgecolor,
            "linewidths": scatter_linewidths,
            "s": scatter_size,
            "alpha": scatter_alpha,
        },
        "line_kws": {
            "color": linecolor,
            "linestyle": linestyle,
            "linewidth": linewidth,
        },
    }

    if hue is not None and palette is not None:
        lmplot_kwargs["palette"] = palette
    elif hue is None and palette is not None:
        lmplot_kwargs["scatter_kws"] = {"color": sb.color_palette(palette)[0]}

    lmplot_kwargs.update(params)

    g = sb.lmplot(**lmplot_kwargs)
    g.fig.set_dpi(config.dpi)

    if title:
        g.fig.suptitle(title, fontsize=config.title_font_size, fontweight=config.title_font_weight, y=1)

    for a in g.axes.flatten():
        a.set_xlabel(xlabel or x, fontsize=xlabel_fontsize, fontweight=config.xlabel_fontweight, labelpad=xlabel_pad)
        a.set_ylabel(ylabel or y, fontsize=ylabel_fontsize, fontweight=config.ylabel_fontweight, labelpad=ylabel_pad)
        a.grid(True, alpha=config.grid_alpha)
        a.set_axisbelow(True)

        if hue is not None:
            a.legend(bbox_to_anchor=(1, 1), loc='upper left') # 범례 위치 조정

        if callback is not None:
            callback(a)

    plt.tight_layout()

    show(save_path)  # type: ignore





# ===================================================================
# 산점도 행렬 시각화
# ===================================================================
def pairplot(
    data: DataFrame,
    x: str | list[str] | None = None,
    y: str | list[str] | None = None,
    hue=None,
    diag_kind: str = "kde",
    reg: bool = False,
    markers: str | list[str] = "o",
    scatter_size: int = 20, 
    scatter_alpha: float = 0.8,
    linecolor: str | None = None, 
    linewidth: float = 1.5, 
    linestyle: str = "-",
    #----- 공통 파라미터 ------
    title: str | None = None,
    palette: str | None = None,
    width: int | None = config.width,
    height: int | None = config.height,
    save_path: str | None = None,
    callback: Callable | None = None,
    **params,
) -> None:
    """
    산점도 행렬 시각화

    Args:
        data (DataFrame): 시각화할 데이터.
        x (str|list[str]|None): 대상 컬럼명.
            - None: 모든 연속형(숫자형) 데이터에 대해 처리.
            - str: 해당 컬럼에 대해서만 처리.
            - list: 주어진 컬럼들에 대해서만 처리.
            기본값은 None.
        y (str|list[str]|None): 대상 컬럼명.
            - None: 모든 연속형(숫자형) 데이터에 대해 처리.
            - str: 해당 컬럼에 대해서만 처리.
            - list: 주어진 컬럼들에 대해서만 처리.
            기본값은 None.
        hue (str|None): 범주 컬럼명. None이면 범주 구분 없이 하나의 색상으로 표시.
        diag_kind (str): 대각선 그래프 종류. 'hist' 또는 'kde'.
        reg (bool): 회귀선 표시 여부. True이면 산점도에 회귀선이 추가되고 diag_kind는 'kde'로 고정됨.
        markers (str|list[str]): 산점도 점 모양. str이면 모든 그래프에 동일한 모양이 적용되고, list이면 hue의 범주별로 순서대로 적용됨.
        scatter_size (int): 산점도 점 크기.
        scatter_alpha (float): 산점도 점 투명도.
        linecolor (str|None): 회귀선 색상
        linewidth (float): 회귀선 굵기.
        linestyle (str): 회귀선 스타일.
        title (str|None): 그래프 제목.
        palette (str|None): 팔레트 이름.
        width (int): 캔버스 가로 픽셀.
        height (int): 캔버스 세로 픽셀.
        save_path (str|None): 이미지 저장 경로. None이면 화면에 표시.
        callback (Callable|None): Axes 후처리 콜백.
        **params: seaborn pairplot 추가 인자.

    Returns:
        None
    """
    # 1) 그래프 초기화
    w = width / 100             # 가로 크기
    h = height / 100            # 세로 크기

    # hue가 지정되지 않았는데 palette와 linecolor가 지정된 경우, 무의미하므로 None으로 설정
    if not hue and palette:
        palette = None

    # 회귀선의 표시 여부에 따라서 plot_kws 분기
    if reg:
        plot_kws = {
            "scatter_kws": { "s": scatter_size, "alpha": scatter_alpha},
            "line_kws": { "color": linecolor, "linewidth": linewidth, "linestyle": linestyle}
        }
    else:
        plot_kws = { "s": scatter_size, "alpha": scatter_alpha }

    # hue가 있을 때만 palette 사용
    pairplot_kwargs = {
        "data": data,
        "hue": hue,
        "markers": markers,
        "palette": palette,
        "kind": "reg" if reg else "scatter",
        "diag_kind": diag_kind,
        "x_vars": x,
        "y_vars": y,
        "plot_kws": plot_kws
    }

    pairplot_kwargs.update(params)

    g = sb.pairplot(**pairplot_kwargs)
    g.fig.set_dpi(config.dpi)
    g.fig.set_figwidth(w)
    g.fig.set_figheight(h)

    if title:
        g.fig.suptitle(title, fontsize=config.title_font_size, fontweight=config.title_font_weight, y=1.02)

    # 3) 개별 그래프 설정
    for ax in g.axes.flatten():
        ax.set_axisbelow(True)  # 격자를 그래프 뒤로 이동
        ax.grid(True, alpha=config.grid_alpha)  # 격자 추가

    show(save_path)  # type: ignore
