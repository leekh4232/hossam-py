from typing import Literal
from kneed import KneeLocator
from pandas import Series
from matplotlib.pyplot import Axes # type: ignore

from . import hs_plot

import numpy as np

def elbow_point(
        x: Series | np.ndarray | list,
        y: Series | np.ndarray | list,
        dir: Literal["left,down", "left,up", "right,down", "right,up"] = "left,down",
        S: float = 0.1,
        plot: bool = True,
        title: str = None,
        marker: str = None,
        width: int = hs_plot.config.width,
        height: int = hs_plot.config.height,
        dpi: int = hs_plot.config.dpi,
        linewidth: int = hs_plot.config.line_width,
        save_path: str | None = None,
        ax: Axes | None = None,
        **params,
) -> tuple:

    """
    엘보우(Elbow) 포인트를 자동으로 탐지하는 함수.

    주어진 x, y 값의 곡선에서 KneeLocator를 활용해 엘보우(혹은 니) 포인트를 탐지하고, 필요시 시각화까지 지원함.

    Args:
        x (Series | np.ndarray | list): x축 값(일반적으로 K값 등).
        y (Series | np.ndarray | list): y축 값(일반적으로 inertia, SSE 등).
        dir (Literal["left,down", "left,up", "right,down", "right,up"], optional):
            곡선의 방향 및 형태 지정. 기본값은 "left,down".
            - "left,down": 왼쪽에서 오른쪽으로 감소(볼록)
            - "left,up": 왼쪽에서 오른쪽으로 증가(오목)
            - "right,down": 오른쪽에서 왼쪽으로 감소(볼록)
            - "right,up": 오른쪽에서 왼쪽으로 증가(오목)
        S (float, optional): KneeLocator의 민감도 파라미터. 기본값 0.1.
        plot (bool, optional): True면 결과를 시각화함. 기본값 True.
        title (str, optional): 플롯 제목.
        marker (str, optional): 마커 스타일.
        width (int, optional): 플롯 가로 크기.
        height (int, optional): 플롯 세로 크기.
        dpi (int, optional): 플롯 해상도.
        linewidth (int, optional): 선 두께.
        save_path (str | None, optional): 저장 경로 지정시 파일로 저장.
        ax (Axes | None, optional): 기존 matplotlib Axes 객체. None이면 새로 생성.
        **params: lineplot에 전달할 추가 파라미터.

    Returns:
        tuple: (best_x, best_y)
            - best_x: 엘보우 포인트의 x값(예: 최적 K)
            - best_y: 엘보우 포인트의 y값

    Examples:
        ```python
        x = [1, 2, 3, 4, 5, 6]
        y = [100, 80, 60, 45, 44, 43]
        elbow_point(x, y)
        ```

    Note:
        - KneeLocator는 kneed 패키지의 클래스로, 곡선의 형태(curve)와 방향(direction)에 따라 엘보우 포인트를 탐지함.
        - dir 파라미터에 따라 curve/direction이 자동 지정됨.
        - plot=True일 때, 엘보우 포인트에 수직/수평선과 텍스트가 표시됨.
    """

    if dir == "left,down":
        curve = "convex"
        direction = "decreasing"
    elif dir == "left,up":
        curve = "concave"
        direction = "increasing"
    elif dir == "right,down":
        curve = "convex"
        direction = "increasing"
    else:
        curve = "concave"
        direction = "decreasing"

    kn = KneeLocator(x=x, y=y, curve=curve, direction=direction, S=S)

    best_x = kn.elbow
    best_y = kn.elbow_y

    if plot:
        def hvline(ax):
            ax.axvline(best_x, color="red", linestyle="--", linewidth=0.7)
            ax.axhline(best_y, color="red", linestyle="--", linewidth=0.7)
            ax.text(
                best_x + 0.1,
                best_y + 0.1,
                "Best K=%d" % best_x,
                fontsize=8,
                ha="left",
                va="bottom",
                color="r",
            )

        hs_plot.lineplot(
                df = None,
                xname = x,
                yname = y,
                title = title,
                marker = marker,
                width = width,
                height = height,
                linewidth = linewidth,
                dpi = dpi,
                save_path = save_path,
                callback = hvline,
                ax = ax,
                **params
        )

    return best_x, best_y