import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

from .config import API_TODAY, API_TOMORROW, PARSE_TTL_SEC, V_THRESHOLD, log
from .image import get_image_cached

# ================= FIXED GRID (ANCHOR + STEP) =================


@dataclass
class FixedGrid:
    cols: int = 24
    rows: int = 12

    # !!! ПІДСТАВ СВОЇ ПРАВИЛЬНІ ЦЕНТРИ !!!
    x0_top: int = 205
    y0_top: int = 450

    x0_bot: int = 205
    y0_bot: int = 1340

    dx: int = 65
    dy: int = 60


GRID = FixedGrid()


def _row_index(queue: int, subqueue: int) -> int:
    return (queue - 1) * 2 + (subqueue - 1)


def _cell_center(half: int, col: int, row: int, grid: FixedGrid) -> Tuple[int, int]:
    if half == 0:
        x0, y0 = grid.x0_top, grid.y0_top
    else:
        x0, y0 = grid.x0_bot, grid.y0_bot
    return x0 + col * grid.dx, y0 + row * grid.dy


def _states_by_threshold(v_vals: List[int], thr: int) -> List[bool]:
    """True = нема світла (темна клітинка)."""
    return [v < thr for v in v_vals]


def read_all_rows_states(img: np.ndarray, grid: FixedGrid) -> List[List[bool]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2]  # uint8

    states_all: List[List[bool]] = []

    for row in range(grid.rows):
        vals: List[int] = []

        for c in range(grid.cols):  # 00-12
            cx, cy = _cell_center(0, c, row, grid)
            vals.append(int(V[cy, cx]))

        for c in range(grid.cols):  # 12-24
            cx, cy = _cell_center(1, c, row, grid)
            vals.append(int(V[cy, cx]))

        states_all.append(_states_by_threshold(vals, V_THRESHOLD))

    return states_all


def intervals_from_states(states_48: List[bool]) -> List[Tuple[str, str]]:
    intervals = []
    start = None
    for i, off in enumerate(states_48):
        if off and start is None:
            start = i
        if (not off) and start is not None:
            intervals.append((start, i))
            start = None
    if start is not None:
        intervals.append((start, 48))

    def slot_to_time(s: int) -> str:
        if s == 48:
            return "24:00"
        m = s * 30
        return f"{m // 60:02d}:{m % 60:02d}"

    return [(slot_to_time(a), slot_to_time(b)) for a, b in intervals]


def states_to_text(states_48: List[bool]) -> str:
    # "1" = off, "0" = on
    return "".join("1" if x else "0" for x in states_48)


def make_fingerprint(q: int, sq: int, row_states: List[bool]) -> str:
    payload = (q, sq, tuple(row_states))
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()


def hash_intervals_by_row(intervals_by_row: List[List[Tuple[str, str]]]) -> str:
    payload = "|".join(
        ",".join(f"{a}-{b}" for a, b in row)
        for row in intervals_by_row
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ================= CACHES (per endpoint) =================

_parsed_cache: Dict[str, Dict[str, Any]] = {
    API_TODAY: {"ts": 0.0, "url": None, "states": None, "intervals_by_row": None},
    API_TOMORROW: {"ts": 0.0, "url": None, "states": None, "intervals_by_row": None},
}


def get_states_cached(api_url: str) -> Tuple[Optional[str], Optional[List[List[bool]]]]:
    """
    Парсить (або віддає з кешу) матрицю states 12x48.
    """
    now = time.time()
    pc = _parsed_cache.setdefault(api_url, {"ts": 0.0, "url": None, "states": None, "intervals_by_row": None})

    if pc["states"] is not None and pc["url"] is not None and (now - pc["ts"]) < PARSE_TTL_SEC:
        return pc["url"], pc["states"]

    try:
        url, img = get_image_cached(api_url)
        if not url or img is None:
            return None, None

        if pc["states"] is not None and pc["url"] == url and (now - pc["ts"]) < PARSE_TTL_SEC:
            return url, pc["states"]

        states = read_all_rows_states(img, GRID)
        intervals_by_row = [intervals_from_states(r) for r in states]

        pc["ts"] = now
        pc["url"] = url
        pc["states"] = states
        pc["intervals_by_row"] = intervals_by_row
        return url, states

    except requests.exceptions.RequestException as e:
        if pc["states"] is not None and pc["url"] is not None:
            log.warning("Network error, using cached states for %s: %s", api_url, e)
            return pc["url"], pc["states"]

        log.warning("Network error, no cache available for %s: %s", api_url, e)
        return None, None


def get_intervals_by_row_cached(api_url: str) -> Tuple[Optional[str], Optional[List[List[Tuple[str, str]]]]]:
    """
    Кешований доступ до інтервалів для всіх 12 рядків.
    """
    now = time.time()
    pc = _parsed_cache.setdefault(api_url, {"ts": 0.0, "url": None, "states": None, "intervals_by_row": None})

    if pc.get("intervals_by_row") is not None and pc.get("url") is not None and (now - pc["ts"]) < PARSE_TTL_SEC:
        return pc["url"], pc["intervals_by_row"]

    url, states = get_states_cached(api_url)
    if not url or not states:
        return None, None

    if pc.get("intervals_by_row") is None:
        pc["intervals_by_row"] = [intervals_from_states(r) for r in states]

    return pc["url"], pc["intervals_by_row"]
