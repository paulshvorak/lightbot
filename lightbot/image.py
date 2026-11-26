import os
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .config import (
    API_TODAY,
    API_TOMORROW,
    CACHE_TTL_SEC,
    SESSION,
    TEST_MODE,
    TEST_TODAY_IMAGE,
    TEST_TOMORROW_IMAGE,
    API_BASE,
)

# ================= API / IMAGE =================


def fetch_latest_image_url_from_api(api_url: str) -> Optional[str]:
    r = SESSION.get(api_url, timeout=30)
    r.raise_for_status()
    data = r.json()

    def _normalize(v: Optional[str]) -> Optional[str]:
        if not isinstance(v, str):
            return None
        v = v.strip()
        if not v:
            return None
        return API_BASE + v if v.startswith("/") else v

    # поточний формат API: голий список [{"value": "...", ...}]
    if isinstance(data, list) and data:
        item = data[0]
        if isinstance(item, dict):
            return _normalize(item.get("value"))
        return None

    if isinstance(data, dict) and isinstance(data.get("hydra:member"), list) and data["hydra:member"]:
        item = data["hydra:member"][0]
        if isinstance(item, dict):
            return _normalize(item.get("value"))

    if isinstance(data, dict):
        return _normalize(data.get("value"))

    return None


def download_image(url: str) -> np.ndarray:
    r = SESSION.get(url, timeout=30)
    r.raise_for_status()
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Не вдалося декодувати зображення")
    return img


def load_local_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Не вдалося прочитати локальне зображення: {path}")
    return img


# ================= CACHES (per endpoint) =================

_cached_img: Dict[str, Dict[str, Any]] = {
    API_TODAY: {"ts": 0.0, "url": None, "img": None},
    API_TOMORROW: {"ts": 0.0, "url": None, "img": None},
}


def get_image_cached(api_url: str) -> Tuple[Optional[str], Optional[np.ndarray]]:
    # ✅ TEST MODE: беремо з диска і не ходимо в мережу
    if TEST_MODE:
        path = TEST_TODAY_IMAGE if api_url == API_TODAY else TEST_TOMORROW_IMAGE
        # "url" тут просто маркер, щоб кеш/парсер розумів "версію"
        # робимо fingerprint по mtime, щоб при заміні файлу вважалося "оновленням"
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return None, None

        fake_url = f"file:{os.path.abspath(path)}?v={int(mtime)}"

        # кеш TTL можна зберегти, але для тестів зручніше реагувати на зміну файлу
        c = _cached_img.setdefault(api_url, {"ts": 0.0, "url": None, "img": None})
        if c["img"] is not None and c["url"] == fake_url and (time.time() - c["ts"]) < CACHE_TTL_SEC:
            return c["url"], c["img"]

        img = load_local_image(path)
        c["ts"] = time.time()
        c["url"] = fake_url
        c["img"] = img
        return fake_url, img

    # ✅ PROD MODE: як було
    now = time.time()
    c = _cached_img.setdefault(api_url, {"ts": 0.0, "url": None, "img": None})

    if c["img"] is not None and c["url"] is not None and (now - c["ts"]) < CACHE_TTL_SEC:
        return c["url"], c["img"]

    url = fetch_latest_image_url_from_api(api_url)
    if not url:
        return None, None

    img = download_image(url)

    c["ts"] = now
    c["url"] = url
    c["img"] = img
    return url, img
