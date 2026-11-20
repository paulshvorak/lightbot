import logging
import os
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ================= CONFIG =================

API_BASE = "https://api-voe-poweron.inneti.net"
API_TODAY = f"{API_BASE}/api/options?option_key=pw_gpv_image_today"
API_TOMORROW = f"{API_BASE}/api/options?option_key=pw_gpv_image_tomorrow"

TEST_MODE = os.getenv("TEST_MODE", "0") == "1"
TEST_TODAY_IMAGE = os.getenv("TEST_TODAY_IMAGE", "./test_images/today.png")
TEST_TOMORROW_IMAGE = os.getenv("TEST_TOMORROW_IMAGE", "./test_images/tomorrow.png")

DB_PATH = os.getenv("DB_PATH", "users.db")

TZ = ZoneInfo("Europe/Uzhgorod")

# Адміни (user_id). У приватному чаті user_id == chat_id.
ADMIN_IDS = {328587643}

# Анти-спам / анти-трафік
CACHE_TTL_SEC = 60          # не качати картинку частіше ніж раз/хв (на весь процес)
PARSE_TTL_SEC = 60          # не парсити OpenCV частіше ніж раз/хв (на весь процес)
NOW_COOLDOWN_SEC = 30       # не обробляти Now/Tomorrow частіше ніж раз/30с на один чат

# Максимум черг, на які може підписатись один користувач
MAX_SUBSCRIPTIONS = 3

# Поріг визначення "темна клітинка" (нема світла) по V каналу HSV (підібраний)
V_THRESHOLD = 185

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("light_bot")

# При бажанні приглушити шумні логи:
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

SESSION = requests.Session()
_retry = Retry(
    total=4,
    connect=4,
    read=4,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
_adapter = HTTPAdapter(max_retries=_retry, pool_connections=10, pool_maxsize=10)
SESSION.mount("https://", _adapter)
SESSION.mount("http://", _adapter)
SESSION.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json,image/*,*/*",
    "Accept-Language": "uk-UA,uk;q=0.9,en;q=0.8",
})
