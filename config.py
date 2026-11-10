"""Project-wide settings. Edit these to match your hardware."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# CSV written by serial_logger.py from the real PV panel
LOGGED_DATA_FILE = DATA_DIR / "solar_data.csv"
# Open dataset: NASA POWER hourly data (raw download + cleaned output)
RAW_DIR = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "cleaned"
CLEAN_HOURLY_FILE = CLEAN_DIR / "solar_hourly_clean.csv"
CLEAN_DAILY_FILE = CLEAN_DIR / "solar_daily_clean.csv"
QUALITY_REPORT_FILE = CLEAN_DIR / "data_quality_report.json"
WORKBOOK_FILE = DATA_DIR / "solar_dataset.xlsx"

# Rated (peak) power of the small PV panel, in watts. Used for capacity factor
# and "sun hours" (time spent above SUN_THRESHOLD of rated power).
PANEL_RATED_W = 5.0
SUN_THRESHOLD = 0.10

# Site used for the open dataset: Tiruchirappalli, Tamil Nadu
SITE_NAME = "Tiruchirappalli"
LATITUDE_DEG = 10.7905
LONGITUDE_DEG = 78.7047

# Study window: the latest four complete weeks NASA POWER had published (hourly
# CERES irradiance lags real time by about two months)
STUDY_START = "2026-08-03"
STUDY_END = "2026-08-30"

# PV model that turns irradiance into the output of the small prototype panel
NOCT_C = 45.0            # nominal operating cell temperature
TEMP_COEFF = -0.004      # power change per degree C above 25 C
SYSTEM_DERATE = 0.90     # dust, wiring and mismatch losses

# Forecasting defaults (all can be changed from the dashboard sidebar)
MA_WINDOW = 3          # days averaged by the moving-average model
BACKTEST_DAYS = 14     # most recent days used for walk-forward evaluation
MIN_TRAIN_DAYS = 7     # ML models need at least this many rows before predicting
RANDOM_STATE = 42

# Column names expected in the raw CSV
COL_TIME = "timestamp"
COL_VOLT = "voltage_V"
COL_CURR = "current_mA"
COL_POWER = "power_mW"
COL_TEMP = "temperature_C"   # optional
COL_LUX = "light_lux"        # optional
