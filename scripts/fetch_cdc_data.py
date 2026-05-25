"""Download CDC COVID-19 case/death data and save to data/raw/cdc_cases_by_state.csv.

Source: https://data.cdc.gov/Case-Surveillance/United-States-COVID-19-Cases-and-Deaths-by-State-o/9mfq-cb36
Endpoint returns CSV via Socrata; large file (~30MB).

Usage:  python scripts/fetch_cdc_data.py
"""

from pathlib import Path
import sys

import requests

URL = "https://data.cdc.gov/api/views/9mfq-cb36/rows.csv?accessType=DOWNLOAD"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "cdc_cases_by_state.csv"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    resp = requests.get(URL, stream=True, timeout=120)
    resp.raise_for_status()
    total = 0
    with OUT.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            total += len(chunk)
    print(f"Wrote {total/1e6:.1f} MB to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
