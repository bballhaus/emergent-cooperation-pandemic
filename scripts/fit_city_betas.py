"""Fit per-state beta from CDC daily new-case data and write to data/processed/city_betas.json.

Assumes scripts/fetch_cdc_data.py has been run. Maps each city in CITY_TABLE to a US
state code (NY, CA, IL, TX, AZ, PA, TX, CA) and fits beta from the state's case curve.
"""

import json
from pathlib import Path

from src.env.calibration import CITY_TABLE, fit_beta_from_cases, load_cdc_state_data, COVID_DEFAULT

REPO = Path(__file__).resolve().parent.parent
CDC_CSV = REPO / "data" / "raw" / "cdc_cases_by_state.csv"
OUT = REPO / "data" / "processed" / "city_betas.json"

CITY_TO_STATE = {
    "New York": "NY",
    "Los Angeles": "CA",
    "Chicago": "IL",
    "Houston": "TX",
    "Phoenix": "AZ",
    "Philadelphia": "PA",
    "San Antonio": "TX",
    "San Diego": "CA",
}


def main() -> int:
    if not CDC_CSV.exists():
        raise SystemExit(f"Missing {CDC_CSV}. Run scripts/fetch_cdc_data.py first.")

    fits: dict[str, float] = {}
    for name, pop, _cap in CITY_TABLE:
        state = CITY_TO_STATE.get(name)
        cases = load_cdc_state_data(CDC_CSV, state)
        if cases is None or len(cases) < 30:
            print(f"  {name}: no data for state={state}, using default beta={COVID_DEFAULT.beta}")
            fits[name] = COVID_DEFAULT.beta
            continue
        # Use the first 90 days of post-Mar-2020 data to fit the ancestral-strain beta.
        window = cases[:90]
        beta = fit_beta_from_cases(window, population=pop, initial_infected=10)
        print(f"  {name}: fitted beta = {beta:.3f}")
        fits[name] = beta

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fits, indent=2))
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
