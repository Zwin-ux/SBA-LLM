"""Baseline logistic regression PD model for Underwriter Copilot."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np

FEATURES_PATH = Path("config/features.json")


@dataclass
class PDModelConfig:
    intercept: float
    coefficients: Dict[str, float]
    lgd_base: float
    lgd_collateral_adjustment: float
    sector_adjustments: Dict[str, float]
    risk_bands: Iterable[Dict[str, float]]


def load_config(path: Path | None = None) -> PDModelConfig:
    path = path or FEATURES_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    coeffs = data["coefficients"]
    lgd = data.get("lgd", {})
    return PDModelConfig(
        intercept=float(data.get("intercept", 0.0)),
        coefficients={k: float(v) for k, v in coeffs.items()},
        lgd_base=float(lgd.get("base", 0.45)),
        lgd_collateral_adjustment=float(lgd.get("collateral_coverage_adjustment", -0.25)),
        sector_adjustments={k: float(v) for k, v in data.get("sector_adjustments", {}).items()},
        risk_bands=list(data.get("risk_bands", [])),
    )


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def probability_of_default(features: Dict[str, float], naics: str | None = None, config: PDModelConfig | None = None) -> Dict[str, float]:
    config = config or load_config()
    total = config.intercept

    # Derived features
    rev_ttm = max(features.get("rev_ttm", 1.0), 1.0)
    log_rev = math.log(rev_ttm)
    feature_values = {
        "log_rev_ttm": log_rev,
        "dscr": features.get("dscr", 1.0),
        "dti": features.get("dti", 0.0),
        "fico_proxy": features.get("fico_proxy", 650) / 100.0,
        "years_in_business": features.get("years_in_business", 0.0),
        "delinquencies": 1.0 if features.get("delinquencies") else 0.0,
        "collateral_coverage": features.get("collateral_coverage", 0.0),
    }

    contributions: Dict[str, float] = {}
    for name, value in feature_values.items():
        weight = config.coefficients.get(name, 0.0)
        contrib = weight * value
        contributions[name] = contrib
        total += contrib

    sector_adj = config.sector_adjustments.get(naics or "", config.sector_adjustments.get("default", 0.0))
    contributions["sector_adjustment"] = sector_adj
    total += sector_adj

    pd = logistic(total)
    lgd = max(0.0, min(1.0, config.lgd_base + config.lgd_collateral_adjustment * (1 - feature_values.get("collateral_coverage", 0.0))))

    band = next((band_info["band"] for band_info in config.risk_bands if pd <= band_info.get("max_pd", 1.0)), "D")

    return {
        "pd": round(pd, 4),
        "lgd": round(lgd, 4),
        "risk_band": band,
        "contributions": {k: round(v, 4) for k, v in contributions.items()},
    }
