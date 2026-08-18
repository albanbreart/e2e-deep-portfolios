#!/usr/bin/env python
"""Download and cache every raw input. Idempotent: safe to re-run.

    python scripts/00_pull_data.py            # everything
    python scripts/00_pull_data.py --quick    # skip the 2.4 GB firm-level archive

Every source is public and free -- the project has no paid data dependency, which
is the point: anyone can clone the repository and reproduce the results.

Open Source Asset Pricing (Chen & Zimmermann)
    ``PredictorPortsFull``      long-short return of each published anomaly.
                                These are the assets we allocate across.
    ``SignalDoc``               what each anomaly is, when it was published, and
                                what the original paper found.
    liquidity-screened variants the same anomalies rebuilt on large caps and on
                                stocks above $5, used for the capacity checks.
    ``signed_predictors_dl_wide`` the firm-level signals (2.4 GB).  We do not
                                model on these; we use them once, to *measure*
                                how much each anomaly's stock book turns over,
                                which is what prices the carrying cost.

Ken French's data library
    The five factors, momentum and the risk-free rate, for the alpha regressions.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xsdp.config import load_config  # noqa: E402
from xsdp.data import openap  # noqa: E402
from xsdp.data.anomaly_costs import compute_turnover_series  # noqa: E402
from xsdp.data.french import load_ff_factors  # noqa: E402

log = logging.getLogger("pull")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true",
                    help="skip the firm-level archive and the turnover measurement")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", stream=sys.stdout)
    cfg = load_config(args.config)
    cfg.paths.mkdirs()

    log.info("=== anomaly returns and metadata ===")
    openap.load_signal_doc(cfg.paths.external)
    for which in ("op", "ex_nyse_p20_me", "ex_price5"):
        openap.load_cz_portfolios(cfg.paths.external, which)

    log.info("=== Fama-French factors (Ken French) ===")
    load_ff_factors(cfg.paths.external)

    if args.quick:
        log.info("--quick: skipping the firm-level archive; costs will fall back to a flat rate")
        return 0

    log.info("=== firm-level signals, for measuring anomaly turnover ===")
    zip_path = openap.download_firm_char(cfg.paths.raw)
    signals = openap.firm_char_to_parquet(
        zip_path, cfg.paths.processed / "cz_firm_signals.parquet"
    )

    log.info("=== measuring each anomaly's decile-book turnover ===")
    compute_turnover_series(signals, cfg.paths.processed / "turnover_series.parquet")

    log.info("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
