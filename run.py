"""
Top-level orchestrator.

Usage:
  python run.py backtest
  python run.py live
"""
import argparse
import logging

from config import Config
from settings import load_settings
from logging_setup import setup_logging
from backtest import run_backtest, export_trades_csv
from scanner import fetch_data, run_live
from broker import PaperBroker, MT5Broker
from risk import RiskManager

log = logging.getLogger("run")


def build_broker(settings):
    if settings.broker == "mt5":
        import os
        login = int(os.environ.get("MT5_LOGIN", "0"))
        pwd = os.environ.get("MT5_PASSWORD", "")
        server = os.environ.get("MT5_SERVER", "")
        path = os.environ.get("MT5_PATH", "")
        return MT5Broker(login, pwd, server, path)
    return PaperBroker()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["backtest", "live", "walkforward"])
    ap.add_argument("--symbol")
    ap.add_argument("--interval")
    ap.add_argument("--period")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    settings = load_settings()
    setup_logging(settings.log_file)

    cfg = Config()
    cfg.interval = args.interval or settings.interval
    cfg.period = args.period or settings.period

    if args.mode == "backtest":
        from metrics import extended_metrics
        symbol = args.symbol or settings.symbols[0]
        df = fetch_data(symbol, cfg.interval, cfg.period)
        cfg.symbol = symbol
        result = run_backtest(df, cfg)
        m = result["metrics"]
        m.update(extended_metrics(result["trades"], result["equity_curve"]))
        log.info("BASE metrics: %s", result["metrics"])
        log.info("EXTENDED metrics: %s", {k: v for k, v in m.items()
                                          if k in ("sharpe", "sortino", "calmar",
                                                   "expectancy_r", "max_dd_pct")})
        export_trades_csv(result)
        return

    if args.mode == "walkforward":
        from walkforward import walk_forward
        symbol = args.symbol or settings.symbols[0]
        df = fetch_data(symbol, cfg.interval, cfg.period)
        wf = walk_forward(df, cfg, symbol=symbol)
        for w in wf["windows"]:
            log.info("WF %s | test R=%.3f | sharpe=%.2f | n=%s",
                     w["test_range"], w["test"].get("expectancy_r", 0),
                     w["test"].get("sharpe", 0), w["test"].get("n_wins", 0) + w["test"].get("n_losses", 0))
        return

    # LIVE
    from adaptive import AdaptiveRegistry
    from kelly import KellySizer
    from correlation import CorrelationGuard
    from news import NewsBlackout
    from scanner import run_live

    broker = build_broker(settings)
    rm = RiskManager(cfg)
    registry = AdaptiveRegistry(path="adaptive.json")
    kelly = KellySizer(window=50, fraction=0.25, cap_mult=2.0)
    corr_guard = CorrelationGuard(threshold=0.7, lookback=200)
    news = NewsBlackout(path="news.csv", before_min=30, after_min=30)

    run_live(cfg, settings, broker, rm, registry, kelly, corr_guard, news)


if __name__ == "__main__":
    main()
