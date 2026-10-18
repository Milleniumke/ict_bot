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
    ap.add_argument("mode", choices=["backtest", "live"])
    ap.add_argument("--symbol", help="override for backtest")
    ap.add_argument("--interval", help="override")
    ap.add_argument("--period", help="override")
    ap.add_argument("--execute", action="store_true",
                    help="live mode: actually route orders to broker")
    args = ap.parse_args()

    settings = load_settings()
    setup_logging(settings.log_file)
    log.info("boot mode=%s broker=%s", args.mode, settings.broker)

    cfg = Config()
    cfg.interval = args.interval or settings.interval
    cfg.period = args.period or settings.period

    if args.mode == "backtest":
        symbol = args.symbol or settings.symbols[0]
        cfg.symbol = symbol
        df = fetch_data(symbol, cfg.interval, cfg.period)
        log.info("loaded %s bars of %s", len(df), symbol)
        result = run_backtest(df, cfg)
        m = result["metrics"]
        if m.get("n_trades", 0) == 0:
            log.warning("no trades generated — loosen thresholds")
            return
        log.info("backtest metrics: %s", m)
        path = export_trades_csv(result)
        log.info("trades exported to %s", path)
        return

    broker = build_broker(settings)
    rm = RiskManager(cfg)
    if args.mode == "live":
        # patch scanner to auto-execute if flag set
        import scanner
        orig = scanner.scan_once
        def wrapped(*a, **kw):
            kw.setdefault("auto_execute", args.execute)
            return orig(*a, **kw)
        scanner.scan_once = wrapped
        run_live(cfg, settings, broker, rm)


if __name__ == "__main__":
    main()
