"""
Entry point.

Usage:
  python main.py backtest --symbol EURUSD=X --interval 15m --period 60d
  python main.py live     --symbol EURUSD=X --interval 15m --period 60d --poll 300
"""
import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

from config import Config
from backtest import run_backtest, export_trades_csv
from signals import generate_signals


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def fetch_data(symbol: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str.lower)
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column '{col}' in data")
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


# ---------------------------------------------------------------------------
# Backtest mode
# ---------------------------------------------------------------------------
def cmd_backtest(cfg: Config, export: bool = True):
    print(f"[*] Fetching {cfg.symbol} {cfg.interval} over {cfg.period} ...")
    df = fetch_data(cfg.symbol, cfg.interval, cfg.period)
    print(f"[*] {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    result = run_backtest(df, cfg)
    m = result["metrics"]

    if m["n_trades"] == 0:
        print("\n[!] No trades generated.")
        print("    Try: looser thresholds (min_confluence_score=2), a different")
        print("    symbol (e.g. BTC-USD, GBPUSD=X), or larger period.")
        return

    print("\n" + "=" * 62)
    print("  BACKTEST RESULTS")
    print("=" * 62)
    for k, v in m.items():
        if isinstance(v, float):
            print(f"  {k:>20} : {v:,.4f}")
        else:
            print(f"  {k:>20} : {v}")

    print("\n  Sample trades (first 10):")
    for t in result["trades"][:10]:
        print(
            f"   {t.entry_time}  entry={t.entry_price:.5f}  "
            f"stop={t.stop:.5f}  pnl={t.pnl:+.2f}  R={t.r_multiple:+.2f}  "
            f"score={t.signal.confluence_score}  exits={t.exits}"
        )

    if export:
        path = export_trades_csv(result)
        print(f"\n[*] Full trade list exported to: {path}")


# ---------------------------------------------------------------------------
# Live mode
# ---------------------------------------------------------------------------
def cmd_live(cfg: Config, poll_seconds: int = 300):
    import time

    seen_signals = set()
    print(f"[*] Live scanner: {cfg.symbol} {cfg.interval} | poll every {poll_seconds}s")
    print(f"[*] Session hours UTC: {cfg.session_start_utc}:00 – {cfg.session_end_utc}:00")
    print(f"[*] Min confluence: {cfg.min_confluence_score}/5")
    print("-" * 62)

    while True:
        try:
            df = fetch_data(cfg.symbol, cfg.interval, cfg.period)
            sigs = generate_signals(df, cfg)
            recent = [s for s in sigs if s.entry_index >= len(df) - 5]

            for s in recent:
                sig_id = df.index[s.entry_index]
                if sig_id in seen_signals:
                    continue
                seen_signals.add(sig_id)
                _print_alert(cfg, df, s)

            if not recent:
                now = datetime.now(timezone.utc).strftime("%H:%M:%S")
                print(f"[{now} UTC] no fresh setups  (last close {df['close'].iloc[-1]:.5f})")
        except Exception as e:
            print(f"[!] live loop error: {e}", file=sys.stderr)

        time.sleep(poll_seconds)


def _print_alert(cfg: Config, df: pd.DataFrame, s):
    print("\n" + "=" * 62)
    print(f"  🎯 SIGNAL  {cfg.symbol}  LONG")
    print("=" * 62)
    print(f"  Bar time     : {df.index[s.entry_index]}")
    print(f"  Entry        : {s.entry_price:.5f}")
    print(f"  Stop         : {s.stop:.5f}")
    print(f"  Targets      : {[round(t, 5) for t in s.targets]}")
    print(f"  Risk (R)     : {s.risk:.5f}")
    print(f"  Confluence   : {s.confluence_score}/5")
    print(f"  Reasons      :")
    for r in s.reasons:
        print(f"     • {r}")
    print(f"  Sweep level  : {s.sweep.level:.5f}  (extreme {s.sweep.extreme:.5f})")
    print(f"  MSS level    : {s.mss.broken_level:.5f}  (body {s.mss.body_atr:.2f}x ATR)")
    print(f"  FVG          : [{s.fvg.bottom:.5f} .. {s.fvg.top:.5f}]  "
          f"({s.fvg.size_atr:.2f}x ATR)")
    print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ICT A+ Setup Hunter")
    ap.add_argument("mode", choices=["backtest", "live"])
    ap.add_argument("--symbol", default="EURUSD=X")
    ap.add_argument("--interval", default="15m")
    ap.add_argument("--period", default="60d")
    ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--no-session", action="store_true",
                    help="Disable killzone hour filter")
    args = ap.parse_args()

    cfg = Config()
    cfg.symbol = args.symbol
    cfg.interval = args.interval
    cfg.period = args.period
    cfg.enforce_session = not args.no_session

    if args.mode == "backtest":
        cmd_backtest(cfg)
    else:
        cmd_live(cfg, args.poll)


if __name__ == "__main__":
    main()
