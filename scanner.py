"""Multi-symbol live scanner — alerts + journal wired in."""
import logging
import time
from datetime import datetime, timezone

from config import Config
from signals import generate_signals
from alerts import Alerter, format_signal
from journal import Journal
from state import SignalState
from broker import Broker, Order
from risk import RiskManager

log = logging.getLogger("scanner")


def fetch_data(symbol: str, interval: str, period: str):
    import yfinance as yf
    df = yf.download(symbol, interval=interval, period=period,
                     progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {symbol}")
    if hasattr(df.columns, "get_level_values"):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing {col}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if not isinstance(df.index, __import__("pandas").DatetimeIndex):
        import pandas as pd
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def scan_once(
    cfg: Config,
    symbols,
    alerter: Alerter,
    journal: Journal,
    state: SignalState,
    broker: Broker,
    rm: RiskManager,
    lookback_bars: int = 5,
    auto_execute: bool = False,
):
    for symbol in symbols:
        try:
            df = fetch_data(symbol, cfg.interval, cfg.period)
        except Exception as e:
            log.warning("fetch failed %s: %s", symbol, e)
            continue

        try:
            sigs = generate_signals(df, cfg)
        except Exception as e:
            log.exception("signal gen failed %s: %s", symbol, e)
            continue

        recent = [s for s in sigs if s.entry_index >= len(df) - lookback_bars]

        for s in recent:
            key = f"{symbol}|{cfg.interval}|{s.entry_time}|{s.direction}"
            if state.has(key):
                continue
            state.add(key)

            text = format_signal(symbol, cfg.interval, s)
            alerter.send(text)

            sig_id = journal.log_signal(symbol, cfg.interval, s)
            log.info("SIGNAL %s %s score=%s id=%s", symbol, s.direction,
                     s.confluence_score, sig_id)

            if auto_execute:
                ok, why = rm.can_open(s.entry_time)
                if not ok:
                    log.info("skip execution (%s)", why)
                    continue
                size = rm.position_size(s.entry_price, s.stop)
                if size <= 0:
                    continue
                order = Order(
                    symbol=symbol,
                    side="buy" if s.direction == "long" else "sell",
                    size=size,
                    entry=s.entry_price,
                    stop=s.stop,
                    targets=list(s.targets),
                )
                fill = broker.place(order)
                if fill:
                    rm.register_open()
                    journal.log_trade(
                        sig_id, symbol, s.direction,
                        str(s.entry_time), s.entry_price, s.stop, size,
                        status="open",
                    )

        if not recent:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            log.info("[%s UTC] %s — no fresh setups (close %.5f)",
                     ts, symbol, float(df["close"].iloc[-1]))


def run_live(cfg: Config, settings, broker: Broker, rm: RiskManager):
    alerter = Alerter(settings.telegram_token, settings.telegram_chat, settings.webhook_url)
    journal = Journal(settings.journal_db)
    state = SignalState(settings.state_file)

    log.info("Live scanner started | symbols=%s interval=%s poll=%ss",
             settings.symbols, cfg.interval, settings.poll_seconds)

    while True:
        try:
            scan_once(cfg, settings.symbols, alerter, journal, state, broker, rm)
        except KeyboardInterrupt:
            log.info("shutdown requested")
            return
        except Exception as e:
            log.exception("scan loop error: %s", e)
        time.sleep(settings.poll_seconds)
