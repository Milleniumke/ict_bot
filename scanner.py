"""Multi-symbol live scanner — alerts + journal wired in."""
import logging
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from correlation import CorrelationGuard
from news import NewsBlackout
from kelly import KellySizer
from adaptive import AdaptiveRegistry
from mtf import mtf_confluence
from volume import volume_confirms

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
    registry: AdaptiveRegistry,
    kelly: KellySizer,
    corr_guard: CorrelationGuard,
    news: NewsBlackout,
    lookback_bars: int = 5,
    auto_execute: bool = False,
    max_workers: int = 8,
):
    # Fetch all symbols in parallel
    data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_data, s, cfg.interval, cfg.period): s for s in symbols}
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                data[sym] = fut.result()
            except Exception as e:
                log.warning("fetch failed %s: %s", sym, e)

    if not data:
        return

    # Update correlation guard
    closes = {sym: df["close"] for sym, df in data.items()}
    corr_guard.update(closes)

    for symbol, df in data.items():
        cfg.symbol = symbol
        cfg.adaptive_registry = registry

        try:
            sigs = generate_signals(df, cfg)
        except Exception as e:
            log.exception("signal gen failed %s: %s", symbol, e)
            continue

        # Adaptive gate
        sigs = [s for s in sigs if s.adaptive_p >= cfg.adaptive_min_p]

        # MTF + volume gates
        filtered = []
        for s in sigs:
            if not volume_confirms(df, s.entry_index, s.direction):
                continue
            mtf_bonus = mtf_confluence(df, s.entry_index, s.direction)
            if mtf_bonus == 0:
                continue
            s.confluence_score = min(5, s.confluence_score + (1 if mtf_bonus >= 2 else 0))
            filtered.append(s)
        sigs = filtered

        recent = [s for s in sigs if s.entry_index >= len(df) - lookback_bars]

        for s in recent:
            key = f"{symbol}|{cfg.interval}|{s.entry_time}|{s.direction}"
            if state.has(key):
                continue
            state.add(key)

            if news.blocked(s.entry_time, symbol):
                log.info("NEWS blackout skip %s %s", symbol, s.entry_time)
                continue

            text = format_signal(symbol, cfg.interval, s)
            alerter.send(text)

            sig_id = journal.log_signal(symbol, cfg.interval, s)
            log.info("SIGNAL %s %s score=%s p=%.2f id=%s",
                     symbol, s.direction, s.confluence_score, s.adaptive_p, sig_id)

            if auto_execute:
                ok, why = rm.can_open(s.entry_time)
                if not ok:
                    log.info("skip execution (%s)", why)
                    continue

                open_syms = [f.order.symbol for f in broker.positions()]
                safe, reason = corr_guard.is_safe(symbol, open_syms)
                if not safe:
                    log.info("correlation skip %s", reason)
                    continue

                mult = kelly.size_multiplier(cfg.risk_per_trade_pct)
                size = rm.position_size(s.entry_price, s.stop) * mult
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

def run_live(cfg, settings, broker, rm, registry, kelly, corr_guard, news):
    alerter = Alerter(settings.telegram_token, settings.telegram_chat, settings.webhook_url)
    journal = Journal(settings.journal_db)
    state = SignalState(settings.state_file)

    log.info("Turbo scanner online | symbols=%s | workers=8", settings.symbols)

    while True:
        try:
            scan_once(cfg, settings.symbols, alerter, journal, state,
                      broker, rm, registry, kelly, corr_guard, news)
        except KeyboardInterrupt:
            return
        except Exception as e:
            log.exception("scan loop error: %s", e)
        import time
        time.sleep(settings.poll_seconds)
