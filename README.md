# ICT A+ Setup Hunter

A complete bot that hunts the ICT long model:
**Liquidity Sweep → Market Structure Shift → FVG pullback entry**.

## Install

```bash
pip install -r requirements.txt
```

## Run a Backtest

```bash
python main.py backtest --symbol EURUSD=X --interval 15m --period 60d
```

Try crypto (24/7):
```bash
python main.py backtest --symbol BTC-USD --interval 15m --period 60d
```

Indices:
```bash
python main.py backtest --symbol ^GSPC --interval 30m --period 60d
```

Results print to console and full trade list exports to `trades.csv`.

## Run Live (paper scanning)

```bash
python main.py live --symbol EURUSD=X --interval 15m --poll 300
```

Prints a formatted alert whenever a fresh A+ setup appears in the last 5 bars.
`--poll` is seconds between scans. Use `--no-session` to disable the killzone filter.

## How It Works

1. **Swing detection** — 5-bar fractal pivots.
2. **Liquidity pools** — clustered equal highs / equal lows (≥2 touches).
3. **Sweep** — a bar wicks through a pool and closes back on the origin side.
4. **MSS** — a body ≥ 1× ATR closes through the prior opposing swing.
5. **FVG** — the imbalance created during the displacement leg.
6. **Entry** — limit into the FVG (proximal / midpoint / distal).
7. **Stop** — below the sweep extreme + 0.3× ATR buffer.
8. **Targets** — 1.5R / 2.5R / 4.0R partial ladder (50 / 30 / 20%).

## Adaptivity & Resilience

- **Regime filter**: ADX ≥ 20 and ATR-percentile ≤ 85 — kills chop setups.
- **HTF bias**: 4H close must be above 50 EMA.
- **Confluence score (0–5)** — only trades scoring ≥ 3 are taken.
- **Session filter** — optional London / NY killzone.
- **Circuit breakers**: 2% daily DD, 6% total DD, 3 consecutive losses, max 2 concurrent.
- **Fractional 1% risk per trade.**

## Extending

- **MT5 broker execution**: replace `fetch_data()` in `main.py` with the MT5 SDK,
  and pipe `_print_alert()` signals into `order_send`.
- **Self-learning weights**: log which `reasons` appeared in winning trades and
  raise `min_confluence_score` when win rate drops (see the Adaptive Decision
  Engine section in the design doc).
- **Multi-symbol**: wrap `cmd_live` in a loop over `cfg.symbols`.
- **Bearish mirror**: duplicate the long path for sweeps of highs → bearish MSS →
  bearish FVG, keeping the same confluence scorer.

## Tuning Cheatsheet

| Symptom | Fix |
|---|---|
| Too few trades | `min_confluence_score=2`, `require_htf_bias=False` |
| Too many losses | `min_confluence_score=4`, `enforce_session=True` |
| Stops too tight | raise `stop_buffer_atr` to 0.5 |
| Wants more runners | extend `tp_r` to `(1.5, 3.0, 6.0)` |

## Disclaimer

Educational tool. No guarantee of profitability. Always forward-test on demo
before risking real capital.
