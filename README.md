# FinBot Web

The Telegram financial bot, as a browser app. Four tabs, all reports rendered inline as HTML
(each one downloadable):

| Tab | What it does | Bot equivalent |
|---|---|---|
| **Analysis** | Type a ticker (fetched via yfinance) or drop one or more TradingView CSVs | send CSV / type ticker |
| **Estimates** | Forward revenue & EPS → implied prices, forward P/E vs TTM, analyst targets | `/estimate` |
| **Backtest** | Strategy Lab: RSI mean-reversion Strategy A and B (MACD filter) × RSI 2/3/4, plus the 2 MACD momentum strategies for reference, compared on the same window | `/backtest`, `/backtestdaily` |
| **Signals** | Subscribe to tickers; alerts appear in the feed | `/notify on/off/list` |

### Analysis tab extras

- **Current price** is the latest close in the CSV (the last row), or yfinance's live price for a
  typed ticker. It drives Mkt Cap, P/E, the current valuation multiples and every price-target
  upside. The header's "Financials as of" date is the last quarter with earnings data.
- **Compare** — upload two or more CSVs at once and a *📊 Compare* tab is added: market cap,
  revenue TTM, revenue growth, EBITDA and FCF margins, net debt / EBITDA, the three current
  multiples (with each company's own historical median) and estimated market share. Market share
  defaults to each company's slice of the group's combined revenue; type a total market size
  ($B) in the report to estimate share of the whole space instead.
- **✈ Send to Telegram** — posts the open report (or all open reports) to a channel as an HTML
  file. Needs `TELEGRAM_BOT_TOKEN` in `.env` (the same token as your bot), and the bot must be an
  **admin of the channel with "Post messages"**. Enter the channel's public link
  (`https://t.me/yourchannel` or `@yourchannel`); links you've sent to are remembered in
  `telegram_links.json` and offered next time. Private invite links (`t.me/+…`) can't be used by
  bots — use the numeric channel ID (`-100…`) for a private channel.

## Backtest tab — Strategy Lab

Eight runs on one scoreboard: **Strategy A** and **Strategy B** for RSI length 2, 3 and 4 (daily
bars), plus the two MACD momentum strategies (Histogram, Crossover + MA50) on the selected timeframe
as a reference.

**Strategy A — pure RSI mean reversion**
- **Buy** when RSI crosses *under* 10 (previous ≥ 10, current < 10), only when flat. Filled at the
  next bar's open. Repeat signals while in a trade are ignored.
- **Arm** a decline watch when RSI crosses *over* 80 (previous ≤ 80, current > 80). It stays armed
  until the trade ends.
- **Sell** on the first bar after arming where RSI < the previous bar's RSI, filled at the next
  bar's open. Example: RSI 75, 83, 86, 81 → arms on 83, sells at the open after the 81 bar.
- No stop-loss, no time exit.

**Strategy B — A + MACD filter**
- Same entry as A.
- From entry, it watches for a MACD (12/26/9, daily) histogram **BUY state**: the first bar of a
  new streak of 2 consecutive rising bars.
- When RSI crosses over 80:
  - if a BUY state has occurred since entry, the exit switches permanently to **close < MA5**,
    checked on every later bar and filled at the next open (the RSI decline watch is ignored);
  - otherwise it uses A's decline watch.
- Each trade records its exit type (*MACD/MA5* or *RSI decline-watch*).

**Fair comparison.** Every run trades inside one common window after all indicators have warmed up,
pays 4 bps per side (3 commission + 1 slippage), invests 100% of equity per trade and is marked to
market daily. Buy & hold is the benchmark, with *return ÷ exposure* because these strategies sit in
cash most of the time. The momentum references keep their 12-combination parameter sweep (treat
their selected set as optimistic); A and B are untuned.

**Reading it.** Scoreboard (★ = best per column; a t-stat below ~2 means the average trade is
indistinguishable from noise), the **Strategy A vs B × RSI length** table (the requested columns,
with CSV downloads for the table and the full trade log), equity curves for A and B (log scale),
Strategy B exit-path cards (how trades split between MACD/MA5 and the decline watch, and what each
path earned), per-strategy details, price chart with trades, returns by year, the momentum sweeps
and every trade log. Pending signals (a buy or sell that fills at tomorrow's open) are shown at the
top.

### Strategy A/B from the command line

```bash
python -m rsi2.ab --ticker AMD                      # RSI 2,3,4 → results/AMD_AB_<timestamp>/
python -m rsi2.ab --csv data/amd.csv --lengths 2,3,4 --out-dir results/amd --repair-ohlc
```

Prints the comparison table and writes `comparison.csv` (RSI length, strategy, total trades, win
rate, avg win %, avg loss %, profit factor, total return %, max drawdown %, avg holding bars, and for
B the % of MACD/MA5 vs RSI decline-watch exits) and `trades.csv` (entry date/price, exit
date/price, RSI length, strategy, exit type, trade return %, bars held; a trade still open at the
end is listed with its status). In Python: `rsi2.ab.run_suite(df)` returns the same two pandas
DataFrames.

## Original RSI(2) spec engine (`rsi2/`)

The earlier RSI(2) spec (200-day trend filter, RSI ≥ 65 AND close > SMA5 exit) is still included as a
standalone engine — pure pandas/numpy, auditable bar loop, no backtesting libraries. The Backtest tab
no longer shows it, but it runs on its own:

```bash
python -m rsi2 --ticker SPY --start 2005-01-01 --end 2025-01-01
python -m rsi2 --csv data/spy.csv --exit-logic OR --no-trend-stop
python -m rsi2 --ticker QQQ --use-stop --stop-pct 6 --out-dir results/qqq
python -m rsi2 --ticker AMD --json-config my_config.json --exit-level 70   # CLI flags override the file
```

Writes `trades.csv`, `equity_daily.csv`, `summary.json` (every statistic + the full resolved config +
code version) and `summary.txt` to `--out-dir` (default `results/<ticker>_<timestamp>/`). Exit codes:
0 success, 1 validation failure, 2 no trades. (`plot.png` from the spec is not implemented — the web
report has the charts.)

| Parameter | Default | Meaning |
|---|---|---|
| `trend_len` / `trend_type` | 200 / SMA | trend filter (SMA or TradingView-seeded EMA) |
| `rsi_len` | 2 | Wilder RSI length, seeded exactly like TradingView |
| `entry_level` | 10 | RSI must cross *down* through this (previous bar ≥ level) |
| `entry_delay` | 1 | bars from signal to fill; fills at that bar's **close** |
| `recheck_filter` | off | cancel the entry if close < trend on the entry bar |
| `fast_len` | 5 | fast SMA for the exit |
| `exit_level` / `exit_logic` | 65 / AND | exit when RSI ≥ level AND/OR close > fast SMA |
| `use_trend_stop` | on | exit on a close below the trend MA (`--no-trend-stop` to disable) |
| `use_stop` / `stop_pct` | off / 8 | intrabar % stop; a gap below it fills at the open |
| `use_timeout` / `max_bars` | off / 10 | time stop |
| `flag_level` | 10 | research tag: RSI still below this the bar after entry |
| `commission_bps` + `slippage_bps` | 3 + 1 | per side; all results are net |
| `initial_capital` / `position_pct` / `whole_shares` | 100,000 / 100 / off | sizing |
| `start_date` / `end_date` | none | entries start at `start_date`; data after `end_date` is ignored |

Other flags: `--repair-ohlc` fixes tiny (≤0.5%) Close-outside-range errors that price adjustment
leaves in Yahoo data (validation is strict by default, as specified). Minimum hold is 1 bar; open
positions at the end are marked to market and reported separately, never force-closed; drawdown,
Sharpe and Sortino come from the daily mark-to-market curve.

**Caveats.** Use split- and dividend-adjusted prices (the loaders do: `auto_adjust=True`) —
unadjusted data creates fake 200-MA breaks and gaps. Backtesting a currently-listed ticker is
survivorship-biased: it exists because it survived.

**Tests.** `pip install pytest && python -m pytest tests` — indicator parity with TradingView seeding,
the no-look-ahead test (truncated vs full data must give identical earlier trades), entry/exit timing,
exit logic and priority, cost accounting, degenerate inputs, the CLI, and Strategy A/B (the
75/83/86/81 example, crossover edges, the MACD BUY state, the permanent MA5 switch, exit-type
logging, and look-ahead checks on the suite and the weekly candles).

**Pine cross-check (spec §12.7) — not done yet.** It needs TradingView: run
`rsi2_mean_reversion.pine` on one ticker and date range, export the List of Trades, and compare with
`trades.csv` from `python -m rsi2 --ticker <same> --start <same> --end <same>`. Trade count and every
entry/exit *date* should match; small price differences are normal (vendor adjustments). A date
mismatch means a timing difference — send me both files.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # optional
python app.py                                            # → http://localhost:5000
```

## Deploy (Render)

Push this folder to GitHub, then **New → Blueprint** in Render and pick the repo — `render.yaml`
does the rest. Health check: `/api/health`.

Things to know before relying on Signals in production:

- **One worker on purpose.** `render.yaml` runs Gunicorn with `--workers 1 --threads 4`. The signal
  thread starts once per worker process, so 2 workers would poll every ticker twice and log
  duplicate alerts (the build spec said `--workers 2`; changed for this reason).
- **Free-tier services sleep** after ~15 min without traffic, and the signal thread sleeps with
  them — no checks run until someone opens the site. Use a paid instance for real monitoring.
- **Free-tier disk is ephemeral**: `signals.json` (subscriptions + feed) resets on every deploy or
  restart. Attach a Render Disk and point `SIGNALS_FILE` at it to keep it.
- **Telegram on a public URL:** anyone who can open the deployed site can post to channels your
  bot administers. Keep the Render URL private, or leave `TELEGRAM_BOT_TOKEN` unset there.
- Yahoo Finance sometimes rate-limits cloud IPs; if ticker lookups start failing on Render but work
  locally, that's the likely cause (CSV upload is unaffected).

## How Signals work

Subscribing runs the same backtest as the bot's `/notify on` and locks in the **winning strategy
and its best parameters** for that ticker (shown under each ticker in the list). The worker then
checks every `SIGNAL_POLL_INTERVAL_S` seconds (default 1 h) on weekly candles, plus once right
after subscribing. Like `signal_monitor.py`, it never logs the same direction twice in a row for a
ticker — so the still-forming weekly candle doesn't re-alert every hour. The page polls the feed
every 60 s.

## Files

```
app.py            Flask routes                    web_reports.py     HTML for the Estimates tab
regime_lab.py     Backtest tab: all 5 strategies  lab_report.py      Backtest tab HTML report
rsi2/             RSI(2) engine, metrics, CLI     tests/             pytest suite (python -m pytest tests)
pipeline.py       CSV → HTML (mirrors bot.py)     compare_report.py  multi-company comparison report
signal_worker.py  background signal thread        telegram_share.py  send reports to Telegram
static/           index.html, app.js, styles.css  wsgi.py            Gunicorn entry point
financial_analysis.py, tradingview_downloader.py                   — copied from the bot, unchanged
html_report.py, macd_backtest.py, estimates_analysis.py            — copied from the bot, changes below
```

## Changes to the bot's modules

These fix real bugs, so they're worth copying back into the Telegram bot too:

**macd_backtest.py**
- `run_all()` never passed `interval` to `fetch_price_data()`, so `/backtestdaily` and
  `interval="2d"` backtests actually ran on **weekly** candles. It now fetches at the requested
  interval (and the crossover sweep carries the interval through as well).
- `BacktestResult` / `CrossoverResult` now default `total_trades`, `win_rate`, `total_pnl`, etc. to 0.
  Before, a parameter set with zero trades had no such attributes and `run_all()` crashed with
  `AttributeError` (tickers with short histories).
- Added `check_current_signal(ticker, ...)` — the live BUY/SELL check used by the Signals tab. Its
  rules match `signal_monitor.py` exactly (verified on 1,440 cases).

**html_report.py** / **pipeline.py** — current price
- `process_financial_data()` keeps only earnings-week rows, so the bot's `run_analysis()` used the
  close from the last *quarter-end week* as "current price" (weeks or months stale), and with it
  stale Mkt Cap, P/E, current multiples and price-target upside. `pipeline.py` now takes the latest
  close / Mkt cap from the last row of the raw CSV (or the live quote for typed tickers). The same
  fix belongs in the bot's `run_analysis()`.
- `_current_multiple()` skipped back to the last *positive* multiple, so a company whose TTM
  EBITDA or FCF had turned negative showed an old multiple as "current". It now shows N/A.

**estimates_analysis.py** (the `eps_estimates` → `earnings_estimate` fix was already in your copy)
- Rating counts read `recs.iloc[-1]`, which is the **3-months-ago** row; Yahoo returns the current
  month first. Now uses the `0m` row.
- Removed a fallback that, whenever Strong Buy was 0, overwrote the **Buy** count with the total
  number of analyst opinions — inflating Buy and the consensus label.
