"""
financial_analysis.py
Core financial analysis logic extracted from company_profile_and_multiple.py.
No Dash/Plotly UI dependencies — pure data processing and text output.
"""

import pandas as pd
import numpy as np
import re
import io


# ──────────────────────────────────────────────
# CSV Parsing
# ──────────────────────────────────────────────

def extract_ticker_from_filename(filename: str) -> str:
    """Extract ticker symbol from a TradingView-style filename."""
    if not filename:
        return "Unknown"
    name = filename.replace('.csv', '').replace(' ', '')
    for pattern in [
        r'_([A-Z]{1,5})[-_,\.]',
        r'^([A-Z]{2,5})[-_,\.]',
        r'\b([A-Z]{2,5})\b',
    ]:
        m = re.search(pattern, name)
        if m:
            return m.group(1)
    return "Unknown"


def parse_csv(csv_bytes: bytes, filename: str = "") -> tuple[pd.DataFrame | None, str | None]:
    """Parse CSV bytes into a DataFrame. Returns (df, error_msg)."""
    try:
        df = pd.read_csv(io.StringIO(csv_bytes.decode('utf-8')))
        return df, None
    except Exception as e:
        return None, str(e)


# ──────────────────────────────────────────────
# Data Processing (mirrors Dash app logic)
# ──────────────────────────────────────────────

REQUIRED_COLS = ['time', 'close', 'Mkt cap', 'Quarterly Revenue',
                 'Gross Profit (B)', 'EBITDA', 'Free cash flow']

FINANCIAL_COLS = ['Mkt cap', 'Quarterly Revenue', 'Gross Profit (B)',
                  'EBITDA', 'Free cash flow', 'Net debt to EBITDA ratio',
                  'EPS estimates', 'Revenue estimates']


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return list of missing required columns."""
    return [c for c in REQUIRED_COLS if c not in df.columns]


def process_financial_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and de-duplicate financial data.

    Handles both quarterly CSVs (one row per quarter) and monthly CSVs
    (three rows per quarter sharing the same Quarterly Revenue value).
    EBITDA and Free cash flow are treated as optional — companies that
    don't report them are still processed with whatever data is present.

    Within each revenue block (quarter), keeps the single row that has the
    most non-null optional financial values (EBITDA, FCF, EPS, etc.).
    """
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    for col in ['close', 'open', 'high', 'low']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in FINANCIAL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Keep rows that have at least Revenue + Gross Profit.
    # EBITDA and FCF are optional — some companies simply don't report them.
    df_valid = df.dropna(subset=['Quarterly Revenue', 'Gross Profit (B)']).copy()

    if df_valid.empty:
        return df.iloc[-1:].reset_index(drop=True)  # caller will catch <4 rows

    # De-duplicate monthly -> quarterly.
    # Monthly CSVs repeat the same Revenue value for all 3 months of a quarter.
    # Identify consecutive revenue blocks, then keep the one row per block that
    # has the most optional financial data (fewest NaNs across EBITDA/FCF/EPS).
    df_valid = df_valid.sort_values('time').reset_index(drop=True)
    df_valid['_rev_block'] = (
        df_valid['Quarterly Revenue']
        .ne(df_valid['Quarterly Revenue'].shift())
        .cumsum()
    )

    optional_cols = [c for c in ['EBITDA', 'Free cash flow', 'EPS estimates',
                                  'Net debt to EBITDA ratio']
                     if c in df_valid.columns]
    df_valid['_null_count'] = (
        df_valid[optional_cols].isnull().sum(axis=1) if optional_cols else 0
    )

    best_idx = df_valid.groupby('_rev_block')['_null_count'].idxmin()
    df_dedup = (df_valid.loc[best_idx]
                .drop(columns=['_rev_block', '_null_count'])
                .sort_values('time')
                .reset_index(drop=True))

    return df_dedup


def calculate_ttm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate TTM (Trailing Twelve Months) rolling metrics."""
    df = df.sort_values('time').reset_index(drop=True).copy()

    df['Revenue_TTM']      = df['Quarterly Revenue'].rolling(4, min_periods=4).sum()
    df['Gross_Profit_TTM'] = df['Gross Profit (B)'].rolling(4, min_periods=4).sum()
    df['EBITDA_TTM']       = df['EBITDA'].rolling(4, min_periods=4).sum()
    df['FCF_TTM']          = df['Free cash flow'].rolling(4, min_periods=4).sum()

    df['Gross_Profit_Margin_TTM'] = (df['Gross_Profit_TTM'] / df['Revenue_TTM']) * 100
    df['EBITDA_Margin_TTM']       = (df['EBITDA_TTM'] / df['Revenue_TTM']) * 100
    df['FCF_Margin_TTM']          = (df['FCF_TTM'] / df['Revenue_TTM']) * 100

    df['Market_Cap_to_Revenue'] = df['Mkt cap'] / df['Revenue_TTM']
    df['Market_Cap_to_EBITDA']  = df['Mkt cap'] / df['EBITDA_TTM']
    df['Market_Cap_to_FCF']     = df['Mkt cap'] / df['FCF_TTM']

    df['Revenue_Growth_YoY'] = df['Revenue_TTM'].pct_change(periods=4) * 100

    if 'EPS estimates' in df.columns:
        df['EPS_TTM']  = df['EPS estimates'].rolling(4, min_periods=4).sum()
        df['PE_Ratio'] = df['close'] / df['EPS_TTM']

    if 'Revenue estimates' in df.columns:
        df['Revenue_Estimates_TTM'] = df['Revenue estimates'].rolling(4, min_periods=4).sum()
        df['Revenue_Surprise_Pct']  = (
            (df['Revenue_TTM'] - df['Revenue_Estimates_TTM']) / df['Revenue_Estimates_TTM']
        ) * 100

    return df


def get_latest_metrics(df: pd.DataFrame) -> dict | None:
    """Extract the most recent available value for each metric."""
    def last(mask_col):
        sub = df[df[mask_col].notna()]
        return sub.iloc[-1] if len(sub) > 0 else None

    rev = last('Revenue_TTM')
    if rev is None:
        return None

    ebitda  = last('EBITDA_TTM')
    fcf     = last('FCF_TTM')
    nd      = last('Net debt to EBITDA ratio')
    eps     = last('EPS_TTM')         if 'EPS_TTM' in df.columns else None
    rev_est = last('Revenue_Estimates_TTM') if 'Revenue_Estimates_TTM' in df.columns else None

    return {
        'ticker_date':          rev['time'],
        'Revenue_TTM':          rev['Revenue_TTM'],
        'Revenue_Growth_YoY':   rev['Revenue_Growth_YoY'],
        'Gross_Profit_Margin':  rev['Gross_Profit_Margin_TTM'],
        'EBITDA_Margin':        ebitda['EBITDA_Margin_TTM']  if ebitda  is not None else np.nan,
        'FCF_TTM':              fcf['FCF_TTM']               if fcf     is not None else np.nan,
        'FCF_Margin':           fcf['FCF_Margin_TTM']        if fcf     is not None else np.nan,
        'Net_Debt_to_EBITDA':   nd['Net debt to EBITDA ratio'] if nd    is not None else np.nan,
        'EPS_TTM':              eps['EPS_TTM']               if eps     is not None else np.nan,
        'PE_Ratio':             eps['PE_Ratio']              if eps     is not None else np.nan,
        'Revenue_Surprise_Pct': rev_est['Revenue_Surprise_Pct'] if rev_est is not None else np.nan,
        # Current price / mkt cap from the very latest row
        'Current_Price':        df.iloc[-1]['close'],
        'Current_Mkt_Cap':      df.iloc[-1]['Mkt cap'],
    }


def calculate_price_predictions(df: pd.DataFrame) -> pd.DataFrame | None:
    """Return a DataFrame of price predictions by scenario."""
    latest = df.iloc[-1]
    current_price   = latest['close']
    current_mkt_cap = latest['Mkt cap']
    shares_out      = current_mkt_cap / current_price

    def _last_ttm(col):
        sub = df[df[col].notna()]
        return sub.iloc[-1][col] if len(sub) > 0 else None

    rev_ttm   = _last_ttm('Revenue_TTM')
    ebitda_ttm = _last_ttm('EBITDA_TTM')
    fcf_ttm    = _last_ttm('FCF_TTM')

    if rev_ttm is None:
        return None

    def stats(series):
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 5:
            return None
        mu, sd = s.mean(), s.std()
        s = s[(s >= mu - 3*sd) & (s <= mu + 3*sd)]
        if len(s) < 5:
            return None
        return dict(min=s.min(), q1=s.quantile(.25), median=s.median(),
                    mean=s.mean(), q3=s.quantile(.75), max=s.max())

    rev_stats   = stats(df[df['Revenue_TTM'].notna()]['Market_Cap_to_Revenue'])
    ebitda_stats = stats(df[df['EBITDA_TTM'].notna()]['Market_Cap_to_EBITDA']) if ebitda_ttm else None
    fcf_stats    = stats(df[df['FCF_TTM'].notna()]['Market_Cap_to_FCF'])       if fcf_ttm   else None

    rows = []
    for label, key in [('Min','min'),('Q1','q1'),('Median','median'),
                        ('Mean','mean'),('Q3','q3'),('Max','max')]:
        row = {'Scenario': label, 'Current Price': f'${current_price:.2f}'}

        if rev_stats:
            p = (rev_stats[key] * rev_ttm) / shares_out
            row['Rev Multiple'] = f"{rev_stats[key]:.1f}x"
            row['Price (Rev)']  = f"${p:.2f}"
            row['Upside (Rev)'] = f"{(p/current_price-1)*100:+.1f}%"
        else:
            row['Rev Multiple'] = row['Price (Rev)'] = row['Upside (Rev)'] = 'N/A'

        if ebitda_stats and ebitda_ttm:
            p = (ebitda_stats[key] * ebitda_ttm) / shares_out
            row['EBITDA Multiple']    = f"{ebitda_stats[key]:.1f}x"
            row['Price (EBITDA)']     = f"${p:.2f}"
            row['Upside (EBITDA)']    = f"{(p/current_price-1)*100:+.1f}%"
        else:
            row['EBITDA Multiple'] = row['Price (EBITDA)'] = row['Upside (EBITDA)'] = 'N/A'

        if fcf_stats and fcf_ttm:
            p = (fcf_stats[key] * fcf_ttm) / shares_out
            row['FCF Multiple']  = f"{fcf_stats[key]:.1f}x"
            row['Price (FCF)']   = f"${p:.2f}"
            row['Upside (FCF)']  = f"{(p/current_price-1)*100:+.1f}%"
        else:
            row['FCF Multiple'] = row['Price (FCF)'] = row['Upside (FCF)'] = 'N/A'

        rows.append(row)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Text Report Formatter  (for Telegram)
# ──────────────────────────────────────────────

def _fmt_b(val) -> str:
    """Format a dollar value in billions."""
    if pd.isna(val):
        return 'N/A'
    return f'${val/1e9:.2f}B'

def _fmt_pct(val, decimals=1) -> str:
    if pd.isna(val):
        return 'N/A'
    return f'{val:.{decimals}f}%'

def _fmt_x(val, decimals=1) -> str:
    if pd.isna(val):
        return 'N/A'
    return f'{val:.{decimals}f}x'

def _fmt_signed_pct(val) -> str:
    if pd.isna(val):
        return 'N/A'
    sign = '+' if val >= 0 else ''
    return f'{sign}{val:.1f}%'

def _leverage_label(ratio) -> str:
    if pd.isna(ratio):
        return ''
    if ratio < 0:
        return ' (Net Cash ✅)'
    if ratio < 2:
        return ' (Strong ✅)'
    if ratio < 3:
        return ' (Moderate ⚠️)'
    return ' (High Leverage ❗)'


def build_telegram_report(ticker: str, metrics: dict, predictions: pd.DataFrame | None) -> str:
    """
    Build a clean, well-structured Telegram message from computed metrics.
    Uses only plain text + emoji — no Markdown that might break.
    """
    m = metrics
    date_str = m['ticker_date'].strftime('%b %Y')

    lines = [
        f"📊 {ticker} — Financial Analysis",
        f"Data as of: {date_str}",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "💰 KEY METRICS (TTM)",
        "━━━━━━━━━━━━━━━━━━━━",
        f"Current Price:      ${m['Current_Price']:.2f}",
        f"Revenue (TTM):      {_fmt_b(m['Revenue_TTM'])}",
        f"Revenue Growth YoY: {_fmt_signed_pct(m['Revenue_Growth_YoY'])}",
        "",
        "📈 MARGINS",
        f"  Gross Profit:  {_fmt_pct(m['Gross_Profit_Margin'])}",
        f"  EBITDA:        {_fmt_pct(m['EBITDA_Margin'])}",
        f"  FCF:           {_fmt_pct(m['FCF_Margin'])}",
        f"  Free Cash Flow:{_fmt_b(m['FCF_TTM'])}",
        "",
        "🏦 BALANCE SHEET",
    ]

    nd = m['Net_Debt_to_EBITDA']
    lines.append(f"  Net Debt/EBITDA: {_fmt_x(nd)}{_leverage_label(nd)}")

    lines += [
        "",
        "📉 VALUATION",
        f"  EPS (TTM): {'${:.2f}'.format(m['EPS_TTM']) if not pd.isna(m['EPS_TTM']) else 'N/A'}",
        f"  P/E Ratio: {_fmt_x(m['PE_Ratio'])}",
    ]

    if not pd.isna(m['Revenue_Surprise_Pct']):
        lines.append(f"  Rev vs Estimates: {_fmt_signed_pct(m['Revenue_Surprise_Pct'])}")

    if predictions is not None:
        lines += [
            "",
            "━━━━━━━━━━━━━━━━━━━━",
            "🎯 PRICE TARGETS (Historical Multiples)",
            "━━━━━━━━━━━━━━━━━━━━",
        ]

        # Show revenue-based targets if available
        has_rev = predictions['Price (Rev)'].iloc[0] != 'N/A'
        has_ebitda = predictions['Price (EBITDA)'].iloc[0] != 'N/A'
        has_fcf = predictions['Price (FCF)'].iloc[0] != 'N/A'

        if has_rev:
            lines.append("Revenue Multiple Targets:")
            for _, row in predictions.iterrows():
                lines.append(f"  {row['Scenario']:<8} {row['Price (Rev)']:>8}  ({row['Upside (Rev)']})")

        if has_ebitda:
            lines.append("EBITDA Multiple Targets:")
            for _, row in predictions.iterrows():
                lines.append(f"  {row['Scenario']:<8} {row['Price (EBITDA)']:>8}  ({row['Upside (EBITDA)']})")

        if has_fcf:
            lines.append("FCF Multiple Targets:")
            for _, row in predictions.iterrows():
                lines.append(f"  {row['Scenario']:<8} {row['Price (FCF)']:>8}  ({row['Upside (FCF)']})")

    lines += [
        "",
        "─────────────────────",
        "⚠️ For informational purposes only. Not financial advice.",
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Main entry point: bytes → report string
# ──────────────────────────────────────────────

def analyze_csv(csv_bytes: bytes, filename: str) -> str:
    """
    Full pipeline: raw CSV bytes → formatted Telegram report string.
    Returns an error string (starting with '❌') on failure.
    """
    ticker = extract_ticker_from_filename(filename)

    df, err = parse_csv(csv_bytes, filename)
    if err:
        return f"❌ Could not parse CSV: {err}"

    missing = validate_columns(df)
    if missing:
        return f"❌ Missing columns in CSV: {', '.join(missing)}"

    try:
        df_clean = process_financial_data(df)
        if len(df_clean) < 4:
            return "❌ Not enough quarterly data (need at least 4 rows with complete financials)."

        df_ttm      = calculate_ttm_metrics(df_clean)
        metrics     = get_latest_metrics(df_ttm)
        if metrics is None:
            return "❌ Could not compute TTM metrics. Check data quality."

        predictions = calculate_price_predictions(df_ttm)
        return build_telegram_report(ticker, metrics, predictions)

    except Exception as exc:
        return f"❌ Analysis error: {exc}"
