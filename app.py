"""
WWBD: What Would Buffett Do?
A classic portfolio built by fixed rules: Python screens and sizes, the AI only argues for and against.

Run:  streamlit run app.py
Educational project. Not financial advice. Not affiliated with Warren Buffett or Berkshire Hathaway.
"""
import json
import os
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")  # hide noisy library deprecation warnings
HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))

# =====================================================================================
# 1. SETTINGS
# =====================================================================================
PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
API_KEY = os.getenv("LLM_API_KEY", "")
ENDPOINT = os.getenv("LLM_ENDPOINT", "")
MODEL = os.getenv("LLM_MODEL", "gpt-6.1-sol")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION", "2024-12-01-preview")
USE_AI = PROVIDER != "mock" and bool(API_KEY)

DATA_DIR = os.path.join(HERE, "data")
PRICES_CSV = os.path.join(DATA_DIR, "prices_snapshot.csv")             # offline safety net only
FUNDAMENTALS_CSV = os.path.join(DATA_DIR, "fundamentals_snapshot.csv")
CASES_JSON = os.path.join(DATA_DIR, "last_cases.json")                 # last AI cases, for a Wi-Fi-proof demo
PRICE_START = "2019-01-01"   # far enough back to replay the COVID crash

CASH_TICKER, INDEX_TICKER = "SHY", "VOO"   # short-term Treasuries; S&P 500 index fund
CASH_SHARE = 0.10
MAX_POSITION = 0.25
MAX_PER_SECTOR = 3
DISCLAIMER = ("Educational project, not financial advice. Not affiliated with or endorsed by Warren Buffett or Berkshire "
              "Hathaway. The rules are our own reading of his published principles.")

# Large US-listed companies plus well-known growth names.
# Banks, insurers and property trusts are left out because these ratios don't work for them.
STOCKS = {
    "MSFT": "Microsoft", "NVDA": "NVIDIA", "AAPL": "Apple", "GOOGL": "Alphabet", "AMZN": "Amazon",
    "META": "Meta Platforms", "AVGO": "Broadcom", "TSM": "TSMC", "ASML": "ASML", "AMD": "AMD",
    "LLY": "Eli Lilly", "NVO": "Novo Nordisk", "CRWD": "CrowdStrike", "SHOP": "Shopify", "MELI": "MercadoLibre",
    "NOW": "ServiceNow", "UBER": "Uber", "NFLX": "Netflix", "COST": "Costco", "V": "Visa",
    "MA": "Mastercard", "ISRG": "Intuitive Surgical", "ANET": "Arista Networks", "ADBE": "Adobe", "NKE": "Nike",
    "ABBV": "AbbVie", "ABT": "Abbott", "ACN": "Accenture", "AMGN": "Amgen", "BA": "Boeing",
    "BKNG": "Booking Holdings", "BMY": "Bristol Myers Squibb", "CAT": "Caterpillar", "CL": "Colgate-Palmolive", "CMCSA": "Comcast",
    "CRM": "Salesforce", "CSCO": "Cisco", "CVS": "CVS Health", "CVX": "Chevron", "DE": "Deere",
    "DHR": "Danaher", "DIS": "Disney", "DUK": "Duke Energy", "GD": "General Dynamics", "GE": "GE Aerospace",
    "GILD": "Gilead", "HD": "Home Depot", "HON": "Honeywell", "IBM": "IBM", "INTC": "Intel",
    "INTU": "Intuit", "LIN": "Linde", "LMT": "Lockheed Martin", "LOW": "Lowe's", "MCD": "McDonald's",
    "MDLZ": "Mondelez", "MDT": "Medtronic", "MMM": "3M", "MO": "Altria", "MRK": "Merck",
    "NEE": "NextEra Energy", "ORCL": "Oracle", "PEP": "PepsiCo", "PFE": "Pfizer", "PM": "Philip Morris",
    "QCOM": "Qualcomm", "RTX": "RTX", "SBUX": "Starbucks", "SO": "Southern Company", "T": "AT&T",
    "TMO": "Thermo Fisher", "TMUS": "T-Mobile", "TSLA": "Tesla", "TXN": "Texas Instruments", "UNH": "UnitedHealth",
    "UNP": "Union Pacific", "UPS": "UPS", "VZ": "Verizon", "WMT": "Walmart", "XOM": "Exxon Mobil",
    "KO": "Coca-Cola", "PG": "Procter & Gamble", "JNJ": "Johnson & Johnson", "UL": "Unilever",
}


def _fact(value, fmt):
    return "no data" if pd.isna(value) else fmt.format(value)


# (short name, the rule in plain English, where the idea comes from, the test, the company's actual figure)
# Graham's original numbers are loosened for today's markets; the point is a fixed rule set anyone can check.
CLASSIC_TESTS = [
    ("Always profitable", "A profit in every year reported (about 4 years)", "Graham: earnings stability",
     lambda d: (d.years_reported >= 3) & (d.loss_years == 0),
     lambda r: f"{r.loss_years:.0f} of {r.years_reported:.0f} years in loss"),
    ("Earnings growing", "Earnings per share higher than at the start of those years", "Graham: earnings growth",
     lambda d: d.eps_latest > d.eps_oldest,
     lambda r: f"EPS {_fact(r.eps_oldest, '{:.2f}')} to {_fact(r.eps_latest, '{:.2f}')}"),
    ("High return on equity", "Return on equity of at least 15%", "Buffett: a good business",
     lambda d: d["roe_%"] >= 15,
     lambda r: _fact(r["roe_%"], "{:.0f}%")),
    ("Strong margins", "Operating margin of at least 15%", "Buffett: pricing power",
     lambda d: d["operating_margin_%"] >= 15,
     lambda r: _fact(r["operating_margin_%"], "{:.0f}%")),
    ("Modest debt", "Debt could be repaid with 4 years of free cash flow or less", "Buffett: conservative debt",
     lambda d: d.debt_to_fcf_years <= 4,
     lambda r: _fact(r.debt_to_fcf_years, "{:.1f} years")),
    ("Bills covered", "Current assets at least match current liabilities", "Graham: financial strength (he asked for double)",
     lambda d: d.current_ratio >= 1,
     lambda r: _fact(r.current_ratio, "{:.2f}x")),
    ("Real cash", "Free cash flow is at least 70% of reported profit", "Piotroski: earnings quality",
     lambda d: d.fcf_to_net_income >= 0.7,
     lambda r: _fact(r.fcf_to_net_income, "{:.0%}")),
    ("Fair price", "Price is no more than 25 times the last year's earnings", "Graham: moderate price (he asked for 15)",
     lambda d: (d.trailing_pe > 0) & (d.trailing_pe <= 25),
     lambda r: _fact(r.trailing_pe, "{:.0f}x earnings")),
]
TEST_NAMES = [t[0] for t in CLASSIC_TESTS]

# =====================================================================================
# 2. MARKET DATA  (live from Yahoo Finance; a CSV snapshot is only a safety net)
# =====================================================================================
def fetch_prices() -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(list(STOCKS) + [CASH_TICKER, INDEX_TICKER], start=PRICE_START, interval="1d", auto_adjust=True, progress=False)
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    close = close.dropna(axis=1, how="all").ffill()
    if close.shape[1] < 0.8 * len(STOCKS) or INDEX_TICKER not in close or CASH_TICKER not in close:
        raise RuntimeError("Yahoo Finance returned too few prices")
    close = close.resample("W-FRI").last()
    today = pd.Timestamp.now().normalize()
    close.index = close.index.where(close.index <= today, today)   # this week's row is dated today, not next Friday
    return close


def fetch_fundamentals() -> pd.DataFrame:
    import yfinance as yf

    def one(t):
        for attempt in range(3):   # Yahoo sometimes returns half-empty records: retry rather than save them
            try:
                tk = yf.Ticker(t)
                i, inc, bal, cf = tk.info, tk.income_stmt, tk.balance_sheet, tk.cashflow
            except Exception:      # no connection: fail fast so the offline snapshot loads straight away
                return None
            if i.get("sector") and not inc.empty and not cf.empty:
                break
            time.sleep(1 + attempt)
        else:
            return None
        pct = lambda k: round(i[k] * 100, 1) if i.get(k) is not None else np.nan
        line = lambda df, k: df.loc[k].dropna().sort_index() if k in df.index else pd.Series(dtype=float)   # oldest first
        last = lambda s: s.iloc[-1] if len(s) else np.nan
        net_income, eps = line(inc, "Net Income"), line(inc, "Diluted EPS")
        ni, fcf = last(net_income), last(line(cf, "Free Cash Flow"))
        debt = last(line(bal, "Total Debt"))
        debt = 0 if pd.isna(debt) else debt   # Yahoo omits the line for companies with no debt
        return {"ticker": t, "name": STOCKS[t], "sector": i.get("sector"),
                "operating_margin_%": pct("operatingMargins"), "roe_%": pct("returnOnEquity"),
                "trailing_pe": round(i["trailingPE"], 1) if i.get("trailingPE") else np.nan,
                "current_ratio": round(i["currentRatio"], 2) if i.get("currentRatio") else np.nan,
                "years_reported": len(net_income), "loss_years": int((net_income <= 0).sum()),
                "eps_oldest": last(eps.iloc[:1]), "eps_latest": last(eps),
                "debt_to_fcf_years": round(debt / fcf, 1) if fcf > 0 else np.nan,
                "fcf_to_net_income": round(fcf / ni, 2) if ni > 0 else np.nan,
                "business": (i.get("longBusinessSummary") or "")[:400]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = [r for r in pool.map(one, STOCKS) if r]
    if len(rows) < 0.8 * len(STOCKS):   # keep the last good snapshot rather than overwrite it with a partial one
        raise RuntimeError("Yahoo Finance returned too few company fundamentals")
    return pd.DataFrame(rows).set_index("ticker")


def _snapshot(fetch, path):
    """Live API first; save a snapshot on success; fall back to the snapshot only if the API fails."""
    try:
        df = fetch()
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(path)
        return df, None
    except Exception as e:
        if not os.path.exists(path):
            raise
        when = pd.Timestamp(os.path.getmtime(path), unit="s").strftime("%d %b %Y")
        return pd.read_csv(path, index_col=0, parse_dates=(path == PRICES_CSV)), f"{e}. Using saved data from {when}"


@st.cache_data(ttl=3600, show_spinner=False)
def load_market():
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(_snapshot, fn, path) for fn, path in [(fetch_prices, PRICES_CSV), (fetch_fundamentals, FUNDAMENTALS_CSV)]]
        (prices, w1), (fund, w2) = [j.result() for j in jobs]
    return prices.ffill(), fund, pd.Timestamp.now().strftime("%d %b %Y %H:%M"), [w for w in (w1, w2) if w]

# =====================================================================================
# 3. THE RULES  (screen the companies, then size the positions)
# =====================================================================================
def run_screen(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Every company against every test, most tests passed then cheapest first. Missing data counts as a fail."""
    df = fundamentals.copy()
    for name, _, _, test, _ in CLASSIC_TESTS:
        df[name] = test(df)
    df["tests_passed"] = df[TEST_NAMES].sum(axis=1)
    df["tests_failed"] = df[TEST_NAMES].apply(lambda r: ", ".join(r.index[~r]) or "none", axis=1)
    return df.sort_values(["tests_passed", "trailing_pe"], ascending=[False, True])


def scorecard(df: pd.DataFrame) -> pd.DataFrame:
    """One row per company: a tick or cross for each test, next to the company's actual figure."""
    cells = {name: [f"{'✓' if r[name] else '✗'} {fact(r)}" for _, r in df.iterrows()] for name, _, _, _, fact in CLASSIC_TESTS}
    return pd.DataFrame({"Company": df["name"].values, "Passed": df.tests_passed.map(lambda n: f"{n}/{len(TEST_NAMES)}").values, **cells},
                        index=df.index)


def build_portfolio(screen: pd.DataFrame, n: int, min_tests: int, amount: float):
    """Pick the top scorers, sector limit applied. Better scores get bigger positions; what's left stays in cash.
    Returns (picks, holdings table, companies that passed but missed out and why)."""
    picks, sectors, skipped = [], {}, []
    for t, r in screen[screen.tests_passed >= min_tests].iterrows():
        if len(picks) >= n:
            skipped.append((t, "portfolio full"))
        elif sectors.get(r.sector, 0) >= MAX_PER_SECTOR:
            skipped.append((t, f"already {MAX_PER_SECTOR} in {r.sector}"))
        else:
            picks.append(t)
            sectors[r.sector] = sectors.get(r.sector, 0) + 1
    chosen = screen.loc[picks]
    points = (chosen.tests_passed - 5).clip(lower=1)   # 8/8 gets three times the money of 6/8
    weights = (points / points.sum() * (1 - CASH_SHARE)).clip(upper=MAX_POSITION) if picks else pd.Series(dtype=float)
    rows = [{"Holding": chosen.loc[t, "name"], "ticker": t, "Sector": chosen.loc[t, "sector"],
             "Tests passed": f"{chosen.loc[t, 'tests_passed']}/{len(TEST_NAMES)}", "usd": round(weights[t] * amount)} for t in picks]
    cash = amount - sum(r["usd"] for r in rows)
    rows.append({"Holding": "Cash reserve (short-term Treasuries)", "ticker": CASH_TICKER, "Sector": "Cash", "Tests passed": "-", "usd": cash})
    holdings = pd.DataFrame(rows)
    holdings["Share"] = holdings.usd / amount * 100
    return chosen, holdings, skipped


def buy_and_hold(prices: pd.DataFrame, weights: pd.Series, start) -> pd.Series:
    """Growth of 1 if bought once on `start` and never touched. A holding not yet listed sits flat until it is."""
    px = prices[list(weights.index)].loc[start:]
    return (px / px.bfill().iloc[0]).fillna(1).mul(weights, axis=1).sum(axis=1) / weights.sum()


def worst_drop(path: pd.Series) -> float:
    return (path / path.cummax() - 1).min()

# =====================================================================================
# 4. THE CASE FOR AND AGAINST  (the AI argues; it does not choose)
# =====================================================================================
CASE_SYSTEM = """You help students learn from Warren Buffett's published investing principles. Fixed rules have already
chosen these companies, and your job is to test that choice. Do not write in Buffett's voice or claim what he would actually do.

For each company give:
- "business": one plain sentence on what it does and how it makes money.
- "case_for": up to 2 short sentences on why it could stay a good business, citing the figures given.
- "case_against": up to 2 short sentences giving the strongest honest reason not to own it. Tackle any failed test first.

Use only the figures given and never invent a number. You may describe the business from general knowledge. No price predictions.
Write plain English a student can read.
Reply with ONLY a JSON object: {"TICKER": {"business": "...", "case_for": "...", "case_against": "..."}}"""


def evidence(picks: pd.DataFrame) -> dict:
    """Everything the AI sees about each company. Add analyst ratings and news here when they join the project."""
    return {t: {"name": r["name"], "sector": r.sector, "business": r.business, "tests_passed": f"{r.tests_passed} of {len(TEST_NAMES)}",
                "tests_failed": r.tests_failed, "figures": {name: fact(r) for name, _, _, _, fact in CLASSIC_TESTS}}
            for t, r in picks.iterrows()}


def get_client():
    if PROVIDER == "azure":
        from openai import AzureOpenAI
        return AzureOpenAI(api_key=API_KEY, azure_endpoint=ENDPOINT, api_version=AZURE_API_VERSION)
    from openai import OpenAI
    return OpenAI(api_key=API_KEY, base_url=ENDPOINT or None)


def ai_cases(picks: pd.DataFrame) -> dict:
    resp = get_client().chat.completions.create(
        model=MODEL, max_completion_tokens=8000,
        messages=[{"role": "system", "content": CASE_SYSTEM}, {"role": "user", "content": json.dumps(evidence(picks), default=str)}])
    found = re.search(r"\{.*\}", resp.choices[0].message.content or "", re.S)
    if not found:
        raise ValueError("no JSON in the reply")
    reply = json.loads(found.group(0))
    return {t: reply[t] for t in picks.index if t in reply}


def rule_case(r: pd.Series) -> dict:
    """The case from the tests alone, used when the AI hasn't been asked or isn't available."""
    said = lambda keep: "; ".join(f"{name.lower()} ({fact(r)})" for name, _, _, _, fact in CLASSIC_TESTS if bool(r[name]) == keep)
    against = said(False)
    return {"business": re.split(r"(?<=[a-z]\.)\s", r.business or "")[0],
            "case_for": f"Passes: {said(True)}." if said(True) else "It passes none of the tests.",
            "case_against": f"Fails: {against}." if against else "Passes all 8 tests, so the risk is what numbers can't show: "
                                                                 "whether the business will still be this strong in 10 years."}


def committee_memo(holdings, picks, cases, amount, min_tests, results, fetched_at) -> str:
    """The recommendation as a Markdown memo the fund's committee can circulate and vote on."""
    lines = [f"# Investment committee memo: ${amount:,.0f}", f"Prepared {pd.Timestamp.now():%d %b %Y} with WWBD. Market data from Yahoo Finance, as of {fetched_at}.",
             "", "## Recommendation", "| Holding | Ticker | Tests passed | Amount | Share |", "|---|---|---|---|---|"]
    lines += [f"| {r.Holding} | {r.ticker} | {r['Tests passed']} | ${r.usd:,.0f} | {r.Share:.0f}% |" for _, r in holdings.iterrows()]
    lines += ["", "## Method", f"Every company faced the same {len(TEST_NAMES)} tests, and only those passing at least {min_tests} were eligible:"]
    lines += [f"- **{name}:** {rule} ({src})" for name, rule, src, _, _ in CLASSIC_TESTS]
    lines += [f"", f"Highest scores first, then lowest price-to-earnings, at most {MAX_PER_SECTOR} per sector and {MAX_POSITION:.0%} per company. "
              f"{CASH_SHARE:.0%} is held in short-term Treasuries.", "", "## The case for and against"]
    for t, r in picks.iterrows():
        c = cases[t]
        lines += [f"### {r['name']} ({t})", c.get("business", ""), f"- **For:** {c.get('case_for', '')}", f"- **Against:** {c.get('case_against', '')}", ""]
    lines += ["## Against the simple alternative", "| Measure | This portfolio | Buffett's 90/10 |", "|---|---|---|"]
    lines += ["| " + " | ".join(r.values()) + " |" for r in results]
    lines += ["", "Hindsight warning: the companies were chosen with today's figures and tested on old prices, so this shows how bumpy "
              "the ride could be, not whether the rules work.", "", f"*{DISCLAIMER}*"]
    return "\n".join(lines)


def saved_cases(tickers) -> dict | None:
    """The last AI cases, if they cover exactly these companies."""
    try:
        with open(CASES_JSON) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return None
    return saved if sorted(saved["cases"]) == sorted(tickers) else None

# =====================================================================================
# 5. THE PAGE
# =====================================================================================
BLUE, ORANGE, GRID = "#2a78d6", "#eb6834", "#9AA5B1"


def md(text) -> str:
    """Text for st.markdown: escape '$' so amounts are not rendered as LaTeX maths."""
    return str(text or "").replace("$", "\\$")


def growth_chart(wide: pd.DataFrame, height: int = 280):
    """Lines over time with a legend and a hover crosshair."""
    d = wide.reset_index(names="Date")
    long = d.melt("Date", var_name="Series", value_name="Value")
    color = alt.Color("Series:N", scale=alt.Scale(domain=list(wide.columns), range=[BLUE, ORANGE]), legend=alt.Legend(orient="top", title=None))
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["Date"], empty=False)
    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(x=alt.X("Date:T", title=None), y=alt.Y("Value:Q", title="USD", axis=alt.Axis(format="~s")), color=color)
    points = lines.mark_point(filled=True, size=64).encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    rule = alt.Chart(d).mark_rule(color=GRID).encode(
        x="Date:T", opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[alt.Tooltip("Date:T", format="%d %b %Y")] + [alt.Tooltip(field=c, type="quantitative", format="$,.0f") for c in wide.columns]
    ).add_params(nearest)
    return (lines + points + rule).properties(height=height)


st.set_page_config(page_title="WWBD: What Would Buffett Do?", layout="wide")
st.markdown("""<style>
h1, h2, h3, h4 { color: #0A3D7A; font-weight: 600; letter-spacing: -0.01em; }
[data-testid="stMetric"] { background: #F2F5F9; border-left: 4px solid #0A3D7A; padding: 10px 14px; border-radius: 4px; }
.block-container { padding-top: 2rem; max-width: 1200px; }
</style>""", unsafe_allow_html=True)
st.markdown("## WWBD: What Would Buffett Do?")
st.markdown("*\"It's far better to buy a wonderful company at a fair price than a fair company at a wonderful price.\"* "
            "Berkshire Hathaway shareholder letter, 1989")
st.markdown("**For student-managed funds and investment clubs:** turn \\$100,000 into a short list of companies the committee can defend, "
            "using fixed rules anyone can check.")
st.caption("The rules screen the companies and size the portfolio. The AI never picks a stock; it only argues for and against. " + DISCLAIMER)

try:
    with st.spinner("Fetching live market data..."):
        prices, fundamentals, fetched_at, data_warnings = load_market()
except Exception as e:
    st.error(f"Couldn't load market data and no saved data exists: {e}")
    st.stop()
for w in data_warnings:
    st.warning(f"Live data unavailable: {w}.")

screen = run_screen(fundamentals.loc[[t for t in fundamentals.index if t in prices]])

c1, c2, c3 = st.columns(3)
amount = c1.number_input("Amount to invest ($)", min_value=1_000, max_value=10_000_000, value=100_000, step=1_000)
n_picks = c2.slider("Companies to own", 5, 10, 8, help="Buffett's 1993 letter describes five to ten companies for investors who understand what they own.")
min_tests = c3.slider("Tests a company must pass", 5, len(TEST_NAMES), 7)
left_out = st.multiselect("Circle of competence: leave out anything you don't understand or wouldn't want to own",
                          options=list(screen.index), format_func=lambda t: f"{t} · {STOCKS[t]}")
pool = screen.drop(index=left_out)
picks, holdings, skipped = build_portfolio(pool, n_picks, min_tests, amount)
passed = pool[pool.tests_passed >= min_tests]

# ---- Step 1: the screen
st.markdown("### 1 · The screen")
st.markdown(f"**{len(passed)} of {len(screen)} companies** pass at least {min_tests} of these {len(TEST_NAMES)} tests.")
st.dataframe(pd.DataFrame([{"Test": n, "Rule": rule, "Idea from": src} for n, rule, src, _, _ in CLASSIC_TESTS]), width="stretch", hide_index=True)
with st.expander(f"See how all {len(screen)} companies scored"):
    st.dataframe(scorecard(screen), width="stretch")
    st.caption(f"Yahoo Finance, as of {fetched_at}. Missing data counts as a fail. Yahoo gives about 4 years of accounts; Graham and Buffett would want far more.")

# ---- Step 2: the portfolio
st.markdown("### 2 · The portfolio")
if picks.empty:
    st.info("No company passes, so the rules hold everything in cash and wait. Lower the number of tests to see more.")
st.dataframe(holdings[["Holding", "ticker", "Sector", "Tests passed", "usd", "Share"]], width="stretch", hide_index=True,
             column_config={"ticker": "Ticker", "usd": st.column_config.NumberColumn("Amount", format="$%d"),
                            "Share": st.column_config.ProgressColumn("Share", format="%.0f%%", min_value=0, max_value=100)})
st.caption(f"Highest scores first, then lowest price-to-earnings, with at most {MAX_PER_SECTOR} per sector. An 8/8 gets three times the money of a 6/8, "
           f"up to {MAX_POSITION:.0%} per company. {CASH_SHARE:.0%} stays in cash, plus anything above that limit.")
if skipped:
    st.caption("Passed but missed out: " + ", ".join(f"{t} ({why})" for t, why in skipped))
if len(picks):
    st.dataframe(scorecard(picks), width="stretch")

# ---- Step 3: the case for and against
if len(picks):
    st.markdown("### 3 · The case for and against")
    if USE_AI and st.button("Ask the AI to argue each case", type="primary"):
        with st.spinner("Writing the case for and against each company..."):
            try:
                found = ai_cases(picks)
                with open(CASES_JSON, "w") as f:
                    json.dump({"saved_at": pd.Timestamp.now().strftime("%d %b %Y %H:%M"), "cases": found}, f)
            except Exception as e:
                st.warning(f"The AI isn't available right now ({str(e)[:80]}...). Showing the summary from the tests instead.")
    saved = saved_cases(picks.index)
    if saved:
        st.caption(f"Written by AI on {saved['saved_at']} from the figures above. Business descriptions come from general knowledge, so check them.")
    else:
        st.caption("Summarised from the tests." + (" Ask the AI for a fuller argument." if USE_AI else " Add an API key for a fuller argument."))
    cases = {t: saved["cases"][t] if saved else rule_case(r) for t, r in picks.iterrows()}
    for t, r in picks.iterrows():
        case = cases[t]
        with st.container(border=True):
            st.markdown(f"**{md(r['name'])}** ({t}) · {md(case.get('business'))}")
            col_for, col_against = st.columns(2)
            col_for.markdown(f"**For.** {md(case.get('case_for'))}")
            col_against.markdown(f"**Against.** {md(case.get('case_against'))}")

# ---- Step 4: what could go wrong
st.markdown("### 4 · What could go wrong" if len(picks) else "### 3 · What could go wrong")
st.markdown("Buffett's 2013 letter gives simple advice for most people: 90% in a low-cost S&P 500 fund and 10% in short-term government bonds. "
            "That's the bar to beat, and the fairest test of whether picking companies is worth the effort.")
weights = holdings.set_index("ticker")["usd"] / amount
simple = pd.Series({INDEX_TICKER: 0.9, CASH_TICKER: 0.1})
start = prices.index[-1] - pd.DateOffset(years=5)
growth = pd.DataFrame({"This portfolio": buy_and_hold(prices, weights, start) * amount, "Buffett's 90/10": buy_and_hold(prices, simple, start) * amount})
st.altair_chart(growth_chart(growth), width="stretch")
windows = {"COVID crash, 2020": ("2020-01-01", "2020-12-31"), "2022 rate shock": ("2022-01-01", "2022-12-31")}
table = [{"Measure": "Change over 5 years", **{c: f"{growth[c].iloc[-1] / amount - 1:+.0%}" for c in growth}}]
for label, (a, b) in windows.items():
    table.append({"Measure": f"Worst fall, {label}", "This portfolio": f"{worst_drop(buy_and_hold(prices, weights, a).loc[:b]):.0%}",
                  "Buffett's 90/10": f"{worst_drop(buy_and_hold(prices, simple, a).loc[:b]):.0%}"})
st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
st.warning("Hindsight warning: these companies were chosen using today's figures, then tested on old prices, so this isn't a fair test in either "
           "direction. Today's strong profits flatter the past, while a low price today can mean the shares have already fallen. "
           "It shows how bumpy the ride could be, not whether the rules work.")

# ---- For the committee
if len(picks):
    st.markdown("### 5 · Take it to the committee")
    st.download_button("Download the committee memo", committee_memo(holdings, picks, cases, amount, min_tests, table, fetched_at),
                       file_name=f"committee-memo-{pd.Timestamp.now():%Y-%m-%d}.md", mime="text/markdown")
    st.caption("The holdings, the rules behind them, the case for and against each company, and the comparison with 90/10, ready to circulate before a vote.")
