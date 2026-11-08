"""
Mandate: your fund's investment policy, turned into a portfolio you can defend.
Fixed rules screen the companies and size the portfolio; the AI only reads each pick's annual report and reports what it finds.

Run:  streamlit run app.py
Educational project. Not financial advice.
"""
import html
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
CHECKS_JSON = os.path.join(DATA_DIR, "annual_report_checks.json")    # saved AI checks, shown to every visitor
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")   # the SEC asks automated tools for a name and contact email
PRICE_START = "2019-01-01"   # far enough back to replay the COVID crash

CASH_TICKER, INDEX_TICKER = "SHY", "VOO"   # short-term Treasuries; S&P 500 index fund
DISCLAIMER = ("Educational project, not financial advice. The default rules are inspired by Benjamin Graham, Warren Buffett "
              "and Joseph Piotroski; not affiliated with or endorsed by any of them.")

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


# The default policy: (test, what the threshold means, default threshold, unit, where the idea comes from,
# the test given the data and a threshold, the company's actual figure).
# The thresholds are my own defaults, loosened from Graham's originals for today's markets; committees can change them in the app.
CLASSIC_TESTS = [
    ("Always profitable", "Most loss-making years allowed (of about 4 reported)", 0, "years", "Graham: earnings stability",
     lambda d, x: (d.years_reported >= 3) & (d.loss_years <= x),
     lambda r: f"{r.loss_years:.0f} of {r.years_reported:.0f} years in loss"),
    ("Earnings growing", "Least growth in earnings per share over those years", 0, "%", "Graham: earnings growth",
     lambda d, x: d.eps_latest > d.eps_oldest + d.eps_oldest.abs() * x / 100,
     lambda r: f"EPS {_fact(r.eps_oldest, '{:.2f}')} to {_fact(r.eps_latest, '{:.2f}')}"),
    ("High return on equity", "Lowest return on equity", 15, "%", "Buffett: a good business",
     lambda d, x: d["roe_%"] >= x,
     lambda r: _fact(r["roe_%"], "{:.0f}%")),
    ("Strong margins", "Lowest operating margin", 15, "%", "Buffett: pricing power",
     lambda d, x: d["operating_margin_%"] >= x,
     lambda r: _fact(r["operating_margin_%"], "{:.0f}%")),
    ("Modest debt", "Most years of free cash flow needed to repay all debt", 4, "years", "Buffett: conservative debt",
     lambda d, x: d.debt_to_fcf_years <= x,
     lambda r: _fact(r.debt_to_fcf_years, "{:.1f} years")),
    ("Bills covered", "Lowest current ratio (current assets ÷ current liabilities)", 1, "x", "Graham: financial strength (he asked for 2)",
     lambda d, x: d.current_ratio >= x,
     lambda r: _fact(r.current_ratio, "{:.2f}x")),
    ("Real cash", "Lowest free cash flow as a share of reported profit", 70, "%", "Piotroski: earnings quality",
     lambda d, x: d.fcf_to_net_income >= x / 100,
     lambda r: _fact(r.fcf_to_net_income, "{:.0%}")),
    ("Fair price", "Highest price-to-earnings ratio", 25, "x", "Graham: moderate price (he asked for 15)",
     lambda d, x: (d.trailing_pe > 0) & (d.trailing_pe <= x),
     lambda r: _fact(r.trailing_pe, "{:.0f}x earnings")),
]
TESTS = {t[0]: t for t in CLASSIC_TESTS}
DEFAULT_POLICY = pd.DataFrame([{"Use": True, "Test": name, "Threshold": float(x), "Unit": unit, "What it means": rule, "Idea from": src}
                               for name, rule, x, unit, src, _, _ in CLASSIC_TESTS])
DEFAULT_RULES = {"n_picks": 8, "min_tests": 7, "cash": 0.10, "max_position": 0.25, "max_per_sector": 3}

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
# 3. THE POLICY  (screen the companies, then size the positions)
# =====================================================================================
def run_screen(fundamentals: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Every company against the fund's tests ({test: threshold}), most passed then cheapest first.
    Missing data counts as a fail."""
    df = fundamentals.copy()
    for name, x in policy.items():
        df[name] = TESTS[name][5](df, x)
    df["tests_passed"] = df[list(policy)].sum(axis=1)
    df["tests_failed"] = df[list(policy)].apply(lambda r: ", ".join(r.index[~r]) or "none", axis=1)
    return df.sort_values(["tests_passed", "trailing_pe"], ascending=[False, True])


def scorecard(df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """One row per company: a tick or cross for each test, next to the company's actual figure."""
    cells = {name: [f"{'✓' if r[name] else '✗'} {TESTS[name][6](r)}" for _, r in df.iterrows()] for name in policy}
    return pd.DataFrame({"Company": df["name"].values, "Passed": df.tests_passed.map(lambda n: f"{n}/{len(policy)}").values, **cells},
                        index=df.index)


def build_portfolio(screen: pd.DataFrame, rules: dict, amount: float):
    """Pick the top scorers within the sector limit. Each test passed above the minimum adds a share of the money;
    anything above the position limit stays in cash. Returns (picks, holdings table, companies that passed but missed out and why)."""
    picks, sectors, skipped = [], {}, []
    for t, r in screen[screen.tests_passed >= rules["min_tests"]].iterrows():
        if len(picks) >= rules["n_picks"]:
            skipped.append((t, "portfolio full"))
        elif sectors.get(r.sector, 0) >= rules["max_per_sector"]:
            skipped.append((t, f"already {rules['max_per_sector']} in {r.sector}"))
        else:
            picks.append(t)
            sectors[r.sector] = sectors.get(r.sector, 0) + 1
    chosen = screen.loc[picks]
    shares = chosen.tests_passed - rules["min_tests"] + 1   # one test above the minimum gets twice the money
    weights = (shares / shares.sum() * (1 - rules["cash"])).clip(upper=rules["max_position"]) if picks else pd.Series(dtype=float)
    rows = [{"Holding": chosen.loc[t, "name"], "ticker": t, "Sector": chosen.loc[t, "sector"],
             "Tests passed": f"{chosen.loc[t, 'tests_passed']}/{rules['n_tests']}", "usd": round(weights[t] * amount)} for t in picks]
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


def policy_lines(policy: dict, rules: dict) -> list:
    """The policy in words, for the memo."""
    unit = lambda name: TESTS[name][3] if TESTS[name][3] in ("%", "x") else " " + TESTS[name][3]
    lines = [f"- **{name}:** {TESTS[name][1]}: {x:g}{unit(name)} ({TESTS[name][4]})" for name, x in policy.items()]
    return lines + [f"- **Portfolio:** up to {rules['n_picks']} companies passing at least {rules['min_tests']} of {len(policy)} tests, "
                    f"highest scores first, then lowest price-to-earnings; at most {rules['max_per_sector']} per sector and "
                    f"{rules['max_position']:.0%} per company; {rules['cash']:.0%} in short-term Treasuries"]

# =====================================================================================
# 4. THE ANNUAL-REPORT CHECK  (the AI reads; it does not choose)
# =====================================================================================
CHECK_SYSTEM = """You help a student investment fund's committee check a company that the fund's rules have already chosen.
Read the annual-report excerpts and answer three questions the financial figures can't:
1. Is the business easy to understand?
2. Does it have a lasting competitive advantage?
3. Does management report candidly and use shareholders' money sensibly?

For each, give a "verdict" of "Strong", "Mixed" or "Weak", an "answer" of up to 2 short sentences in plain English, and a "quote":
one sentence copied word for word from the excerpts that best supports the answer. Never paraphrase the quote.
Judge only from the excerpts and the figures given. Don't predict the share price or claim to speak for any famous investor.
Reply with ONLY a JSON object: {"answers": [{"question": "...", "verdict": "...", "answer": "...", "quote": "..."}]}"""

SECTIONS_10K = {"Business": r"item\s*1\s*[\.:\-–—]?\s*business",
                "Risk factors": r"item\s*1a\s*[\.:\-–—]?\s*risk factors",
                "Management's discussion": r"item\s*7\s*[\.:\-–—]?\s*management.s discussion"}
NEXT_ITEM = re.compile(r"^\s*item\s*\d{1,2}[a-c]?\s*[\.:\-–—]", re.I | re.M)
KEYWORDS = re.compile(r"compet|advantage|brand|patent|switching|network effect|pricing|market share|repurchase|buyback|dividend|"
                      r"acquisition|capital allocation|return on", re.I)
SECTION_CHARS = 30_000   # per section, to keep each AI call to a sensible size


def evidence(picks: pd.DataFrame, policy: dict) -> dict:
    """The figures the AI sees about each company. Add analyst ratings and news here when they join the project."""
    return {t: {"name": r["name"], "sector": r.sector, "tests_passed": f"{r.tests_passed} of {len(policy)}",
                "tests_failed": r.tests_failed, "figures": {name: TESTS[name][6](r) for name in policy}}
            for t, r in picks.iterrows()}


def _sec_get(url: str):
    import requests
    r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT}, timeout=60)
    r.raise_for_status()
    return r


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def sec_ciks() -> dict:
    """Ticker -> SEC company number."""
    return {v["ticker"]: v["cik_str"] for v in _sec_get("https://www.sec.gov/files/company_tickers.json").json().values()}


def _plain_text(page: str) -> str:
    """Filing HTML -> text, one paragraph per line."""
    t = re.sub(r"<(script|style|ix:header)[^>]*>.*?</\1>", " ", page, flags=re.S | re.I)   # ix:header is hidden machine-readable data
    t = re.sub(r"<(br|/p|/div|/tr|/li|/h\d)[^>]*>", "\n", t, flags=re.I)
    t = html.unescape(re.sub(r"<[^>]+>", " ", t))
    return re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t\xa0]+", " ", t))


def filing_excerpts(text: str) -> dict:
    """The business, risk and management-discussion sections of a 10-K, or the most relevant paragraphs of any other filing."""
    out = {}
    for label, pattern in SECTIONS_10K.items():
        parts = []
        for m in re.finditer(r"^\s*" + pattern, text, re.I | re.M):   # some filings repeat the heading on every page
            nxt = NEXT_ITEM.search(text, m.end())
            part = text[m.end(): nxt.start() if nxt else len(text)]
            if len(part) > 500:   # skip the contents-page entries
                parts.append(part)
        if sum(map(len, parts)) > 2000:
            out[label] = " ".join(parts)[:SECTION_CHARS]
    if len(out) < 2:   # 20-F and other layouts: keep the blocks of text that talk most about competition and capital
        blocks, block = [], ""
        for line in text.split("\n"):
            block += line.strip() + " "
            if len(block) > 1500:
                blocks.append(block)
                block = ""
        relevant = [b for b in blocks if len(KEYWORDS.findall(b)) >= 2]
        out = {"Key passages": "\n".join(relevant)[:3 * SECTION_CHARS]}
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def annual_report(t: str) -> dict:
    """The company's latest 10-K or 20-F from SEC EDGAR: where it is, the full text (for checking quotes) and the excerpts the AI reads."""
    cik = sec_ciks()[t]
    index = _sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()["filings"]
    pages = [index["recent"]] + [f["name"] for f in index.get("files", [])]   # busy filers' older filings are on extra pages
    for page in pages:
        f = page if isinstance(page, dict) else _sec_get(f"https://data.sec.gov/submissions/{page}").json()
        for form, acc, doc, filed in zip(f["form"], f["accessionNumber"], f["primaryDocument"], f["filingDate"]):
            if form in ("10-K", "20-F"):
                url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
                text = _plain_text(_sec_get(url).text)
                return {"form": form, "filed": filed, "url": url, "text": text, "excerpts": filing_excerpts(text)}
    raise LookupError(f"no annual report on EDGAR for {t}")


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s).replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')).strip().lower()


def quote_found(quote: str, filing_norm: str) -> bool:
    """True if every part of the quote (split at any '...') appears word for word in the filing."""
    parts = [p.strip(" .") for p in re.split(r"\.\.\.|\u2026", _norm(quote))]
    parts = [p for p in parts if len(p) > 15]
    return bool(parts) and all(p in filing_norm for p in parts)


def get_client():
    if PROVIDER == "azure":
        from openai import AzureOpenAI
        return AzureOpenAI(api_key=API_KEY, azure_endpoint=ENDPOINT, api_version=AZURE_API_VERSION)
    from openai import OpenAI
    return OpenAI(api_key=API_KEY, base_url=ENDPOINT or None)


def check_company(t: str, facts: dict) -> dict:
    """Read one annual report and answer the three questions, with each quote checked against the filing."""
    report = annual_report(t)
    user = json.dumps({"company": facts, "filing": f"{report['form']} filed {report['filed']}", "excerpts": report["excerpts"]})
    resp = get_client().chat.completions.create(
        model=MODEL, max_completion_tokens=4000,
        messages=[{"role": "system", "content": CHECK_SYSTEM}, {"role": "user", "content": user}])
    found = re.search(r"\{.*\}", resp.choices[0].message.content or "", re.S)
    if not found:
        raise ValueError("no JSON in the reply")
    filing_norm = _norm(report["text"])
    answers = [{**a, "quote": " ".join(str(a.get("quote", "")).split()), "quote_found": quote_found(a.get("quote", ""), filing_norm)}
               for a in json.loads(found.group(0)).get("answers", [])[:3]]
    return {"form": report["form"], "filed": report["filed"], "url": report["url"],
            "checked_at": pd.Timestamp.now().strftime("%d %b %Y"), "answers": answers}


def load_checks() -> dict:
    try:
        with open(CHECKS_JSON) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def run_checks(picks: pd.DataFrame, policy: dict) -> dict:
    """Check every pick in parallel and save the results for every visitor. Returns {ticker: error} for any that failed."""
    facts, errors, results = evidence(picks, policy), {}, {}
    sec_ciks()   # fetch the ticker list once, before the threads need it
    with ThreadPoolExecutor(max_workers=4) as pool:   # the SEC allows 10 requests a second
        jobs = {t: pool.submit(check_company, t, facts[t]) for t in picks.index}
    for t, job in jobs.items():
        try:
            results[t] = job.result()
        except Exception as e:
            errors[t] = str(e)[:80]
    if results:
        with open(CHECKS_JSON, "w") as f:
            json.dump({**load_checks(), **results}, f, indent=1)
    return errors


def rule_case(r: pd.Series, policy: dict) -> dict:
    """The case for and against from the tests alone."""
    said = lambda keep: "; ".join(f"{name.lower()} ({TESTS[name][6](r)})" for name in policy if bool(r[name]) == keep)
    against = said(False)
    first_sentence = re.split(r"(?<=[a-z]\.)(?<!Inc\.)(?<!Corp\.)(?<!Co\.)(?<!Ltd\.)\s+(?=[A-Z])", r.business or "")[0]
    return {"business": first_sentence,
            "case_for": f"Passes: {said(True)}." if said(True) else "It passes none of the tests.",
            "case_against": f"Fails: {against}." if against else f"Passes all {len(policy)} tests, so the risk is what numbers can't show: "
                                                                 "whether the business will still be this strong in 10 years."}


def committee_memo(holdings, picks, cases, checks, amount, policy, rules, results, fetched_at) -> str:
    """The recommendation as a Markdown memo the fund's committee can circulate and vote on."""
    lines = [f"# Investment committee memo: ${amount:,.0f}", f"Prepared {pd.Timestamp.now():%d %b %Y} with Mandate. Market data from Yahoo Finance, as of {fetched_at}.",
             "", "## Recommendation", "| Holding | Ticker | Tests passed | Amount | Share |", "|---|---|---|---|---|"]
    lines += [f"| {r.Holding} | {r.ticker} | {r['Tests passed']} | ${r.usd:,.0f} | {r.Share:.0f}% |" for _, r in holdings.iterrows()]
    lines += ["", "## The policy applied", *policy_lines(policy, rules), "", "## The case for and against"]
    for t, r in picks.iterrows():
        c = cases[t]
        lines += [f"### {r['name']} ({t})", c.get("business", ""), f"- **For:** {c.get('case_for', '')}", f"- **Against:** {c.get('case_against', '')}"]
        if t in checks:
            k = checks[t]
            lines += [f"- **Annual report check** ([{k['form']} filed {k['filed']}]({k['url']}), read by AI):"]
            lines += [f"  - **{a.get('verdict', '?')}:** {a.get('question', '')} {a.get('answer', '')} "
                      f"Quote: \"{a.get('quote', '')}\"{'' if a.get('quote_found') else ' (not found in the filing)'}" for a in k["answers"]]
        lines += [""]
    lines += ["## Against the simple alternative", "| Measure | This portfolio | Buffett's 90/10 |", "|---|---|---|"]
    lines += ["| " + " | ".join(r.values()) + " |" for r in results]
    lines += ["", "Hindsight warning: the companies were chosen with today's figures and tested on old prices, so this shows how bumpy "
              "the ride could be, not whether the policy works.", "", f"*{DISCLAIMER}*"]
    return "\n".join(lines)

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


st.set_page_config(page_title="Mandate", layout="wide")
st.markdown("""<style>
h1, h2, h3, h4 { color: #0A3D7A; font-weight: 600; letter-spacing: -0.01em; }
[data-testid="stMetric"] { background: #F2F5F9; border-left: 4px solid #0A3D7A; padding: 10px 14px; border-radius: 4px; }
.block-container { padding-top: 2rem; max-width: 1200px; }
</style>""", unsafe_allow_html=True)
st.markdown("## Mandate")
st.markdown("**Your fund's investment policy, turned into a portfolio you can defend.**")
st.markdown("Student-managed funds invest real money, but their members graduate every year and the fund's judgement leaves with them. "
            "Mandate writes the policy down as rules, applies them the same way every year, and records why each company was chosen, "
            "so any holding can be challenged against the exact test it passed or failed.")
st.caption("The rules choose the companies and size the portfolio. The AI never picks a stock: it reads each pick's annual report and answers "
           "three questions the figures can't, quoting its evidence. " + DISCLAIMER)

try:
    with st.spinner("Fetching live market data..."):
        prices, fundamentals, fetched_at, data_warnings = load_market()
except Exception as e:
    st.error(f"Couldn't load market data and no saved data exists: {e}")
    st.stop()
for w in data_warnings:
    st.warning(f"Live data unavailable: {w}.")

step = iter(range(1, 10))

# ---- Step 1: the policy
st.markdown(f"### {next(step)} · Your fund's policy")
st.caption("The tests and thresholds below are my defaults, loosened from Graham's originals for today's markets. "
           "Change any threshold, or untick a test, to match your own fund's policy.")
edited = st.data_editor(DEFAULT_POLICY, key="policy", hide_index=True, width="stretch",
                        disabled=["Test", "Unit", "What it means", "Idea from"],
                        column_config={"Use": st.column_config.CheckboxColumn("Use", width="small"),
                                       "Threshold": st.column_config.NumberColumn("Threshold", min_value=0.0, step=0.5, width="small"),
                                       "Unit": st.column_config.TextColumn("Unit", width="small")})
policy = {r.Test: float(r.Threshold if pd.notna(r.Threshold) else TESTS[r.Test][2]) for r in edited.itertuples() if r.Use}
if not policy:
    st.warning("Tick at least one test to build a portfolio.")
    st.stop()

c1, c2, c3, c4, c5, c6 = st.columns(6)
amount = c1.number_input("Amount to invest ($)", min_value=1_000, max_value=10_000_000, value=100_000, step=1_000)
rules = {"n_tests": len(policy),
         "n_picks": c2.number_input("Companies to own", 1, 20, DEFAULT_RULES["n_picks"]),
         "min_tests": c3.number_input(f"Tests to pass (of {len(policy)})", 1, len(policy), min(DEFAULT_RULES["min_tests"], len(policy))),
         "cash": c4.number_input("Cash reserve (%)", 0, 100, int(DEFAULT_RULES["cash"] * 100)) / 100,
         "max_position": c5.number_input("Most in one company (%)", 1, 100, int(DEFAULT_RULES["max_position"] * 100)) / 100,
         "max_per_sector": c6.number_input("Most per sector", 1, 20, DEFAULT_RULES["max_per_sector"])}
left_out = st.multiselect("Leave out any company the committee doesn't understand or won't own",
                          options=list(fundamentals.index), format_func=lambda t: f"{t} · {STOCKS.get(t, t)}")

screen = run_screen(fundamentals.loc[[t for t in fundamentals.index if t in prices]], policy)
pool = screen.drop(index=left_out, errors="ignore")
picks, holdings, skipped = build_portfolio(pool, rules, amount)
passed = pool[pool.tests_passed >= rules["min_tests"]]

# ---- Step 2: the screen
st.markdown(f"### {next(step)} · The screen")
st.markdown(f"**{len(passed)} of {len(screen)} companies** pass at least {rules['min_tests']} of your {len(policy)} tests.")
st.dataframe(scorecard(pool.head(15), policy), width="stretch")
with st.expander(f"See how all {len(screen)} companies scored"):
    st.dataframe(scorecard(screen, policy), width="stretch")
st.caption(f"Yahoo Finance, as of {fetched_at}. Missing data counts as a fail. Yahoo gives about 4 years of accounts; Graham wanted 10.")

# ---- Step 3: the portfolio
st.markdown(f"### {next(step)} · The portfolio")
if picks.empty:
    st.info("No company meets the policy, so it all stays in cash. Loosen a threshold or lower the number of tests to see more.")
st.dataframe(holdings[["Holding", "ticker", "Sector", "Tests passed", "usd", "Share"]], width="stretch", hide_index=True,
             column_config={"ticker": "Ticker", "usd": st.column_config.NumberColumn("Amount", format="$%d"),
                            "Share": st.column_config.ProgressColumn("Share", format="%.0f%%", min_value=0, max_value=100)})
st.caption(f"Highest scores first, then lowest price-to-earnings, with at most {rules['max_per_sector']} per sector. A company passing one test more "
           f"than the minimum gets twice the money, up to {rules['max_position']:.0%} each. {rules['cash']:.0%} stays in cash, plus anything above that limit.")
if skipped:
    st.caption("Passed but missed out: " + ", ".join(f"{t} ({why})" for t, why in skipped))

# ---- Step 4: the case for and against
if len(picks):
    st.markdown(f"### {next(step)} · The case for and against")
    st.caption("For and against come from the tests. The annual-report check is done by AI: it reads each company's latest annual report "
               "from SEC EDGAR and judges what the figures can't. Every quote is checked word for word against the filing.")
    checks = load_checks()
    todo = [t for t in picks.index if t not in checks]
    if todo and USE_AI and SEC_USER_AGENT:
        if st.button(f"Read the annual reports for {len(todo)} {'company' if len(todo) == 1 else 'companies'} (AI)", type="primary"):
            with st.spinner("Reading annual reports from SEC EDGAR..."):
                errors = run_checks(picks.loc[todo], policy)
            for t, e in errors.items():
                st.warning(f"{t}: couldn't check the annual report ({e}).")
            checks = load_checks()
    elif todo:
        st.caption(f"{len(todo)} of {len(picks)} not checked yet. Running the check needs an AI key and SEC_USER_AGENT in .env.")
    cases = {t: rule_case(r, policy) for t, r in picks.iterrows()}
    for t, r in picks.iterrows():
        case = cases[t]
        with st.container(border=True):
            st.markdown(f"**{md(r['name'])}** ({t}) · {md(case.get('business'))}")
            col_for, col_against = st.columns(2)
            col_for.markdown(f"**For.** {md(case.get('case_for'))}")
            col_against.markdown(f"**Against.** {md(case.get('case_against'))}")
            if t in checks:
                k = checks[t]
                st.markdown(f"**Annual report check** · [{k['form']} filed {k['filed']}]({k['url']}) · read by AI on {k['checked_at']}")
                for a in k["answers"]:
                    colour = {"Strong": "green", "Mixed": "orange", "Weak": "red"}.get(a.get("verdict"), "gray")
                    st.markdown(f":{colour}[**{a.get('verdict', '?')}**] · *{md(a.get('question'))}* {md(a.get('answer'))}")
                    found = "found in the filing" if a.get("quote_found") else "⚠ not found in the filing, so treat this answer with caution"
                    st.caption(f"\u201c{md(a.get('quote'))}\u201d ({found})")
            else:
                st.caption("Annual report not checked yet.")

# ---- Step 5: what could go wrong
st.markdown(f"### {next(step)} · What could go wrong")
st.markdown("Buffett's 2013 shareholder letter gives simple advice for most people: 90% in a low-cost S&P 500 fund and 10% in short-term "
            "government bonds. That's the bar any fund picking companies has to beat.")
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
           "It shows how bumpy the ride could be, not whether the policy works.")

# ---- Step 6: for the committee
if len(picks):
    st.markdown(f"### {next(step)} · Take it to the committee")
    st.download_button("Download the committee memo", committee_memo(holdings, picks, cases, checks, amount, policy, rules, table, fetched_at),
                       file_name=f"committee-memo-{pd.Timestamp.now():%Y-%m-%d}.md", mime="text/markdown")
    st.caption("The holdings, the exact policy applied, the case for and against each company with its annual-report check, and the comparison with 90/10, "
               "ready to circulate before a vote.")
