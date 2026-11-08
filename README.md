# Mandate

**Your fund's investment policy, turned into a portfolio you can defend.**

Student-managed investment funds invest real money, but their members graduate every year and the fund's judgement leaves with them. Mandate writes the investment policy down as rules and applies them the same way every year. It records why each company was chosen, so any holding can be challenged against the exact test it passed or failed.

Started as an entry to the Microsoft x LSEG Hackathon 2026, whose brief was: *"A client has $100,000 to invest. Using public data and LLMs, build a tool that recommends where the money should go and explains why."*

*Educational project, not financial advice. The default rules are inspired by Benjamin Graham, Warren Buffett and Joseph Piotroski; not affiliated with or endorsed by any of them.*

## How it works
1. **Your fund's policy:** 8 tests on fundamentals, each with a threshold the committee can change or switch off. The defaults are my own, loosened from Graham's originals for today's markets:

   | Test | Default | Idea from |
   |---|---|---|
   | Always profitable | no loss-making years (of about 4) | Graham: earnings stability |
   | Earnings growing | earnings per share up over those years | Graham: earnings growth |
   | High return on equity | at least 15% | Buffett: a good business |
   | Strong margins | operating margin of at least 15% | Buffett: pricing power |
   | Modest debt | repayable from 4 years of free cash flow | Buffett: conservative debt |
   | Bills covered | current ratio of at least 1 (Graham asked for 2) | Graham: financial strength |
   | Real cash | free cash flow at least 70% of profit | Piotroski: earnings quality |
   | Fair price | at most 25 times earnings (Graham asked for 15) | Graham: moderate price |

   The portfolio rules are editable too: how many companies to own, how many tests they must pass, the cash reserve, the most in one company and the most per sector.
2. **The screen:** about 80 large companies are tested, and each result shows the company's actual figure. Missing data counts as a fail.
3. **The portfolio:** the highest scorers, cheapest first, within the sector limit. A company passing one test more than the minimum gets twice the money, up to the position limit. The cash reserve sits in short-term Treasuries.
4. **The case for and against:** the case for and against comes from the tests. Then an LLM runs an **annual-report check**: it reads each pick's latest 10-K or 20-F from SEC EDGAR and answers three questions the figures can't:
   - Is the business easy to understand?
   - Does it have a lasting competitive advantage?
   - Does management report candidly and use shareholders' money sensibly?

   Each answer gets a verdict of Strong, Mixed or Weak, plus a quote from the filing. Python checks every quote word for word against the filing, and flags any it can't find. The AI never chooses or sizes a holding. Results are saved in `data/annual_report_checks.json`, so visitors see them without a key.
5. **What could go wrong:** the portfolio against Buffett's advice for most people (2013 letter): 90% S&P 500, 10% short-term government bonds. This includes the 2020 and 2022 crashes.
6. **Committee memo:** a downloadable summary of the holdings, the exact policy applied, the cases and the comparison, ready to circulate before a vote.

**Data:** Yahoo Finance prices, ratios and about 4 years of company accounts, through `yfinance`, and annual reports from SEC EDGAR. A snapshot in `data/` keeps the app working if Yahoo is unreachable.

## Limitations, and the next step
- The backtest uses today's picks on old prices, so it is not a fair test in either direction. **Next step:** a point-in-time backtest that re-runs the screen each year using only the accounts published by then, from SEC EDGAR filings.
- Yahoo gives about 4 years of accounts; Graham looked at 10 or more.
- The tests can't judge competitive advantage or management quality. The annual-report check covers that, but it's an AI reading one year's filing, so treat it as a starting point for the committee's discussion, not a verdict.
- The annual-report check was built for the hackathon's Azure OpenAI endpoint, which has since been shut down. Add any working key to run it.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env     # optional: add an AI key and SEC contact for the annual-report check
streamlit run app.py
```
`LLM_PROVIDER=azure` uses Azure OpenAI. Any other value uses the OpenAI API, or an OpenAI-compatible service set through `LLM_ENDPOINT`. `SEC_USER_AGENT` is your name and email, which the SEC asks every automated tool to send. With no key, everything except the annual-report check still works.

## Deploy
The app runs on Streamlit Community Cloud from this repo, with `app.py` as the entry point, and redeploys on every push to `main`. Leave the key out of a public deployment, so visitors see the saved checks (`data/annual_report_checks.json`) and can't spend your credits. Never commit `.env`.

## Where to edit
- `CLASSIC_TESTS`: the tests and their default thresholds
- `DEFAULT_RULES`: the default portfolio rules
- `STOCKS`: the companies on offer
- `CHECK_SYSTEM`: the annual-report questions the AI answers
- `evidence()`: the figures the AI sees, and the place to add analyst ratings and news
