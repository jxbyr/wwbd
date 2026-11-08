# WWBD: What Would Buffett Do?

**For student-managed investment funds and investment clubs:** turn $100,000 into a short list of companies the committee can defend, using fixed rules anyone can check.

Started as an entry to the Microsoft x LSEG Hackathon 2026, whose brief was: *"A client has $100,000 to invest. Using public data and LLMs, build a tool that recommends where the money should go and explains why."*

*Educational project, not financial advice. Not affiliated with or endorsed by Warren Buffett or Berkshire Hathaway. The rules are our own reading of his published principles.*

## The client
Student funds, investment clubs and small charities invest real money through a committee, and every holding has to be defended to members, trustees or a faculty adviser. Their problem is not finding stock tips. It is having a process that is consistent, documented and easy to challenge. WWBD gives them one, and it compares itself honestly with the simple alternative of an index fund.

## How it works
1. **Screen:** about 80 large companies face 8 tests on fundamentals, inspired by Graham, Buffett and Piotroski:
   - always profitable
   - earnings growing
   - return on equity
   - operating margin
   - debt against free cash flow
   - current ratio
   - free cash flow against reported profit
   - price-to-earnings

   Missing data counts as a fail.
2. **Portfolio:** the highest scorers, cheapest first, up to 3 per sector. Better scores get bigger positions, capped at 25% each. 10% stays in short-term Treasuries, and anything above a cap stays there too.
3. **Case for and against:** an LLM argues each pick from the figures. It never chooses or sizes a holding. Without an API key, the case is summarised from the tests.
4. **What could go wrong:** the portfolio against Buffett's own advice for most people (2013 letter): 90% S&P 500, 10% short-term government bonds. This includes the 2020 and 2022 crashes.
5. **Committee memo:** a downloadable summary of the holdings, the rules, the cases and the comparison, ready to circulate before a vote.

**Data:** Yahoo Finance prices, ratios and about 4 years of company accounts, through `yfinance`. A snapshot in `data/` keeps the app working if Yahoo is unreachable.

## Limitations, and the next step
- The backtest uses today's picks on old prices, so it is not a fair test in either direction. **Next step:** a point-in-time backtest that re-runs the screen each year using only the accounts published by then, from SEC EDGAR filings.
- Yahoo gives about 4 years of accounts; Graham and Buffett looked at a decade or more.
- The tests can't judge what Buffett cared about most: a lasting competitive advantage and honest management.

## Run it
```bash
pip install -r requirements.txt
cp .env.example .env     # optional: add an Azure OpenAI or OpenAI key for the AI cases
streamlit run app.py
```
`LLM_PROVIDER=azure` uses Azure OpenAI. Any other value uses the OpenAI API, or an OpenAI-compatible service set through `LLM_ENDPOINT`. With no key, everything except the AI cases still works.

## Deploy
Push to GitHub, then create an app on [Streamlit Community Cloud](https://share.streamlit.io) from the repo, with `app.py` as the entry point. Leave the key out of a public deployment, so visitors see the saved cases (`data/last_cases.json`) and can't spend your credits. Never commit `.env`.

## Where to edit
- `CLASSIC_TESTS`: the rules and thresholds
- `STOCKS`: the companies on offer
- `CASH_SHARE`, `MAX_POSITION`, `MAX_PER_SECTOR`: how the portfolio is built
- `evidence()`: what the AI sees, and the place to add analyst ratings and news
