"""Small, explicit research proxy universe, not personalized investability approval."""
PROXIES = {
    'equity_us': ('SPY', 'USD', 'US large-cap equity proxy'),
    'government_bond_us_intermediate': ('IEF', 'USD', 'US 7-10 year Treasury proxy'),
    'commodity_gold': ('GLD', 'USD', 'Physically backed gold trust proxy'),
    'listed_reit_us': ('VNQ', 'USD', 'US listed real estate proxy'),
}
CASH_TICKER = '449170.KS'  # TIGER KOFR active (synthetic); not an insured deposit.
FX_TICKER = 'KRW=X'       # KRW per USD; never multiply a KRW instrument by this again.
SYMBOLS = {**{v[0]: v[1] for v in PROXIES.values()}, CASH_TICKER: 'KRW', FX_TICKER: 'KRW'}
SOURCE_ID = 'yahoo_finance_research'


def series_id(ticker):
    return 'yf:' + ticker + (':close' if ticker == FX_TICKER else ':adj_close')
