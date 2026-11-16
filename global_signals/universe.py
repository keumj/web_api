"""Explicit provider series and analysis bindings for every catalogue exposure.

Context bindings never imply a matching tradable product, duration, or credit grade.
"""
from .catalog import EXPOSURES

CASH_TICKER = '449170.KS'
FX_TICKER = 'KRW=X'
SOURCE_ID = 'yahoo_finance_research'
SERIES = {}
BINDINGS = {}


def yahoo(ticker, currency, label, kind='adjusted_price', url=None):
    SERIES[ticker] = dict(provider='yahoo', code=ticker, currency=currency, label=label,
                          kind=kind, frequency='daily', max_age_days=7,
                          url=url or 'https://finance.yahoo.com/quote/' + ticker)
    return ticker


def fred(code, label, kind='yield', frequency='monthly'):
    key = 'fred:' + code
    SERIES[key] = dict(provider='fred', code=code, currency='PCT' if kind == 'yield' else 'INDEX',
                       label=label, kind=kind, frequency=frequency,
                       max_age_days={'daily': 10, 'monthly': 120, 'quarterly': 270}[frequency],
                       url='https://fred.stlouisfed.org/series/' + code)
    return key


def bis_policy(country, label):
    key = 'bis:policy:' + country
    SERIES[key] = dict(provider='bis', code='D.' + country, currency='PCT', label=label,
                       kind='yield', frequency='daily', max_age_days=14,
                       url='https://data.bis.org/topics/CBPOL')
    return key


yahoo(CASH_TICKER, 'KRW', 'TIGER KOFR active synthetic cash ETF')
yahoo(FX_TICKER, 'KRW', 'KRW per USD', 'fx_spot')
# Cross legs are explicitly oriented. Never treat JPY as a 100-yen quote.
FX_LEGS = {'USD': (FX_TICKER, 'direct'), 'KRW': (None, 'identity')}
for currency, ticker, side in [
    ('EUR', 'EURUSD=X', 'usd_per_foreign'), ('GBP', 'GBPUSD=X', 'usd_per_foreign'),
    ('AUD', 'AUDUSD=X', 'usd_per_foreign'), ('JPY', 'JPY=X', 'foreign_per_usd'),
    ('CHF', 'CHF=X', 'foreign_per_usd'), ('CAD', 'CAD=X', 'foreign_per_usd'),
    ('CNY', 'CNY=X', 'foreign_per_usd'), ('SGD', 'SGD=X', 'foreign_per_usd'),
    ('HKD', 'HKD=X', 'foreign_per_usd'),
]:
    quote = 'USD' if side == 'usd_per_foreign' else currency
    yahoo(ticker, quote, f'{currency}/USD cross leg ({side})', 'fx_spot')
    FX_LEGS[currency] = (ticker, side)


def fx_dependencies(currency):
    ticker, side = FX_LEGS[currency]
    if side == 'identity':
        return []
    return [FX_TICKER] if side == 'direct' else [FX_TICKER, ticker]


def proxy(exposure, ticker, currency, label, note='', kind='adjusted_price'):
    yahoo(ticker, currency, label, kind)
    BINDINGS[exposure] = dict(mode='forecast', primary=ticker, quote_currency=currency,
                              series=list(dict.fromkeys([ticker, *fx_dependencies(currency), CASH_TICKER])),
                              note=note or 'Representative research proxy; personal access is unverified.')


def context(exposure, sources, note):
    BINDINGS[exposure] = dict(mode='context', primary=sources[0], series=list(dict.fromkeys(sources)), note=note)


for country, ticker in {
    'kr': 'EWY', 'us': 'SPY', 'jp': 'EWJ', 'cn': 'MCHI', 'hk': 'EWH', 'tw': 'EWT',
    'in': 'INDA', 'au': 'EWA', 'de': 'EWG', 'gb': 'EWU', 'fr': 'EWQ', 'ch': 'EWL',
    'ca': 'EWC', 'br': 'EWZ', 'mx': 'EWW', 'za': 'EZA',
}.items():
    proxy('equity_' + country, ticker, 'USD', f'{country.upper()} country equity ETF proxy',
          'US-listed country equity proxy. Its USD return already includes underlying currency moves; convert USD to KRW once.')

for duration, ticker in [('short', 'SHY'), ('intermediate', 'IEF'), ('long', 'TLT')]:
    proxy('government_bond_us_' + duration, ticker, 'USD', 'US Treasury ' + duration,
          {'short': '1–3 year US Treasury ETF.', 'intermediate': '7–10 year US Treasury ETF.', 'long': '20+ year US Treasury ETF.'}[duration])

for country in ('KR', 'JP', 'DE', 'GB', 'AU', 'CA'):
    short = (fred('IRSTCI01GBM156N', 'GBP overnight/interbank market rate') if country == 'GB' else
             fred(f'IR3TIB01{country}M156N', f'{country} 3-month interbank rate (not a government bond return)'))
    long = fred(f'IRLTLT01{country}M156N', f'{country} 10-year government yield')
    for duration in ('short', 'intermediate', 'long'):
        context(f'government_bond_{country.lower()}_{duration}', [short, long],
                'Money-market rate / 10-year government yield context only. A matching sovereign maturity-bucket total-return instrument is not connected.')

proxy('corporate_bond_usd_ig', 'LQD', 'USD', 'USD investment-grade corporate bond ETF')
proxy('corporate_bond_usd_hy', 'HYG', 'USD', 'USD high-yield corporate bond ETF')
proxy('corporate_bond_eur_ig', 'IEAC.AS', 'EUR', 'EUR investment-grade corporate bond ETF')
proxy('corporate_bond_eur_hy', 'IHYG.L', 'EUR', 'EUR high-yield corporate bond ETF (EUR quote on LSE)')
proxy('corporate_bond_gbp_ig', 'SLXX.L', 'GBP', 'Sterling corporate bond ETF')
for currency, country in [('GBP', 'GB'), ('KRW', 'KR'), ('JPY', 'JP')]:
    rate = fred(f'IRLTLT01{country}M156N', f'{country} 10-year government yield')
    for grade in ('ig', 'hy'):
        exposure = f'corporate_bond_{currency.lower()}_{grade}'
        if exposure not in BINDINGS:
            context(exposure, [rate, *fx_dependencies(currency)],
                    'Interest-rate / FX context only, not a credit spread or corporate bond return. Matching credit-grade product data is still required.')

for country, ticker, currency, note in [
    ('us', 'VNQ', 'USD', 'US listed real estate ETF.'),
    ('kr', '329200.KS', 'KRW', 'Korean REIT/infrastructure income basket; includes infrastructure and is not a pure REIT index.'),
    ('jp', '1343.T', 'JPY', 'Tokyo Stock Exchange REIT ETF.'),
    ('gb', 'IUKP.L', 'GBP', 'UK listed property ETF; includes property companies beyond REITs.'),
    ('au', 'VAP.AX', 'AUD', 'Australian A-REIT index ETF.'),
    ('ca', 'XRE.TO', 'CAD', 'Canadian capped REIT index ETF.'),
    ('sg', 'CLR.SI', 'SGD', 'Singapore S-REIT ETF.'),
]:
    proxy('listed_reit_' + country, ticker, currency, note, note)

for country in ('KR', 'US', 'JP', 'GB', 'DE', 'FR', 'AU', 'CA', 'SG'):
    residential = fred(f'Q{country}N628BIS', f'{country} nominal residential property price index (2010=100)', 'property_index', 'quarterly')
    context('physical_property_' + country.lower(), [residential],
            'BIS quarterly residential price context; excludes rent, transaction costs and commercial property. No one-month trading signal.')
    if country in ('DE', 'FR'):
        context('listed_reit_' + country.lower(), [residential],
                'Residential property context only; this is not a listed REIT return. A country-specific listed property proxy is still required.')

for name, ticker in [('gold', 'GLD'), ('silver', 'SLV'), ('copper', 'CPER'), ('oil', 'USO'), ('gas', 'UNG'), ('agriculture', 'DBA')]:
    proxy('commodity_' + name, ticker, 'USD', name.title() + ' exchange-traded proxy',
          'Physically backed trust, net of fund expenses.' if name in ('gold', 'silver') else
          'Futures-based fund return includes its roll/collateral mechanics; it is not the raw spot commodity return.')

for currency in ('USD', 'EUR', 'JPY', 'GBP', 'CHF', 'AUD', 'CAD', 'CNY', 'SGD', 'HKD'):
    BINDINGS[f'fx_{currency.lower()}_krw'] = dict(mode='fx', primary=FX_LEGS[currency][0], quote_currency=currency,
        series=list(dict.fromkeys([*fx_dependencies(currency), CASH_TICKER])),
        note='Spot FX price exposure only, expressed as KRW per one foreign unit. No deposit interest, forward carry, or hedging is included.')

for name in ('btc', 'eth'):
    proxy('crypto_' + name, name.upper() + '-USD', 'USD', name.upper() + ' USD aggregate spot price',
          'Provider aggregate USD spot price; no staking income or Korean exchange premium. Not an executable venue-specific quote.', 'spot_price')

context('cash_krw', [CASH_TICKER], 'KRW cash-like synthetic KOFR ETF comparator; not an insured deposit. No signal against itself.')

SHORT_RATES = {'USD': fred('SOFR', 'US secured overnight financing rate', frequency='daily'),
               'EUR': fred('ECBDFR', 'ECB deposit facility policy rate', frequency='daily')}
for currency, country in [('JPY', 'JP'), ('GBP', 'GB'), ('AUD', 'AU'), ('CAD', 'CA')]:
    SHORT_RATES[currency] = fred(f'IRSTCI01{country}M156N', f'{currency} overnight/interbank market rate')
SHORT_RATES['CNY'] = fred('IR3TIB01CNM156N', 'CNY 3-month interbank rate')
SHORT_RATES['CHF'] = bis_policy('CH', 'Swiss National Bank policy rate (policy context, not a deposit quote)')
SHORT_RATES['HKD'] = bis_policy('HK', 'Hong Kong Monetary Authority base rate (policy context, not HIBOR)')
for currency in ('USD', 'EUR', 'JPY', 'GBP', 'CHF', 'AUD', 'CAD', 'CNY', 'SGD', 'HKD'):
    sources = ([SHORT_RATES[currency]] if currency in SHORT_RATES else []) + fx_dependencies(currency)
    for product in ('fixed_term_deposit', 'treasury_bill', 'money_market_fund'):
        context(f'foreign_cash_{currency.lower()}_{product}', sources,
                'Market rate / FX context only. ' + ('No local money-market rate is connected. ' if currency not in SHORT_RATES else '') +
                {'fixed_term_deposit': 'Bank, contractual rate, maturity, fees and investor access are required for a deposit return.',
                 'treasury_bill': 'Issuer, bill identifier, purchase price and matching redemption date are required for a bill return.',
                 'money_market_fund': 'Fund identifier, NAV, distributions and expenses are required for an MMF return.'}[product])

if set(BINDINGS) != {e['exposure_id'] for e in EXPOSURES}:
    raise RuntimeError('Every catalogue exposure must have exactly one analysis binding')

# Compatibility exports for callers that need only the tradable proxy subset.
PROXIES = {key: (b['primary'], b['quote_currency'], SERIES[b['primary']]['label'])
           for key, b in BINDINGS.items() if b['mode'] == 'forecast'}
SYMBOLS = {key: spec['currency'] for key, spec in SERIES.items()}


def series_id(ticker):
    spec = SERIES[ticker]
    if spec['provider'] != 'yahoo':
        return ticker
    return 'yf:' + ticker + (':adj_close' if spec['kind'] == 'adjusted_price' else ':close')


def source_id(ticker):
    return {'fred': 'fred', 'bis': 'bis_policy_rates', 'yahoo': SOURCE_ID}[SERIES[ticker]['provider']]
