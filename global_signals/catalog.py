"""Candidate coverage for the isolated global asset monitor.

This registry is a research inventory, not an instrument list or a statement of
data availability.  No prices, investment recommendations, or computed signals
are defined here.  Reference currencies describe the intended exposure; the
quote currency of a future source series must be verified independently.
"""

__all__ = ["SOURCES", "EXPOSURES"]


SOURCES = [
    {
        "source_id": "fred",
        "label": "Federal Reserve Economic Data",
        "category": "economic_and_market_statistics",
        "documentation_url": "https://fred.stlouisfed.org/docs/api/fred/",
        "access_notes": (
            "Public data service; API registration and series-specific terms apply. "
            "A yield or spread is not a holding-period total return. ICE BofA series "
            "have only three years of observations available from April 2026; "
            "long histories and redistribution rights must be checked per series."
        ),
    },
    {
        "source_id": "alfred",
        "label": "ALFRED historical data vintages",
        "category": "economic_data_vintages",
        "documentation_url": "https://alfred.stlouisfed.org/help/downloaddata",
        "access_notes": (
            "Versions of data available on historical dates, where archived. "
            "Preserve observation date and information-availability date separately. "
            "Do not assume all series have complete historical vintages."
        ),
    },
    {
        "source_id": "bis_rpp",
        "label": "BIS residential property prices",
        "category": "physical_property_context",
        "documentation_url": "https://data.bis.org/topics/RPP",
        "access_notes": (
            "Public residential price statistics with country-specific coverage, "
            "frequencies, revisions, and publication lags. These are price indicators, "
            "not rental-income-inclusive tradable returns or daily trading signals."
        ),
    },
    {
        "source_id": "bis_cpp",
        "label": "BIS commercial property prices",
        "category": "physical_property_context",
        "documentation_url": "https://data.bis.org/topics/CPP",
        "access_notes": (
            "Public commercial property indicators; property type, geography, "
            "frequency, and methodology vary by country. Coverage for each "
            "candidate requires verification and must remain separate from REITs."
        ),
    },
    {
        "source_id": "imf_pcps",
        "label": "IMF Primary Commodity Price System",
        "category": "low_frequency_commodity_context",
        "documentation_url": "https://data.imf.org/Datasets/PCPS",
        "access_notes": (
            "Candidate source for monthly commodity-price context. Published "
            "period-average benchmark prices are not daily closes, futures roll "
            "returns, or ETF total returns. Verify available commodities, units, "
            "history, and current access conditions before collection."
        ),
    },
    {
        "source_id": "imf_api",
        "label": "IMF SDMX data APIs",
        "category": "economic_statistics_access",
        "documentation_url": "https://data.imf.org/en/Resource-Pages/IMF-API",
        "access_notes": (
            "Official SDMX API documentation; dataset identifiers, dimensions, "
            "registration requirements, and release dates need connector-level "
            "verification. Documentation does not establish daily asset-return coverage."
        ),
    },
    {
        "source_id": "fed_fx",
        "label": "Federal Reserve H.10 foreign exchange rates",
        "category": "foreign_exchange_statistics",
        "documentation_url": "https://www.federalreserve.gov/releases/h10/",
        "access_notes": (
            "Official daily observations released on a weekly schedule. Quote "
            "directions vary by currency; normalize units and retain publication "
            "time. Reference rates are not executable prices or deposit returns."
        ),
    },
    {
        "source_id": "nareit",
        "label": "Nareit historical monthly index values and returns",
        "category": "listed_real_estate_returns",
        "documentation_url": (
            "https://www.reit.com/data-research/reit-market-data/report/"
            "monthly-index-values-returns"
        ),
        "access_notes": (
            "Public US REIT monthly history; verify selected index, return type, "
            "file format, and permitted use. This source does not establish daily "
            "coverage for all countries or direct-property investment returns."
        ),
    },
    {
        "source_id": "bok_ecos",
        "label": "Bank of Korea ECOS Open API",
        "category": "korean_economic_statistics",
        "documentation_url": "https://ecos.bok.or.kr/api/",
        "access_notes": (
            "Candidate public source for Korean rates and exchange-rate context. "
            "API key, statistic codes, units, history, and permitted use require "
            "verification. An interest-rate series is not a cash or bond total-return index."
        ),
    },
    {
        "source_id": "gdelt",
        "label": "GDELT public news and event data",
        "category": "news_and_event_context",
        "documentation_url": "https://www.gdeltproject.org/data.html",
        "access_notes": (
            "Optional news/event context, not a price or financial-return source. "
            "Retain publication and retrieval times, verify archive coverage and "
            "access limits, and handle duplicate reports and language differences."
        ),
    },
    {
        "source_id": "coinbase_exchange",
        "label": "Coinbase Exchange public candle data",
        "category": "venue_specific_crypto_prices",
        "documentation_url": (
            "https://docs.cdp.coinbase.com/api-reference/exchange-api/"
            "rest-api/products/get-product-candles"
        ),
        "access_notes": (
            "Candidate venue-specific OHLCV source; at most 300 candles per request. "
            "Intervals without trades may be absent. Verify pair coverage and history; "
            "do not treat USD, stablecoins, or Korean exchange prices as interchangeable."
        ),
    },
]


def _candidate(
    exposure_id,
    label,
    asset_class,
    country_code,
    reference_currency,
    *,
    monitor_only=False,
    desired_frequency="daily",
    **attributes,
):
    """Keep intended coverage distinct from a verified source-series binding."""
    return {
        "exposure_id": exposure_id,
        "label": label,
        "asset_class": asset_class,
        "country_code": country_code,
        "reference_currency": reference_currency,
        "monitor_only": monitor_only,
        "attributes": {
            "data_status": "unverified",
            "desired_frequency": desired_frequency,
            "reporting_currency": "KRW",
            "quote_currency": None,
            "quote_currency_status": "unverified",
            "source_series_id": None,
            "source_id": None,
            "signal_status": "not_computed",
            "tactical_horizon": None if monitor_only else "1m",
            **attributes,
        },
    }


EXPOSURES = []

# Country exposure is distinct from listing venue and the source quote currency.
_EQUITY_COUNTRIES = [
    ("KR", "South Korea", "KRW"),
    ("US", "United States", "USD"),
    ("JP", "Japan", "JPY"),
    ("CN", "China", "CNY"),
    ("HK", "Hong Kong", "HKD"),
    ("TW", "Taiwan", "TWD"),
    ("IN", "India", "INR"),
    ("AU", "Australia", "AUD"),
    ("DE", "Germany", "EUR"),
    ("GB", "United Kingdom", "GBP"),
    ("FR", "France", "EUR"),
    ("CH", "Switzerland", "CHF"),
    ("CA", "Canada", "CAD"),
    ("BR", "Brazil", "BRL"),
    ("MX", "Mexico", "MXN"),
    ("ZA", "South Africa", "ZAR"),
]
for country, label, currency in _EQUITY_COUNTRIES:
    EXPOSURES.append(
        _candidate(
            f"equity_{country.lower()}",
            f"{label} equities",
            "equity",
            country,
            currency,
            desired_measure="total_return",
            dividend_treatment="unverified",
            geographic_definition="country_equity_exposure",
            investable_proxy_status="unverified",
        )
    )

_GOVERNMENT_COUNTRIES = [
    ("US", "United States", "USD"),
    ("KR", "South Korea", "KRW"),
    ("JP", "Japan", "JPY"),
    ("DE", "Germany", "EUR"),
    ("GB", "United Kingdom", "GBP"),
    ("AU", "Australia", "AUD"),
    ("CA", "Canada", "CAD"),
]
for country, label, currency in _GOVERNMENT_COUNTRIES:
    for duration in ("short", "intermediate", "long"):
        EXPOSURES.append(
            _candidate(
                f"government_bond_{country.lower()}_{duration}",
                f"{label} government bonds: {duration} duration",
                "government_bond",
                country,
                currency,
                desired_measure="total_return",
                duration_bucket=duration,
                duration_bucket_boundaries="unverified",
                debt_currency_scope="local_currency",
                yield_is_total_return=False,
                candidate_context_source_ids=["fred", "imf_api"],
                investable_proxy_status="unverified",
            )
        )

# Currency buckets do not imply issuer nationality or investor access.
for currency in ("USD", "EUR", "GBP", "KRW", "JPY"):
    for grade, grade_label in (("ig", "investment grade"), ("hy", "high yield")):
        EXPOSURES.append(
            _candidate(
                f"corporate_bond_{currency.lower()}_{grade}",
                f"{currency}-denominated corporate bonds: {grade_label}",
                "corporate_bond",
                None,
                currency,
                desired_measure="total_return",
                credit_bucket=grade,
                denomination_currency=currency,
                issuer_geography="unverified",
                duration_bucket="unverified",
                investor_access="unverified",
                yield_is_total_return=False,
                candidate_context_source_ids=["fred"],
                investable_proxy_status="unverified",
            )
        )

_PROPERTY_COUNTRIES = [
    ("KR", "South Korea", "KRW"),
    ("US", "United States", "USD"),
    ("JP", "Japan", "JPY"),
    ("GB", "United Kingdom", "GBP"),
    ("DE", "Germany", "EUR"),
    ("FR", "France", "EUR"),
    ("AU", "Australia", "AUD"),
    ("CA", "Canada", "CAD"),
    ("SG", "Singapore", "SGD"),
]
for country, label, currency in _PROPERTY_COUNTRIES:
    EXPOSURES.append(
        _candidate(
            f"listed_reit_{country.lower()}",
            f"{label} listed REITs",
            "listed_reit",
            country,
            currency,
            desired_measure="total_return",
            property_type="listed_reit_equity",
            geographic_definition="unverified",
            dividend_treatment="unverified",
            candidate_context_source_ids=["nareit"] if country == "US" else [],
            investable_proxy_status="unverified",
        )
    )
    EXPOSURES.append(
        _candidate(
            f"physical_property_{country.lower()}",
            f"{label} physical property context",
            "physical_property",
            country,
            currency,
            monitor_only=True,
            desired_frequency="quarterly",
            desired_measure="property_price_index",
            candidate_context_source_ids=["bis_rpp", "bis_cpp"],
            property_type="unverified",
            series_separation_required=["residential", "commercial"],
            rental_income_included=False,
            publication_lag="unverified",
            tactical_signal_allowed=False,
        )
    )

for name, label in (
    ("gold", "Gold"),
    ("silver", "Silver"),
    ("copper", "Copper"),
    ("oil", "Crude oil"),
    ("gas", "Natural gas"),
    ("agriculture", "Agriculture basket"),
):
    EXPOSURES.append(
        _candidate(
            f"commodity_{name}",
            f"Global {label.lower()} exposure",
            "commodity",
            None,
            "USD",
            desired_measure="total_return",
            benchmark_grade_or_basket="unverified",
            instrument_structure="unverified",
            roll_and_collateral_treatment="unverified",
            investable_proxy_status="unverified",
            candidate_context_source_ids=["imf_pcps"],
            context_desired_frequency="monthly",
            context_measure="period_average_commodity_price",
            context_is_daily_return_source=False,
        )
    )

for currency in ("USD", "EUR", "JPY", "GBP", "CHF", "AUD", "CAD", "CNY", "SGD", "HKD"):
    EXPOSURES.append(
        _candidate(
            f"fx_{currency.lower()}_krw",
            f"{currency} relative to KRW",
            "foreign_exchange",
            None,
            currency,
            desired_measure="spot_exchange_rate",
            target_quote_convention=f"KRW per 1 {currency}",
            fx_base_currency=currency,
            fx_quote_currency="KRW",
            includes_interest_or_carry=False,
            candidate_context_source_ids=["fed_fx", "bok_ecos"],
            observation_fixing="unverified",
        )
    )

for asset, label in (("btc", "Bitcoin"), ("eth", "Ether")):
    EXPOSURES.append(
        _candidate(
            f"crypto_{asset}",
            label,
            "cryptoasset",
            None,
            "USD",
            desired_measure="spot_price_return",
            candidate_context_source_ids=["coinbase_exchange"],
            venue="unverified",
            trading_pair="unverified",
            daily_cutoff="unverified",
            staking_income_included=False,
            stablecoin_is_usd=False,
            local_krw_exchange_premium_included=False,
        )
    )

EXPOSURES.append(
    _candidate(
        "cash_krw",
        "KRW cash reference",
        "cash",
        "KR",
        "KRW",
        desired_measure="cash_total_return",
        candidate_context_source_ids=["bok_ecos"],
        accrual_convention="unverified",
        rate_is_total_return=False,
        investable_proxy_status="unverified",
    )
)


# Foreign-currency cash candidates are separate from spot FX exposure.  A local
# short rate supplies context only; actual product terms and cash flows remain
# unverified.  One calendar month must not be silently replaced with 30 days.
_FOREIGN_CASH_CURRENCIES = (
    "USD", "EUR", "JPY", "GBP", "CHF", "AUD", "CAD", "CNY", "SGD", "HKD"
)
_FOREIGN_CASH_INSTRUMENTS = (
    (
        "fixed_term_deposit",
        "fixed-term deposit",
        {
            "cash_flow_certainty": (
                "Contractual maturity amount is conditional on verified fixed-rate "
                "terms, accrual convention, fees, and holding to maturity; repayment "
                "is not guaranteed by this classification."
            ),
            "credit_risk": "Bank counterparty risk; applicable protection is unverified.",
            "liquidity_risk": "Early withdrawal may be restricted or change interest and fees.",
            "deposit_protection_status": "unverified",
        },
    ),
    (
        "treasury_bill",
        "treasury bill",
        {
            "cash_flow_certainty": (
                "Contractual redemption amount is known only after bill terms are "
                "verified and maturity matches the planning horizon, assuming no "
                "default; sale proceeds before maturity are unknown."
            ),
            "credit_risk": "Sovereign issuer risk; issuer and repayment terms are unverified.",
            "liquidity_risk": "A pre-maturity sale introduces market-price and bid-ask risk.",
            "maturity_matches_horizon": "unverified",
        },
    ),
    (
        "money_market_fund",
        "money market fund",
        {
            "cash_flow_certainty": "Future redemption value and distributions are unknown.",
            "credit_risk": "Underlying portfolio credit and concentration risks are unverified.",
            "liquidity_risk": "Redemption terms, settlement timing, and liquidity limits are unverified.",
            "bank_deposit_equivalent": False,
            "bank_deposit_guarantee_assumed": False,
            "fund_nav_structure": "unverified",
        },
    ),
)

for currency in _FOREIGN_CASH_CURRENCIES:
    for instrument_category, label, instrument_attributes in _FOREIGN_CASH_INSTRUMENTS:
        EXPOSURES.append(
            _candidate(
                f"foreign_cash_{currency.lower()}_{instrument_category}",
                f"{currency} {label} candidate",
                "foreign_cash",
                None,
                currency,
                desired_measure="cash_total_return",
                instrument_category=instrument_category,
                planning_horizon="1_calendar_month",
                planning_horizon_calendar_months=1,
                horizon_day_count="calendar_dates_not_fixed_30_days",
                denomination_currency=currency,
                issuer_country="unverified",
                retail_access="unverified",
                investable_proxy_status="unverified",
                tenor="unverified",
                day_count_convention="unverified",
                compounding_convention="unverified",
                product_fees="unverified",
                tax_treatment="unverified",
                repayment_guarantee_status="unverified_do_not_assume",
                indicative_short_rate_is_product_return=False,
                rate_series_role="indicative_local_short_rate_context_only",
                rate_is_total_return=False,
                candidate_context_source_ids=["fred", "imf_api"],
                yield_fee_basis="unverified",
                fee_handling=(
                    "Verify gross versus net yield; deduct each fee once. Do not "
                    "subtract a fee already included in a published net yield."
                ),
                hedge_modes=["unhedged", "krw_hedged"],
                hedge_reference_currency="KRW",
                hedge_instrument="unverified",
                hedge_cost_status="unverified",
                hedge_notional_policy="unverified",
                return_estimate_status="not_computed",
                **instrument_attributes,
            )
        )
