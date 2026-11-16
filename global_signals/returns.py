"""Currency conversion for observed total returns, not a forecasting model."""
import calendar
import datetime
import math
from dataclasses import dataclass


def krw_total_return(quote_total_return, krw_per_quote_start, krw_per_quote_end):
    """Convert exactly once from the instrument's quote currency into KRW.

    Example: for a USD-quoted Japan ETF, use the ETF's USD total return and
    KRW per USD. Its underlying JPY exposure is already in the USD return.
    Reinvestment and other distributions must already be in quote_total_return.
    """
    values = (quote_total_return, krw_per_quote_start, krw_per_quote_end)
    if not all(math.isfinite(x) for x in values):
        raise ValueError('Returns and exchange rates must be finite')
    if quote_total_return < -1 or min(krw_per_quote_start, krw_per_quote_end) <= 0:
        raise ValueError('Invalid total return or exchange rate')
    return (1 + quote_total_return) * krw_per_quote_end / krw_per_quote_start - 1


def next_calendar_month(day: datetime.date):
    """Nominal horizon; a trading calendar must subsequently set executable time."""
    year, month = (day.year + 1, 1) if day.month == 12 else (day.year, day.month + 1)
    return day.replace(year=year, month=month, day=min(day.day, calendar.monthrange(year, month)[1]))


def _validate_cost(cost):
    if not math.isfinite(cost) or cost < 0:
        raise ValueError('Explicit costs must be a known nonnegative fraction of initial KRW capital')


def foreign_cash_unhedged_return(local_period_return, spot_start, spot_end, *, explicit_cost_fraction):
    """Observed/assumed foreign money-market return plus FX, less explicit costs.

    local_period_return includes the instrument's distributions and net expenses.
    Do not deduct expenses again when already embedded in the input return.
    All spot inputs are KRW per one unit of the actual quote currency.
    """
    _validate_cost(explicit_cost_fraction)
    return krw_total_return(local_period_return, spot_start, spot_end) - explicit_cost_fraction


def known_cashflow_hedged_return(local_period_return, spot_ask, forward_bid, *, explicit_cost_fraction):
    """Hedge a contractually known terminal payment, including known interest.

    Sell (1 + local_period_return) foreign units forward per initial foreign unit.
    The deposit/bill and forward must mature together, and payment/default risk
    is outside this cash-flow calculation. MMF future returns are not known.
    Forward points already contain rate differential and basis; do not deduct
    a second 'interest differential cost'. Explicit costs exclude bid/ask spreads
    already included in spot_ask and forward_bid. Margin/funding costs, if any,
    belong in the explicit cost fraction of initial KRW capital.
    """
    _validate_cost(explicit_cost_fraction)
    return krw_total_return(local_period_return, spot_ask, forward_bid) - explicit_cost_fraction


def static_principal_hedge_return(quote_total_return, spot_start, spot_end, forward_rate,
                                  *, hedge_ratio=1.0, explicit_cost_fraction):
    """Static forward on initial principal; unknown gains/income retain FX risk.

    Per initial 1 foreign unit: asset pays (1+R)*S_T KRW, forward pays h*(F-S_T).
    This is a single-maturity, constant-notional overlay. A rolling/dynamic hedge
    requires actual rebalance cash flows and costs, not this formula.
    Currency conversion/spreads at maturity must be consistently reflected in
    the rates or explicit costs; the forward settling and asset valuation fixings
    are assumed identical. Do not overlay again on an already KRW-hedged return.
    """
    unhedged = krw_total_return(quote_total_return, spot_start, spot_end)
    _validate_cost(explicit_cost_fraction)
    if not math.isfinite(forward_rate) or forward_rate <= 0:
        raise ValueError('Forward rate must be finite and positive')
    if not math.isfinite(hedge_ratio) or not 0 <= hedge_ratio <= 1:
        raise ValueError('Initial principal hedge ratio must be between 0 and 1')
    forward_pnl = hedge_ratio * (forward_rate - spot_end) / spot_start
    return unhedged + forward_pnl - explicit_cost_fraction


@dataclass(frozen=True)
class IndicativeForward:
    krw_per_foreign: float
    pricing_basis: str = 'covered_interest_parity'
    source_quality: str = 'model_estimate'


def indicative_cip_forward(spot, krw_simple_rate, foreign_simple_rate,
                           *, krw_year_fraction, foreign_year_fraction):
    """Research-only CIP reference, without basis/liquidity/execution spreads.

    Inputs are matched-tenor simple annual rates and explicit day-count fractions;
    a policy rate is not automatically a tradable deposit or forward quote.
    The typed estimate is deliberately marked ineligible as an executable quote.
    """
    values = (spot, krw_simple_rate, foreign_simple_rate, krw_year_fraction, foreign_year_fraction)
    if not all(math.isfinite(v) for v in values) or spot <= 0:
        raise ValueError('Rates and spot must be finite; spot must be positive')
    if min(krw_year_fraction, foreign_year_fraction) <= 0:
        raise ValueError('Both day-count year fractions must be positive')
    domestic_growth = 1 + krw_simple_rate * krw_year_fraction
    foreign_growth = 1 + foreign_simple_rate * foreign_year_fraction
    if min(domestic_growth, foreign_growth) <= 0:
        raise ValueError('Rate-period combinations imply invalid repayment factors')
    return IndicativeForward(spot * domestic_growth / foreign_growth)
