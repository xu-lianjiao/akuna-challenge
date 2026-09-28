import math
import random
from collections import defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final
import numpy as np

AJARAI_NAME: Final[str] = "AJR"
AJARAI_UNDERLYING_ID: Final[int] = 2
FED_FUNDS_RATE_NAME: Final[str] = "FED"
FED_FUNDS_RATE_UNDERLYING_ID: Final[int] = 1
RATE_STRIKE_GRID: Final[float] = 0.25
THERIODIC_NAME: Final[str] = "THR"
THERIODIC_UNDERLYING_ID: Final[int] = 3

UNDERLYING_NAME_BY_ID: Final[dict[int, str]] = {
    AJARAI_UNDERLYING_ID: AJARAI_NAME,
    FED_FUNDS_RATE_UNDERLYING_ID: FED_FUNDS_RATE_NAME,
    THERIODIC_UNDERLYING_ID: THERIODIC_NAME,
}


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class BinaryOption:
    legs: "tuple[OptionLeg, ...]"
    option_id: int
    steps_until_expiry: int
    strike: float

    def __post_init__(self) -> None:
        if self.steps_until_expiry < 0:
            raise ValueError("Steps until expiry must be non-negative")

        if not self.legs:
            raise ValueError("Binary option must have at least one leg")

        underlying_ids: list[int] = [leg.underlying_id for leg in self.legs]
        if len(underlying_ids) != len(set(underlying_ids)):
            raise ValueError("Binary option legs must reference distinct underlyings")

        if any(leg.weight == 0 for leg in self.legs):
            raise ValueError("Binary option leg weights must be non-zero")

    def __str__(self) -> str:
        terms: list[str] = []
        for index, leg in enumerate(self.legs):
            name: str = UNDERLYING_NAME_BY_ID.get(leg.underlying_id, str(leg.underlying_id))
            magnitude: float = abs(leg.weight)
            magnitude_str: str = "" if magnitude == 1 else f"{magnitude:.2f}*"
            if index == 0:
                sign: str = "-" if leg.weight < 0 else ""
            else:
                sign = " - " if leg.weight < 0 else " + "
            terms.append(f"{sign}{magnitude_str}{name}")
        observable_expression: str = "".join(terms)
        return f"{self.option_id} ({self.steps_until_expiry}d {observable_expression} >= {self.strike:.2f})"

    def advance_step(self) -> "BinaryOption":
        if self.steps_until_expiry == 0:
            return self

        return replace(self, steps_until_expiry=self.steps_until_expiry - 1)

    def contract_matches(self, other: "BinaryOption") -> bool:
        return replace(other, option_id=self.option_id) == self

    def expiry_valuation(self, value_by_underlying_id: dict[int, float]) -> float:
        return 1.0 if self.observable_value(value_by_underlying_id) >= self.strike else 0.0

    def observable_value(self, value_by_underlying_id: dict[int, float]) -> float:
        return sum(leg.weight * value_by_underlying_id[leg.underlying_id] for leg in self.legs)


@dataclass(frozen=True)
class FokOrder:
    counterparty_id: int
    option_id: int
    order_type: "OrderType"
    price: float
    quantity: int

    def __post_init__(self) -> None:
        if self.price < 0:
            raise ValueError("FOK order price must be non-negative")

        if self.quantity <= 0:
            raise ValueError("FOK order quantity must be positive")


@dataclass(frozen=True)
class MarketHistory:
    values_by_underlying_id: dict[int, tuple[float, ...]]

    def __post_init__(self) -> None:
        lengths: set[int] = {len(values) for values in self.values_by_underlying_id.values()}
        if len(lengths) > 1:
            raise ValueError("All underlyings must have the same number of historical days")

        if lengths and next(iter(lengths)) <= 0:
            raise ValueError("Market history must contain at least one day")

    @property
    def num_days(self) -> int:
        if not self.values_by_underlying_id:
            return 0
        return len(next(iter(self.values_by_underlying_id.values())))


@dataclass(frozen=True)
class MarketParameters:
    ajarai_drift: float
    ajarai_idio_std_dev: float
    ajarai_rate_beta: float
    ajarai_sector_beta: float
    rate_down_probability: float
    rate_reversion_strength: float
    rate_up_probability: float
    sector_std_dev: float
    theriodic_drift: float
    theriodic_idio_std_dev: float
    theriodic_rate_beta: float
    theriodic_sector_beta: float

    rate_step: float = 0.25
    rate_target: float = 2.0

    def __post_init__(self) -> None:
        if self.rate_step <= 0:
            raise ValueError("Rate step must be positive")

        if self.rate_up_probability <= 0 or self.rate_down_probability <= 0:
            raise ValueError("Rate up/down probabilities must both be positive")

        if self.rate_up_probability + self.rate_down_probability > 1:
            raise ValueError("Rate up/down probabilities must not sum to more than 1")

        if self.rate_target < 0:
            raise ValueError("Rate target must be non-negative")

        if not (0 <= self.rate_reversion_strength <= 1):
            raise ValueError("Rate reversion strength must be between 0 and 1")

        if self.ajarai_idio_std_dev < 0 or self.theriodic_idio_std_dev < 0 or self.sector_std_dev < 0:
            raise ValueError("Standard deviations must be non-negative")

    def advance_company_value(
        self,
        current_value: float,
        rate_change: float,
        sector_shock: float,
        *,
        drift: float,
        rate_beta: float,
        sector_beta: float,
        idio_std_dev: float,
    ) -> float:
        idiosyncratic_shock: float = random.gauss(mu=0.0, sigma=idio_std_dev)
        log_return: float = drift + (rate_beta * rate_change) + (sector_beta * sector_shock) + idiosyncratic_shock
        return round(current_value * math.exp(log_return), 2)

    def advance_rate(self, rate_value: float) -> float:
        up_probability, down_probability = self.tilted_rate_probabilities(rate_value)
        draw: float = random.random()
        if draw < up_probability:
            return self.next_rate_value(rate_value, 1)

        if draw < up_probability + down_probability:
            return self.next_rate_value(rate_value, -1)

        return rate_value

    def advance_step(self, value_by_underlying_id: dict[int, float]) -> dict[int, float]:
        current_rate_value: float = value_by_underlying_id[FED_FUNDS_RATE_UNDERLYING_ID]
        rate_value: float = self.advance_rate(current_rate_value)
        rate_change: float = round(rate_value - current_rate_value, 2)
        sector_shock: float = random.gauss(mu=0.0, sigma=self.sector_std_dev)
        return {
            FED_FUNDS_RATE_UNDERLYING_ID: rate_value,
            AJARAI_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[AJARAI_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.ajarai_drift,
                rate_beta=self.ajarai_rate_beta,
                sector_beta=self.ajarai_sector_beta,
                idio_std_dev=self.ajarai_idio_std_dev,
            ),
            THERIODIC_UNDERLYING_ID: self.advance_company_value(
                value_by_underlying_id[THERIODIC_UNDERLYING_ID],
                rate_change,
                sector_shock,
                drift=self.theriodic_drift,
                rate_beta=self.theriodic_rate_beta,
                sector_beta=self.theriodic_sector_beta,
                idio_std_dev=self.theriodic_idio_std_dev,
            ),
        }

    def next_rate_value(self, rate_value: float, num_grid_steps: int) -> float:
        return max(round(rate_value + num_grid_steps * self.rate_step, 2), 0.0)

    def tilted_rate_probabilities(self, rate_value: float) -> tuple[float, float]:
        tilt: float = self.rate_reversion_strength * (self.rate_target - rate_value)
        up_probability: float = min(max(self.rate_up_probability + tilt, 0.0), 1.0)
        down_probability: float = min(max(self.rate_down_probability - tilt, 0.0), 1.0 - up_probability)
        return up_probability, down_probability


@dataclass(frozen=True)
class OptionLeg:
    underlying_id: int
    weight: float


class OrderType(StrEnum):
    BUY = "buy"
    SELL = "sell"


class Position:
    def __init__(self) -> None:
        self.option_quantity_by_option_id: dict[int, int] = defaultdict(int)

    def add_option_quantity(self, option_id: int, quantity: int) -> None:
        self.option_quantity_by_option_id[option_id] += quantity


@dataclass(frozen=True)
class Quote:
    bid_price: float
    bid_quantity: int
    offer_price: float
    offer_quantity: int

    def __post_init__(self) -> None:
        if self.bid_quantity <= 0 or self.offer_quantity <= 0:
            raise ValueError("Quote quantities must be positive")

        if not (0.0 <= self.bid_price <= 1.0 and 0.0 <= self.offer_price <= 1.0):
            raise ValueError("Quote prices must be between 0 and 1")

        if self.bid_price >= self.offer_price:
            raise ValueError("Quote bid price must be less than offer price")

        if any(abs(round(price * 100) - price * 100) > 1e-6 for price in (self.bid_price, self.offer_price)):
            raise ValueError("Quote prices must be in whole pennies (multiples of 0.01)")


@dataclass(frozen=True)
class Underlying:
    name: str
    underlying_id: int
    value: float

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Underlying):
            return False
        return self.underlying_id == other.underlying_id

# ============================================================================
# YOUR MARKET MAKER -- fill in the six stubbed methods below
# ============================================================================


class MarketMaker:
    def __init__(
        self,
        underlying_initial_state: list[Underlying],
        option_initial_state: list[BinaryOption],
        cash_balance: float,
    ) -> None:
        self.underlying_state: list[Underlying] = underlying_initial_state
        self.active_option_state: list[BinaryOption] = option_initial_state
        self.cash_balance: float = cash_balance
        self.position: Position = Position()

    def on_step_advance(self, new_underlying_state: list[Underlying], new_option_state: list[BinaryOption]) -> None:
        self.underlying_state = new_underlying_state
        self.active_option_state = new_option_state

    def on_trade(self, option: BinaryOption, price: float, quantity: int, counterparty_id: int) -> None:
        self.position.add_option_quantity(option.option_id, quantity)

    @property
    def name(self) -> str:  # type: ignore[empty-body]
        # TODO: return a unique display name for your market maker.
        return "Julian A Xiao"

    def price_option(self, option: BinaryOption) -> float:  # type: ignore[empty-body]
        # TODO: return your own theoretical probability (in [0, 1]) that `option` expires in
        # the money, using whatever you estimated in `warm_up` and the current
        # `self.underlying_state`. Called whenever you quote or price a FOK, and by the grader
        # to log what you thought an option was worth, so keep it free of side effects.
        
        # price_option_from_parameters is assumed to already be correctly implemented.
        # warm_up stores our fitted MarketParameters in self.estimated_market_parameters.
        return self.price_option_from_parameters(
            self.estimated_market_parameters,
            option
        )

    def price_option_from_parameters(  # type: ignore[empty-body]
        self, market_parameters: MarketParameters, option: BinaryOption
    ) -> float:
        # TODO: return the theoretical probability (in [0, 1]) that `option` expires in the
        # money, given `market_parameters` and the current `self.underlying_state`. Only the
        # THEO test calls this, handing you the true parameters; `price_option` above is what
        # prices your live trading.

        values = {
            underlying.underlying_id: float(underlying.value)
            for underlying in self.underlying_state
        }

        steps = option.steps_until_expiry

        if steps == 0:
            return float(option.expiry_valuation(values))

        params = market_parameters

        current_rate = values[FED_FUNDS_RATE_UNDERLYING_ID]
        current_ajr = values[AJARAI_UNDERLYING_ID]
        current_thr = values[THERIODIC_UNDERLYING_ID]

        # 1. Exact terminal distribution of FED.
        rate_probabilities = {round(current_rate, 2): 1.0}

        for _ in range(steps):
            next_probabilities = defaultdict(float)

            for rate_value, probability in rate_probabilities.items():
                tilt = params.rate_reversion_strength * (
                    params.rate_target - rate_value
                )

                up_probability = min(
                    max(params.rate_up_probability + tilt, 0.0),
                    1.0,
                )

                down_probability = min(
                    max(params.rate_down_probability - tilt, 0.0),
                    1.0 - up_probability,
                )

                flat_probability = (
                    1.0 - up_probability - down_probability
                )

                up_rate = max(
                    round(rate_value + params.rate_step, 2),
                    0.0,
                )

                down_rate = max(
                    round(rate_value - params.rate_step, 2),
                    0.0,
                )

                next_probabilities[up_rate] += (
                    probability * up_probability
                )
                next_probabilities[down_rate] += (
                    probability * down_probability
                )
                next_probabilities[round(rate_value, 2)] += (
                    probability * flat_probability
                )

            rate_probabilities = dict(next_probabilities)

        # 2. Conditional log-normal distributions of AJR and THR.
        #
        # Conditional on terminal FED rate r:
        #
        # log(A_T) ~ N(m_A, v_A)
        # log(T_T) ~ N(m_T, v_T)
        #
        # The common sector shock gives them covariance.
        ajr_variance = steps * (
            params.ajarai_sector_beta ** 2
            * params.sector_std_dev ** 2
            + params.ajarai_idio_std_dev ** 2
        )

        thr_variance = steps * (
            params.theriodic_sector_beta ** 2
            * params.sector_std_dev ** 2
            + params.theriodic_idio_std_dev ** 2
        )

        ajr_thr_covariance = steps * (
            params.ajarai_sector_beta
            * params.theriodic_sector_beta
            * params.sector_std_dev ** 2
        )

        ajr_sd = math.sqrt(max(ajr_variance, 0.0))
        thr_sd = math.sqrt(max(thr_variance, 0.0))

        theo = 0.0

        # 3. Sum over terminal FED states.
        for terminal_rate, rate_probability in rate_probabilities.items():

            ajr_mean = (
                math.log(current_ajr)
                + steps * params.ajarai_drift
                + params.ajarai_rate_beta
                * (terminal_rate - current_rate)
            )

            thr_mean = (
                math.log(current_thr)
                + steps * params.theriodic_drift
                + params.theriodic_rate_beta
                * (terminal_rate - current_rate)
            )

            conditional_probability = 0.0

            # Single-legged option.
            if len(option.legs) == 1:
                leg = option.legs[0]
                weight = leg.weight
                strike = option.strike

                if leg.underlying_id == FED_FUNDS_RATE_UNDERLYING_ID:
                    conditional_probability = (
                        1.0
                        if weight * terminal_rate >= strike
                        else 0.0
                    )

                elif leg.underlying_id == AJARAI_UNDERLYING_ID:
                    if weight > 0.0:
                        threshold = strike / weight

                        if threshold <= 0.0:
                            conditional_probability = 1.0
                        elif ajr_sd <= 1e-15:
                            conditional_probability = (
                                1.0
                                if math.exp(ajr_mean) >= threshold
                                else 0.0
                            )
                        else:
                            z = (
                                math.log(threshold) - ajr_mean
                            ) / ajr_sd

                            conditional_probability = 1.0 - 0.5 * (
                                1.0 + math.erf(z / math.sqrt(2.0))
                            )

                    else:
                        threshold = strike / weight

                        if threshold <= 0.0:
                            conditional_probability = 0.0
                        elif ajr_sd <= 1e-15:
                            conditional_probability = (
                                1.0
                                if math.exp(ajr_mean) <= threshold
                                else 0.0
                            )
                        else:
                            z = (
                                math.log(threshold) - ajr_mean
                            ) / ajr_sd

                            conditional_probability = 0.5 * (
                                1.0 + math.erf(z / math.sqrt(2.0))
                            )

                elif leg.underlying_id == THERIODIC_UNDERLYING_ID:
                    if weight > 0.0:
                        threshold = strike / weight

                        if threshold <= 0.0:
                            conditional_probability = 1.0
                        elif thr_sd <= 1e-15:
                            conditional_probability = (
                                1.0
                                if math.exp(thr_mean) >= threshold
                                else 0.0
                            )
                        else:
                            z = (
                                math.log(threshold) - thr_mean
                            ) / thr_sd

                            conditional_probability = 1.0 - 0.5 * (
                                1.0 + math.erf(z / math.sqrt(2.0))
                            )

                    else:
                        threshold = strike / weight

                        if threshold <= 0.0:
                            conditional_probability = 0.0
                        elif thr_sd <= 1e-15:
                            conditional_probability = (
                                1.0
                                if math.exp(thr_mean) <= threshold
                                else 0.0
                            )
                        else:
                            z = (
                                math.log(threshold) - thr_mean
                            ) / thr_sd

                            conditional_probability = 0.5 * (
                                1.0 + math.erf(z / math.sqrt(2.0))
                            )

                else:
                    raise ValueError(
                        "Unsupported single-legged binary option"
                    )

            # AJR/THR spread.
            #
            # We have:
            #
            #   a*A_T - b*THR_T >= K
            #
            # with a,b > 0.
            #
            # Conditional on log(negative_leg), the log of the positive leg
            # remains Gaussian. We integrate the resulting conditional
            # probability with deterministic Gauss-Hermite quadrature.
            elif len(option.legs) == 2:
                first = option.legs[0]
                second = option.legs[1]

                if {
                    first.underlying_id,
                    second.underlying_id,
                } != {
                    AJARAI_UNDERLYING_ID,
                    THERIODIC_UNDERLYING_ID,
                }:
                    raise ValueError(
                        "Two-legged binaries must be AJR/THR spreads"
                    )

                if first.weight * second.weight >= 0.0:
                    raise ValueError(
                        "AJR/THR spread weights must have opposite signs"
                    )

                # Identify the positive and negative legs.
                if first.weight > 0.0:
                    positive_weight = first.weight
                    negative_weight = -second.weight

                    positive_mean = (
                        ajr_mean
                        if first.underlying_id == AJARAI_UNDERLYING_ID
                        else thr_mean
                    )
                    negative_mean = (
                        thr_mean
                        if second.underlying_id == THERIODIC_UNDERLYING_ID
                        else ajr_mean
                    )

                    positive_variance = (
                        ajr_variance
                        if first.underlying_id == AJARAI_UNDERLYING_ID
                        else thr_variance
                    )
                    negative_variance = (
                        thr_variance
                        if second.underlying_id == THERIODIC_UNDERLYING_ID
                        else ajr_variance
                    )
                else:
                    positive_weight = second.weight
                    negative_weight = -first.weight

                    positive_mean = (
                        thr_mean
                        if second.underlying_id == THERIODIC_UNDERLYING_ID
                        else ajr_mean
                    )
                    negative_mean = (
                        ajr_mean
                        if first.underlying_id == AJARAI_UNDERLYING_ID
                        else thr_mean
                    )

                    positive_variance = (
                        thr_variance
                        if second.underlying_id == THERIODIC_UNDERLYING_ID
                        else ajr_variance
                    )
                    negative_variance = (
                        ajr_variance
                        if first.underlying_id == AJARAI_UNDERLYING_ID
                        else thr_variance
                    )

                # Determine covariance according to which asset is the
                # positive/negative leg. Covariance itself is symmetric.
                covariance = ajr_thr_covariance

                nodes, quadrature_weights = np.polynomial.hermite.hermgauss(32)

                if negative_variance <= 1e-15:
                    negative_samples = [
                        (math.exp(negative_mean), 1.0)
                    ]
                else:
                    negative_sd = math.sqrt(negative_variance)

                    negative_samples = [
                        (
                            math.exp(
                                negative_mean
                                + math.sqrt(2.0)
                                * negative_sd
                                * node
                            ),
                            weight / math.sqrt(math.pi),
                        )
                        for node, weight in zip(
                            nodes,
                            quadrature_weights,
                        )
                    ]

                conditional_probability = 0.0

                for negative_value, quadrature_weight in negative_samples:
                    threshold = (
                        option.strike
                        + negative_weight * negative_value
                    ) / positive_weight

                    if threshold <= 0.0:
                        spread_probability = 1.0

                    elif positive_variance <= 1e-15:
                        spread_probability = (
                            1.0
                            if math.exp(positive_mean) >= threshold
                            else 0.0
                        )

                    elif negative_variance <= 1e-15:
                        z = (
                            math.log(threshold)
                            - positive_mean
                        ) / math.sqrt(positive_variance)

                        spread_probability = 1.0 - 0.5 * (
                            1.0 + math.erf(
                                z / math.sqrt(2.0)
                            )
                        )

                    else:
                        conditional_mean = (
                            positive_mean
                            + covariance
                            / negative_variance
                            * (
                                math.log(negative_value)
                                - negative_mean
                            )
                        )

                        conditional_variance = max(
                            positive_variance
                            - covariance ** 2 / negative_variance,
                            0.0,
                        )

                        if conditional_variance <= 1e-15:
                            spread_probability = (
                                1.0
                                if math.exp(conditional_mean) >= threshold
                                else 0.0
                            )
                        else:
                            conditional_sd = math.sqrt(
                                conditional_variance
                            )

                            z = (
                                math.log(threshold)
                                - conditional_mean
                            ) / conditional_sd

                            spread_probability = 1.0 - 0.5 * (
                                1.0 + math.erf(
                                    z / math.sqrt(2.0)
                                )
                            )

                    conditional_probability += (
                        quadrature_weight
                        * spread_probability
                    )

            else:
                raise ValueError(
                    "Unsupported binary option structure"
                )

            theo += (
                rate_probability
                * conditional_probability
            )

        return max(0.0, min(1.0, theo))
    
    
    def quote(self, option: BinaryOption, counterparty_id: int) -> Quote:  # type: ignore[empty-body]
        # Reconcile the last reservation using the position change recorded by on_trade().
        if self.pending_risk_option_id is not None:
            current_position = self.position.option_quantity_by_option_id.get(
                self.pending_risk_option_id, 0
            )
            position_change = current_position - self.pending_risk_position_before
            actual_cash_used = 0.0
            if position_change > 0:
                fill_price = (
                    self.pending_risk_fok_price
                    if self.pending_risk_fok_type is not None
                    else self.pending_risk_bid_price
                )
                actual_cash_used = position_change * float(fill_price)
            elif position_change < 0:
                fill_price = (
                    self.pending_risk_fok_price
                    if self.pending_risk_fok_type is not None
                    else self.pending_risk_offer_price
                )
                actual_cash_used = (-position_change) * (1.0 - float(fill_price))

            self.risk_cash_remaining += self.pending_risk_reservation - actual_cash_used
            self.risk_cash_remaining = max(
                0.0, min(0.95 * self.cash_balance, self.risk_cash_remaining)
            )
            self.pending_risk_reservation = 0.0
            self.pending_risk_option_id = None
            self.pending_risk_fok_type = None

        p = max(0.0, min(1.0, float(self.price_option(option))))
        payoff_variance = p * (1.0 - p)
        variance_denominator = max(payoff_variance, 0.01)
        option_time = max(float(option.steps_until_expiry), 1.0)

        # Approximate portfolio inventory in units of this option.  For contracts
        # on the same linear observable, nesting gives the same-expiry binary
        # covariance exactly; different expiries are attenuated by sqrt(Tmin/Tmax).
        # AJR-vs-THR single-leg exposure uses their estimated return correlation.
        effective_inventory = 0.0
        active_by_id = {active.option_id: active for active in self.active_option_state}
        current_leg_map = {leg.underlying_id: leg.weight for leg in option.legs}

        for held_id, held_quantity in self.position.option_quantity_by_option_id.items():
            if held_quantity == 0:
                continue
            held_option = active_by_id.get(held_id)
            if held_option is None:
                continue
            if held_id == option.option_id:
                effective_inventory += held_quantity
                continue

            held_leg_map = {leg.underlying_id: leg.weight for leg in held_option.legs}
            same_observable = set(current_leg_map) == set(held_leg_map)
            company_pair = (
                len(option.legs) == 1
                and len(held_option.legs) == 1
                and {option.legs[0].underlying_id, held_option.legs[0].underlying_id}
                == {AJARAI_UNDERLYING_ID, THERIODIC_UNDERLYING_ID}
            )
            if not same_observable and not company_pair:
                continue

            state_key = tuple(
                (u.underlying_id, round(float(u.value), 6))
                for u in self.underlying_state
            )
            cache_key = (held_id, held_option.steps_until_expiry, state_key)
            held_p = self.inventory_price_cache.get(cache_key)
            if held_p is None:
                held_p = max(0.0, min(1.0, float(self.price_option(held_option))))
                self.inventory_price_cache[cache_key] = held_p
            held_variance = held_p * (1.0 - held_p)
            held_time = max(float(held_option.steps_until_expiry), 1.0)
            maturity_factor = math.sqrt(min(option_time, held_time) / max(option_time, held_time))
            correlation = 0.0

            if same_observable:
                dot = sum(current_leg_map[key] * held_leg_map[key] for key in current_leg_map)
                norm_current = math.sqrt(sum(value * value for value in current_leg_map.values()))
                norm_held = math.sqrt(sum(value * value for value in held_leg_map.values()))
                cosine = dot / max(norm_current * norm_held, 1e-12)
                if abs(cosine) > 0.999 and payoff_variance > 1e-10 and held_variance > 1e-10:
                    joint_probability = (
                        min(p, held_p)
                        if cosine > 0.0
                        else max(0.0, p + held_p - 1.0)
                    )
                    correlation = (
                        (joint_probability - p * held_p)
                        / math.sqrt(payoff_variance * held_variance)
                    ) * maturity_factor

            elif company_pair:
                current_leg = option.legs[0]
                held_leg = held_option.legs[0]
                direction = 1.0 if current_leg.weight * held_leg.weight > 0.0 else -1.0
                binary_sensitivity = min(
                    1.0,
                    4.0 * math.sqrt(max(payoff_variance * held_variance, 0.0)),
                )
                correlation = (
                    direction
                    * self.company_return_correlation
                    * maturity_factor
                    * binary_sensitivity
                )

            correlation = max(-0.98, min(0.98, correlation))
            covariance_proxy = correlation * math.sqrt(max(payoff_variance * held_variance, 0.0))
            effective_inventory += held_quantity * covariance_proxy / variance_denominator

        effective_inventory = max(-20.0, min(20.0, effective_inventory))

        # Mean-variance reservation value for a binary payoff.
        reservation_price = (
            p
            - self.inventory_risk_aversion
            * effective_inventory
            * payoff_variance
        )

        # A small fixed spread plus extra protection for short-dated, uncertain binaries.
        uncertainty_edge = min(
            0.0075,
            self.uncertainty_scale
            * math.sqrt(payoff_variance / max(float(self.warmup_observations), 10.0)),
        )
        expiry_edge = self.expiry_edge_scale * (4.0 * payoff_variance) / math.sqrt(option_time)
        half_spread = self.rfq_base_half_spread + uncertainty_edge + expiry_edge

        bid = math.floor((reservation_price - half_spread) * 100.0 + 1e-9) / 100.0
        offer = math.ceil((reservation_price + half_spread) * 100.0 - 1e-9) / 100.0
        bid = max(0.0, min(0.99, bid))
        offer = max(0.01, min(1.0, offer))
        if bid >= offer:
            bid = max(0.0, min(0.99, round(reservation_price - 0.01, 2)))
            offer = min(1.0, max(0.01, round(reservation_price + 0.01, 2)))
            if bid >= offer:
                bid, offer = 0.0, 1.0

        # Inventory-aware size: quote more on the side that reduces portfolio risk.
        base_quantity = 5
        inventory_units = min(5, int(round(abs(effective_inventory))))
        if effective_inventory > 0.0:
            bid_quantity = max(1, base_quantity - inventory_units)
            offer_quantity = min(self.max_quote_quantity, base_quantity + inventory_units)
        elif effective_inventory < 0.0:
            bid_quantity = min(self.max_quote_quantity, base_quantity + inventory_units)
            offer_quantity = max(1, base_quantity - inventory_units)
        else:
            bid_quantity = base_quantity
            offer_quantity = base_quantity

        available_cash = max(self.risk_cash_remaining, 0.0)
        if bid > 0.0:
            bid_quantity = min(bid_quantity, int(available_cash / bid))
        if offer < 1.0:
            offer_quantity = min(offer_quantity, int(available_cash / (1.0 - offer)))

        if bid_quantity < 1:
            bid, bid_quantity = 0.0, 1
        if offer_quantity < 1:
            offer, offer_quantity = 1.0, 1
        if bid >= offer:
            bid, offer, bid_quantity, offer_quantity = 0.0, 1.0, 1, 1

        reservation = max(
            bid_quantity * bid,
            offer_quantity * (1.0 - offer),
        )
        if reservation > self.risk_cash_remaining + 1e-12:
            bid, offer, bid_quantity, offer_quantity, reservation = 0.0, 1.0, 1, 1, 0.0

        self.risk_cash_remaining -= reservation
        self.pending_risk_reservation = reservation
        self.pending_risk_option_id = option.option_id
        self.pending_risk_position_before = self.position.option_quantity_by_option_id.get(option.option_id, 0)
        self.pending_risk_bid_price = bid
        self.pending_risk_offer_price = offer
        self.pending_risk_fok_price = None
        self.pending_risk_fok_type = None

        return Quote(
            bid_price=round(bid, 2),
            bid_quantity=int(bid_quantity),
            offer_price=round(offer, 2),
            offer_quantity=int(offer_quantity),
        )

    def respond_to_fok(self, option: BinaryOption, fok_order: FokOrder) -> bool:  # type: ignore[empty-body]
        # Reconcile the last reservation using the position change recorded by on_trade().
        if self.pending_risk_option_id is not None:
            current_position = self.position.option_quantity_by_option_id.get(
                self.pending_risk_option_id, 0
            )
            position_change = current_position - self.pending_risk_position_before
            actual_cash_used = 0.0
            if position_change > 0:
                fill_price = (
                    self.pending_risk_fok_price
                    if self.pending_risk_fok_type is not None
                    else self.pending_risk_bid_price
                )
                actual_cash_used = position_change * float(fill_price)
            elif position_change < 0:
                fill_price = (
                    self.pending_risk_fok_price
                    if self.pending_risk_fok_type is not None
                    else self.pending_risk_offer_price
                )
                actual_cash_used = (-position_change) * (1.0 - float(fill_price))

            self.risk_cash_remaining += self.pending_risk_reservation - actual_cash_used
            self.risk_cash_remaining = max(
                0.0, min(0.95 * self.cash_balance, self.risk_cash_remaining)
            )
            self.pending_risk_reservation = 0.0
            self.pending_risk_option_id = None
            self.pending_risk_fok_type = None

        fair_value = max(0.0, min(1.0, float(self.price_option(option))))
        payoff_variance = fair_value * (1.0 - fair_value)
        variance_denominator = max(payoff_variance, 0.01)
        option_time = max(float(option.steps_until_expiry), 1.0)

        # Same lightweight cross-contract inventory approximation as quote().
        effective_inventory = 0.0
        active_by_id = {active.option_id: active for active in self.active_option_state}
        current_leg_map = {leg.underlying_id: leg.weight for leg in option.legs}

        for held_id, held_quantity in self.position.option_quantity_by_option_id.items():
            if held_quantity == 0:
                continue
            held_option = active_by_id.get(held_id)
            if held_option is None:
                continue
            if held_id == option.option_id:
                effective_inventory += held_quantity
                continue

            held_leg_map = {leg.underlying_id: leg.weight for leg in held_option.legs}
            same_observable = set(current_leg_map) == set(held_leg_map)
            company_pair = (
                len(option.legs) == 1
                and len(held_option.legs) == 1
                and {option.legs[0].underlying_id, held_option.legs[0].underlying_id}
                == {AJARAI_UNDERLYING_ID, THERIODIC_UNDERLYING_ID}
            )
            if not same_observable and not company_pair:
                continue

            state_key = tuple(
                (u.underlying_id, round(float(u.value), 6))
                for u in self.underlying_state
            )
            cache_key = (held_id, held_option.steps_until_expiry, state_key)
            held_p = self.inventory_price_cache.get(cache_key)
            if held_p is None:
                held_p = max(0.0, min(1.0, float(self.price_option(held_option))))
                self.inventory_price_cache[cache_key] = held_p
            held_variance = held_p * (1.0 - held_p)
            held_time = max(float(held_option.steps_until_expiry), 1.0)
            maturity_factor = math.sqrt(min(option_time, held_time) / max(option_time, held_time))
            correlation = 0.0

            if same_observable:
                dot = sum(current_leg_map[key] * held_leg_map[key] for key in current_leg_map)
                norm_current = math.sqrt(sum(value * value for value in current_leg_map.values()))
                norm_held = math.sqrt(sum(value * value for value in held_leg_map.values()))
                cosine = dot / max(norm_current * norm_held, 1e-12)
                if abs(cosine) > 0.999 and payoff_variance > 1e-10 and held_variance > 1e-10:
                    joint_probability = (
                        min(fair_value, held_p)
                        if cosine > 0.0
                        else max(0.0, fair_value + held_p - 1.0)
                    )
                    correlation = (
                        (joint_probability - fair_value * held_p)
                        / math.sqrt(payoff_variance * held_variance)
                    ) * maturity_factor

            elif company_pair:
                current_leg = option.legs[0]
                held_leg = held_option.legs[0]
                direction = 1.0 if current_leg.weight * held_leg.weight > 0.0 else -1.0
                binary_sensitivity = min(
                    1.0,
                    4.0 * math.sqrt(max(payoff_variance * held_variance, 0.0)),
                )
                correlation = (
                    direction
                    * self.company_return_correlation
                    * maturity_factor
                    * binary_sensitivity
                )

            correlation = max(-0.98, min(0.98, correlation))
            covariance_proxy = correlation * math.sqrt(max(payoff_variance * held_variance, 0.0))
            effective_inventory += held_quantity * covariance_proxy / variance_denominator

        effective_inventory = max(-20.0, min(20.0, effective_inventory))
        common_center = (
            fair_value
            - self.inventory_risk_aversion
            * effective_inventory
            * payoff_variance
        )

        uncertainty_edge = min(
            0.0075,
            self.uncertainty_scale
            * math.sqrt(payoff_variance / max(float(self.warmup_observations), 10.0)),
        )
        expiry_edge = self.expiry_edge_scale * (4.0 * payoff_variance) / math.sqrt(option_time)
        required_edge = self.fok_base_edge + uncertainty_edge + expiry_edge
        quantity = int(fok_order.quantity)

        # Finite-size risk term values the post-trade inventory, preventing a large
        # FOK from receiving an inventory-reduction concession after it crosses zero.
        if fok_order.order_type == OrderType.BUY:
            minimum_sell_price = (
                common_center
                + 0.5 * self.inventory_risk_aversion * quantity * payoff_variance
                + required_edge
            )
            if fok_order.price + 1e-12 < minimum_sell_price:
                return False
            required_cash = quantity * (1.0 - fok_order.price)
        else:
            maximum_buy_price = (
                common_center
                - 0.5 * self.inventory_risk_aversion * quantity * payoff_variance
                - required_edge
            )
            if fok_order.price - 1e-12 > maximum_buy_price:
                return False
            required_cash = quantity * fok_order.price

        # Assume the whole FOK can fill.  This is conservative but bankruptcy-safe.
        if required_cash > self.risk_cash_remaining + 1e-12:
            return False

        self.risk_cash_remaining -= required_cash
        self.pending_risk_reservation = required_cash
        self.pending_risk_option_id = option.option_id
        self.pending_risk_position_before = self.position.option_quantity_by_option_id.get(option.option_id, 0)
        self.pending_risk_bid_price = None
        self.pending_risk_offer_price = None
        self.pending_risk_fok_price = fok_order.price
        self.pending_risk_fok_type = fok_order.order_type
        return True

    def warm_up(self, market_history: MarketHistory) -> None:
        # Strategy/risk constants.  Keep the live path cheap: no Monte Carlo,
        # no live refitting, no trade ledger, and no optimizer.
        self.risk_cash_remaining = 0.95 * self.cash_balance
        self.pending_risk_reservation = 0.0
        self.pending_risk_option_id = None
        self.pending_risk_position_before = 0
        self.pending_risk_bid_price = None
        self.pending_risk_offer_price = None
        self.pending_risk_fok_price = None
        self.pending_risk_fok_type = None

        self.inventory_risk_aversion = 0.02
        self.rfq_base_half_spread = 0.025
        self.fok_base_edge = 0.005
        self.uncertainty_scale = 0.06
        self.expiry_edge_scale = 0.0075
        self.max_quote_quantity = 10
        self.inventory_price_cache = {}

        fed = np.asarray(
            market_history.values_by_underlying_id[FED_FUNDS_RATE_UNDERLYING_ID],
            dtype=float,
        )
        ajr = np.asarray(
            market_history.values_by_underlying_id[AJARAI_UNDERLYING_ID],
            dtype=float,
        )
        thr = np.asarray(
            market_history.values_by_underlying_id[THERIODIC_UNDERLYING_ID],
            dtype=float,
        )
        n = len(fed)
        self.warmup_observations = max(n - 1, 1)

        if n < 2:
            self.company_return_correlation = 0.0
            self.estimated_market_parameters = MarketParameters(
                ajarai_drift=0.0,
                ajarai_idio_std_dev=0.01,
                ajarai_rate_beta=0.0,
                ajarai_sector_beta=0.0,
                rate_down_probability=0.25,
                rate_reversion_strength=0.0,
                rate_up_probability=0.25,
                sector_std_dev=0.0,
                theriodic_drift=0.0,
                theriodic_idio_std_dev=0.01,
                theriodic_rate_beta=0.0,
                theriodic_sector_beta=0.0,
            )
            return

        rate_changes = np.diff(fed)
        ajr_returns = np.log(ajr[1:] / ajr[:-1])
        thr_returns = np.log(thr[1:] / thr[:-1])
        m = len(rate_changes)

        # Lightweight FED fit.  For each candidate mean-reversion strength, infer
        # baseline up/down probabilities by moments, then score the exact clipped
        # transition likelihood.  At FED=0 a latent down move and a flat move both
        # produce the observed state 0, so their probabilities are added.
        best_log_likelihood = -float("inf")
        best_rate_up = 0.25
        best_rate_down = 0.25
        best_reversion = 0.0
        eps = 1e-8

        for k_index in range(101):
            k = k_index / 100.0
            x = 2.0 - fed[:-1]
            p_up = float(np.mean((rate_changes > 0).astype(float) - k * x))
            positive_rate_mask = fed[:-1] > 0.0
            if np.any(positive_rate_mask):
                p_down = float(np.mean(
                    (rate_changes[positive_rate_mask] < 0).astype(float)
                    + k * x[positive_rate_mask]
                ))
            else:
                p_down = 0.25

            p_up = max(eps, min(1.0 - eps, p_up))
            p_down = max(eps, min(1.0 - eps, p_down))
            if p_up + p_down >= 1.0:
                scale = (1.0 - eps) / (p_up + p_down)
                p_up *= scale
                p_down *= scale

            log_likelihood = 0.0
            for t in range(m):
                rate_value = float(fed[t])
                tilt = k * (2.0 - rate_value)
                up_probability = min(max(p_up + tilt, 0.0), 1.0)
                down_probability = min(
                    max(p_down - tilt, 0.0),
                    1.0 - up_probability,
                )
                flat_probability = 1.0 - up_probability - down_probability

                observed_next = round(float(fed[t + 1]), 2)
                up_rate = max(round(rate_value + 0.25, 2), 0.0)
                down_rate = max(round(rate_value - 0.25, 2), 0.0)
                flat_rate = round(rate_value, 2)
                probability = (
                    (up_probability if up_rate == observed_next else 0.0)
                    + (down_probability if down_rate == observed_next else 0.0)
                    + (flat_probability if flat_rate == observed_next else 0.0)
                )
                log_likelihood += math.log(max(probability, eps))

            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_rate_up = p_up
                best_rate_down = p_down
                best_reversion = k

        # OLS company dynamics: log return = drift + rate_beta * dFED + residual.
        mean_x = float(np.mean(rate_changes))
        centered_x = rate_changes - mean_x
        var_x = float(np.dot(centered_x, centered_x))

        mean_ajr = float(np.mean(ajr_returns))
        mean_thr = float(np.mean(thr_returns))
        if var_x > 1e-15:
            ajr_rate_beta = float(np.dot(centered_x, ajr_returns - mean_ajr) / var_x)
            thr_rate_beta = float(np.dot(centered_x, thr_returns - mean_thr) / var_x)
        else:
            ajr_rate_beta = 0.0
            thr_rate_beta = 0.0

        ajr_drift = mean_ajr - ajr_rate_beta * mean_x
        thr_drift = mean_thr - thr_rate_beta * mean_x
        ajr_residuals = ajr_returns - ajr_drift - ajr_rate_beta * rate_changes
        thr_residuals = thr_returns - thr_drift - thr_rate_beta * rate_changes

        var_ajr_resid = float(np.mean((ajr_residuals - np.mean(ajr_residuals)) ** 2))
        var_thr_resid = float(np.mean((thr_residuals - np.mean(thr_residuals)) ** 2))
        cov_resid = float(np.mean(
            (ajr_residuals - np.mean(ajr_residuals))
            * (thr_residuals - np.mean(thr_residuals))
        ))
        var_ajr_resid = max(var_ajr_resid, 0.0)
        var_thr_resid = max(var_thr_resid, 0.0)
        max_abs_cov = math.sqrt(var_ajr_resid * var_thr_resid)
        cov_resid = max(-max_abs_cov, min(max_abs_cov, cov_resid)) if max_abs_cov > 0.0 else 0.0

        # Store one simple cross-company correlation for inventory approximation.
        ajr_std = float(np.std(ajr_returns))
        thr_std = float(np.std(thr_returns))
        if ajr_std > 1e-12 and thr_std > 1e-12:
            self.company_return_correlation = float(np.corrcoef(ajr_returns, thr_returns)[0, 1])
            self.company_return_correlation = max(-0.95, min(0.95, self.company_return_correlation))
        else:
            self.company_return_correlation = 0.0

        # Pricing-equivalent Gaussian decomposition of the residual covariance.
        if abs(cov_resid) <= 1e-15 or var_ajr_resid <= 1e-15 or var_thr_resid <= 1e-15:
            sector_std_dev = 0.0
            ajr_sector_beta = 0.0
            thr_sector_beta = 0.0
            ajr_idio_var = var_ajr_resid
            thr_idio_var = var_thr_resid
        else:
            sector_std_dev = 1.0
            abs_cov = abs(cov_resid)
            ajr_common_var = min(
                abs_cov * math.sqrt(var_ajr_resid / var_thr_resid),
                var_ajr_resid,
            )
            thr_common_var = min(
                abs_cov * math.sqrt(var_thr_resid / var_ajr_resid),
                var_thr_resid,
            )
            ajr_sector_beta = math.sqrt(max(ajr_common_var, 0.0))
            thr_sector_beta = math.sqrt(max(thr_common_var, 0.0))
            if cov_resid < 0.0:
                thr_sector_beta *= -1.0
            ajr_idio_var = max(var_ajr_resid - ajr_sector_beta ** 2, 0.0)
            thr_idio_var = max(var_thr_resid - thr_sector_beta ** 2, 0.0)

        self.estimated_market_parameters = MarketParameters(
            ajarai_drift=ajr_drift,
            ajarai_idio_std_dev=math.sqrt(ajr_idio_var),
            ajarai_rate_beta=ajr_rate_beta,
            ajarai_sector_beta=ajr_sector_beta,
            rate_down_probability=best_rate_down,
            rate_reversion_strength=best_reversion,
            rate_up_probability=best_rate_up,
            sector_std_dev=sector_std_dev,
            theriodic_drift=thr_drift,
            theriodic_idio_std_dev=math.sqrt(thr_idio_var),
            theriodic_rate_beta=thr_rate_beta,
            theriodic_sector_beta=thr_sector_beta,
        )
