from datamodel import (
    Listing,
    Observation,
    Order,
    OrderDepth,
    ProsperityEncoder,
    Symbol,
    Trade,
    TradingState,
)
from typing import List, Any
import jsonpickle
import json
import numpy as np
import math


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(
        self,
        state: TradingState,
        orders: dict[Symbol, list[Order]],
        conversions: int,
        trader_data: str,
    ) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(
                        state, self.truncate(state.traderData, max_item_length)
                    ),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(
        self, order_depths: dict[Symbol, OrderDepth]
    ) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUIDINK = "SQUID_INK"
PRODUCTS = [RESIN, KELP, SQUIDINK]

DEFAULT_PRICES = {RESIN: 10000, KELP: 2000, SQUIDINK: 2000}
POSITION_LIMIT = {RESIN: 50, KELP: 50, SQUIDINK: 50}

logger = Logger()


class Trader:
    def __init__(self):

        self.ema_factor = 0.6
        #  ema_factor = 2 / (period + 1)
        self.resin_factor = 0.00985
        self.kelp_factor = 0.4
        self.squidink_factor = 0.6
        self.squidink_frequency = self.squidink_cooldown = 140

        # calculated resin std to be 1.4965920476507861 from provided data
        self.kelp_mm_volume_threshold = 20
        self.resin_spread = 1.4965920476507861
        self.kelp_spread = 1
        self.squidink_spread = 1
        # self.spread = 2

    # default_price should be set to the previous ema if available
    def get_mid_price(self, product, state: TradingState) -> float:
        default_price = DEFAULT_PRICES[product]

        if product not in state.order_depths:
            return default_price
        order_depth = state.order_depths[product]

        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            return (best_bid + best_ask) / 2
        else:
            return default_price

    def resin_strat(self, state: TradingState, storedData):
        position = state.position.get(RESIN, 0)
        s = self.resin_spread

        bid_volume = POSITION_LIMIT[RESIN] - position
        ask_volume = -POSITION_LIMIT[RESIN] - position

        orders: List[Order] = []

        # orders.append(Order(RESIN, DEFAULT_PRICES[RESIN] - s, bid_volume))
        # orders.append(Order(RESIN, DEFAULT_PRICES[RESIN] + s, ask_volume))
        orders.append(
            Order(RESIN, math.floor(storedData[RESIN]["ema"] - s), bid_volume // 2)
        )
        orders.append(
            Order(RESIN, math.ceil(storedData[RESIN]["ema"] + s), ask_volume // 2)
        )

        return orders

    def kelp_strat(self, state: TradingState, storedData):
        position = state.position.get(KELP, 0)
        s = self.kelp_spread

        bid_volume = POSITION_LIMIT[KELP] - position
        ask_volume = -POSITION_LIMIT[KELP] - position

        orders: List[Order] = []

        orders.append(
            Order(KELP, math.floor(storedData[KELP]["fairprice"] - s), bid_volume // 2)
        )
        orders.append(
            Order(KELP, math.ceil(storedData[KELP]["fairprice"] + s), ask_volume // 2)
        )

        return orders

    def squidink_strat(self, state: TradingState, storedData):
        if self.squidink_cooldown > 0:
            self.squidink_cooldown -= 1
            return []

        # position = state.position.get(SQUIDINK, 0)
        # s = self.squidink_spread

        # bid_volume = POSITION_LIMIT[SQUIDINK] - position  # buy_volume
        # ask_volume = -POSITION_LIMIT[SQUIDINK] - position  # sell_volume

        orders: List[Order] = []
        order_depth = state.order_depths[SQUIDINK]

        best_bid, best_ask = None, None
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)

        current_mprice = storedData[SQUIDINK]["midprice"]
        prev_mprice = storedData[SQUIDINK]["prevN_midprice"]
        buy_momentum = (current_mprice - 1.1 * prev_mprice) / self.squidink_frequency
        sell_momentum = (current_mprice - prev_mprice) / self.squidink_frequency
        if buy_momentum > 0 and storedData[SQUIDINK]["obv"] > 0:
            # When momentum > 0 buy
            volume = max(order_depth.sell_orders[best_ask], -5)  # example cap
            orders.append(Order(SQUIDINK, best_ask, -volume))
        elif sell_momentum < 0 and storedData[SQUIDINK]["obv"] < 0:
            # When momentum < 0 sell
            volume = min(order_depth.buy_orders[best_bid], 5)
            orders.append(Order(SQUIDINK, best_bid, -volume))

        self.squidink_cooldown = self.squidink_frequency
        storedData[SQUIDINK]["prevN_midprice"] = current_mprice
        return orders

    # main trading function
    def run(self, state: TradingState):
        # store all state that needs to be preserved in the state.traderData variable
        storedData = {
            RESIN: {
                "midprice": None,
                "ema": None,
                "ema_factor": self.resin_factor,
            },
            KELP: {
                "midprice": None,
                "fairprice": None,
                "ema": None,
                "ema_factor": self.kelp_factor,
            },
            SQUIDINK: {
                "prevN_midprice": None,
                "midprice": None,
                "ema": None,
                "ema_factor": self.kelp_factor,
                "obv": 0,
            },
        }
        if state.traderData != "":
            storedData = jsonpickle.decode(state.traderData)

        result = {}

        # update midprice and exponential moving averages for each product
        for product in PRODUCTS:
            # store data for all products
            # - orderbook midprice
            # - exponential moving average
            mid_price = self.get_mid_price(product, state)
            storedData[product]["midprice"] = mid_price

            if storedData[product]["ema"] is None:
                storedData[product]["ema"] = mid_price
            else:
                prevEma = storedData[product]["ema"]
                k = storedData[product]["ema_factor"]
                storedData[product]["ema"] = mid_price * k + prevEma * (1 - k)

            if product == KELP:
                # set fair price to mid price of orderbook by default.
                storedData[product]["fairprice"] = storedData[product]["ema"]

                order_depth = state.order_depths[product]
                buys = sorted(order_depth.buy_orders.items(), reverse=True)
                sells = sorted(order_depth.sell_orders.items())
                mm_bid = next(
                    (
                        (price, vol)
                        for price, vol in buys
                        if vol >= self.kelp_mm_volume_threshold
                    ),
                    None,
                )
                mm_ask = next(
                    (
                        (price, vol)
                        for price, vol in sells
                        if vol >= self.kelp_mm_volume_threshold
                    ),
                    None,
                )
                if mm_bid and mm_ask:
                    storedData[product]["fairprice"] = (mm_bid + mm_ask) / 2
            elif product == SQUIDINK:
                mid_price = self.get_mid_price(product, state)
                prev_mid_price = storedData[product]["midprice"]

                order_depth = state.order_depths[product]
                if (
                    prev_mid_price is not None
                    and mid_price > prev_mid_price
                    and order_depth.buy_orders
                    and order_depth.sell_orders
                ):
                    volume = sum(order_depth.buy_orders.values()) - sum(
                        order_depth.sell_orders.values()
                    )
                    storedData[product]["obv"] += volume
                elif (
                    prev_mid_price is not None
                    and mid_price < prev_mid_price
                    and order_depth.buy_orders
                    and order_depth.sell_orders
                ):
                    # make volume account for other bot trades too
                    volume = sum(order_depth.buy_orders.values()) - sum(
                        order_depth.sell_orders.values()
                    )
                    storedData[product]["obv"] -= volume

                storedData[product]["midprice"] = mid_price
                if storedData[product]["prevN_midprice"] is None:
                    storedData[product]["prevN_midprice"] = mid_price

        # get orders for each product
        result[RESIN] = self.resin_strat(state, storedData)
        result[KELP] = self.kelp_strat(state, storedData)
        result[SQUIDINK] = self.squidink_strat(state, storedData)

        # save data to traderData for next cycle
        traderData = jsonpickle.encode(storedData)

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
