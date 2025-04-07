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

        self.ema_factor = 0.5
        self.resin_factor = 3
        self.kelp_factor = 1
        self.squidink_factor = 1

        self.resin_spread = 1
        self.kelp_spread = 2
        self.squidink_spread = 2
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
        orders.append(Order(RESIN, DEFAULT_PRICES[RESIN] - s, bid_volume))
        orders.append(Order(RESIN, DEFAULT_PRICES[RESIN] + s, ask_volume))

        return orders

    def kelp_strat(self, state: TradingState, storedData):
        position = state.position.get(KELP, 0)
        s = self.kelp_spread

        bid_volume = POSITION_LIMIT[KELP] - position
        ask_volume = -POSITION_LIMIT[KELP] - position

        orders: List[Order] = []
        if position == 0:
            orders.append(
                Order(KELP, math.floor(storedData[KELP]["ema"] - s), bid_volume)
            )
            orders.append(
                Order(KELP, math.ceil(storedData[KELP]["ema"] + s), ask_volume)
            )

        if position > 0:
            # Long position
            orders.append(
                Order(
                    KELP,
                    math.floor(storedData[KELP]["ema"] - (2 * s)),
                    bid_volume,
                )
            )
            orders.append(Order(KELP, math.ceil(storedData[KELP]["ema"]), ask_volume))

        if position < 0:
            # Short position
            orders.append(Order(KELP, math.floor(storedData[KELP]["ema"]), bid_volume))
            orders.append(
                Order(
                    KELP,
                    math.ceil(storedData[KELP]["ema"] + (2 * s)),
                    ask_volume,
                )
            )

        return orders

    def squidink_strat(self, state: TradingState, storedData):
        position = state.position.get(SQUIDINK, 0)
        s = self.squidink_spread

        bid_volume = POSITION_LIMIT[SQUIDINK] - position
        ask_volume = -POSITION_LIMIT[SQUIDINK] - position

        orders: List[Order] = []

        if position == 0:
            orders.append(
                Order(
                    SQUIDINK,
                    math.floor(storedData[SQUIDINK]["ema"] - s),
                    bid_volume,
                )
            )
            orders.append(
                Order(
                    SQUIDINK,
                    math.ceil(storedData[SQUIDINK]["ema"] + s),
                    ask_volume,
                )
            )

        if position > 0:
            # Long position
            orders.append(
                Order(
                    SQUIDINK,
                    math.floor(storedData[SQUIDINK]["ema"] - (2 * s)),
                    bid_volume,
                )
            )
            orders.append(
                Order(SQUIDINK, math.ceil(storedData[SQUIDINK]["ema"]), ask_volume)
            )

        if position < 0:
            # Short position
            orders.append(
                Order(SQUIDINK, math.floor(storedData[SQUIDINK]["ema"]), bid_volume)
            )
            orders.append(
                Order(
                    SQUIDINK,
                    math.ceil(storedData[SQUIDINK]["ema"] + (2 * s)),
                    ask_volume,
                )
            )

        return orders

    # main trading function
    def run(self, state: TradingState):
        # store all state that needs to be preserved in the state.traderData variable
        storedData = {
            RESIN: {"midprice": None, "ema": None, "ema_factor": self.resin_factor},
            KELP: {"midprice": None, "ema": None, "ema_factor": self.kelp_factor},
            SQUIDINK: {
                "midprice": None,
                "ema": None,
                "ema_factor": self.squidink_factor,
                # "fast_ema": None,
                # "slow_ema": None,
            },
        }
        if state.traderData != "":
            storedData = jsonpickle.decode(state.traderData)

        result = {}

        # update midprice and exponential moving averages for each product
        for product in PRODUCTS:
            mid_price = self.get_mid_price(product, state)
            storedData[product]["midprice"] = mid_price
            if storedData[product]["ema"] is None:
                storedData[product]["ema"] = mid_price
            else:
                prevPrice = storedData[product]["ema"]
                period = storedData[product]["ema_factor"]
                k = 2 / (period + 1)
                storedData[product]["ema"] = mid_price * k + prevPrice * (1 - k)

        # get orders for each product
        result[RESIN] = self.resin_strat(state, storedData)
        result[KELP] = self.kelp_strat(state, storedData)
        result[SQUIDINK] = self.squidink_strat(state, storedData)

        # save data to traderData for next cycle
        traderData = jsonpickle.encode(storedData)

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
