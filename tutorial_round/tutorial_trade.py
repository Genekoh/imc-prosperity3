from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import numpy as np
import math

RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
PRODUCTS = [RESIN, KELP]

DEFAULT_PRICES = {RESIN: 10000, KELP: 2000}
POSITION_LIMIT = {RESIN: 50, KELP: 50}


class Trader:
    def __init__(self):
        self.ema_factor = 0.6
        self.resin_spread = 2
        self.kelp_spread = 6
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

        bid_volume = POSITION_LIMIT[RESIN] - position
        ask_volume = -POSITION_LIMIT[RESIN] - position

        orders: List[Order] = []
        if position == 0:
            orders.append(
                Order(KELP, math.floor(storedData[KELP]["ema_price"] - s), bid_volume)
            )
            orders.append(
                Order(KELP, math.ceil(storedData[KELP]["ema_price"] + s), ask_volume)
            )

        if position > 0:
            # Long position
            orders.append(
                Order(
                    KELP,
                    math.floor(storedData[KELP]["ema_price"] - (2 * s)),
                    bid_volume,
                )
            )
            orders.append(
                Order(KELP, math.ceil(storedData[KELP]["ema_price"]), ask_volume)
            )

        if position < 0:
            # Short position
            orders.append(
                Order(KELP, math.floor(storedData[KELP]["ema_price"]), bid_volume)
            )
            orders.append(
                Order(
                    KELP,
                    math.ceil(storedData[KELP]["ema_price"] + (2 * s)),
                    ask_volume,
                )
            )

        return orders

    def run(self, state: TradingState):
        storedData = {
            RESIN: {"ema_price": None},
            KELP: {"ema_price": None},
        }
        if state.traderData != "":
            storedData = jsonpickle.decode(state.traderData)

        result = {}

        for product in PRODUCTS:
            mid_price = self.get_mid_price(product, state)
            if storedData[product]["ema_price"] is None:
                storedData[product]["ema_price"] = mid_price
            else:
                prevPrice = storedData[product]["ema_price"]
                k = self.ema_factor
                storedData[product]["ema_price"] = mid_price * k + prevPrice * (1 - k)

        result[RESIN] = self.resin_strat(state, storedData)
        result[KELP] = self.kelp_strat(state, storedData)

        traderData = jsonpickle.encode(storedData)

        conversions = 1
        return result, conversions, traderData
