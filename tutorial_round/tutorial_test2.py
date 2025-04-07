from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import jsonpickle
import numpy as np

RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
PRODUCTS = [RESIN, KELP]

DEFAULT_PRICES = {RESIN: 10000, KELP: 2000}
POSITION_LIMIT = {RESIN: 50, KELP: 50}


class Trader:
    def __init__(self):
        self.ema_factor = 0.5

    # default_price should be set to the previous ema if available
    def get_ema(self, product, state: TradingState, default_price=None):
        if default_price = None
            default_price = DEFAULT_PRICES[product]

        if product not in state.order_depth:
            return default_price
        order_depth = state.order_depths[product]

        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            return (best_bid + best_ask) // 2
        else:
            return default_price

    # def get_micro_price(self, order_depth: OrderDepth):
    #     if order_depth.buy_orders and order_depth.sell_orders:
    #         bid_volume = 0
    #         ask_volume = 0
    #         volume = 0
    #         for p, v in order_depth.buy_orders.items():
    #             bid_volume += p * v
    #             volume += v
    #         for p, v in order_depth.sell_orders.items():
    #             ask_volume += p * v
    #             volume -= v
    #         return (bid_volume + ask_volume) / volume
    #     else:
    #         return None

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        tradeHistories = {}
        if state.traderData != "":
            tradeHistories = jsonpickle.decode(state.traderData)
        else:
            for products in state.listings:
                tradeHistories[products] = []

        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []

            fair_price = self.get_micro_price(order_depth)
            if fair_price == None:
                print("imintrouble")
            x = state.listings[product].product
            position = 0
            if x in state.position.keys():
                position = state.position[x]
            tradeHistories[product].append(fair_price)
            # if product in state.market_trades.keys():
            #     for trade in state.market_trades[product]:
            #         if trade.timestamp == state.timestamp:
            #             tradeHistories[product].append(trade.price)
            print("meanprice " + str(np.mean(tradeHistories[product])))

            # risk_multiplier = 1.0
            # n = 10
            # spread = 3

            # if len(tradeHistories[product]) > n:
            #     volatility = float(np.std(tradeHistories[product][-n:]))
            #     spread = int(2 * volatility * risk_multiplier)
            # qty = 2

            # if position < 50:
            #     buy_price = fair_price - spread // 2
            #     orders.append(Order(product, buy_price, qty))

            # if position > 50:
            #     sell_price = fair_price + spread // 2
            #     orders.append(Order(product, sell_price, -qty))

            if len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = order_depth.sell_orders[best_ask]
                if best_ask < fair_price:
                    orders.append(Order(product, best_ask, -best_ask_volume))

            if len(order_depth.buy_orders) != 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                if best_bid > fair_price:
                    orders.append(Order(product, best_bid, -best_bid_volume))

            result[product] = orders

        traderData = jsonpickle.encode(
            tradeHistories
        )  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1
        return result, conversions, traderData
