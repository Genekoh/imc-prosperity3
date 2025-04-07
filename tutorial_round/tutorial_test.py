from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Time, Symbol
import jsonpickle
import numpy as np


class TraderObject:
    def __init__(self):
        # list of tuples (timestamp, price/sma)
        self.midprices: List[tuple[Time, float]] = []
        self.sma: List[tuple[Time, float]] = []


class Trader:
    def get_mid_price(
        self, trader_object: TraderObject, order_depth: OrderDepth
    ) -> float:
        prevMidPrice = 0.0
        if len(trader_object.midprices) > 0:
            prevMidPrice = trader_object.midprices[-1][1]

        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = np.max(order_depth.buy_orders)
            best_ask = np.min(order_depth.sell_orders)
            return (best_bid + best_ask) / 2
        else:
            return prevMidPrice

    def get_moving_average(
        self, prices: Dict[Time, float], currentTime: Time, window=500
    ) -> float:
        prices_in_window = dict(
            filter(lambda x: x[0] >= (currentTime - window), prices)
        )
        return np.mean(list(prices_in_window.values()))

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        traderObjects: Dict[Symbol, TraderObject] = {}
        if state.traderData != "":
            traderObjects = jsonpickle.decode(
                state.traderData, classes=Dict[Symbol, TraderObject]
            )
        else:
            for product in state.listings:
                traderObjects[product] = TraderObject()

        # Initialize the method output dict as an empty dict
        result = {}

        # Iterate over all the keys (the available products) contained in the order dephts
        for product in state.order_depths.keys():
            # Retrieve the Order Depth containing all the market BUY and SELL orders
            order_depth: OrderDepth = state.order_depths[product]

            traderObjects[product].midprices.append(
                (
                    state.timestamp,
                    self.get_mid_price(traderObjects[product], order_depth),
                )
            )
            traderObjects[product].sma.append(
                (
                    state.timestamp,
                    self.get_moving_average(
                        traderObjects[product].midprices, state.timestamp
                    ),
                )
            )

            # Initialize the list of Orders to be sent as an empty list
            orders: list[Order] = []

            # Note that this value of 1 is just a dummy value, you should likely change it!
            acceptable_price = 10

            # If statement checks if there are any SELL orders in the market
            if len(order_depth.sell_orders) > 0:

                # Sort all the available sell orders by their price,
                # and select only the sell order with the lowest price
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_volume = order_depth.sell_orders[best_ask]

                # Check if the lowest ask (sell order) is lower than the above defined fair value
                if best_ask < acceptable_price:

                    # In case the lowest ask is lower than our fair value,
                    # This presents an opportunity for us to buy cheaply
                    # The code below therefore sends a BUY order at the price level of the ask,
                    # with the same quantity
                    # We expect this order to trade with the sell order
                    print("BUY", str(-best_ask_volume) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_volume))

            # The below code block is similar to the one above,
            # the difference is that it find the highest bid (buy order)
            # If the price of the order is higher than the fair value
            # This is an opportunity to sell at a premium
            if len(order_depth.buy_orders) != 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_volume = order_depth.buy_orders[best_bid]
                if best_bid > acceptable_price:
                    print("SELL", str(best_bid_volume) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_volume))

            # Add all the above the orders to the result dict
            result[product] = orders

        traderData = jsonpickle.encode(
            traderObjects
        )  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.

        conversions = 1

        # Return the dict of orders
        # These possibly contain buy or sell orders
        # Depending on the logic above

        print("result", result)
        return result, conversions, traderData
