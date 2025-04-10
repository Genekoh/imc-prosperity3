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
POSITION_LIMIT = {RESIN: 50, KELP: 50, SQUIDINK: 50}
# params below
PARAMETERS = {
    "DEFAULT_PRICES": {RESIN: 10000, KELP: 2000, SQUIDINK: 1980},
    KELP: {
        "KELP_FACTOR": 0.4,  # ema_factor = 2 / (period + 1)
        "KELP_UNFAVOURABLE_VOLUME": 20,
        "KELP_TAKE_WIDTH": 1,
        "KELP_CLEAR_WIDTH": 0,
        "KELP_DISREGARD_EDGE": 1,
        "KELP_JOIN_EDGE": 0,
        "KELP_DEFAULT_EDGE": 1,
        "KELP_MM_VOLUME_THRESHOLD": 20,
    },
    RESIN: {
        "RESIN_FACTOR": 0.00985,  # 0.00985
        "RESIN_EDGE": 1.4965920476507861,  # 1.4965920476507861
    },
    SQUIDINK: {
        "SQUIDINK_FACTOR": 0.4,  # 0.4
        "PERIOD": 30,
        "RSI_PERIOD": 25,
        "STOP_LOSS_THRESHOLD": 1,
    },
}


logger = Logger()


class Trader:
    def __init__(self, params=PARAMETERS):
        self.params = params

    def get_mid_price(self, product, state: TradingState) -> float:
        default_price = self.params["DEFAULT_PRICES"][product]

        if product not in state.order_depths:
            return default_price
        order_depth = state.order_depths[product]

        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders)
            best_ask = min(order_depth.sell_orders)
            return (best_bid + best_ask) / 2
        else:
            return default_price

    def set_mid_price(self, product, state, storedData):
        mid_price = self.get_mid_price(product, state)
        storedData[product]["midprice"] = mid_price

    def set_fair_price_midprice(self, product, state, storedData):
        storedData[product]["fairprice"] = storedData[product]["midprice"]

    def set_fair_price_vsmm(self, product, state, storedData):
        order_depth = state.order_depths[product]
        buys = sorted(order_depth.buy_orders.items(), reverse=True)
        sells = sorted(order_depth.sell_orders.items())
        mm_bid = next(
            (
                (price, vol)
                for price, vol in buys
                if vol >= storedData[product]["mm_volume_threshold"]
            ),
            None,
        )
        mm_ask = next(
            (
                (price, vol)
                for price, vol in sells
                if vol >= storedData[product]["mm_volume_threshold"]
            ),
            None,
        )
        if mm_bid and mm_ask:
            storedData[product]["fairprice"] = (mm_bid + mm_ask) / 2
        else:
            storedData[product]["fairprice"] = storedData[product]["midprice"]

    def set_ema(self, product, state, storedData):
        mid_price = storedData[product]["midprice"]
        if storedData[product]["ema"] is None:
            storedData[product]["ema"] = mid_price
        else:
            prevEma = storedData[product]["ema"]
            k = storedData[product]["ema_factor"]
            storedData[product]["ema"] = mid_price * k + prevEma * (1 - k)

    def set_obv(self, product, state, storedData):
        mid_price = self.get_mid_price(product, state)  # the problem
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

    def resin_strat(self, state: TradingState, storedData):
        position = state.position.get(RESIN, 0)
        edge = self.params[RESIN]["RESIN_EDGE"]

        bid_volume = POSITION_LIMIT[RESIN] - position
        ask_volume = -POSITION_LIMIT[RESIN] - position

        orders: List[Order] = []

        # orders.append(Order(RESIN, DEFAULT_PRICES[RESIN] - s, bid_volume))
        # orders.append(Order(RESIN, DEFAULT_PRICES[RESIN] + s, ask_volume))
        orders.append(
            Order(RESIN, math.floor(storedData[RESIN]["ema"] - edge), bid_volume // 2)
        )
        orders.append(
            Order(RESIN, math.ceil(storedData[RESIN]["ema"] + edge), ask_volume // 2)
        )

        return orders

    def kelp_strat(self, state: TradingState, storedData):
        orders: List[Order] = []
        order_depth = state.order_depths[KELP]
        position = state.position.get(KELP, 0)
        position_limit = POSITION_LIMIT[KELP]
        buy_order_volume, sell_order_volume = 0, 0
        fair_price = storedData[KELP]["fairprice"]

        # take in favourable orders
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders)
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if (
                abs(best_ask_amount) <= self.params[KELP]["KELP_UNFAVOURABLE_VOLUME"]
                and best_ask <= fair_price - self.params[KELP]["KELP_TAKE_WIDTH"]
            ):
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order(KELP, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if (
                abs(best_bid_amount) <= self.params[KELP]["KELP_UNFAVOURABLE_VOLUME"]
                and best_bid >= fair_price + self.params[KELP]["KELP_TAKE_WIDTH"]
            ):
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order(KELP, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]

        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_price - self.params[KELP]["KELP_CLEAR_WIDTH"])
        fair_for_ask = round(fair_price + self.params[KELP]["KELP_CLEAR_WIDTH"])

        # clear orders in a favourable manner
        max_buy_volume = position_limit - (position + buy_order_volume)
        max_sell_volume = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(max_sell_volume, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(KELP, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(max_buy_volume, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(KELP, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_price + self.params[KELP]["KELP_DISREGARD_EDGE"]
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_price - self.params[KELP]["KELP_DISREGARD_EDGE"]
        ]
        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        ask = round(fair_price + self.params[KELP]["KELP_DEFAULT_EDGE"])  # ceil?
        if best_ask_above_fair is not None:
            if (
                abs(best_ask_above_fair - fair_price)
                <= self.params[KELP]["KELP_JOIN_EDGE"]
            ):
                ask = best_ask_above_fair  # join
            else:
                ask = best_ask_above_fair - 1  # penny

        bid = round(fair_price - self.params[KELP]["KELP_DEFAULT_EDGE"])  # floor?
        if best_bid_below_fair is not None:
            if (
                abs(fair_price - best_bid_below_fair)
                <= self.params[KELP]["KELP_JOIN_EDGE"]
            ):
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        # marketmake
        max_buy_volume = position_limit - (position + buy_order_volume)
        max_sell_volume = position_limit + (position - sell_order_volume)

        if max_buy_volume > 0:
            orders.append(Order(KELP, bid, max_buy_volume))
        if max_sell_volume > 0:
            orders.append(Order(KELP, ask, -max_sell_volume))

        return orders

    def squidink_strat(self, state: TradingState, storedData):
        period = self.params[SQUIDINK]["PERIOD"]
        if (state.timestamp / 100) % period != 0:
            return []

        orders: List[Order] = []
        order_depth = state.order_depths[SQUIDINK]

        best_bid, best_ask = None, None
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)

        position = state.position.get(SQUIDINK, 0)
        entry_price = storedData[SQUIDINK]["entry_price"]
        # entry_price = storedData[SQUIDINK]["entry_price"]
        stop_loss_threshold = self.params[SQUIDINK]["STOP_LOSS_THRESHOLD"]
        current_mprice = storedData[SQUIDINK]["midprice"]
        prev_mprice = storedData[SQUIDINK]["prevN_midprice"]
        # buy_momentum good 0.2 * 20 ?
        buy_momentum = current_mprice - prev_mprice
        sell_momentum = current_mprice - prev_mprice
        obv = storedData[SQUIDINK]["obv"]
        rsi = storedData[SQUIDINK]["rsi"]

        # stop loss, risk management
        if position != 0 and entry_price is not None:
            pct_change = 100 * (current_mprice - entry_price) / entry_price
            if (position > 0 and pct_change < -stop_loss_threshold) or (
                position < 0 and pct_change > stop_loss_threshold
            ):
                # Stop loss triggered, close position
                if position > 0:
                    volume = -position  # Sell to close long
                    orders.append(Order(SQUIDINK, best_bid, volume))
                else:
                    volume = -position  # Buy to close short
                    orders.append(Order(SQUIDINK, best_ask, volume))
                return orders

        # entry conditions
        if position == 0:
            if buy_momentum > 0 and obv > 0 and rsi > 55:
                # When momentum > 0 buy
                volume = min(order_depth.sell_orders[best_ask], -5)  # example cap
                orders.append(Order(SQUIDINK, best_ask - 1, -volume))
            elif sell_momentum < -0 and obv < 0 and rsi < 45:
                # When momentum < 0 sell
                volume = max(order_depth.buy_orders[best_bid], 5)
                orders.append(Order(SQUIDINK, best_bid + 1, -volume))

        # set new previous mid price
        storedData[SQUIDINK]["prevN_midprice"] = current_mprice
        return orders

    # main trading function
    def run(self, state: TradingState):
        # store all state that needs to be preserved in the state.traderData variable
        storedData = {
            RESIN: {
                "midprice": None,
                "fairprice": None,
                "ema": None,
                "ema_factor": self.params[RESIN]["RESIN_FACTOR"],
            },
            KELP: {
                "midprice": None,
                "fairprice": 0,
                "ema": None,
                "ema_factor": self.params[KELP]["KELP_FACTOR"],
                "mm_volume_threshold": self.params[KELP]["KELP_MM_VOLUME_THRESHOLD"],
            },
            SQUIDINK: {
                "prevN_midprice": None,
                "midprice": None,
                "fairprice": None,
                "ema": None,
                "ema_factor": self.params[SQUIDINK]["SQUIDINK_FACTOR"],
                "obv": 0,
                "prev_close": None,
                "close": None,
                "u_ema": None,  # used to calculate rsi
                "d_ema": None,  # used to calculate rsi
                "rsi": 0,
                "entry_price": None,
            },
        }
        if state.traderData != "":
            storedData = jsonpickle.decode(state.traderData)

        result = {}

        # update midprice and exponential moving averages for each product
        for product in PRODUCTS:
            if product == SQUIDINK:
                mid_price = self.get_mid_price(product, state)  # the problem
                self.set_obv(product, state, storedData)

                storedData[product]["midprice"] = mid_price
                if storedData[product]["prevN_midprice"] is None:
                    storedData[product]["prevN_midprice"] = mid_price

                market_trades = state.market_trades.get(product, None)
                if market_trades:
                    if storedData[product]["close"] is not None:
                        storedData[product]["prev_close"] = storedData[product]["close"]
                    storedData[product]["close"] = market_trades[0].price

                close, prev_close = (
                    storedData[product]["close"],
                    storedData[product]["prev_close"],
                )
                if close is not None and prev_close is not None:
                    k = 1 / self.params[product]["RSI_PERIOD"]
                    u, d = 0, 0
                    if close > prev_close:
                        u, d = close - prev_close, 0
                    elif close < prev_close:
                        u, d = 0, prev_close - close

                    if storedData[product]["u_ema"] is None:
                        storedData[product]["u_ema"] = u
                    else:
                        prevEma = storedData[product]["u_ema"]
                        storedData[product]["u_ema"] = u * k + prevEma * (1 - k)
                    if storedData[product]["d_ema"] is None:
                        storedData[product]["d_ema"] = d
                    else:
                        prevEma = storedData[product]["d_ema"]
                        storedData[product]["d_ema"] = d * k + prevEma * (1 - k)

                u_ema = storedData[product]["u_ema"]
                d_ema = storedData[product]["d_ema"]
                if u_ema is not None and d_ema is not None:
                    if d_ema == 0:
                        storedData[product]["rsi"] = 100
                    else:
                        rs = u_ema / d_ema
                        storedData[product]["rsi"] = 100 - 100 / (1 + rs)
                if product in state.own_trades:
                    storedData[product]["entry_price"] = state.own_trades[product][
                        0
                    ].price

            # store data for all products
            # - orderbook midprice
            # - exponential moving average
            self.set_mid_price(product, state, storedData)
            self.set_ema(product, state, storedData)
            if product == KELP:
                self.set_fair_price_vsmm(product, state, storedData)
            else:
                self.set_fair_price_midprice(product, state, storedData)

        # get orders for each product
        result[RESIN] = self.resin_strat(state, storedData)
        result[KELP] = self.kelp_strat(state, storedData)
        result[SQUIDINK] = self.squidink_strat(state, storedData)

        # save data to traderData for next cycle
        traderData = jsonpickle.encode(storedData)

        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData
