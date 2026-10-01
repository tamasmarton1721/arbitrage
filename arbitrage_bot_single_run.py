"""
Triangulációs arbitrázs szkenner - EGYSZERI LEFUTÁSÚ verzió
===============================================================
Ugyanaz a logika, mint az arbitrage_bot_v2.py-ban, de NEM végtelen
ciklusban fut, hanem egyszer lefut és kilép -- ez kell ahhoz, hogy
GitHub Actions (vagy bármilyen cron-szerű ütemező) tudja futtatni.

Minden futás hozzáfűz egy sort (vagy sorokat) az
arbitrage_opportunities.csv fájlhoz, amit a workflow visszacommitol
a repóba -- így idővel összegyűlik egy elemezhető adatsor.
"""

import asyncio
import csv
import itertools
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import ccxt.async_support as ccxt_async

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("arbitrage")

DRY_RUN = True
EXCHANGE_ID = "kraken"
BASE_CURRENCY = "USDT"
DEFAULT_TAKER_FEE = 0.001
MIN_PROFIT_THRESHOLD = 0.002
TEST_AMOUNT = 50.0
ORDER_BOOK_DEPTH = 20
MAX_TRIANGLES = 40
LOG_CSV_PATH = "arbitrage_opportunities.csv"


@dataclass
class Opportunity:
    path: list
    net_multiplier: float
    expected_profit_pct: float
    max_tradable_base: float


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def ensure_csv_header():
    if not os.path.exists(LOG_CSV_PATH):
        with open(LOG_CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "path", "net_multiplier", "expected_profit_pct", "max_tradable_base", "above_threshold"]
            )


def log_to_csv(opp: Opportunity, above_threshold: bool):
    with open(LOG_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [now_iso(), "->".join(opp.path), f"{opp.net_multiplier:.6f}",
             f"{opp.expected_profit_pct:.4f}", f"{opp.max_tradable_base:.4f}", above_threshold]
        )


def build_triangles(markets: dict, base: str, max_triangles: int):
    active_symbols = {s for s, m in markets.items() if m.get("active", True)}
    direct = set()
    for s in active_symbols:
        b, q = s.split("/")
        if b == base:
            direct.add(q)
        elif q == base:
            direct.add(b)

    triangles = []
    for x, y in itertools.permutations(direct, 2):
        candidates = [f"{x}/{y}", f"{y}/{x}"]
        middle = next((c for c in candidates if c in active_symbols), None)
        if middle is None:
            continue
        leg1 = f"{base}/{x}" if f"{base}/{x}" in active_symbols else f"{x}/{base}"
        leg3 = f"{y}/{base}" if f"{y}/{base}" in active_symbols else f"{base}/{y}"
        if leg1 in active_symbols and leg3 in active_symbols:
            triangles.append((leg1, middle, leg3))
        if len(triangles) >= max_triangles:
            break
    return triangles


async def fetch_order_book_safe(exchange, symbol):
    try:
        return symbol, await exchange.fetch_order_book(symbol, limit=ORDER_BOOK_DEPTH)
    except Exception as e:
        log.debug("Nem sikerült lekérni %s order bookot: %s", symbol, e)
        return symbol, None


async def fetch_all_order_books(exchange, symbols):
    unique_symbols = list(set(symbols))
    results = await asyncio.gather(*[fetch_order_book_safe(exchange, s) for s in unique_symbols])
    return {symbol: ob for symbol, ob in results if ob is not None}


def walk_book(levels, amount_needed, is_buy):
    remaining = amount_needed
    total_cost = 0.0
    filled = 0.0
    for level in levels:
        price, size = level[0], level[1]
        take = min(remaining, size)
        total_cost += take * price
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled == 0:
        return None, 0.0
    return total_cost / filled, filled


def evaluate_path(path, order_books, base, taker_fee, test_amount):
    leg1, leg2, leg3 = path
    ob1, ob2, ob3 = order_books.get(leg1), order_books.get(leg2), order_books.get(leg3)
    if not all([ob1, ob2, ob3]):
        return None
    try:
        b1, q1 = leg1.split("/")
        if q1 == base:
            price, filled = walk_book(ob1["asks"], test_amount / ob1["asks"][0][0], True)
            amount_x = filled
        else:
            price, filled = walk_book(ob1["bids"], test_amount, False)
            amount_x = filled * price if price else 0
        if not amount_x:
            return None
        amount_x *= (1 - taker_fee)

        price2, filled2 = walk_book(ob2["bids"], amount_x, False)
        if price2 is None:
            return None
        amount_y = filled2 * price2 * (1 - taker_fee)

        b3, q3 = leg3.split("/")
        price3, filled3 = walk_book(ob3["bids"] if b3 != base else ob3["asks"], amount_y, False)
        if price3 is None:
            return None
        final_base = filled3 * price3 * (1 - taker_fee)

        net_multiplier = final_base / test_amount
        return Opportunity(
            path=list(path),
            net_multiplier=net_multiplier,
            expected_profit_pct=(net_multiplier - 1) * 100,
            max_tradable_base=test_amount,
        )
    except (IndexError, ZeroDivisionError, KeyError):
        return None


async def get_taker_fee(exchange):
    try:
        fees = await exchange.fetch_trading_fees()
        sample = next(iter(fees.values()), None)
        if sample and "taker" in sample:
            return sample["taker"]
    except Exception:
        pass
    return DEFAULT_TAKER_FEE


async def run_once():
    ensure_csv_header()
    exchange = getattr(ccxt_async, EXCHANGE_ID)({"enableRateLimit": True})
    try:
        markets = await exchange.load_markets()
        taker_fee = await get_taker_fee(exchange)
        triangles = build_triangles(markets, BASE_CURRENCY, MAX_TRIANGLES)
        log.info("Díj: %.4f%% | Talált hurkok: %d", taker_fee * 100, len(triangles))

        if not triangles:
            log.error("Nem található hurok '%s' bázisból.", BASE_CURRENCY)
            return

        all_symbols = set(s for tri in triangles for s in tri)
        start = time.time()
        order_books = await fetch_all_order_books(exchange, all_symbols)
        log.info("Order bookok lekérve %.2f mp alatt (%d szimbólum).", time.time() - start, len(order_books))

        found_any = False
        for path in triangles:
            opp = evaluate_path(path, order_books, BASE_CURRENCY, taker_fee, TEST_AMOUNT)
            if opp is None:
                continue
            above = opp.expected_profit_pct / 100 > MIN_PROFIT_THRESHOLD
            log_to_csv(opp, above_threshold=above)
            if above:
                found_any = True
                log.info("LEHETŐSÉG: %s | profit: %.3f%%", " -> ".join(opp.path), opp.expected_profit_pct)

        if not found_any:
            log.info("Ebben a futásban nem volt küszöb feletti lehetőség.")

    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(run_once())
