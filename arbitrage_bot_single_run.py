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
import json
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
MIN_PROFIT_THRESHOLD = -1
TEST_AMOUNT = 50.0
ORDER_BOOK_DEPTH = 20
MAX_TRIANGLES = 40
LOG_CSV_PATH = "arbitrage_opportunities.csv"
BALANCE_FILE = "balance.json"
TRADES_CSV_PATH = "virtual_trades.csv"
STARTING_BALANCE = 1000.0  # ennyivel indul a virtuális egyenleg, ha még nincs balance.json


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


def load_balance():
    """Betölti a virtuális egyenleget a balance.json-ból; ha nem létezik,
    létrehozza STARTING_BALANCE értékkel."""
    if os.path.exists(BALANCE_FILE):
        with open(BALANCE_FILE, "r") as f:
            data = json.load(f)
    else:
        data = {
            "balance": STARTING_BALANCE,
            "currency": BASE_CURRENCY,
            "started_at": now_iso(),
            "last_updated": now_iso(),
            "total_trades": 0,
        }
    return data


def save_balance(data: dict):
    data["last_updated"] = now_iso()
    with open(BALANCE_FILE, "w") as f:
        json.dump(data, f, indent=2)


def ensure_trades_csv_header():
    if not os.path.exists(TRADES_CSV_PATH):
        with open(TRADES_CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["timestamp", "path", "trade_amount", "net_multiplier",
                 "profit_pct", "profit_amount", "balance_before", "balance_after"]
            )


def log_trade(opp: Opportunity, trade_amount: float, balance_before: float, balance_after: float):
    with open(TRADES_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [now_iso(), "->".join(opp.path), f"{trade_amount:.4f}", f"{opp.net_multiplier:.6f}",
             f"{opp.expected_profit_pct:.4f}", f"{balance_after - balance_before:.4f}",
             f"{balance_before:.4f}", f"{balance_after:.4f}"]
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


def sell_base_get_quote(bids, amount_base):
    """Eladjuk `amount_base` mennyiségű bázis devizát a bid oldalon; visszaadja
    (megkapott jegyzett deviza mennyisége, ténylegesen eladott bázis mennyiség)."""
    remaining = amount_base
    quote_total = 0.0
    filled = 0.0
    for level in bids:
        price, size = level[0], level[1]
        take = min(remaining, size)
        quote_total += take * price
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    return quote_total, filled


def buy_base_with_quote(asks, amount_quote):
    """Elköltünk `amount_quote` mennyiségű jegyzett devizát az ask oldalon, hogy
    bázis devizát vegyünk; visszaadja (megkapott bázis mennyiség, elköltött jegyzett
    deviza mennyiség)."""
    remaining = amount_quote
    base_total = 0.0
    quote_spent = 0.0
    for level in asks:
        price, size = level[0], level[1]
        cost_level = price * size
        take_quote = min(remaining, cost_level)
        take_base = take_quote / price
        base_total += take_base
        quote_spent += take_quote
        remaining -= take_quote
        if remaining <= 0:
            break
    return base_total, quote_spent


def convert(symbol, order_book, have_currency, amount_have, fee):
    """Átvált `amount_have` mennyiségű `have_currency` devizát a `symbol` pár
    (pl. 'BTC/USDT') másik devizájára, a könyv megfelelő oldalát használva.
    Visszaadja (új deviza, új mennyiség díj levonása után), vagy None-t, ha
    a pár nem kapcsolódik a birtokolt devizához, vagy üres a könyv."""
    base, quote = symbol.split("/")
    if have_currency == base:
        quote_received, base_filled = sell_base_get_quote(order_book["bids"], amount_have)
        if base_filled <= 0:
            return None
        return quote, quote_received * (1 - fee)
    elif have_currency == quote:
        base_received, quote_spent = buy_base_with_quote(order_book["asks"], amount_have)
        if quote_spent <= 0:
            return None
        return base, base_received * (1 - fee)
    else:
        return None


def evaluate_path(path, order_books, base, taker_fee, test_amount):
    leg1, leg2, leg3 = path
    ob1, ob2, ob3 = order_books.get(leg1), order_books.get(leg2), order_books.get(leg3)
    if not all([ob1, ob2, ob3]):
        return None
    try:
        currency = base
        amount = test_amount
        for symbol, ob in ((leg1, ob1), (leg2, ob2), (leg3, ob3)):
            result = convert(symbol, ob, currency, amount, taker_fee)
            if result is None:
                return None
            currency, amount = result

        if currency != base:
            # A hurok nem zárult vissza a bázis devizára -> hibás/nem valódi triangle
            return None

        net_multiplier = amount / test_amount
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
    ensure_trades_csv_header()
    balance_data = load_balance()
    balance = balance_data["balance"]

    exchange = getattr(ccxt_async, EXCHANGE_ID)({"enableRateLimit": True})
    try:
        markets = await exchange.load_markets()
        taker_fee = await get_taker_fee(exchange)
        triangles = build_triangles(markets, BASE_CURRENCY, MAX_TRIANGLES)
        log.info("Egyenleg: %.4f %s | Díj: %.4f%% | Talált hurkok: %d",
                  balance, BASE_CURRENCY, taker_fee * 100, len(triangles))

        if not triangles:
            log.error("Nem található hurok '%s' bázisból.", BASE_CURRENCY)
            return

        all_symbols = set(s for tri in triangles for s in tri)
        start = time.time()
        order_books = await fetch_all_order_books(exchange, all_symbols)
        log.info("Order bookok lekérve %.2f mp alatt (%d szimbólum).", time.time() - start, len(order_books))

        best_opp = None
        for path in triangles:
            opp = evaluate_path(path, order_books, BASE_CURRENCY, taker_fee, TEST_AMOUNT)
            if opp is None:
                continue
            above = opp.expected_profit_pct / 100 > MIN_PROFIT_THRESHOLD
            log_to_csv(opp, above_threshold=above)
            if above and (best_opp is None or opp.expected_profit_pct > best_opp.expected_profit_pct):
                best_opp = opp

        if best_opp:
            trade_amount = min(balance, TEST_AMOUNT)
            if trade_amount <= 0:
                log.warning("Nincs elkölthető egyenleg, kihagyott lehetőség: %s", " -> ".join(best_opp.path))
            else:
                profit = trade_amount * (best_opp.net_multiplier - 1)
                balance_before = balance
                balance = balance + profit
                log.info(
                    "VIRTUÁLIS KÖTÉS: %s | tétel: %.4f | profit: %.3f%% (%.4f) | egyenleg: %.4f -> %.4f",
                    " -> ".join(best_opp.path), trade_amount, best_opp.expected_profit_pct,
                    profit, balance_before, balance,
                )
                log_trade(best_opp, trade_amount, balance_before, balance)
                balance_data["total_trades"] = balance_data.get("total_trades", 0) + 1
        else:
            log.info("Ebben a futásban nem volt küszöb feletti lehetőség, nincs kötés.")

        balance_data["balance"] = balance
        save_balance(balance_data)

    finally:
        await exchange.close()


if __name__ == "__main__":
    asyncio.run(run_once())
