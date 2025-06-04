"""
api_wrapper.py – MEXC Spot REST & WS 封装（支持 clientOrderId、Cancel‑All、Market Sell）
"""
import time, hmac, hashlib, urllib.parse, json, threading, requests, re
from decimal import Decimal, ROUND_DOWN, getcontext
getcontext().prec = 28

BASE_URL = "https://api.mexc.com"
WS_USER  = "wss://wbs-api.mexc.com/ws"

ACCESS_KEY = "mx0vgl13KGD6ZtZh3e"          # 测试 key
SECRET_KEY = "a90ff106e2034991860eac7ce5d32952"

class MexcRest:
    def __init__(self, symbol="ALEOUSDT", key=ACCESS_KEY, secret=SECRET_KEY):
        self.symbol = symbol.upper()
        self.key, self.secret = key, secret.encode()
        self.tick_size, self.step_size = self._get_precision()

    # ----------- 基础 HTTP ----------
    def _request(self, m:str, path:str, *, params=None, signed=False):
        params = params or {}
        if signed:
            params["timestamp"] = int(time.time()*1000)
            params["signature"] = self._sign(params)
        if   m == "GET":    r = requests.get   (BASE_URL+path, params=params, headers={"X-MEXC-APIKEY": self.key}, timeout=10)
        elif m == "DELETE": r = requests.delete(BASE_URL+path, params=params, headers={"X-MEXC-APIKEY": self.key}, timeout=10)
        else:               r = requests.post  (BASE_URL+path, params=params, headers={"X-MEXC-APIKEY": self.key}, timeout=10)
        if not r.ok: print("[API]", r.status_code, r.text[:120])
        r.raise_for_status(); return r.json()

    def _sign(self, p:dict):
        qs = urllib.parse.urlencode(p, quote_via=urllib.parse.quote)
        return hmac.new(self.secret, qs.encode(), hashlib.sha256).hexdigest()

    # ----------- 精度 ----------
    def _get_precision(self):
        info = self._request("GET", "/api/v3/exchangeInfo")
        tgt  = next(s for s in info["symbols"]
                    if re.sub(r"[-_]", "", s["symbol"]) == self.symbol)
        tick = next((Decimal(f["tickSize"]) for f in tgt["filters"]
                     if Decimal(f.get("tickSize", "0")) > 0), None) \
               or Decimal(1).scaleb(-int(tgt["quoteAssetPrecision"]))
        step = next((Decimal(f["stepSize"]) for f in tgt["filters"]
                     if Decimal(f.get("stepSize", "0")) > 0), None) \
               or Decimal(1).scaleb(-int(tgt["baseAssetPrecision"]))
        print(f"[Init] {self.symbol} tick={tick} step={step}")
        return tick, step

    # ----------- 下单 ----------
    def quant_p(self, p): return (Decimal(str(p))/self.tick_size)\
           .to_integral_value(ROUND_DOWN)*self.tick_size
    def quant_q(self, q): return (Decimal(str(q))/self.step_size)\
           .to_integral_value(ROUND_DOWN)*self.step_size

    def place_limit(self, side:str, price:float, qty:float, idx:int):
        cli = f"G{idx}_{side[0]}"             # clientOrderId
        return self._request("POST", "/api/v3/order", signed=True, params={
            "symbol": self.symbol, "side": side, "type":"LIMIT", "timeInForce":"GTC",
            "price": format(self.quant_p(price), "f"),
            "quantity": format(self.quant_q(qty), "f"),
            "newClientOrderId": cli
        })

    def place_market_sell(self, qty:Decimal):
        return self._request("POST", "/api/v3/order", signed=True, params={
            "symbol": self.symbol, "side":"SELL", "type":"MARKET",
            "quantity": format(self.quant_q(qty), "f")
        })

    def cancel_all(self):
        try:
            self._request("DELETE", "/api/v3/openOrders",
                          signed=True, params={"symbol": self.symbol})
        except requests.HTTPError as e:
            print("[CANCEL‑ALL]", e)

    def open_orders(self):
        return self._request("GET", "/api/v3/openOrders",
                             params={"symbol": self.symbol}, signed=True)

    # ----------- 辅助 ----------
    def get_price(self):
        return Decimal(self._request("GET", "/api/v3/ticker/price",
                           params={"symbol": self.symbol})["price"])
    def get_balances(self):
        acc = self._request("GET", "/api/v3/account", signed=True)
        return {b["asset"]: Decimal(b["free"])
                for b in acc["balances"] if Decimal(b["free"]) > 0}
    def balance_of(self, asset): return self.get_balances().get(asset, Decimal(0))
    def start_user_stream(self):
        return self._request("POST", "/api/v3/userDataStream", signed=True)["listenKey"]

# ----------- WebSocket -----------
import websocket
class MexcUserWebSocket:
    def __init__(self, rest: MexcRest, on_deal):
        self.rest, self.on_deal = rest, on_deal
    def _msg(self, _, msg):
        d = json.loads(msg)
        if d.get("channel") == "spot@private.deals.v3.api.pb":
            for item in d["data"]:
                self.on_deal(item)
    def run(self):
        key = self.rest.start_user_stream()
        ws  = websocket.WebSocketApp(f"{WS_USER}?listenKey={key}",
                                     on_message=self._msg)
        threading.Thread(target=ws.run_forever, daemon=True).start()
