"""
多数据源OHLCV获取器 (15分钟)
- 支持 Yahoo Finance / Huobi REST / 多交易所 CCXT
- 自动在可用数据源之间切换
- 支持分批抓取获得更大数据量（默认 5000 根 K 线）
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np
import yfinance as yf

# 新增依赖
try:
    import ccxt  # type: ignore
except ImportError:
    ccxt = None  # 在缺失时降级使用其它数据源

import warnings
warnings.filterwarnings('ignore')


class MultiSourceDataFetcher:
    """统一的多数据源 15m OHLCV 获取器"""

    # 默认尝试的 ccxt 交易所列表（可根据需要调整顺序）
    DEFAULT_EXCHANGES = [
        "binance",  # 流动性最好，优先尝试
        "kucoin",
        "okx",
    ]

    def __init__(self, symbol: str = "DOGEUSDT") -> None:
        self.symbol = symbol.upper()

    # --------------------------- 对外主入口 ---------------------------
    def fetch_large_dataset(self, target_points: int = 5000, maximize: bool = False):
        """尝试多数据源，尽量返回满足目标的最大数据集。

        参数
        ------
        target_points : int
            期望的最少 K 线数量；可设置成非常大，如 100_000 来获取尽可能多的数据。
        maximize : bool
            设置为 True 时，不论是否达到 target_points，都继续在其它数据源中寻找更长的数据，
            最终返回获得记录数最多的那一个数据集。
        """

        print(f"📊 正在获取 {self.symbol} 15m 数据, 目标 {target_points} 根 K 线 (maximize={maximize}) ...")

        best_source: str | None = None
        best_df: pd.DataFrame | None = None

        def _update_best(src: str, data: pd.DataFrame):
            nonlocal best_source, best_df
            if data is None or data.empty:
                return
            if (best_df is None) or (len(data) > len(best_df)):
                best_source, best_df = src, data

        # ---------- 1. Yahoo Finance ----------
        df = self._fetch_yahoo_finance(target_points)
        if df is not None:
            print(f"✅ Yahoo Finance 获取成功: {len(df)} 条")
            if len(df) >= target_points and not maximize:
                return "yahoo_finance", df
            _update_best("yahoo_finance", df)

        # ---------- 2. Huobi ----------
        df = self._fetch_huobi_15m(target_points)
        if df is not None:
            print(f"✅ Huobi API 获取成功: {len(df)} 条")
            if len(df) >= target_points and not maximize:
                return "huobi_api", df
            _update_best("huobi_api", df)

        # ---------- 3. ccxt 交易所 ----------
        if ccxt is not None:
            for exch in self.DEFAULT_EXCHANGES:
                df = self._fetch_ccxt_exchange(exch, target_points)
                if df is not None:
                    print(f"✅ {exch} (ccxt) 获取成功: {len(df)} 条")
                    if len(df) >= target_points and not maximize:
                        return f"ccxt_{exch}", df
                    _update_best(f"ccxt_{exch}", df)
        else:
            print("⚠️  未安装 ccxt，跳过交易所数据源 …")

        if best_df is not None:
            print(f"📈 已返回当前获取到的最大数据集: {len(best_df)} 条 (source={best_source})")
            return best_source, best_df

        print("❌ 所有数据源均未能满足要求")
        return None, None

    # --------------------------- 工具方法 ---------------------------
    @staticmethod
    def _is_valid(df: pd.DataFrame | None, target: int) -> bool:
        """验证 DataFrame 是否有效且长度可接受"""
        return (df is not None) and (len(df) >= min(300, target * 0.6))

    # --------------------------- 各数据源实现 ---------------------------
    def _fetch_yahoo_finance(self, target_points: int):
        """通过 yfinance 获取最多一个月*96*30 ≈ 2880 条 (15m)"""
        try:
            yahoo_symbol = self._convert_to_yahoo_symbol()
            end_dt = datetime.utcnow()
            # Yahoo Finance 15m 数据只能获取最近 60 天；5000 根 ≈ 52 天
            # 按目标数量动态计算所需区间，并额外预留一点冗余
            span_minutes = target_points * 15 + 60  # +60 分钟冗余
            # 最多不超过 60 天 (60*24*60 = 86400 分)
            max_minutes = 60 * 24 * 60
            span_minutes = min(span_minutes, max_minutes - 15)  # 留 15 分钟保险
            start_dt = end_dt - timedelta(minutes=span_minutes)

            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_dt, end=end_dt, interval="15m", auto_adjust=True, prepost=False)
            if df.empty:
                return None

            df = df.rename(columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })
            df.index = df.index.tz_localize(None)
            return df[["open", "high", "low", "close", "volume"]].dropna().tail(target_points)
        except Exception as e:
            print(f"Yahoo Finance 失败: {e}")
            return None

    def _fetch_huobi_15m(self, target_points: int):
        """Huobi 公共 API: 每次 size≤2000，故无需分页即可满足中等数据量"""
        try:
            symbol = self.symbol.replace("USDT", "usdt").lower()
            url = "https://api.huobi.pro/market/history/kline"
            params = {"symbol": symbol, "period": "15min", "size": min(2000, target_points)}
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "ok" or not data.get("data"):
                return None
            df = pd.DataFrame(data["data"])
            df.rename(columns={"id": "timestamp", "vol": "volume"}, inplace=True)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("timestamp", inplace=True)
            return df[["open", "high", "low", "close", "volume"]].sort_index().tail(target_points)
        except Exception as e:
            print(f"Huobi API 失败: {e}")
            return None

    def _fetch_ccxt_exchange(self, exchange_name: str, target_points: int):
        """利用 ccxt 分批拉取大数据量，自动限速"""
        try:
            if ccxt is None:
                return None

            if not hasattr(ccxt, exchange_name):
                print(f"⚠️  ccxt 未支持 {exchange_name}")
                return None
            # 实例化交易所
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({"enableRateLimit": True})
            if not exchange.has.get("fetchOHLCV", False):
                return None

            market_symbol = self.symbol.replace("USDT", "/USDT")
            timeframe = "15m"
            limit_per_call = 1000  # 大部分交易所限制 1000

            end_ts = exchange.milliseconds()  # 当前毫秒时间戳
            all_ohlcv: list[list] = []

            while len(all_ohlcv) < target_points:
                since = end_ts - limit_per_call * 15 * 60 * 1000  # 向前推 limit*15m
                ohlcv = exchange.fetch_ohlcv(market_symbol, timeframe=timeframe, since=since, limit=limit_per_call)
                if not ohlcv:
                    break
                # 因部分交易所返回从旧→新，需要保证顺序
                ohlcv_sorted = sorted(ohlcv, key=lambda x: x[0])
                all_ohlcv = ohlcv_sorted + all_ohlcv  # 前插，保持升序
                end_ts = ohlcv_sorted[0][0] - 1  # 下一轮结束时间设为当前最老的再往前 1ms
                # Sleep 尊重 rateLimit
                time.sleep(exchange.rateLimit / 1000 + 0.05)
                if len(ohlcv) < limit_per_call:  # 已到最早记录
                    break

            if not all_ohlcv:
                return None

            columns = ["timestamp", "open", "high", "low", "close", "volume"]
            df = pd.DataFrame(all_ohlcv[-target_points:], columns=columns)  # 只保留最新 target_points 条
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df[["open", "high", "low", "close", "volume"]].sort_index()
        except Exception as e:
            print(f"{exchange_name} 失败: {e}")
            return None

    # --------------------------- 辅助 ---------------------------
    def _convert_to_yahoo_symbol(self):
        if self.symbol.endswith("USDT"):
            return self.symbol.replace("USDT", "-USD")
        return self.symbol


if __name__ == "__main__":
    fetcher = MultiSourceDataFetcher("DOGEUSDT")
    src, df = fetcher.fetch_large_dataset(target_points=5000)
    if df is not None:
        print(src, df.shape, df.index[0], "->", df.index[-1])
    else:
        print("获取数据失败") 