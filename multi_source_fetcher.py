"""
优化版多数据源获取器 - 只保留真正有效的数据源
获取DOGE等币种的15分钟交易数据，支持一年历史数据
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np
import zipfile
import io
import os
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

class EnhancedMultiSourceDataFetcher:
    """优化版多数据源获取器 - 只保留有效数据源"""
    
    def __init__(self, symbol="DOGEUSDT"):
        self.symbol = symbol
        # 只保留真正有效的数据源
        self.data_sources = [
            "yahoo_finance",       # ✅ 已验证：真实DOGE数据，1年历史
            "kraken_ohlc",         # ✅ 稳定：官方API，良好数据
            "huobi_api",           # ✅ 可靠：15分钟数据，高密度
        ]
        
    def fetch_all_sources(self, target_points=2000):
        """尝试所有有效数据源，获取最大数据量"""
        
        print("🚀 启动优化版多数据源获取")
        print("=" * 60)
        print(f"🎯 交易对: {self.symbol}")
        print(f"📊 目标数据量: {target_points} 条")
        print(f"🔄 有效数据源: {len(self.data_sources)} 个")
        
        all_data = []
        successful_sources = []
        
        # 1. 尝试 Yahoo Finance (混合策略)
        print(f"\n📊 1. Yahoo Finance (混合策略)")
        print("-" * 50)
        try:
            data = self._fetch_yahoo_finance()
            if data is not None and len(data) > 100:
                time_span = (data.index[-1] - data.index[0]).total_seconds() / 86400
                print(f"✅ 成功: {len(data)}条数据, {time_span:.1f}天跨度")
                all_data.append(("yahoo_finance", data))
                successful_sources.append("Yahoo Finance")
            else:
                print(f"❌ 失败: 数据不足或无数据")
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            
        # 2. 尝试 Kraken OHLC (稳定API)
        print(f"\n📊 2. Kraken OHLC API (稳定数据源)")
        print("-" * 50)
        try:
            data = self._fetch_kraken_ohlc()
            if data is not None and len(data) > 100:
                time_span = (data.index[-1] - data.index[0]).total_seconds() / 86400
                print(f"✅ 成功: {len(data)}条数据, {time_span:.1f}天跨度")
                all_data.append(("kraken_ohlc", data))
                successful_sources.append("Kraken OHLC")
            else:
                print(f"❌ 失败: 数据不足或无数据")
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            
        # 3. 尝试 Huobi API (高密度15分钟数据)
        print(f"\n📊 3. Huobi API (高密度15分钟数据)")
        print("-" * 50)
        try:
            data = self._fetch_huobi_15min()
            if data is not None and len(data) > 100:
                time_span = (data.index[-1] - data.index[0]).total_seconds() / 86400
                print(f"✅ 成功: {len(data)}条数据, {time_span:.1f}天跨度")
                all_data.append(("huobi_api", data))
                successful_sources.append("Huobi API")
            else:
                print(f"❌ 失败: 数据不足或无数据")
        except Exception as e:
            print(f"❌ 错误: {str(e)}")
            
        # 合并所有成功的数据源
        if all_data:
            print(f"\n🎉 数据获取总结")
            print("=" * 60)
            print(f"✅ 成功数据源: {len(successful_sources)} 个")
            print(f"📋 成功列表: {', '.join(successful_sources)}")
            
            # 选择数据量最大的源
            best_source, best_data = max(all_data, key=lambda x: len(x[1]))
            
            print(f"\n🏆 最佳数据源: {best_source}")
            print(f"📊 最佳数据量: {len(best_data)} 条")
            
            # 尝试合并数据（修复时区问题）
            if len(all_data) > 1:
                print(f"\n🔄 尝试合并多个数据源...")
                combined_data = self._merge_multiple_sources_fixed(all_data)
                if combined_data is not None and len(combined_data) > len(best_data):
                    print(f"✅ 数据合并成功: {len(combined_data)} 条 (增加了 {len(combined_data)-len(best_data)} 条)")
                    best_source = "merged_sources"
                    best_data = combined_data
                else:
                    print(f"⚠️ 数据合并后无明显增加，使用单一最佳数据源")
            
            time_span_days = (best_data.index[-1] - best_data.index[0]).total_seconds()/86400
            time_span_years = time_span_days / 365
            
            print(f"⏰ 时间跨度: {time_span_days:.1f}天 ({time_span_years:.2f}年)")
            print(f"📅 时间范围: {best_data.index[0]} 到 {best_data.index[-1]}")
            print(f"💰 价格范围: {best_data['close'].min():.6f} - {best_data['close'].max():.6f}")
            
            return best_source, best_data
        else:
            print("❌ 所有数据源都失败了")
            return None, None
    
    def _fetch_yahoo_finance(self):
        """获取 Yahoo Finance 的DOGE数据（混合策略：15分钟+1小时）"""
        try:
            print("   🔄 尝试 Yahoo Finance (混合策略)...")
            
            # 使用已验证的DOGE符号
            symbol = "DOGE-USD"
            ticker = yf.Ticker(symbol)
            
            print(f"   📥 获取符号: {symbol}")
            
            # 策略1: 获取近期60天的15分钟高精度数据
            print("   📊 策略1: 获取近期60天15分钟数据...")
            try:
                data_15m = ticker.history(period="60d", interval="15m", auto_adjust=False, prepost=False)
                if len(data_15m) > 100:
                    days_15m = (data_15m.index[-1] - data_15m.index[0]).total_seconds() / 86400
                    print(f"   ✅ 15分钟数据: {len(data_15m)}条, {days_15m:.1f}天")
                    
                    # 转换格式
                    df_15m = pd.DataFrame()
                    df_15m['open'] = data_15m['Open']
                    df_15m['high'] = data_15m['High'] 
                    df_15m['low'] = data_15m['Low']
                    df_15m['close'] = data_15m['Close']
                    df_15m['volume'] = data_15m['Volume']
                    
                    # 统一时区处理
                    if data_15m.index.tz is not None:
                        df_15m.index = data_15m.index.tz_convert('UTC').tz_localize(None)
                    else:
                        df_15m.index = pd.to_datetime(data_15m.index)
                    
                    # 策略2: 获取1年的1小时数据作为补充
                    print("   📊 策略2: 获取1年1小时数据作为历史补充...")
                    try:
                        data_1h = ticker.history(period="1y", interval="1h", auto_adjust=False, prepost=False)
                        if len(data_1h) > 100:
                            days_1h = (data_1h.index[-1] - data_1h.index[0]).total_seconds() / 86400
                            print(f"   ✅ 1小时数据: {len(data_1h)}条, {days_1h:.1f}天")
                            
                            # 转换格式
                            df_1h = pd.DataFrame()
                            df_1h['open'] = data_1h['Open']
                            df_1h['high'] = data_1h['High'] 
                            df_1h['low'] = data_1h['Low']
                            df_1h['close'] = data_1h['Close']
                            df_1h['volume'] = data_1h['Volume']
                            
                            # 统一时区处理
                            if data_1h.index.tz is not None:
                                df_1h.index = data_1h.index.tz_convert('UTC').tz_localize(None)
                            else:
                                df_1h.index = pd.to_datetime(data_1h.index)
                            
                            # 策略3: 智能合并数据
                            print("   🔄 智能合并高精度和长期数据...")
                            
                            # 找到15分钟数据的开始时间
                            cutoff_time = df_15m.index[0]
                            
                            # 只保留15分钟数据开始时间之前的1小时数据
                            df_1h_historical = df_1h[df_1h.index < cutoff_time]
                            
                            if len(df_1h_historical) > 0:
                                # 合并数据：历史1小时数据 + 近期15分钟数据
                                combined_df = pd.concat([df_1h_historical, df_15m]).sort_index()
                                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                                
                                total_days = (combined_df.index[-1] - combined_df.index[0]).total_seconds() / 86400
                                historical_days = (df_1h_historical.index[-1] - df_1h_historical.index[0]).total_seconds() / 86400
                                
                                print(f"   🎉 混合数据合并成功!")
                                print(f"      📈 总数据量: {len(combined_df)}条")
                                print(f"      ⏰ 总时间跨度: {total_days:.1f}天 ({total_days/365:.2f}年)")
                                print(f"      📊 历史部分: {len(df_1h_historical)}条1小时数据 ({historical_days:.1f}天)")
                                print(f"      🔍 近期部分: {len(df_15m)}条15分钟数据 ({days_15m:.1f}天)")
                                
                                return combined_df
                            else:
                                print("   ⚠️ 无历史数据可合并，返回15分钟数据")
                                return df_15m
                        else:
                            print("   ⚠️ 1小时数据获取失败，返回15分钟数据")
                            return df_15m
                    except Exception as e:
                        print(f"   ⚠️ 1小时数据获取失败: {str(e)}，返回15分钟数据")
                        return df_15m
                else:
                    print("   ❌ 15分钟数据获取失败")
            except Exception as e:
                print(f"   ❌ 15分钟数据获取失败: {str(e)}")
            
            # 备用策略: 如果15分钟数据失败，尝试获取1小时数据
            print("   📊 备用策略: 尝试1小时数据...")
            try:
                data_1h = ticker.history(period="1y", interval="1h", auto_adjust=False, prepost=False)
                if len(data_1h) > 100:
                    days = (data_1h.index[-1] - data_1h.index[0]).total_seconds() / 86400
                    print(f"   ✅ 备用1小时数据: {len(data_1h)}条, {days:.1f}天")
                    
                    # 转换格式
                    df = pd.DataFrame()
                    df['open'] = data_1h['Open']
                    df['high'] = data_1h['High'] 
                    df['low'] = data_1h['Low']
                    df['close'] = data_1h['Close']
                    df['volume'] = data_1h['Volume']
                    
                    # 统一时区处理
                    if data_1h.index.tz is not None:
                        df.index = data_1h.index.tz_convert('UTC').tz_localize(None)
                    else:
                        df.index = pd.to_datetime(data_1h.index)
                    
                    return df
            except Exception as e:
                print(f"   ❌ 备用1小时数据失败: {str(e)}")
            
            print(f"   ❌ Yahoo Finance 所有策略都失败")
            return None
            
        except Exception as e:
            print(f"   ❌ Yahoo Finance 整体失败: {str(e)}")
            return None
    
    def _fetch_kraken_ohlc(self):
        """获取 Kraken OHLC 数据"""
        try:
            print("   🔄 尝试 Kraken API...")
            
            # 尝试多个Kraken的DOGE交易对
            pairs = ["DOGEUSD", "XDGUSD"]
            
            for pair in pairs:
                try:
                    print(f"   📡 尝试 Kraken {pair}...")
                    
                    url = "https://api.kraken.com/0/public/OHLC"
                    params = {
                        "pair": pair,
                        "interval": 15,  # 15分钟间隔
                        "since": int((datetime.now() - timedelta(days=30)).timestamp())
                    }
                    
                    response = requests.get(url, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data.get("error") == [] and data.get("result"):
                            # Kraken 返回的key可能不同
                            result_key = None
                            for key in data["result"].keys():
                                if key != "last":
                                    result_key = key
                                    break
                            
                            if result_key and len(data["result"][result_key]) > 10:
                                df = self._convert_kraken_format(data["result"][result_key])
                                if df is not None:
                                    print(f"   ✅ Kraken {pair} 成功: {len(df)}条数据")
                                    return df
                        else:
                            print(f"   ⚠️ Kraken {pair} API错误: {data.get('error', 'Unknown')}")
                    else:
                        print(f"   ⚠️ Kraken {pair} HTTP错误: {response.status_code}")
                    
                except Exception as e:
                    print(f"   ⚠️ Kraken {pair} 失败: {str(e)}")
                    continue
            
            print(f"   ❌ Kraken 所有交易对都失败")
            return None
            
        except Exception as e:
            print(f"   ❌ Kraken 整体失败: {str(e)}")
            return None

    def _convert_kraken_format(self, data):
        """转换 Kraken OHLC 数据格式"""
        try:
            # Kraken OHLC格式: [timestamp, open, high, low, close, vwap, volume, count]
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
            
            # 转换数据类型
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            # 统一时区处理
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']].sort_index()
            
        except Exception as e:
            print(f"   ❌ Kraken 格式转换失败: {str(e)}")
            return None

    def _fetch_huobi_15min(self):
        """获取Huobi 15分钟数据"""
        try:
            symbol = self.symbol.replace("USDT", "usdt").lower()
            
            print("   🔄 获取Huobi数据(最优15分钟)...")
            
            # 直接获取15分钟高质量数据
            url = "https://api.huobi.pro/market/history/kline"
            params = {
                "symbol": symbol,
                "period": "15min",
                "size": 2000  # 获取最大可能的数据量
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            result = response.json()
            
            if result.get("status") == "ok" and result.get("data"):
                data = result["data"]
                
                if len(data) > 100:
                    df = self._convert_huobi_data(data)
                    
                    # 显示数据信息
                    min_ts = min(item['id'] for item in data)
                    max_ts = max(item['id'] for item in data)
                    min_time = datetime.fromtimestamp(min_ts)
                    max_time = datetime.fromtimestamp(max_ts)
                    time_span = (max_ts - min_ts) / (24 * 3600)  # 天数
                    
                    print(f"   ✅ Huobi 15分钟数据成功: {len(df)}条")
                    print(f"      📅 时间范围: {min_time} 到 {max_time}")
                    print(f"      ⏰ 时间跨度: {time_span:.1f}天")
                    print(f"      💰 价格范围: {df['close'].min():.6f} - {df['close'].max():.6f} USDT")
                    print(f"      📊 数据密度: {len(df)/time_span:.1f} 条/天")
                    
                    return df
                else:
                    print(f"   ❌ Huobi数据量不足: {len(data)}条")
            else:
                print(f"   ❌ Huobi API返回错误: {result.get('err-msg', '未知错误')}")
                
        except Exception as e:
            print(f"   ❌ Huobi数据获取失败: {str(e)}")
        
        return None
    
    def _convert_huobi_data(self, data):
        """转换Huobi数据格式"""
        df = pd.DataFrame(data)
        
        # Huobi数据格式: {id, open, close, low, high, amount, vol, count}
        df = df.rename(columns={
            'id': 'timestamp',
            'vol': 'volume'
        })
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col])
        
        # 统一时区处理 - 去除时区信息
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        
        return df[['open', 'high', 'low', 'close', 'volume']].sort_index()
    
    def _merge_multiple_sources_fixed(self, all_data):
        """合并多个数据源的数据（修复时区问题）"""
        try:
            print("   🔄 合并数据源...")
            
            # 按数据量排序，最大的在前
            sorted_data = sorted(all_data, key=lambda x: len(x[1]), reverse=True)
            
            # 从最大的数据源开始
            combined_df = sorted_data[0][1].copy()
            base_source = sorted_data[0][0]
            
            # 确保基础数据没有时区信息
            if combined_df.index.tz is not None:
                combined_df.index = combined_df.index.tz_localize(None)
            
            print(f"   📊 基础数据源: {base_source} ({len(combined_df)}条)")
            
            # 合并其他数据源
            for source_name, df in sorted_data[1:]:
                print(f"   🔄 合并 {source_name} ({len(df)}条)...")
                
                # 确保要合并的数据也没有时区信息
                df_to_merge = df.copy()
                if df_to_merge.index.tz is not None:
                    df_to_merge.index = df_to_merge.index.tz_localize(None)
                
                # 去重合并：只添加时间戳不重复的数据
                before_merge = len(combined_df)
                
                # 找到不重复的时间戳
                new_timestamps = df_to_merge.index.difference(combined_df.index)
                
                if len(new_timestamps) > 0:
                    new_data = df_to_merge.loc[new_timestamps]
                    combined_df = pd.concat([combined_df, new_data]).sort_index()
                    
                    print(f"   ✅ 添加了 {len(new_data)} 条新数据")
                else:
                    print(f"   ⚠️ 无新数据可添加")
            
            print(f"   🎉 合并完成: 总计 {len(combined_df)} 条数据")
            
            return combined_df
            
        except Exception as e:
            print(f"   ❌ 数据合并失败: {str(e)}")
            return None

def test_enhanced_fetcher():
    """测试优化版数据获取器"""
    print("🧪 测试优化版多数据源获取器")
    print("=" * 60)
    
    fetcher = EnhancedMultiSourceDataFetcher("DOGEUSDT")
    source, data = fetcher.fetch_all_sources(target_points=2000)
    
    if data is not None:
        print(f"\n🎉 多数据源获取成功！")
        print("=" * 60)
        
        time_span_days = (data.index[-1] - data.index[0]).total_seconds()/86400
        time_span_years = time_span_days / 365
        
        print(f"📊 最终数据获取结果:")
        print(f"   🏆 最佳数据源: {source}")
        print(f"   📈 数据量: {len(data):,}条")
        print(f"   ⏰ 时间跨度: {time_span_days:.1f}天 ({time_span_years:.2f}年)")
        print(f"   📅 时间范围: {data.index[0]} 到 {data.index[-1]}")
        print(f"   💰 价格范围: {data['close'].min():.6f} - {data['close'].max():.6f}")
        print(f"   📊 平均日密度: {len(data)/time_span_days:.1f} 条/天")
        
        # 数据质量评估
        if len(data) >= 4000:
            print(f"   🏆 数据质量: 卓越 (4,000+条数据)")
        elif len(data) >= 2000:
            print(f"   🥇 数据质量: 优秀 (2,000+条数据)")
        elif len(data) >= 1000:
            print(f"   👍 数据质量: 良好 (1,000+条数据)")
        else:
            print(f"   ⚠️ 数据质量: 一般 (少于1,000条)")
            
        # 数据时间跨度评估
        if time_span_years >= 0.8:
            print(f"   📅 时间跨度: 优秀 (接近1年数据)")
        elif time_span_years >= 0.5:
            print(f"   📅 时间跨度: 良好 (半年以上数据)")
        else:
            print(f"   📅 时间跨度: 一般 (少于半年数据)")
        
        return True
    else:
        print("❌ 所有数据源都失败了")
        return False

# 保持向后兼容性
class MultiSourceDataFetcher(EnhancedMultiSourceDataFetcher):
    """向后兼容的数据获取器"""
    
    def fetch_large_dataset(self, target_points=2000):
        """保持向后兼容的方法，直接调用新的多源获取"""
        return self.fetch_all_sources(target_points)

def test_huobi_fetcher():
    """保持向后兼容的测试函数"""
    return test_enhanced_fetcher()

if __name__ == "__main__":
    print("🚀 开始测试优化后的数据源...")
    print("💡 只保留真正有效的数据源")
    print("📊 目标：获取DOGE 15分钟数据，最好一年历史")
    print()
    
    test_enhanced_fetcher() 