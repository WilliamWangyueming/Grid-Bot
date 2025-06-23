"""
优化版多数据源获取器 - 智能数据源选择，避免重复数据
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
    """优化版多数据源获取器 - 智能数据源选择"""
    
    def __init__(self, symbol="DOGEUSDT"):
        self.symbol = symbol
        # 数据源按优先级排序：质量 + 数据量 + 稳定性
        self.data_sources = [
            "yahoo_finance",       # ✅ 最优：真实DOGE数据，1年历史，多时间颗粒度
            "huobi_api",           # ✅ 高质量：15分钟数据，高密度，稳定
            "kraken_ohlc",         # ✅ 稳定：官方API，但数据量较少
        ]
        
    def fetch_all_sources(self, target_points=2000):
        """智能获取数据：优先选择最佳数据源，避免不必要的重复"""
        
        print("🚀 启动智能多数据源获取")
        print("=" * 60)
        print(f"🎯 交易对: {self.symbol}")
        print(f"📊 目标数据量: {target_points} 条")
        print(f"🧠 智能策略: 优先级选择 + 质量评估")
        
        best_source = None
        best_data = None
        all_sources_data = []
        
        # 1. Yahoo Finance - 最高优先级（混合策略）
        print(f"\n📊 1. Yahoo Finance (智能混合策略)")
        print("-" * 50)
        try:
            data = self._fetch_yahoo_finance_smart()
            if data is not None and len(data) > 100:
                time_span = (data.index[-1] - data.index[0]).total_seconds() / 86400
                data_density = len(data) / time_span
                
                print(f"✅ Yahoo Finance 成功")
                print(f"   📈 数据量: {len(data)}条")
                print(f"   ⏰ 时间跨度: {time_span:.1f}天")
                print(f"   📊 数据密度: {data_density:.1f} 条/天")
                
                # 评估Yahoo数据质量
                quality_score = self._evaluate_data_quality(data, time_span)
                print(f"   🏆 质量评分: {quality_score:.1f}/10")
                
                all_sources_data.append(("yahoo_finance", data, quality_score))
                
                # 如果Yahoo数据已经很好，可能不需要其他数据源
                if len(data) >= target_points and time_span >= 30:  # 至少1个月数据
                    print(f"   🎯 Yahoo数据已满足需求，跳过其他数据源")
                    best_source = "yahoo_finance"
                    best_data = data
                    return self._finalize_result(best_source, best_data)
                    
        except Exception as e:
            print(f"❌ Yahoo Finance 错误: {str(e)}")
        
        # 2. Huobi API - 如果Yahoo不够好才使用
        print(f"\n📊 2. Huobi API (15分钟高密度数据)")
        print("-" * 50)
        try:
            data = self._fetch_huobi_15min()
            if data is not None and len(data) > 100:
                time_span = (data.index[-1] - data.index[0]).total_seconds() / 86400
                data_density = len(data) / time_span
                
                print(f"✅ Huobi API 成功")
                print(f"   📈 数据量: {len(data)}条")
                print(f"   ⏰ 时间跨度: {time_span:.1f}天")
                print(f"   📊 数据密度: {data_density:.1f} 条/天")
                
                quality_score = self._evaluate_data_quality(data, time_span)
                print(f"   🏆 质量评分: {quality_score:.1f}/10")
                
                all_sources_data.append(("huobi_api", data, quality_score))
                
        except Exception as e:
            print(f"❌ Huobi API 错误: {str(e)}")
        
        # 3. Kraken OHLC - 作为补充
        print(f"\n📊 3. Kraken OHLC API (稳定补充数据)")
        print("-" * 50)
        try:
            data = self._fetch_kraken_ohlc()
            if data is not None and len(data) > 100:
                time_span = (data.index[-1] - data.index[0]).total_seconds() / 86400
                data_density = len(data) / time_span
                
                print(f"✅ Kraken OHLC 成功")
                print(f"   📈 数据量: {len(data)}条")
                print(f"   ⏰ 时间跨度: {time_span:.1f}天")
                print(f"   📊 数据密度: {data_density:.1f} 条/天")
                
                quality_score = self._evaluate_data_quality(data, time_span)
                print(f"   🏆 质量评分: {quality_score:.1f}/10")
                
                all_sources_data.append(("kraken_ohlc", data, quality_score))
                
        except Exception as e:
            print(f"❌ Kraken OHLC 错误: {str(e)}")
        
        # 智能选择最佳数据源
        if all_sources_data:
            print(f"\n🧠 智能数据源选择")
            print("=" * 60)
            
            # 按质量评分排序
            all_sources_data.sort(key=lambda x: x[2], reverse=True)
            
            for i, (source_name, data, score) in enumerate(all_sources_data, 1):
                time_span = (data.index[-1] - data.index[0]).total_seconds() / 86400
                print(f"{i}. {source_name}: {len(data)}条, {time_span:.1f}天, 评分{score:.1f}")
            
            # 选择最佳数据源
            best_source, best_data, best_score = all_sources_data[0]
            
            print(f"\n🏆 选择最佳数据源: {best_source}")
            print(f"📊 最终数据: {len(best_data)}条, 评分{best_score:.1f}")
            
            # 智能合并策略：只有在有明显补充价值时才合并
            if len(all_sources_data) > 1:
                best_time_span = (best_data.index[-1] - best_data.index[0]).total_seconds() / 86400
                
                # 检查是否需要合并其他数据源
                should_merge = False
                for source_name, data, score in all_sources_data[1:]:
                    time_span = (data.index[-1] - data.index[0]).total_seconds() / 86400
                    
                    # 只有在能显著增加时间跨度或数据量时才合并
                    if (time_span > best_time_span * 1.2 or  # 时间跨度增加20%以上
                        len(data) > len(best_data) * 0.5):   # 数据量是最佳源的50%以上
                        should_merge = True
                        break
                
                if should_merge:
                    print(f"\n🔄 执行智能数据合并...")
                    combined_data = self._smart_merge_sources(all_sources_data)
                    if combined_data is not None and len(combined_data) > len(best_data) * 1.1:
                        improvement = len(combined_data) - len(best_data)
                        print(f"✅ 合并成功: +{improvement}条数据")
                        best_source = "merged_smart"
                        best_data = combined_data
                    else:
                        print(f"⚠️ 合并收益不明显，保持单一数据源")
                else:
                    print(f"⚠️ 其他数据源无显著补充价值，保持最佳单一数据源")
            
            # 统一将Yahoo最佳数据重采样至严格15分钟时间轴，避免后续重复填充
            best_data = self._resample_to_15m(best_data)
            return self._finalize_result(best_source, best_data)
        
        else:
            print("❌ 所有数据源都失败了")
            return None, None
    
    def _evaluate_data_quality(self, data, time_span_days):
        """评估数据质量 (0-10分)"""
        try:
            score = 0.0
            
            # 1. 数据量评分 (0-3分)
            if len(data) >= 4000:
                score += 3.0
            elif len(data) >= 2000:
                score += 2.5
            elif len(data) >= 1000:
                score += 2.0
            elif len(data) >= 500:
                score += 1.5
            else:
                score += len(data) / 500.0
            
            # 2. 时间跨度评分 (0-3分)
            if time_span_days >= 300:  # 接近1年
                score += 3.0
            elif time_span_days >= 180:  # 半年
                score += 2.5
            elif time_span_days >= 90:   # 3个月
                score += 2.0
            elif time_span_days >= 30:   # 1个月
                score += 1.5
            else:
                score += time_span_days / 30.0
            
            # 3. 数据密度评分 (0-2分)
            density = len(data) / time_span_days
            if density >= 96:    # 15分钟级别 (96条/天)
                score += 2.0
            elif density >= 24:  # 小时级别 (24条/天)
                score += 1.5
            elif density >= 4:   # 6小时级别
                score += 1.0
            else:
                score += density / 24.0
            
            # 4. 数据完整性评分 (0-2分)
            # 检查是否有缺失值
            missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
            if missing_ratio == 0:
                score += 2.0
            elif missing_ratio < 0.01:
                score += 1.5
            elif missing_ratio < 0.05:
                score += 1.0
            else:
                score += max(0, 1.0 - missing_ratio)
            
            return min(10.0, score)
            
        except:
            return 5.0  # 默认中等评分
    
    def _smart_merge_sources(self, all_sources_data):
        """
        智能合并多数据源，统一重采样到15分钟时间轴
        """
        try:
            print("🔄 开始智能数据源合并...")
            
            all_data = []
            
            # 预处理每个数据源
            for source_name, data, score in all_sources_data:
                # 确保数据有时间索引
                if not isinstance(data.index, pd.DatetimeIndex):
                    data.index = pd.to_datetime(data.index)
                
                # 删除重复时间戳，保留最后一个
                data_clean = data[~data.index.duplicated(keep='last')]
                
                # 统一重采样到15分钟
                try:
                    resampled = data_clean.resample('15T').agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min', 
                        'close': 'last',
                        'volume': 'sum'
                    })
                    
                    # 前向填充缺失值
                    resampled = resampled.fillna(method='ffill')
                    
                    # 删除NaN行
                    resampled = resampled.dropna()
                    
                    if len(resampled) > 50:  # 确保有足够数据
                        all_data.append((source_name, resampled, score))
                        print(f"   ✅ {source_name}: 重采样到 {len(resampled)} 条15分钟数据")
                    else:
                        print(f"   ⚠️ {source_name}: 重采样后数据不足，跳过")
                        
                except Exception as e:
                    print(f"   ❌ {source_name}: 重采样失败 - {str(e)}")
                    continue
            
            if len(all_data) < 2:
                print("   ⚠️ 可用数据源不足，无法合并")
                return None
            
            # 找到时间范围的交集和并集
            earliest_start = min([data.index[0] for _, data, _ in all_data])
            latest_end = max([data.index[-1] for _, data, _ in all_data])
            
            print(f"   📅 合并时间范围: {earliest_start} 到 {latest_end}")
            
            # 创建统一的15分钟时间轴
            unified_timeline = pd.date_range(
                start=earliest_start, 
                end=latest_end, 
                freq='15T'
            )
            
            # 基于质量评分排序，优先使用高质量数据
            all_data.sort(key=lambda x: x[2], reverse=True)
            
            # 从最高质量数据源开始
            primary_source, primary_data, primary_score = all_data[0]
            print(f"   🏆 主要数据源: {primary_source} (评分: {primary_score:.1f})")
            
            # 重建索引到统一时间轴
            merged_data = primary_data.reindex(unified_timeline, method='nearest', tolerance='30T')
            
            # 用其他数据源填补缺失
            for source_name, data, score in all_data[1:]:
                print(f"   🔗 合并 {source_name} (评分: {score:.1f})")
                
                # 找到缺失的时间点
                missing_mask = merged_data.isnull().any(axis=1)
                missing_times = merged_data[missing_mask].index
                
                if len(missing_times) > 0:
                    # 用当前数据源填补
                    fill_data = data.reindex(missing_times, method='nearest', tolerance='30T')
                    
                    # 只填补有效数据
                    valid_fill = fill_data.dropna()
                    if len(valid_fill) > 0:
                        merged_data.loc[valid_fill.index] = valid_fill
                        filled_count = len(valid_fill)
                        print(f"     ↪️ 填补了 {filled_count} 个缺失时间点")
                    else:
                        print(f"     ⚠️ 无有效数据可填补")
                else:
                    print(f"     ✅ 无需填补")
            
            # 最终清理
            merged_data = merged_data.dropna()
            
            # 数据质量检查
            if len(merged_data) > 0:
                # 确保价格逻辑性
                merged_data['high'] = merged_data[['open', 'high', 'close']].max(axis=1)
                merged_data['low'] = merged_data[['open', 'low', 'close']].min(axis=1)
                merged_data['volume'] = merged_data['volume'].clip(lower=0)  # 成交量非负
                
                time_span = (merged_data.index[-1] - merged_data.index[0]).total_seconds() / 86400
                data_density = len(merged_data) / time_span if time_span > 0 else 0
                
                print(f"   ✅ 合并完成:")
                print(f"     📊 最终数据量: {len(merged_data)} 条")
                print(f"     ⏰ 时间跨度: {time_span:.1f} 天")
                print(f"     📈 数据密度: {data_density:.1f} 条/天")
                
                return merged_data
            else:
                print("   ❌ 合并后无有效数据")
                return None
                
        except Exception as e:
            print(f"❌ 智能合并失败: {str(e)}")
            return None
    
    def _finalize_result(self, source, data):
        """最终结果处理和展示"""
        if data is None:
            return None, None
        
        time_span_days = (data.index[-1] - data.index[0]).total_seconds()/86400
        time_span_years = time_span_days / 365
        
        print(f"\n🎉 数据获取完成")
        print("=" * 60)
        print(f"🏆 最终数据源: {source}")
        print(f"📈 数据量: {len(data):,} 条")
        print(f"⏰ 时间跨度: {time_span_days:.1f}天 ({time_span_years:.2f}年)")
        print(f"📅 时间范围: {data.index[0]} 到 {data.index[-1]}")
        print(f"💰 价格范围: {data['close'].min():.6f} - {data['close'].max():.6f}")
        print(f"📊 数据密度: {len(data)/time_span_days:.1f} 条/天")
        
        # 数据质量评估
        quality_score = self._evaluate_data_quality(data, time_span_days)
        if quality_score >= 8.5:
            print(f"🏆 数据质量: 卓越 ({quality_score:.1f}/10)")
        elif quality_score >= 7.0:
            print(f"🥇 数据质量: 优秀 ({quality_score:.1f}/10)")
        elif quality_score >= 5.5:
            print(f"👍 数据质量: 良好 ({quality_score:.1f}/10)")
        else:
            print(f"⚠️ 数据质量: 一般 ({quality_score:.1f}/10)")
        
        return source, data
    
    def _fetch_yahoo_finance_smart(self):
        """智能获取 Yahoo Finance 数据 - 优先15分钟，1小时补充历史"""
        try:
            print("   🧠 智能 Yahoo Finance 策略...")
            
            symbol = "DOGE-USD"
            ticker = yf.Ticker(symbol)
            print(f"   📥 获取符号: {symbol}")
            
            # 策略选择：根据需求智能选择最佳策略
            strategies = []
            
            # 策略1: 最大范围15分钟数据（优先策略）
            print("   📊 策略1: 最大范围15分钟数据...")
            df_15m_max = None
            for period in ["60d", "30d", "7d"]:  # 尝试不同时间范围
                try:
                    print(f"      🔍 尝试15分钟数据 ({period})...")
                    data_15m = ticker.history(period=period, interval="15m", auto_adjust=False, prepost=False)
                    if len(data_15m) > 100:
                        df_15m_temp = self._convert_yahoo_data(data_15m)
                        time_span = (df_15m_temp.index[-1] - df_15m_temp.index[0]).total_seconds() / 86400
                        print(f"         ✅ 15分钟数据 ({period}): {len(df_15m_temp)}条, {time_span:.1f}天")
                        
                        # 选择数据最多的15分钟数据集
                        if df_15m_max is None or len(df_15m_temp) > len(df_15m_max):
                            df_15m_max = df_15m_temp
                            
                except Exception as e:
                    print(f"         ❌ 15分钟数据 ({period}) 失败: {str(e)}")
            
            if df_15m_max is not None:
                time_span_15m = (df_15m_max.index[-1] - df_15m_max.index[0]).total_seconds() / 86400
                strategies.append(("15min_max", df_15m_max, len(df_15m_max), time_span_15m))
                print(f"      🏆 最佳15分钟数据: {len(df_15m_max)}条, {time_span_15m:.1f}天")
            
            # 策略2: 1年1小时数据（补充历史用）
            print("   📊 策略2: 1年1小时历史数据...")
            df_1h = None
            for period in ["1y", "max"]:  # 尝试获取最长历史
                try:
                    print(f"      🔍 尝试1小时数据 ({period})...")
                    data_1h = ticker.history(period=period, interval="1h", auto_adjust=False, prepost=False)
                    if len(data_1h) > 100:
                        df_1h_temp = self._convert_yahoo_data(data_1h)
                        time_span = (df_1h_temp.index[-1] - df_1h_temp.index[0]).total_seconds() / 86400
                        print(f"         ✅ 1小时数据 ({period}): {len(df_1h_temp)}条, {time_span:.1f}天")
                        
                        # 选择时间跨度最长的1小时数据
                        if df_1h is None or time_span > (df_1h.index[-1] - df_1h.index[0]).total_seconds() / 86400:
                            df_1h = df_1h_temp
                            
                except Exception as e:
                    print(f"         ❌ 1小时数据 ({period}) 失败: {str(e)}")
            
            if df_1h is not None:
                time_span_1h = (df_1h.index[-1] - df_1h.index[0]).total_seconds() / 86400
                strategies.append(("1hour_max", df_1h, len(df_1h), time_span_1h))
                print(f"      🏆 最佳1小时数据: {len(df_1h)}条, {time_span_1h:.1f}天")
            
            # 策略3: 智能融合 - 15分钟优先 + 1小时补充历史
            if df_15m_max is not None and df_1h is not None:
                print("   📊 策略3: 智能融合 (15分钟优先 + 1小时补充)...")
                try:
                    # 分析15分钟数据的时间范围
                    min_15m = df_15m_max.index.min()
                    max_15m = df_15m_max.index.max()
                    
                    print(f"      📅 15分钟数据范围: {min_15m} 到 {max_15m}")
                    
                    # 从1小时数据中提取15分钟数据范围之外的历史数据
                    df_1h_historical = df_1h[df_1h.index < min_15m]
                    df_1h_future = df_1h[df_1h.index > max_15m]  # 也可能有未来数据
                    
                    print(f"      📊 1小时历史数据: {len(df_1h_historical)}条")
                    print(f"      📊 1小时未来数据: {len(df_1h_future)}条")
                    
                    # 智能融合策略
                    fusion_parts = []
                    
                    # 1. 添加1小时历史数据（15分钟数据之前的时间段）
                    if len(df_1h_historical) > 0:
                        fusion_parts.append(df_1h_historical)
                        print(f"      ✅ 添加历史1小时数据: {len(df_1h_historical)}条")
                    
                    # 2. 添加15分钟高精度数据（主要数据）
                    fusion_parts.append(df_15m_max)
                    print(f"      ✅ 添加15分钟高精度数据: {len(df_15m_max)}条")
                    
                    # 3. 添加1小时未来数据（如果有的话）
                    if len(df_1h_future) > 0:
                        fusion_parts.append(df_1h_future)
                        print(f"      ✅ 添加未来1小时数据: {len(df_1h_future)}条")
                    
                    # 合并所有数据
                    if len(fusion_parts) > 1:
                        combined_df = pd.concat(fusion_parts).sort_index()
                        # 去除重复时间戳（优先保留15分钟数据）
                        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                        
                        time_span_combined = (combined_df.index[-1] - combined_df.index[0]).total_seconds() / 86400
                        
                        # 计算数据构成
                        historical_days = 0
                        if len(df_1h_historical) > 0:
                            historical_days = (df_1h_historical.index[-1] - df_1h_historical.index[0]).total_seconds() / 86400
                        
                        main_days = (df_15m_max.index[-1] - df_15m_max.index[0]).total_seconds() / 86400
                        
                        strategies.append(("fusion_optimal", combined_df, len(combined_df), time_span_combined))
                        
                        print(f"      🎉 融合数据创建成功!")
                        print(f"         📈 总数据量: {len(combined_df)}条")
                        print(f"         ⏰ 总时间跨度: {time_span_combined:.1f}天 ({time_span_combined/365:.2f}年)")
                        print(f"         📊 历史1小时段: {len(df_1h_historical)}条 ({historical_days:.1f}天)")
                        print(f"         🔍 高精度15分钟段: {len(df_15m_max)}条 ({main_days:.1f}天)")
                        if len(df_1h_future) > 0:
                            future_days = (df_1h_future.index[-1] - df_1h_future.index[0]).total_seconds() / 86400
                            print(f"         📈 未来1小时段: {len(df_1h_future)}条 ({future_days:.1f}天)")
                        
                        # 检查是否达到1年目标
                        if time_span_combined >= 300:  # 约10个月
                            print(f"         🎯 已达到长期历史目标 ({time_span_combined/365:.2f}年)")
                        else:
                            print(f"         ⚠️ 时间跨度未达到1年目标，但已最大化利用可用数据")
                    
                except Exception as e:
                    print(f"      ❌ 融合策略失败: {str(e)}")
            
            # 策略4: 如果只有15分钟数据
            elif df_15m_max is not None:
                print("   📊 策略4: 仅15分钟数据 (无1小时补充)...")
                strategies.append(("15min_only", df_15m_max, len(df_15m_max), time_span_15m))
                print(f"      ⚠️ 仅有15分钟数据，无法获取1小时历史补充")
            
            # 策略5: 如果只有1小时数据  
            elif df_1h is not None:
                print("   📊 策略5: 仅1小时数据 (无15分钟数据)...")
                strategies.append(("1hour_only", df_1h, len(df_1h), time_span_1h))
                print(f"      ⚠️ 仅有1小时数据，无法获取15分钟高精度数据")
            
            # 选择最佳策略 - 优先融合策略
            if strategies:
                print("   🧠 评估策略优劣...")
                
                # 计算每个策略的综合评分，融合策略额外加分
                scored_strategies = []
                for strategy_name, df, count, time_span in strategies:
                    base_score = self._evaluate_data_quality(df, time_span)
                    
                    # 融合策略额外奖励
                    if "fusion" in strategy_name:
                        bonus_score = 1.0  # 融合策略额外1分
                        print(f"      {strategy_name}: {count}条, {time_span:.1f}天, 评分{base_score:.1f}+{bonus_score:.1f}(融合奖励)={base_score+bonus_score:.1f}")
                        scored_strategies.append((strategy_name, df, base_score + bonus_score))
                    else:
                        scored_strategies.append((strategy_name, df, base_score))
                        print(f"      {strategy_name}: {count}条, {time_span:.1f}天, 评分{base_score:.1f}")
                
                # 选择最高评分的策略
                best_strategy = max(scored_strategies, key=lambda x: x[2])
                strategy_name, best_df, best_score = best_strategy
                
                print(f"   🏆 选择策略: {strategy_name} (评分: {best_score:.1f})")
                # 统一将Yahoo最佳数据重采样至严格15分钟时间轴，避免后续重复填充
                best_df = self._resample_to_15m(best_df)
                return best_df
            
            print("   ❌ 所有Yahoo策略都失败了")
            return None
            
        except Exception as e:
            print(f"   ❌ Yahoo Finance 智能策略失败: {str(e)}")
            return None
    
    def _convert_yahoo_data(self, data):
        """转换Yahoo Finance数据格式"""
        df = pd.DataFrame()
        df['open'] = data['Open']
        df['high'] = data['High'] 
        df['low'] = data['Low']
        df['close'] = data['Close']
        df['volume'] = data['Volume']
        
        # 统一时区处理
        if data.index.tz is not None:
            df.index = data.index.tz_convert('UTC').tz_localize(None)
        else:
            df.index = pd.to_datetime(data.index)
        
        return df.sort_index()

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

    def _resample_to_15m(self, data):
        """将数据重采样到严格15分钟时间轴"""
        try:
            print("🔄 开始重采样到15分钟时间轴...")
            
            # 确保数据有时间索引
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # 删除重复时间戳，保留最后一个
            data_clean = data[~data.index.duplicated(keep='last')]
            
            # 统一重采样到15分钟
            try:
                resampled = data_clean.resample('15T').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min', 
                    'close': 'last',
                    'volume': 'sum'
                })
                
                # 前向填充缺失值
                resampled = resampled.fillna(method='ffill')
                
                # 删除NaN行
                resampled = resampled.dropna()
                
                if len(resampled) > 50:  # 确保有足够数据
                    print(f"   ✅ 重采样到15分钟时间轴成功: {len(resampled)} 条15分钟数据")
                    return resampled
                else:
                    print(f"   ⚠️ 重采样后数据不足，跳过")
                    
            except Exception as e:
                print(f"   ❌ 重采样失败 - {str(e)}")
                return None
            
        except Exception as e:
            print(f"❌ 重采样失败: {str(e)}")
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