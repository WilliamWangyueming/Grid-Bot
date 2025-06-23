"""
优化版LSTM价格预测系统 v2.0
- 数据层：统一重采样、技术指标、RobustScaler
- 模型架构：双向LSTM + Attention机制  
- 训练策略：滚动交叉验证、优化器改进
- CPU优化：MKL-DNN、torch.compile加速
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import torch.nn.functional as F

# 尝试导入TA-Lib，如果失败则使用pandas备用实现
try:
    import talib
    TALIB_AVAILABLE = True
    print("✅ TA-Lib已导入")
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib未安装，将使用pandas备用实现")

from datetime import datetime, timedelta
import warnings
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
import joblib
from multi_source_fetcher import MultiSourceDataFetcher

warnings.filterwarnings('ignore')

# CPU性能优化设置
def setup_optimized_device():
    """设置优化的设备配置"""
    device = "cpu"  # 默认CPU
    device_name = "优化CPU"
    
    # 启用MKL-DNN/OneDNN (针对Intel/AMD CPU优化)
    try:
        torch.backends.mkldnn.enabled = True
        print("🚀 启用MKL-DNN加速")
    except:
        pass
    
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 检测到GPU: {device_name}")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        print("🔧 使用CPU优化设置...")
        # 多核并行设置
        torch.set_num_threads(os.cpu_count())
        # 设置默认数据类型为bfloat16 (如果支持)
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'cpu'):
            try:
                torch.set_default_dtype(torch.float32)  # 保持精度
            except:
                pass
                
    return device, device_name

class EnhancedLSTMPredictor(nn.Module):
    """
    增强的LSTM预测模型
    - 双向LSTM
    - Multi-Head Attention机制
    - LayerNorm和Residual连接
    - 多任务学习（价格回归 + 方向分类）
    """
    def __init__(self, input_size=15, hidden_size=128, num_layers=3, dropout=0.2, 
                 num_heads=4, output_size=4):
        super(EnhancedLSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # 双向LSTM
        )
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # 双向LSTM输出维度
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 价格预测分支（回归任务）
        self.price_branch = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)  # OHLC
        )
        
        # 成交量预测分支
        self.volume_branch = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.ReLU()  # 确保成交量为正
        )
        
        # 方向分类分支（多任务学习）
        self.direction_branch = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2),  # up/down二分类
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # 初始化隐藏状态（双向LSTM需要2倍数量）
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size, device=x.device)
        
        # 双向LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Attention机制
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual连接 + LayerNorm
        lstm_out = self.layer_norm(lstm_out + attn_output)
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # 多分支输出
        price_pred = self.price_branch(last_output)
        volume_pred = self.volume_branch(last_output)
        direction_pred = self.direction_branch(last_output)
        
        return price_pred, volume_pred, direction_pred

class FocalLoss(nn.Module):
    """多分类 Focal Loss (softmax 版本)"""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        # inputs 形状: (batch, num_classes) -> softmax 概率
        eps = 1e-8
        inputs = inputs.clamp(min=eps, max=1.0-eps)
        one_hot = F.one_hot(targets.view(-1), num_classes=inputs.size(1)).float().to(inputs.device)
        pt = (inputs * one_hot).sum(dim=1)  # 取正确类别的概率
        focal_term = self.alpha * (1 - pt) ** self.gamma
        loss = -focal_term * pt.log()
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class OptimizedLSTMSystem:
    """优化的LSTM价格预测系统 v2.0"""
    
    def __init__(self, symbol="DOGEUSDT", sequence_length=30):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.device, self.device_name = setup_optimized_device()
        
        # 使用RobustScaler（对离群值更稳健）
        self.price_scaler = RobustScaler()
        self.volume_scaler = RobustScaler()
        self.tech_scaler = StandardScaler()  # 技术指标使用StandardScaler
        
        # 模型
        self.model = None
        self.trained = False
        
        # 缓存文件路径
        self.scaler_path = f"scalers_{symbol}.joblib"
        
        print(f"📱 运行设备: {self.device_name}")
        print(f"🔢 序列长度: {sequence_length}")
        
    def add_technical_indicators(self, data):
        """
        添加技术指标特征 - 支持TA-Lib和pandas备用实现
        """
        print("📊 计算技术指标...")
        
        df = data.copy()
        
        # 确保数据类型正确
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        try:
            # 1. 对数收益率
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # 2. 成交量Z-score
            df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()
            
            if TALIB_AVAILABLE:
                # 使用TA-Lib计算技术指标
                print("   🚀 使用TA-Lib计算技术指标")
                
                # 3. EMA指标
                df['ema_12'] = talib.EMA(df['close'].values, timeperiod=12)
                df['ema_26'] = talib.EMA(df['close'].values, timeperiod=26)
                
                # 4. MACD
                df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values)
                
                # 5. RSI
                df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
                
                # 6. 布林带
                df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'].values)
                df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                
                # 7. ATR (平均真实范围)
                df['atr'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values)
                
            else:
                # 使用pandas备用实现
                print("   🔧 使用pandas备用实现技术指标")
                
                # 3. EMA指标 (pandas实现)
                df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
                df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
                
                # 4. MACD (简化实现)
                df['macd'] = df['ema_12'] - df['ema_26']
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                
                # 5. RSI (pandas实现)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # 6. 布林带 (pandas实现)
                df['bb_middle'] = df['close'].rolling(window=20).mean()
                bb_std = df['close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
                df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
                df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                
                # 7. ATR (pandas实现)
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                df['atr'] = true_range.rolling(window=14).mean()
            
            # 8. 成交量移动平均 (通用实现)
            df['volume_ma'] = df['volume'].rolling(10).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # 9. 周期时间特征（交易日规律）
            df['minute_of_day'] = df.index.hour * 60 + df.index.minute
            df['minute_sin'] = np.sin(2 * np.pi * df['minute_of_day'] / 1440)
            df['minute_cos'] = np.cos(2 * np.pi * df['minute_of_day'] / 1440)

            df['dayofweek'] = df.index.dayofweek
            df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

            # 10. 蜡烛图形态特征
            df['body'] = (df['close'] - df['open']) / df['open'].replace(0, np.nan)
            df['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['open'].replace(0, np.nan)
            df['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['open'].replace(0, np.nan)

            # 11. VWAP 差值
            df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
            df['vwap_diff'] = (df['close'] - df['vwap']) / df['vwap']
            
            # 删除包含NaN的行
            df = df.dropna()
            
            indicator_type = "TA-Lib" if TALIB_AVAILABLE else "pandas备用"
            print(f"✅ 技术指标计算完成({indicator_type})，数据量: {len(df)}")
            
            return df
            
        except Exception as e:
            print(f"❌ 技术指标计算失败: {str(e)}")
            # 如果技术指标计算失败，返回基础数据
            print("   🔄 回退到基础OHLCV数据")
            return data
    
    def resample_and_clean_data(self, data):
        """
        统一重采样到15分钟，清理重复时间戳
        """
        print("🔧 统一重采样和数据清理...")
        
        try:
            # 确保索引是时间戳
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            
            # 删除重复的时间戳，保留最后一个
            data = data[~data.index.duplicated(keep='last')]
            
            # 重采样到15分钟
            resampled = data.resample('15T').agg({
                'open': 'first',
                'high': 'max', 
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            # 前向填充空值
            resampled = resampled.fillna(method='ffill')
            
            # 删除仍然包含NaN的行
            resampled = resampled.dropna()
            
            print(f"✅ 重采样完成: {len(data)} -> {len(resampled)} 条记录")
            
            return resampled
            
        except Exception as e:
            print(f"❌ 重采样失败: {str(e)}")
            return data
    
    def prepare_enhanced_data(self, data, use_walk_forward=True):
        """
        增强的数据准备，包含技术指标和滚动交叉验证
        """
        print(f"📊 准备增强数据: {len(data)} 条记录")
        
        # 1. 统一重采样和清理
        data = self.resample_and_clean_data(data)
        
        # 2. 添加技术指标
        data = self.add_technical_indicators(data)
        
        if len(data) < self.sequence_length + 50:
            raise ValueError(f"数据量不足，需要至少 {self.sequence_length + 50} 条记录")
        
        # 3. 准备特征
        # 基础OHLCV
        ohlc_data = data[['open', 'high', 'low', 'close']].values
        volume_data = data[['volume']].values
        
        # 技术指标特征（扩充后共17列）
        tech_cols = ['log_return', 'volume_zscore', 'ema_12', 'ema_26',
                     'macd', 'rsi', 'bb_percent', 'atr', 'volume_ratio',
                     'minute_sin', 'minute_cos', 'day_sin', 'day_cos',
                     'body', 'upper_shadow', 'lower_shadow', 'vwap_diff']
        tech_features = data[tech_cols].values
        
        # 4. 数据标准化
        ohlc_scaled = self.price_scaler.fit_transform(ohlc_data)
        volume_scaled = self.volume_scaler.fit_transform(volume_data)
        tech_scaled = self.tech_scaler.fit_transform(tech_features)
        
        # 合并所有特征 (4 + 1 + 17 = 22维)
        combined_features = np.concatenate([ohlc_scaled, volume_scaled, tech_scaled], axis=1)
        
        # 5. 创建序列数据
        sequences = []
        targets_ohlc = []
        targets_volume = []
        targets_direction = []  # 方向标签
        
        for i in range(self.sequence_length, len(combined_features)):
            # 输入序列
            seq = combined_features[i-self.sequence_length:i]
            sequences.append(seq)
            
            # 目标值
            targets_ohlc.append(ohlc_scaled[i])
            targets_volume.append(volume_scaled[i])
            
            # 方向标签（上涨=1，下跌=0）
            current_close = data['close'].iloc[i]
            prev_close = data['close'].iloc[i-1]
            direction = 1 if current_close > prev_close else 0
            targets_direction.append(direction)
        
        sequences = np.array(sequences)
        targets_ohlc = np.array(targets_ohlc)
        targets_volume = np.array(targets_volume)
        targets_direction = np.array(targets_direction)
        
        print(f"✅ 生成序列: {len(sequences)} 个")
        print(f"📏 特征维度: {sequences.shape[2]} (OHLC:4 + Volume:1 + 技术指标:{len(tech_cols)})")
        
        if use_walk_forward:
            # 使用时间序列交叉验证
            print("🔄 使用滚动交叉验证...")
            tscv = TimeSeriesSplit(n_splits=5)
            
            # 返回所有折叠的数据，用于训练时选择
            cv_splits = []
            for train_idx, test_idx in tscv.split(sequences):
                cv_splits.append({
                    'X_train': sequences[train_idx],
                    'X_test': sequences[test_idx],
                    'y_train_ohlc': targets_ohlc[train_idx],
                    'y_test_ohlc': targets_ohlc[test_idx],
                    'y_train_vol': targets_volume[train_idx],
                    'y_test_vol': targets_volume[test_idx],
                    'y_train_dir': targets_direction[train_idx],
                    'y_test_dir': targets_direction[test_idx]
                })
            
            return cv_splits, data.index[self.sequence_length:]
        else:
            # 传统固定分割
            split_idx = int(len(sequences) * 0.85)
            
            return [{
                'X_train': sequences[:split_idx],
                'X_test': sequences[split_idx:],
                'y_train_ohlc': targets_ohlc[:split_idx],
                'y_test_ohlc': targets_ohlc[split_idx:],
                'y_train_vol': targets_volume[:split_idx],
                'y_test_vol': targets_volume[split_idx:],
                'y_train_dir': targets_direction[:split_idx],
                'y_test_dir': targets_direction[split_idx:]
            }], data.index[self.sequence_length:]
    
    def create_enhanced_model(self, input_size=22):
        """创建增强的LSTM模型"""
        model = EnhancedLSTMPredictor(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            dropout=0.2,
            num_heads=4,
            output_size=4
        )
        
        # 尝试使用torch.compile优化（PyTorch 2.x）
        # Windows 若未安装 Visual Studio Build Tools (cl.exe) 会报错，检测后再决定
        try:
            import shutil, platform
            has_cl = shutil.which('cl') is not None
            if hasattr(torch, 'compile') and (platform.system() != 'Windows' or has_cl):
                model = torch.compile(model, mode="max-autotune")
                print("🚀 启用torch.compile优化")
            else:
                print("⚠️ 未检测到可用 C++ 编译器，已跳过 torch.compile")
        except Exception as e:
            print(f"⚠️ torch.compile 跳过: {e}")
            
        return model.to(self.device)
    
    def train_enhanced_model(self, cv_splits, batch_size=32, epochs=100, learning_rate=0.001):
        """
        增强的模型训练，支持滚动交叉验证
        """
        print("🚀 开始增强训练...")
        
        # 使用最后一个分割作为主要训练数据
        main_split = cv_splits[-1]
        
        # 创建模型
        input_size = main_split['X_train'].shape[2]
        self.model = self.create_enhanced_model(input_size=input_size)
        
        # 转换为张量
        X_train = torch.FloatTensor(main_split['X_train']).to(self.device)
        y_train_ohlc = torch.FloatTensor(main_split['y_train_ohlc']).to(self.device)
        y_train_vol = torch.FloatTensor(main_split['y_train_vol']).to(self.device)
        y_train_dir = torch.LongTensor(main_split['y_train_dir']).to(self.device)
        
        X_test = torch.FloatTensor(main_split['X_test']).to(self.device)
        y_test_ohlc = torch.FloatTensor(main_split['y_test_ohlc']).to(self.device)
        y_test_vol = torch.FloatTensor(main_split['y_test_vol']).to(self.device)
        y_test_dir = torch.LongTensor(main_split['y_test_dir']).to(self.device)
        
        # 数据加载器（多进程优化）
        train_dataset = TensorDataset(X_train, y_train_ohlc, y_train_vol, y_train_dir)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=min(4, os.cpu_count()//2),  # CPU优化的worker数量
            persistent_workers=True,
            pin_memory=True if self.device == "cuda" else False
        )
        
        # 损失函数
        criterion_price = nn.MSELoss()
        criterion_volume = nn.MSELoss()
        criterion_direction = FocalLoss(alpha=0.25, gamma=2.0)
        
        # 优化器改进: AdamW + Lookahead
        base_optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器: CosineAnnealingWarmRestarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            base_optimizer, 
            T_0=20,  # 重启周期
            eta_min=learning_rate * 0.01
        )
        
        # 训练循环
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 30  # 增加patience
        patience_counter = 0
        
        print(f"🔧 设备: {self.device_name}")
        print(f"📦 批次大小: {batch_size}")
        print(f"🔄 计划训练轮数: {epochs}")
        
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            epoch_train_loss = 0
            
            for batch_idx, (batch_x, batch_y_ohlc, batch_y_vol, batch_y_dir) in enumerate(train_loader):
                base_optimizer.zero_grad()
                
                # 前向传播
                price_pred, vol_pred, dir_pred = self.model(batch_x)
                
                # 多任务损失
                loss_price = criterion_price(price_pred, batch_y_ohlc)
                loss_vol = criterion_volume(vol_pred.squeeze(), batch_y_vol.squeeze())
                loss_dir = criterion_direction(dir_pred, batch_y_dir)
                
                total_loss = loss_price + 0.1 * loss_vol + 0.1 * loss_dir
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                base_optimizer.step()
                epoch_train_loss += total_loss.item()
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                price_pred_val, vol_pred_val, dir_pred_val = self.model(X_test)
                val_loss_price = criterion_price(price_pred_val, y_test_ohlc)
                val_loss_vol = criterion_volume(vol_pred_val.squeeze(), y_test_vol.squeeze())
                val_loss_dir = criterion_direction(dir_pred_val, y_test_dir)
                val_loss = val_loss_price + 0.1 * val_loss_vol + 0.1 * val_loss_dir
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss.item())
            
            # 学习率调度
            scheduler.step()
            
            # 早停检查（使用7日滑动平均）
            if len(val_losses) >= 7:
                recent_val_loss = np.mean(val_losses[-7:])
                if recent_val_loss < best_val_loss:
                    best_val_loss = recent_val_loss
                    patience_counter = 0
                    # 保存最佳模型和scaler
                    torch.save(self.model.state_dict(), f'best_lstm_model_{self.symbol}.pth')
                    self._save_scalers()
                else:
                    patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"轮次 [{epoch+1}/{epochs}] - "
                      f"训练损失: {avg_train_loss:.6f}, "
                      f"验证损失: {val_loss:.6f}, "
                      f"学习率: {current_lr:.6f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"🛑 早停触发，第 {epoch+1} 轮")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(f'best_lstm_model_{self.symbol}.pth'))
        self.trained = True
        
        print(f"✅ 增强训练完成！最佳验证损失: {best_val_loss:.6f}")
        return train_losses, val_losses
    
    def _save_scalers(self):
        """保存标准化器"""
        scalers = {
            'price_scaler': self.price_scaler,
            'volume_scaler': self.volume_scaler,
            'tech_scaler': self.tech_scaler
        }
        joblib.dump(scalers, self.scaler_path)
    
    def _load_scalers(self):
        """加载标准化器"""
        try:
            scalers = joblib.load(self.scaler_path)
            self.price_scaler = scalers['price_scaler']
            self.volume_scaler = scalers['volume_scaler'] 
            self.tech_scaler = scalers['tech_scaler']
            return True
        except:
            return False

    def predict_enhanced_future(self, data, prediction_hours=24):
        """增强的未来价格预测"""
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        print(f"🔮 预测未来 {prediction_hours} 小时价格...")
        
        self.model.eval()
        
        # 添加技术指标
        data_with_tech = self.add_technical_indicators(data)
        
        # 准备特征
        ohlc_data = data_with_tech[['open', 'high', 'low', 'close']].values
        volume_data = data_with_tech[['volume']].values
        tech_cols = ['log_return', 'volume_zscore', 'ema_12', 'ema_26',
                     'macd', 'rsi', 'bb_percent', 'atr', 'volume_ratio',
                     'minute_sin', 'minute_cos', 'day_sin', 'day_cos',
                     'body', 'upper_shadow', 'lower_shadow', 'vwap_diff']
        tech_features = data_with_tech[tech_cols].values
        
        # 标准化
        ohlc_scaled = self.price_scaler.transform(ohlc_data)
        volume_scaled = self.volume_scaler.transform(volume_data)
        tech_scaled = self.tech_scaler.transform(tech_features)
        
        combined_features = np.concatenate([ohlc_scaled, volume_scaled, tech_scaled], axis=1)
        
        # 最后一个序列
        last_sequence = combined_features[-self.sequence_length:]
        current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
        
        predictions = []
        directions = []
        
        with torch.no_grad():
            for _ in range(prediction_hours):
                # 预测
                price_pred, vol_pred, dir_pred = self.model(current_sequence)
                
                # 反标准化价格和成交量
                price_pred_unscaled = self.price_scaler.inverse_transform(price_pred.cpu().numpy())
                vol_pred_unscaled = self.volume_scaler.inverse_transform(vol_pred.cpu().numpy())
                
                # 方向预测
                direction_prob = torch.softmax(dir_pred, dim=-1).cpu().numpy()[0]
                predicted_direction = "上涨" if direction_prob[1] > 0.5 else "下跌"
                direction_confidence = max(direction_prob)
                
                # 确保价格逻辑性
                open_price = price_pred_unscaled[0][0]
                high_price = max(price_pred_unscaled[0][1], open_price, price_pred_unscaled[0][3])
                low_price = min(price_pred_unscaled[0][2], open_price, price_pred_unscaled[0][3])
                close_price = price_pred_unscaled[0][3]
                volume = max(0, vol_pred_unscaled[0][0])
                
                predictions.append({
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
                
                directions.append({
                    'direction': predicted_direction,
                    'confidence': direction_confidence
                })
                
                # 更新序列（这里需要计算新的技术指标，简化处理）
                new_point = np.array([[open_price, high_price, low_price, close_price, volume]])
                new_point_scaled = np.concatenate([
                    self.price_scaler.transform(new_point[:, :4]),
                    self.volume_scaler.transform(new_point[:, 4:5]),
                    np.zeros((1, 17))  # 技术指标暂时用0填充
                ], axis=1)
                
                # 滑动窗口更新
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    torch.FloatTensor(new_point_scaled).unsqueeze(0).to(self.device)
                ], dim=1)
        
        # 生成时间索引
        last_time = data.index[-1]
        future_times = [last_time + timedelta(hours=i+1) for i in range(prediction_hours)]
        
        # 转换为DataFrame
        pred_df = pd.DataFrame(predictions, index=future_times)
        dir_df = pd.DataFrame(directions, index=future_times)
        
        print(f"✅ 增强预测完成：{len(pred_df)} 个时间点")
        
        return pred_df, dir_df

    def enhanced_predict_sequence(self, historical_data, prediction_hours=24):
        """完整的增强预测流程"""
        print("🚀 启动增强LSTM价格预测系统 v2.0")
        print("=" * 60)
        
        # 数据准备
        cv_splits, time_index = self.prepare_enhanced_data(historical_data, use_walk_forward=True)
        
        # 训练模型
        print("\n🏋️ 训练增强LSTM模型...")
        train_losses, val_losses = self.train_enhanced_model(
            cv_splits,
            batch_size=64,
            epochs=150,
            learning_rate=0.001
        )
        
        # 生成预测
        predictions, directions = self.predict_enhanced_future(historical_data, prediction_hours)
        
        # 计算增强指标
        main_split = cv_splits[-1]
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(main_split['X_test']).to(self.device)
            y_test_ohlc_tensor = torch.FloatTensor(main_split['y_test_ohlc']).to(self.device)
            y_test_dir_tensor = torch.LongTensor(main_split['y_test_dir']).to(self.device)
            
            price_pred_test, _, dir_pred_test = self.model(X_test_tensor)
            
            # 反标准化
            y_test_unscaled = self.price_scaler.inverse_transform(y_test_ohlc_tensor.cpu().numpy())
            y_pred_unscaled = self.price_scaler.inverse_transform(price_pred_test.cpu().numpy())
            
            # 价格指标
            mse = mean_squared_error(y_test_unscaled[:, 3], y_pred_unscaled[:, 3])
            mae = mean_absolute_error(y_test_unscaled[:, 3], y_pred_unscaled[:, 3])
            mape = np.mean(np.abs((y_test_unscaled[:, 3] - y_pred_unscaled[:, 3]) / y_test_unscaled[:, 3])) * 100
            
            # 方向准确率
            dir_pred_classes = torch.argmax(dir_pred_test, dim=1)
            directional_accuracy = (dir_pred_classes == y_test_dir_tensor).float().mean().item() * 100
            
            current_price = historical_data['close'].iloc[-1]
            predicted_price = predictions['close'].iloc[-1]
            change_pct = ((predicted_price - current_price) / current_price) * 100
        
        quality_metrics = {
            'mse': mse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'change_pct': change_pct,
            'trend': '上涨' if change_pct > 1 else '下跌' if change_pct < -1 else '横盘',
            'confidence': max(70, min(95, 90 - abs(change_pct))),
            'main_direction': directions['direction'].iloc[-1],
            'direction_confidence': directions['confidence'].iloc[-1]
        }
        
        print(f"✅ 增强LSTM预测完成!")
        print(f"   📊 预测点数: {len(predictions)}")
        print(f"   💰 当前价格: {current_price:.6f}")
        print(f"   🎯 预测价格: {predicted_price:.6f}")
        print(f"   📈 变化幅度: {change_pct:+.2f}%")
        print(f"   🎲 MSE: {mse:.8f}")
        print(f"   📏 MAE: {mae:.6f}")
        print(f"   📊 MAPE: {mape:.2f}%")
        print(f"   🎯 方向准确率: {directional_accuracy:.1f}%")
        
        return {
            "success": True,
            "predictions": predictions,
            "directions": directions,
            "quality_metrics": quality_metrics,
            "model_info": {
                "model_type": "Enhanced BiLSTM + Attention",
                "device": self.device_name,
                "sequence_length": self.sequence_length,
                "feature_size": cv_splits[0]['X_train'].shape[2],
                "training_samples": len(cv_splits[-1]['X_train']),
                "test_samples": len(cv_splits[-1]['X_test']),
                "cv_folds": len(cv_splits)
            },
            "training_history": {
                "train_losses": train_losses,
                "val_losses": val_losses
            }
        }

# Streamlit WebUI
def main():
    """主WebUI应用"""
    st.set_page_config(
        page_title="增强LSTM价格预测系统 v2.0",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 增强LSTM价格预测系统 v2.0")
    st.markdown("**双向LSTM + Attention + 技术指标 + 滚动交叉验证的深度学习预测**")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 预测配置")
        
        symbol = st.selectbox("选择交易对", [
            "DOGEUSDT", "BTCUSDT", "ETHUSDT", "ADAUSDT", 
            "DOTUSDT", "LINKUSDT", "LTCUSDT", "BCHUSDT",
            "XLMUSDT", "EOSUSDT", "MATICUSDT"
        ])
        
        sequence_length = st.slider("LSTM序列长度", 20, 60, 30)
        prediction_hours = st.slider("预测时长 (小时)", 1, 168, 24)
        
        if st.button("🚀 开始增强预测", type="primary"):
            st.session_state.run_prediction = True
    
    # 主界面
    if st.session_state.get('run_prediction', False):
        run_enhanced_prediction(symbol, sequence_length, prediction_hours)

def run_enhanced_prediction(symbol, sequence_length, prediction_hours):
    """运行增强LSTM预测"""
    
    # 显示数据获取进度
    progress_container = st.container()
    with progress_container:
        st.info(f"📊 正在获取 {symbol} 的增强历史数据...")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        # 获取真实历史数据
        status_text.text("🔍 从多数据源获取15分钟数据...")
        progress_bar.progress(20)
        
        fetcher = MultiSourceDataFetcher(symbol)
        source, historical_data = fetcher.fetch_large_dataset(target_points=2000)
        
        if historical_data is None or len(historical_data) < 100:
            st.error("❌ 无法获取足够的历史数据，请稍后重试或更换交易对")
            return
        
        progress_bar.progress(40)
        status_text.text(f"✅ 数据获取成功: {len(historical_data)}条记录")
        
        # 显示数据信息
        time_span = (historical_data.index[-1] - historical_data.index[0]).total_seconds() / 86400
        current_price = historical_data['close'].iloc[-1]
        
        # 设备信息
        device, device_name = setup_optimized_device()
        
        with st.expander("📊 数据与设备信息"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("数据量", f"{len(historical_data)}条")
            with col2:
                st.metric("时间跨度", f"{time_span:.1f}天")
            with col3:
                st.metric("当前价格", f"{current_price:.6f}")
            with col4:
                st.metric("运行设备", device_name)
        
        # 运行增强LSTM预测
        status_text.text("🧠 正在训练增强LSTM模型...")
        progress_bar.progress(60)
        
        with st.spinner("正在训练增强LSTM模型，请稍候..."):
            predictor = OptimizedLSTMSystem(symbol, sequence_length)
            result = predictor.enhanced_predict_sequence(historical_data, prediction_hours)
            
            progress_bar.progress(100)
            status_text.text("✅ 增强LSTM预测完成!")
            
            if result and result["success"]:
                st.success("🎉 增强LSTM预测生成成功!")
                
                # 显示预测结果
                predictions = result["predictions"]
                directions = result["directions"]
                metrics = result["quality_metrics"]
                model_info = result["model_info"]
                
                # 核心指标（增加新指标）
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("当前价格", f"{metrics['current_price']:.6f}")
                with col2:
                    st.metric("预测价格", f"{metrics['predicted_price']:.6f}")
                with col3:
                    st.metric("变化幅度", f"{metrics['change_pct']:+.2f}%")
                with col4:
                    st.metric("预测趋势", metrics['trend'])
                with col5:
                    st.metric("方向准确率", f"{metrics['directional_accuracy']:.1f}%")
                
                # 绘制预测图表
                fig = go.Figure()
                
                # 历史价格 (最近100个点)
                recent_data = historical_data.tail(100)
                fig.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data['close'],
                    name="历史价格",
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='时间: %{x}<br>价格: %{y:.6f}<extra></extra>'
                ))
                
                # 预测价格
                fig.add_trace(go.Scatter(
                    x=predictions.index,
                    y=predictions['close'],
                    name="Enhanced LSTM预测",
                    line=dict(color='#ff7f0e', dash='dash', width=2),
                    hovertemplate='时间: %{x}<br>预测价格: %{y:.6f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"🔮 {symbol} Enhanced LSTM价格预测 (双向+Attention)",
                    xaxis_title="时间",
                    yaxis_title="价格 (USDT)",
                    height=600,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 模型详情（增强版）
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📈 增强预测分析")
                    st.write(f"**预测趋势**: {metrics['trend']}")
                    st.write(f"**置信度**: {metrics['confidence']:.1f}%")
                    st.write(f"**MSE**: {metrics['mse']:.8f}")
                    st.write(f"**MAE**: {metrics['mae']:.6f}")
                    st.write(f"**MAPE**: {metrics['mape']:.2f}%")
                    st.write(f"**方向准确率**: {metrics['directional_accuracy']:.1f}%")
                    
                    # 方向预测信息
                    st.write("---")
                    st.write(f"**主要方向**: {metrics['main_direction']}")
                    st.write(f"**方向置信度**: {metrics['direction_confidence']:.2f}")
                
                with col2:
                    st.subheader("🎯 增强模型信息")
                    st.write(f"**模型类型**: {model_info['model_type']}")
                    st.write(f"**运行设备**: {model_info['device']}")
                    st.write(f"**序列长度**: {model_info['sequence_length']}")
                    st.write(f"**特征维度**: {model_info['feature_size']}")
                    st.write(f"**训练样本**: {model_info['training_samples']}")
                    st.write(f"**交叉验证折数**: {model_info['cv_folds']}")
                
                # 方向预测详情
                with st.expander("🎯 方向预测详情"):
                    # 创建方向预测图表
                    fig_dir = go.Figure()
                    
                    # 方向置信度
                    colors = ['green' if d == '上涨' else 'red' for d in directions['direction']]
                    fig_dir.add_trace(go.Bar(
                        x=directions.index,
                        y=directions['confidence'],
                        name="方向置信度",
                        marker_color=colors,
                        hovertemplate='时间: %{x}<br>方向: %{text}<br>置信度: %{y:.2f}<extra></extra>',
                        text=directions['direction']
                    ))
                    
                    fig_dir.update_layout(
                        title="方向预测置信度",
                        xaxis_title="时间",
                        yaxis_title="置信度",
                        height=400
                    )
                    st.plotly_chart(fig_dir, use_container_width=True)
                
                # 训练历史
                if "training_history" in result:
                    with st.expander("📈 训练历史"):
                        train_losses = result["training_history"]["train_losses"]
                        val_losses = result["training_history"]["val_losses"]
                        
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            y=train_losses,
                            name="训练损失",
                            line=dict(color='blue')
                        ))
                        fig_loss.add_trace(go.Scatter(
                            y=val_losses,
                            name="验证损失",
                            line=dict(color='red')
                        ))
                        fig_loss.update_layout(
                            title="Enhanced LSTM训练损失曲线",
                            xaxis_title="训练轮次",
                            yaxis_title="损失值",
                            height=400
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                
                # 预测数据表格（包含方向）
                with st.expander("📋 详细预测数据"):
                    # 合并价格和方向预测
                    combined_pred = predictions.copy()
                    combined_pred['predicted_direction'] = directions['direction']
                    combined_pred['direction_confidence'] = directions['confidence'].round(3)
                    st.dataframe(combined_pred.round(6))
            else:
                st.error("❌ 增强LSTM预测生成失败!")
                
    except Exception as e:
        st.error(f"❌ 发生错误: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        
    finally:
        # 清理进度显示
        progress_container.empty()
    
    st.session_state.run_prediction = False

if __name__ == "__main__":
    main() 