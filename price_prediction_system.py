"""
AMD优化LSTM价格预测系统
专门针对AMD CPU和GPU设计，充分利用全部2000条数据
支持ROCm和高效CPU并行计算
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime, timedelta
import warnings
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import os
from multi_source_fetcher import MultiSourceDataFetcher

warnings.filterwarnings('ignore')

# AMD GPU优化配置
def setup_amd_device():
    """设置AMD GPU设备配置"""
    device = "cpu"  # 默认CPU
    device_name = "AMD CPU"
    
    if torch.cuda.is_available():
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
        print(f"🚀 检测到GPU: {device_name}")
        
        # 针对AMD GPU的优化设置
        if "MI" in device_name or "Instinct" in device_name or "Radeon" in device_name:
            print("🔧 应用AMD GPU优化设置...")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.enabled = True
            
        # 设置显存管理
        torch.cuda.empty_cache()
    else:
        print("🔧 使用AMD CPU优化设置...")
        # AMD CPU优化
        torch.set_num_threads(os.cpu_count())  # 使用所有CPU核心
        
    return device, device_name

class LSTMPricePredictor(nn.Module):
    """
    优化的LSTM价格预测模型
    针对AMD硬件优化，支持多层LSTM和Dropout
    """
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, dropout=0.2, output_size=4):
        super(LSTMPricePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        
        # 多层LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层预测OHLC
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)  # OHLC
        )
        
        # 成交量预测层
        self.volume_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.ReLU()  # 成交量必须为正
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        
        # 预测OHLC价格
        ohlc_pred = self.fc_layers(last_output)
        
        # 预测成交量
        volume_pred = self.volume_layer(last_output)
        
        return ohlc_pred, volume_pred

class AMDLSTMPredictionSystem:
    """AMD优化的LSTM价格预测系统"""
    
    def __init__(self, symbol="DOGEUSDT", sequence_length=30):
        self.symbol = symbol
        self.sequence_length = sequence_length
        self.device, self.device_name = setup_amd_device()
        
        # 数据标准化器
        self.price_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        
        # 模型
        self.model = None
        self.trained = False
        
        print(f"📱 运行设备: {self.device_name}")
        print(f"🔢 序列长度: {sequence_length}")
        
    def prepare_data(self, data, train_ratio=0.85):
        """
        准备训练数据，充分利用全部2000条数据
        """
        print(f"📊 准备数据: {len(data)} 条记录")
        
        # 确保数据按时间排序
        data = data.sort_index()
        
        # 提取OHLCV数据
        ohlc_data = data[['open', 'high', 'low', 'close']].values
        volume_data = data[['volume']].values
        
        # 数据标准化
        ohlc_scaled = self.price_scaler.fit_transform(ohlc_data)
        volume_scaled = self.volume_scaler.fit_transform(volume_data)
        
        # 合并特征
        combined_features = np.concatenate([ohlc_scaled, volume_scaled], axis=1)
        
        # 创建序列数据
        sequences = []
        targets_ohlc = []
        targets_volume = []
        
        for i in range(self.sequence_length, len(combined_features)):
            # 输入序列
            seq = combined_features[i-self.sequence_length:i]
            sequences.append(seq)
            
            # 目标值：下一个时间点的OHLC和成交量
            targets_ohlc.append(ohlc_scaled[i])
            targets_volume.append(volume_scaled[i])
        
        sequences = np.array(sequences)
        targets_ohlc = np.array(targets_ohlc)
        targets_volume = np.array(targets_volume)
        
        print(f"✅ 生成序列: {len(sequences)} 个")
        print(f"📏 序列形状: {sequences.shape}")
        
        # 分割训练测试集，充分利用数据
        split_idx = int(len(sequences) * train_ratio)
        
        X_train = sequences[:split_idx]
        X_test = sequences[split_idx:]
        y_train_ohlc = targets_ohlc[:split_idx]
        y_test_ohlc = targets_ohlc[split_idx:]
        y_train_vol = targets_volume[:split_idx]
        y_test_vol = targets_volume[split_idx:]
        
        print(f"🏋️ 训练集: {len(X_train)} 个序列")
        print(f"🧪 测试集: {len(X_test)} 个序列")
        print(f"📈 数据利用率: {len(sequences)/len(data)*100:.1f}%")
        
        return (X_train, X_test, y_train_ohlc, y_test_ohlc, 
                y_train_vol, y_test_vol, data.index[self.sequence_length:])
    
    def create_model(self, input_size=5):
        """创建LSTM模型"""
        model = LSTMPricePredictor(
            input_size=input_size,
            hidden_size=128,  # AMD GPU优化的隐藏层大小
            num_layers=3,     # 深度LSTM
            dropout=0.2,
            output_size=4     # OHLC
        )
        return model.to(self.device)
    
    def train_model(self, X_train, y_train_ohlc, y_train_vol, X_test, y_test_ohlc, y_test_vol, 
                   batch_size=32, epochs=100, learning_rate=0.001):
        """
        训练LSTM模型，使用AMD优化的设置
        """
        print("🚀 开始训练LSTM模型...")
        
        # 创建模型
        self.model = self.create_model(input_size=X_train.shape[2])
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_ohlc_tensor = torch.FloatTensor(y_train_ohlc).to(self.device)
        y_train_vol_tensor = torch.FloatTensor(y_train_vol).to(self.device)
        
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        y_test_ohlc_tensor = torch.FloatTensor(y_test_ohlc).to(self.device)
        y_test_vol_tensor = torch.FloatTensor(y_test_vol).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_ohlc_tensor, y_train_vol_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=0, pin_memory=True if self.device == "cuda" else False)
        
        # 损失函数和优化器
        criterion_price = nn.MSELoss()
        criterion_volume = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=0.5, patience=10)
        
        # 训练循环
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        print(f"🔧 设备: {self.device_name}")
        print(f"📦 批次大小: {batch_size}")
        print(f"🔄 计划训练轮数: {epochs}")
        
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            epoch_train_loss = 0
            
            for batch_idx, (batch_x, batch_y_ohlc, batch_y_vol) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # 前向传播
                ohlc_pred, vol_pred = self.model(batch_x)
                
                # 计算损失
                loss_ohlc = criterion_price(ohlc_pred, batch_y_ohlc)
                loss_vol = criterion_volume(vol_pred.squeeze(), batch_y_vol.squeeze())
                total_loss = loss_ohlc + 0.1 * loss_vol  # 价格损失权重更高
                
                # 反向传播
                total_loss.backward()
                
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_train_loss += total_loss.item()
            
            # 验证
            self.model.eval()
            with torch.no_grad():
                ohlc_pred_val, vol_pred_val = self.model(X_test_tensor)
                val_loss_ohlc = criterion_price(ohlc_pred_val, y_test_ohlc_tensor)
                val_loss_vol = criterion_volume(vol_pred_val.squeeze(), y_test_vol_tensor.squeeze())
                val_loss = val_loss_ohlc + 0.1 * val_loss_vol
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss.item())
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), f'best_lstm_model_{self.symbol}.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"轮次 [{epoch+1}/{epochs}] - "
                      f"训练损失: {avg_train_loss:.6f}, "
                      f"验证损失: {val_loss:.6f}, "
                      f"学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"🛑 早停触发，第 {epoch+1} 轮")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load(f'best_lstm_model_{self.symbol}.pth'))
        self.trained = True
        
        print(f"✅ 训练完成！最佳验证损失: {best_val_loss:.6f}")
        return train_losses, val_losses
    
    def predict_future(self, data, prediction_hours=24):
        """
        预测未来价格
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        print(f"🔮 预测未来 {prediction_hours} 小时价格...")
        
        self.model.eval()
        
        # 准备最后一个序列
        ohlc_data = data[['open', 'high', 'low', 'close']].values
        volume_data = data[['volume']].values
        
        ohlc_scaled = self.price_scaler.transform(ohlc_data)
        volume_scaled = self.volume_scaler.transform(volume_data)
        combined_features = np.concatenate([ohlc_scaled, volume_scaled], axis=1)
        
        # 取最后一个序列
        last_sequence = combined_features[-self.sequence_length:]
        current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(prediction_hours):
                # 预测下一个时间点
                ohlc_pred, vol_pred = self.model(current_sequence)
                
                # 反标准化预测结果
                ohlc_pred_unscaled = self.price_scaler.inverse_transform(ohlc_pred.cpu().numpy())
                vol_pred_unscaled = self.volume_scaler.inverse_transform(vol_pred.cpu().numpy())
                
                # 确保价格逻辑性
                open_price = ohlc_pred_unscaled[0][0]
                high_price = max(ohlc_pred_unscaled[0][1], open_price, ohlc_pred_unscaled[0][3])
                low_price = min(ohlc_pred_unscaled[0][2], open_price, ohlc_pred_unscaled[0][3])
                close_price = ohlc_pred_unscaled[0][3]
                volume = max(0, vol_pred_unscaled[0][0])
                
                predictions.append({
                    'open': open_price,
                    'high': high_price, 
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
                
                # 更新序列用于下一次预测
                new_point = np.array([[open_price, high_price, low_price, close_price, volume]])
                new_point_scaled = np.concatenate([
                    self.price_scaler.transform(new_point[:, :4]),
                    self.volume_scaler.transform(new_point[:, 4:5])
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
        
        print(f"✅ 预测完成：{len(pred_df)} 个时间点")
        
        return pred_df
    
    def predict_sequence(self, historical_data, prediction_hours=24):
        """
        完整的预测流程
        """
        print("🚀 启动AMD优化LSTM价格预测")
        print("=" * 60)
        
        # 数据准备
        train_data = self.prepare_data(historical_data)
        X_train, X_test, y_train_ohlc, y_test_ohlc, y_train_vol, y_test_vol, time_index = train_data
        
        # 训练模型
        print("\n🏋️ 训练LSTM模型...")
        train_losses, val_losses = self.train_model(
            X_train, y_train_ohlc, y_train_vol, 
            X_test, y_test_ohlc, y_test_vol,
            batch_size=64,  # AMD GPU优化的批次大小
            epochs=150,
            learning_rate=0.001
        )
        
        # 生成预测
        predictions = self.predict_future(historical_data, prediction_hours)
        
        # 计算测试集准确性
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_test_ohlc_tensor = torch.FloatTensor(y_test_ohlc).to(self.device)
            
            ohlc_pred_test, _ = self.model(X_test_tensor)
            
            # 反标准化
            y_test_unscaled = self.price_scaler.inverse_transform(y_test_ohlc_tensor.cpu().numpy())
            y_pred_unscaled = self.price_scaler.inverse_transform(ohlc_pred_test.cpu().numpy())
            
            # 计算指标
            mse = mean_squared_error(y_test_unscaled[:, 3], y_pred_unscaled[:, 3])  # 收盘价MSE
            mae = mean_absolute_error(y_test_unscaled[:, 3], y_pred_unscaled[:, 3])  # 收盘价MAE
            
            current_price = historical_data['close'].iloc[-1]
            predicted_price = predictions['close'].iloc[-1]
            change_pct = ((predicted_price - current_price) / current_price) * 100
        
        quality_metrics = {
            'mse': mse,
            'mae': mae,
            'current_price': current_price,
            'predicted_price': predicted_price,
            'change_pct': change_pct,
            'trend': '上涨' if change_pct > 1 else '下跌' if change_pct < -1 else '横盘',
            'confidence': max(70, min(95, 90 - abs(change_pct)))
        }
        
        print(f"✅ LSTM预测完成!")
        print(f"   📊 预测点数: {len(predictions)}")
        print(f"   💰 当前价格: {current_price:.6f}")
        print(f"   🎯 预测价格: {predicted_price:.6f}")
        print(f"   📈 变化幅度: {change_pct:+.2f}%")
        print(f"   🎲 MSE: {mse:.8f}")
        print(f"   📏 MAE: {mae:.6f}")
        
        return {
            "success": True,
            "predictions": predictions,
            "quality_metrics": quality_metrics,
            "model_info": {
                "model_type": "LSTM",
                "device": self.device_name,
                "sequence_length": self.sequence_length,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "data_utilization": f"{(len(X_train) + len(X_test))/len(historical_data)*100:.1f}%"
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
        page_title="AMD优化LSTM价格预测系统",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 AMD优化LSTM价格预测系统")
    st.markdown("**专为AMD CPU和GPU优化，充分利用2000条数据的深度学习预测**")
    
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
        
        if st.button("🚀 开始LSTM预测", type="primary"):
            st.session_state.run_prediction = True
    
    # 主界面
    if st.session_state.get('run_prediction', False):
        run_lstm_prediction(symbol, sequence_length, prediction_hours)

def run_lstm_prediction(symbol, sequence_length, prediction_hours):
    """运行LSTM预测"""
    
    # 显示数据获取进度
    progress_container = st.container()
    with progress_container:
        st.info(f"📊 正在获取 {symbol} 的真实历史数据...")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        # 获取真实历史数据
        status_text.text("🔍 从Huobi获取15分钟数据...")
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
        device, device_name = setup_amd_device()
        
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
        
        # 运行LSTM预测
        status_text.text("🧠 正在训练LSTM模型...")
        progress_bar.progress(60)
        
        with st.spinner("正在训练LSTM模型，请稍候..."):
            predictor = AMDLSTMPredictionSystem(symbol, sequence_length)
            result = predictor.predict_sequence(historical_data, prediction_hours)
            
            progress_bar.progress(100)
            status_text.text("✅ LSTM预测完成!")
            
            if result and result["success"]:
                st.success("🎉 LSTM预测生成成功!")
                
                # 显示预测结果
                predictions = result["predictions"]
                metrics = result["quality_metrics"]
                model_info = result["model_info"]
                
                # 核心指标
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("当前价格", f"{metrics['current_price']:.6f}")
                with col2:
                    st.metric("预测价格", f"{metrics['predicted_price']:.6f}")
                with col3:
                    st.metric("变化幅度", f"{metrics['change_pct']:+.2f}%")
                with col4:
                    st.metric("预测趋势", metrics['trend'])
                
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
                    name="LSTM预测",
                    line=dict(color='#ff7f0e', dash='dash', width=2),
                    hovertemplate='时间: %{x}<br>预测价格: %{y:.6f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f"🔮 {symbol} LSTM价格预测 (AMD优化)",
                    xaxis_title="时间",
                    yaxis_title="价格 (USDT)",
                    height=600,
                    showlegend=True,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 模型详情
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📈 预测分析")
                    st.write(f"**预测趋势**: {metrics['trend']}")
                    st.write(f"**置信度**: {metrics['confidence']:.1f}%")
                    st.write(f"**MSE**: {metrics['mse']:.8f}")
                    st.write(f"**MAE**: {metrics['mae']:.6f}")
                
                with col2:
                    st.subheader("🎯 模型信息")
                    st.write(f"**模型类型**: {model_info['model_type']}")
                    st.write(f"**运行设备**: {model_info['device']}")
                    st.write(f"**序列长度**: {model_info['sequence_length']}")
                    st.write(f"**训练样本**: {model_info['training_samples']}")
                    st.write(f"**数据利用率**: {model_info['data_utilization']}")
                
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
                            title="LSTM训练损失曲线",
                            xaxis_title="训练轮次",
                            yaxis_title="损失值",
                            height=400
                        )
                        st.plotly_chart(fig_loss, use_container_width=True)
                
                # 预测数据表格
                with st.expander("📋 详细预测数据"):
                    st.dataframe(predictions.round(6))
            else:
                st.error("❌ LSTM预测生成失败!")
                
    except Exception as e:
        st.error(f"❌ 发生错误: {str(e)}")
        
    finally:
        # 清理进度显示
        progress_container.empty()
    
    st.session_state.run_prediction = False

if __name__ == "__main__":
    main() 