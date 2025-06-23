"""
精简版LSTM价格预测系统
- 核心LSTM模型
- 基础数据处理
- 简化的Web界面
"""

# 解决PyTorch与Streamlit的兼容性问题
import sys
import os
import importlib.util

# 在导入torch之前设置环境变量
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '0'

# 禁用Streamlit的文件监视器对torch模块的检查
if 'torch' not in sys.modules:
    # 预加载torch以避免Streamlit的模块路径检查问题
    try:
        import torch
        # 修复torch.classes的__path__问题
        if hasattr(torch, 'classes'):
            if not hasattr(torch.classes, '__path__'):
                torch.classes.__path__ = []
    except ImportError:
        pass

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
from multi_source_fetcher import MultiSourceDataFetcher

warnings.filterwarnings('ignore')

class SimpleLSTMPredictor(nn.Module):
    """简化的LSTM预测模型"""
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super(SimpleLSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)  # 只预测收盘价
        
    def forward(self, x):
        batch_size = x.size(0)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        
        return out

class SimpleLSTMSystem:
    """简化的LSTM价格预测系统"""
    
    def __init__(self, symbol="DOGEUSDT", sequence_length=30):
        self.symbol = symbol
        self.sequence_length = sequence_length
        # 自动检测 GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: SimpleLSTMPredictor | None = None
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, data):
        """准备训练数据"""
        # 只使用OHLCV基础数据
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        data = data[feature_columns].copy()
        
        # 标准化数据
        scaled_data = self.scaler.fit_transform(data)
        
        # 创建序列数据
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 3])  # 收盘价
            
        return np.array(X), np.array(y)
    
    def train_model(self, X, y, epochs=50, learning_rate=0.001):
        """训练模型"""
        # 分割训练和测试数据
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 转换为PyTorch张量并移动到 device
        X_train = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=self.device).unsqueeze(1)
        X_test = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_test = torch.tensor(y_test, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # 初始化模型
        self.model = SimpleLSTMPredictor(input_size=X.shape[2]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练
        train_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
        
        # 评估
        self.model.eval()
        with torch.no_grad():
            test_pred = self.model(X_test)
            test_loss = criterion(test_pred, y_test).item()
        
        return train_losses, test_loss
    
    def predict_future(self, data, prediction_hours=24):
        """预测未来价格"""
        if self.model is None:
            return None
            
        # 准备最后一段数据
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        last_data = data[feature_columns].tail(self.sequence_length)
        scaled_data = self.scaler.transform(last_data)
        
        # 预测
        predictions = []
        current_seq = scaled_data.copy()
        
        self.model.eval()
        with torch.no_grad():
            for i in range(prediction_hours):
                X = torch.tensor(current_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
                pred = self.model(X).cpu()
                predictions.append(pred.item())
                
                # 更新序列 - 更智能的方式
                next_row = current_seq[-1].copy()
                
                # 考虑价格的合理变动范围，避免过度预测
                if i < prediction_hours - 1:
                    # 限制单步预测的变化幅度
                    pred_value = pred.item()
                    last_pred = predictions[-2] if len(predictions) > 1 else current_seq[-1][3]
                    max_change = 0.1  # 限制单步最大变化为10%
                    
                    if abs(pred_value - last_pred) / abs(last_pred) > max_change:
                        if pred_value > last_pred:
                            pred_value = last_pred * (1 + max_change)
                        else:
                            pred_value = last_pred * (1 - max_change)
                        predictions[-1] = pred_value
                    
                    next_row[3] = pred_value  # 更新收盘价
                else:
                    next_row[3] = pred.item()
                
                # 更新其他价格字段（简单估算）
                current_close = next_row[3]
                next_row[0] = current_close * (0.999 + np.random.uniform(-0.001, 0.001))  # open
                next_row[1] = current_close * (1.0 + abs(np.random.uniform(0, 0.002)))   # high
                next_row[2] = current_close * (1.0 - abs(np.random.uniform(0, 0.002)))   # low
                
                current_seq = np.vstack([current_seq[1:], next_row])
        
        # 反标准化预测结果
        dummy_data = np.zeros((len(predictions), 5))
        dummy_data[:, 3] = predictions
        predictions_unscaled = self.scaler.inverse_transform(dummy_data)[:, 3]
        
        # 创建时间索引
        last_time = data.index[-1]
        time_index = pd.date_range(
            start=last_time + timedelta(minutes=15),
            periods=prediction_hours,
            freq='15T'
        )
        
        return pd.DataFrame({
            'predicted_close': predictions_unscaled
        }, index=time_index)

def main():
    """主WebUI应用"""
    st.set_page_config(
        page_title="LSTM价格预测系统",
        page_icon="🔮",
        layout="wide"
    )
    
    st.title("🔮 LSTM价格预测系统")
    st.markdown("**基于LSTM神经网络的加密货币价格预测**")
    
    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 预测配置")
        
        symbol = st.selectbox("选择交易对", [
            "DOGEUSDT", "BTCUSDT", "ETHUSDT", "ADAUSDT"
        ])
        
        sequence_length = st.slider("序列长度", 20, 60, 30)
        prediction_hours = st.slider("预测时长 (小时)", 1, 72, 24)
        epochs = st.slider("训练轮数", 20, 200, 50)
        target_points = st.number_input("历史K线数量", 500, 10000, 5000, step=500)
        
        if st.button("🚀 开始预测", type="primary"):
            st.session_state.run_prediction = True
            st.session_state.target_points = int(target_points)
    
    # 主界面
    if st.session_state.get('run_prediction', False):
        run_prediction(symbol, sequence_length, prediction_hours, epochs, st.session_state.get('target_points', 5000))

def run_prediction(symbol, sequence_length, prediction_hours, epochs, target_points):
    """运行预测"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # 获取数据
        status_text.text("📊 正在获取历史数据...")
        progress_bar.progress(20)
        
        fetcher = MultiSourceDataFetcher(symbol)
        source, historical_data = fetcher.fetch_large_dataset(target_points=target_points)
        
        if historical_data is None or len(historical_data) < 100:
            st.error("❌ 无法获取足够的历史数据")
            return
            
        # 验证数据质量
        if historical_data['close'].std() < 0.000001:
            st.error("❌ 数据质量问题：价格变化太小，可能是无效数据")
            return
            
        # 检查是否有异常值
        price_range = historical_data['close'].max() - historical_data['close'].min()
        if price_range < historical_data['close'].mean() * 0.01:
            st.warning("⚠️ 警告：价格变化范围很小，预测结果可能不准确")
        
        progress_bar.progress(40)
        current_price = historical_data['close'].iloc[-1]
        
        # 显示数据信息和基本统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("数据量", f"{len(historical_data)}条")
        with col2:
            st.metric("当前价格", f"{current_price:.6f}")
        with col3:
            st.metric("数据源", source)
        
        # 显示数据基本信息（调试用）
        st.write("📊 数据概览:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最低价", f"{historical_data['close'].min():.6f}")
        with col2:
            st.metric("最高价", f"{historical_data['close'].max():.6f}")
        with col3:
            st.metric("价格标准差", f"{historical_data['close'].std():.6f}")
        with col4:
            st.metric("数据时间跨度", f"{(historical_data.index[-1] - historical_data.index[0]).days}天")
        
        # 训练模型
        status_text.text("🧠 正在训练LSTM模型...")
        progress_bar.progress(60)
        
        predictor = SimpleLSTMSystem(symbol, sequence_length)
        X, y = predictor.prepare_data(historical_data)
        
        train_losses, test_loss = predictor.train_model(X, y, epochs)
        
        progress_bar.progress(80)
        
        # 预测
        status_text.text("🔮 正在生成预测...")
        predictions = predictor.predict_future(historical_data, prediction_hours)
        
        progress_bar.progress(100)
        status_text.text("✅ 预测完成!")
        
        if predictions is not None:
            predicted_price = predictions['predicted_close'].iloc[-1]
            change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # 显示预测结果
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("预测价格", f"{predicted_price:.6f}")
            with col2:
                st.metric("变化幅度", f"{change_pct:+.2f}%")
            with col3:
                st.metric("测试损失", f"{test_loss:.6f}")
            
            # 绘制图表
            fig = go.Figure()
            
            # 获取更多历史数据以显示更好的趋势
            recent_data = historical_data.tail(200)  # 增加数据点
            
            # 确保索引是datetime格式
            if not isinstance(recent_data.index, pd.DatetimeIndex):
                recent_data.index = pd.to_datetime(recent_data.index)
            if not isinstance(predictions.index, pd.DatetimeIndex):
                predictions.index = pd.to_datetime(predictions.index)
            
            # 历史价格
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['close'],
                name="历史价格",
                line=dict(color='blue', width=2),
                mode='lines'
            ))
            
            # 预测价格
            fig.add_trace(go.Scatter(
                x=predictions.index,
                y=predictions['predicted_close'],
                name="预测价格",
                line=dict(color='red', dash='dash', width=2),
                mode='lines'
            ))
            
            # 添加当前价格连接线
            connection_x = [recent_data.index[-1], predictions.index[0]]
            connection_y = [recent_data['close'].iloc[-1], predictions['predicted_close'].iloc[0]]
            fig.add_trace(go.Scatter(
                x=connection_x,
                y=connection_y,
                mode='lines',
                line=dict(color='orange', width=2, dash='dot'),
                name="连接线",
                showlegend=False
            ))
            
            fig.update_layout(
                title=f"{symbol} LSTM价格预测",
                xaxis_title="时间",
                yaxis_title="价格 (USDT)",
                height=600,
                xaxis=dict(
                    type='date',
                    tickformat='%H:%M<br>%m/%d'
                ),
                yaxis=dict(
                    tickformat='.6f'
                ),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示预测数据的统计信息
            with st.expander("🔍 预测详情"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**历史数据（最近20条）:**")
                    st.dataframe(recent_data[['close']].tail(20), height=200)
                with col2:
                    st.write("**预测数据:**")
                    pred_df = predictions.copy()
                    pred_df['time'] = pred_df.index.strftime('%H:%M %m/%d')
                    st.dataframe(pred_df[['predicted_close', 'time']].head(24), height=200)
            
            # 训练历史
            with st.expander("📈 训练损失"):
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    y=train_losses,
                    name="训练损失",
                    line=dict(color='blue')
                ))
                fig_loss.update_layout(
                    title="LSTM训练损失曲线",
                    xaxis_title="轮数",
                    yaxis_title="损失值"
                )
                st.plotly_chart(fig_loss, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ 预测过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main() 