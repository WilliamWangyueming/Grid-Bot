# 🤖 网格交易机器人 (Grid Trading Bot)

## 📋 功能特性

- **真实价格模拟交易**: 使用真实 API 价格数据进行模拟网格交易
- **自动网格管理**: 根据价格自动调整买卖单
- **实时监控**: 实时显示交易状态、收益和挂单情况
- **风险控制**: 支持止损、追踪止盈等功能
- **多币种支持**: 支持各种 USDT 交易对

## 🚀 快速开始

### 1. 环境要求
```bash
Python 3.8+
streamlit
pandas
numpy
requests
```

### 2. 配置设置
编辑 `config.yaml` 文件：
```yaml
symbol: "DOGEUSDT"      # 交易对
lower_price: 0.18       # 网格下限
upper_price: 0.26       # 网格上限  
grids: 30              # 网格数量
mode: "geometric"       # 网格模式 (geometric/arithmetic)
qty: 50                # 每格数量
fee_rate: 0.001        # 手续费率
trailing_k: 3          # 追踪参数
```

### 3. 启动网格交易
```bash
streamlit run real_price_app.py
```

## 📁 文件说明

- **`grid_config.py`** - 网格配置和计算逻辑
- **`real_price_grid_engine.py`** - 网格交易引擎核心
- **`real_price_simulation.py`** - 真实价格模拟交易
- **`real_price_app.py`** - Web 界面应用
- **`api_wrapper.py`** - API 接口封装
- **`config.yaml`** - 配置文件

## ⚙️ 使用说明

### 网格交易策略
1. 在价格区间内设置多个网格
2. 在低价格处挂买单，高价格处挂卖单
3. 价格波动时自动成交获利
4. 支持几何和算术网格模式

### 风险提示
- 仅为模拟交易，使用虚拟资金
- 实际交易需要谨慎评估风险
- 市场波动可能导致损失

## 🔧 技术架构

```
网格交易机器人
├── 配置模块 (grid_config.py)
├── 交易引擎 (real_price_grid_engine.py)  
├── 价格模拟 (real_price_simulation.py)
├── API 接口 (api_wrapper.py)
└── Web 界面 (real_price_app.py)
```

## 📊 监控指标

- **实时收益**: 总盈亏和年化收益率
- **交易对数**: 完成的买卖对数量  
- **挂单状态**: 当前网格挂单情况
- **价格位置**: 当前价格在网格中的位置

## 🎯 优化建议

1. **网格范围**: 根据历史波动率设置合理范围
2. **网格密度**: 平衡交易频率和单次收益
3. **资金管理**: 合理分配资金避免过度集中
4. **参数调整**: 根据市场状况动态调整参数 