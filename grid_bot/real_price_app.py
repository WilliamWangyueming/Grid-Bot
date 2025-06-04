# real_price_app.py – 使用真实价格的模拟交易GUI界面
import streamlit as st
import time
import threading
import pandas as pd
from decimal import Decimal
from streamlit_autorefresh import st_autorefresh
from grid_config import GridConfig
from real_price_grid_engine import RealPriceGridEngine

# 页面配置
st.set_page_config(
    page_title="真实价格网格交易模拟器",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自动刷新
st_autorefresh(interval=3000, key="refresh")  # 3秒刷新一次

st.title("📈 真实价格网格交易模拟器")
st.markdown("*使用真实API价格数据进行网格交易模拟，1,000,000 USDT虚拟资金*")

# 侧边栏配置
st.sidebar.header("📊 网格配置")

# 检查是否有运行中的模拟
running = st.session_state.get("real_simulation_engine") is not None

# 获取当前价格并根据币种自动调整网格范围
def get_auto_grid_range(symbol):
    """根据币种和当前价格自动计算合理的网格范围"""
    try:
        from api_wrapper import MexcRest
        temp_api = MexcRest(symbol)
        current_price = float(temp_api.get_price())
        
        # 根据价格范围确定合理的网格宽度
        if current_price >= 50000:  # BTC类 (>50k)
            grid_width = 0.15  # 15%宽度
            lower = current_price * (1 - grid_width)
            upper = current_price * (1 + grid_width)
        elif current_price >= 1000:  # ETH类 (1k-50k)
            grid_width = 0.20  # 20%宽度
            lower = current_price * (1 - grid_width)
            upper = current_price * (1 + grid_width)
        elif current_price >= 1:  # 主流币 (1-1000)
            grid_width = 0.25  # 25%宽度
            lower = current_price * (1 - grid_width)
            upper = current_price * (1 + grid_width)
        else:  # 小币 (<1)
            grid_width = 0.30  # 30%宽度
            lower = current_price * (1 - grid_width)
            upper = current_price * (1 + grid_width)
        
        return current_price, lower, upper
    except Exception as e:
        st.sidebar.error(f"获取价格失败: {e}")
        return 0.2, 0.18, 0.26

with st.sidebar:
    symbol = st.selectbox("交易对", ["ALEOUSDT", "BTCUSDT", "ETHUSDT"], disabled=running)
    
    # 检查是否切换了币种，只有切换时才更新建议价格
    if "last_symbol" not in st.session_state or st.session_state.last_symbol != symbol:
        current_real_price, suggested_lower, suggested_upper = get_auto_grid_range(symbol)
        st.session_state.last_symbol = symbol
        st.session_state.suggested_lower = suggested_lower
        st.session_state.suggested_upper = suggested_upper
        st.session_state.current_price = current_real_price
        
        # 只在切换币种时更新默认值
        if "user_lower_price" not in st.session_state:
            st.session_state.user_lower_price = suggested_lower
        if "user_upper_price" not in st.session_state:
            st.session_state.user_upper_price = suggested_upper
    else:
        # 只更新当前价格，不更新建议价格
        try:
            from api_wrapper import MexcRest
            temp_api = MexcRest(symbol)
            st.session_state.current_price = float(temp_api.get_price())
        except:
            pass
    
    # 显示当前价格
    st.metric("当前真实价格", f"{st.session_state.get('current_price', 0.2):.6f}")
    
    # 价格范围输入，使用session state保存用户输入
    col1, col2 = st.columns(2)
    with col1:
        # 使用回调函数保存用户输入
        def update_lower_price():
            st.session_state.user_lower_price = st.session_state.lower_price_input
            
        lower_price = st.number_input(
            "下界价格", 
            value=st.session_state.get('user_lower_price', st.session_state.get('suggested_lower', 0.18)), 
            format="%.6f", 
            disabled=running,
            help=f"建议值: {st.session_state.get('suggested_lower', 0.18):.6f}",
            key="lower_price_input",
            on_change=update_lower_price
        )
        
    with col2:
        def update_upper_price():
            st.session_state.user_upper_price = st.session_state.upper_price_input
            
        upper_price = st.number_input(
            "上界价格", 
            value=st.session_state.get('user_upper_price', st.session_state.get('suggested_upper', 0.26)), 
            format="%.6f", 
            disabled=running,
            help=f"建议值: {st.session_state.get('suggested_upper', 0.26):.6f}",
            key="upper_price_input",
            on_change=update_upper_price
        )
    
    # 添加重置按钮
    if st.button("🔄 使用建议价格", disabled=running):
        st.session_state.user_lower_price = st.session_state.get('suggested_lower', 0.18)
        st.session_state.user_upper_price = st.session_state.get('suggested_upper', 0.26)
        st.rerun()
    
    grids = st.number_input("网格数量", value=20, min_value=5, max_value=300, disabled=running)
    
    # 检查网格密度并给出建议
    if 'current_price' in st.session_state:
        current_price = st.session_state.get('current_price', 0.2)
        suggested_lower = st.session_state.get('suggested_lower', 0.18)
        suggested_upper = st.session_state.get('suggested_upper', 0.26)
        
        price_range = suggested_upper - suggested_lower
        grid_spacing = price_range / grids
        spacing_percent = (grid_spacing / current_price) * 100
        
        if spacing_percent < 0.1:  # 网格间距小于0.1%
            st.warning(f"⚠️ 网格过密！当前间距仅{spacing_percent:.3f}%，建议减少到{max(5, int(price_range / current_price * 1000))}个网格")
        elif spacing_percent < 0.5:  # 网格间距小于0.5%
            st.info(f"💡 网格较密，当前间距{spacing_percent:.2f}%，可能产生较多小额订单")
        else:
            st.success(f"✅ 网格间距合理：{spacing_percent:.2f}%")
    
    mode = st.selectbox("网格模式", ["geometric", "arithmetic"], disabled=running)
    fee_rate = st.number_input("手续费率(%)", value=0.05, min_value=0.0, step=0.01, disabled=running) / 100
    
    st.markdown("---")
    st.info("💡 网格范围已根据当前价格自动建议，你可以自定义修改")
    
    # 显示价格范围状态
    current_price = st.session_state.get('current_price', 0.2)
    if current_price < lower_price or current_price > upper_price:
        st.warning("⚠️ 当前价格超出网格范围，建议调整!")
    else:
        st.success("✅ 当前价格在网格范围内")

# 控制按钮
col1, col2, col3 = st.columns(3)

def start_real_simulation():
    """启动真实价格模拟"""
    cfg = GridConfig(
        symbol=symbol,
        lower_price=lower_price,
        upper_price=upper_price,
        grids=grids,
        mode=mode,
        fee_rate=fee_rate,
        qty=10.0  # 会在bootstrap中重新计算
    )
    
    engine = RealPriceGridEngine(
        cfg=cfg,
        invest_usdt=Decimal("10000"),
        required_base=Decimal("0")
    )
    
    # 启动
    engine.bootstrap()
    st.session_state["real_simulation_engine"] = engine
    st.session_state["real_start_time"] = time.time()
    st.success("🚀 真实价格模拟交易已启动！")

def stop_real_simulation():
    """停止真实价格模拟"""
    if "real_simulation_engine" in st.session_state:
        engine = st.session_state["real_simulation_engine"]
        engine.stop()
        del st.session_state["real_simulation_engine"]
        if "real_start_time" in st.session_state:
            del st.session_state["real_start_time"]
        st.success("⏹️ 真实价格模拟交易已停止！")

def reset_real_simulation():
    """重置真实价格模拟"""
    stop_real_simulation()
    # 清理状态文件
    import os
    for file in ["real_price_grid_state.json", "real_price_simulation_state.json", "real_price_simulation_balance.pkl"]:
        if os.path.exists(file):
            os.remove(file)
    st.success("🔄 真实价格模拟状态已重置！")

with col1:
    if st.button("🚀 开始模拟", disabled=running):
        start_real_simulation()

with col2:
    if st.button("⏹️ 停止模拟", disabled=not running):
        stop_real_simulation()

with col3:
    if st.button("🔄 重置状态", disabled=running):
        reset_real_simulation()

# 显示状态
if running:
    engine = st.session_state.get("real_simulation_engine")
    
    if engine is None:
        st.error("模拟引擎丢失，请重新启动")
        if "real_simulation_engine" in st.session_state:
            del st.session_state["real_simulation_engine"]
        if "real_start_time" in st.session_state:
            del st.session_state["real_start_time"]
        st.rerun()
    
    try:
        stats = engine.get_detailed_stats()
        
        # 运行时间
        runtime = time.time() - st.session_state.get("real_start_time", time.time())
        
        # 主要指标
        st.header("📈 实时数据（真实价格）")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "当前价格", 
                f"{stats['current_price']:.6f}",
                delta="真实价格"
            )
        
        with col2:
            st.metric(
                "总价值(USDT)", 
                f"{stats['total_value']:.2f}",
                f"{stats['total_value'] - 10000:+.2f}"
            )
        
        with col3:
            profit_pct = ((stats['total_value'] / 10000) - 1) * 100
            st.metric(
                "收益率", 
                f"{profit_pct:.4f}%",
                f"{stats['roi']:.2%} (年化)"
            )
        
        with col4:
            st.metric(
                "完成交易对", 
                stats['pairs'],
                f"{stats['total_trades']} 总交易"
            )
        
        with col5:
            st.metric(
                "运行时间", 
                f"{runtime/60:.1f} 分钟",
                f"{stats['trades_per_hour']:.1f} 交易/小时"
            )
        
        # 详细余额
        st.header("💰 模拟账户余额")
        balance_col1, balance_col2, balance_col3 = st.columns(3)
        
        with balance_col1:
            st.metric("USDT余额", f"{stats['usdt_balance']:.2f}")
        
        with balance_col2:
            base_asset = symbol.replace("USDT", "")
            st.metric(f"{base_asset}余额", f"{stats['base_balance']:.6f}")
        
        with balance_col3:
            st.metric("开放订单", stats['open'])
        
        # 当前挂单可视化
        st.header("📋 当前挂单")
        open_orders = engine.get_open_orders()
        if open_orders:
            # 转换为DataFrame显示
            orders_data = []
            for order_id, (idx, side) in open_orders.items():
                try:
                    # 检查网格索引是否在有效范围内
                    if 0 <= idx <= engine.cfg.N:
                        grid_price = float(engine.cfg.grid_price(idx))
                        orders_data.append({
                            "订单ID": order_id,
                            "网格索引": idx,
                            "方向": side,
                            "价格": f"{grid_price:.6f}",
                            "数量": f"{engine.cfg.qty:.6f}",
                            "价值(USDT)": f"{grid_price * float(engine.cfg.qty):.2f}"
                        })
                    else:
                        # 网格索引超出范围，显示错误信息
                        orders_data.append({
                            "订单ID": order_id,
                            "网格索引": f"{idx} (越界!)",
                            "方向": side,
                            "价格": "错误",
                            "数量": "错误",
                            "价值(USDT)": "错误"
                        })
                        print(f"[ERROR] 订单 {order_id} 的网格索引 {idx} 超出范围 [0, {engine.cfg.N}]")
                except Exception as e:
                    print(f"[ERROR] 处理订单 {order_id} 时出错: {e}")
                    orders_data.append({
                        "订单ID": order_id,
                        "网格索引": f"{idx} (错误)",
                        "方向": side,
                        "价格": "错误",
                        "数量": "错误",
                        "价值(USDT)": "错误"
                    })
            
            if orders_data:
                orders_df = pd.DataFrame(orders_data)
                
                # 高亮显示买单和卖单
                def color_orders(row):
                    if row['方向'] == 'BUY':
                        return ['background-color: #90EE90'] * len(row)  # 浅绿色
                    else:
                        return ['background-color: #FFB6C1'] * len(row)  # 浅红色
                
                st.dataframe(
                    orders_df.style.apply(color_orders, axis=1),
                    use_container_width=True
                )
                
                # 检查是否有越界订单
                invalid_orders = [order for order in orders_data if "越界" in str(order["网格索引"]) or "错误" in str(order["网格索引"])]
                if invalid_orders:
                    st.error(f"⚠️ 发现 {len(invalid_orders)} 个无效订单，建议重置状态！")
                    st.info("💡 点击'重置状态'按钮清理无效订单")
                else:
                    st.info(f"💡 当前有 {len(orders_data)} 个挂单等待成交（绿色=买单，粉色=卖单）")
            else:
                st.warning("⚠️ 当前没有有效挂单！")
        else:
            st.warning("⚠️ 当前没有挂单！这可能是以下原因：")
            st.write("1. 模拟刚启动，订单正在设置中")
            st.write("2. 价格超出网格范围")
            st.write("3. 余额不足无法下单")
            st.write("4. 订单检查逻辑需要调试")
        
        # 收益分析
        st.header("📊 收益分析")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.metric("已实现收益", f"{stats['pnl']:.4f} USDT")
            st.metric("未实现收益", f"{stats['unreal']:.4f} USDT")
        
        with analysis_col2:
            if stats['pairs'] > 0:
                avg_profit = stats['pnl'] / stats['pairs']
                st.metric("平均每对收益", f"{avg_profit:.4f} USDT")
            else:
                st.metric("平均每对收益", "0.0000 USDT")
            
            if stats['total_trades'] > 0:
                avg_trade_value = stats['total_value'] / stats['total_trades']
                st.metric("平均交易价值", f"{avg_trade_value:.2f} USDT")
        
        # 网格状态表格
        st.header("🎯 网格状态")
        
        try:
            current_price = stats['current_price']
            current_idx = engine.cfg.price_to_grid_index(current_price)
            
            # 检查当前价格索引是否有效
            if current_idx < 0 or current_idx >= engine.cfg.N:
                st.error(f"⚠️ 当前价格 {current_price:.6f} 超出网格范围，无法显示网格状态")
                st.info(f"当前网格范围: {engine.cfg.lo} - {engine.cfg.hi}")
                st.info("建议调整网格价格范围以包含当前价格")
            else:
                # 构建网格表格数据
                grid_data = []
                for i in range(engine.cfg.N):
                    try:
                        buy_price = float(engine.cfg.grid_price(i))
                        sell_price = float(engine.cfg.grid_price(i + 1))
                        profit = float(engine.cfg.profit_per_grid(i))
                        
                        # 检查是否有订单在这个网格
                        has_buy_order = any(idx == i and side == "BUY" for idx, side in open_orders.values() if 0 <= idx <= engine.cfg.N)
                        has_sell_order = any(idx == i + 1 and side == "SELL" for idx, side in open_orders.values() if 0 <= idx <= engine.cfg.N)
                        
                        status = ""
                        if has_buy_order:
                            status += "🟢买单 "
                        if has_sell_order:
                            status += "🔴卖单 "
                        if not status:
                            status = "⚪空闲"
                        
                        grid_data.append({
                            "网格": i,
                            "买入价": f"{buy_price:.6f}",
                            "卖出价": f"{sell_price:.6f}",
                            "利润": f"{profit:.4f}",
                            "状态": status,
                            "当前位置": "🎯" if i == current_idx else ""
                        })
                    except Exception as e:
                        print(f"[ERROR] 处理网格 {i} 时出错: {e}")
                        continue
                
                if grid_data:
                    grid_df = pd.DataFrame(grid_data)
                    
                    # 高亮当前价格所在网格
                    def highlight_current_grid(s):
                        current_grid = s.name == current_idx
                        if current_grid:
                            return ['background-color: yellow'] * len(s)
                        else:
                            return [''] * len(s)
                    
                    st.dataframe(
                        grid_df.style.apply(highlight_current_grid, axis=1),
                        use_container_width=True,
                        height=400
                    )
                    
                    st.info(f"💡 当前价格 {current_price:.6f} 位于网格 {current_idx}（黄色高亮）")
                else:
                    st.error("无法显示网格数据")
            
        except Exception as e:
            st.error(f"网格状态显示错误: {e}")
            st.info("建议重置状态以清理无效数据")
            
        # 最近交易记录
        st.header("📈 最近交易记录")
        try:
            trade_history = engine.get_trade_history()
            if trade_history:
                recent_trades = trade_history[-10:]  # 显示最近10笔交易
                trades_data = []
                for trade in recent_trades:
                    trades_data.append({
                        "时间": time.strftime("%H:%M:%S", time.localtime(trade.get('timestamp', time.time()))),
                        "方向": trade['side'],
                        "价格": f"{float(trade['price']):.6f}",
                        "数量": f"{float(trade['quantity']):.6f}",
                        "网格": trade.get('grid_index', 'N/A')
                    })
                
                trades_df = pd.DataFrame(trades_data)
                st.dataframe(trades_df, use_container_width=True)
            else:
                st.info("还没有交易记录")
        except Exception as e:
            st.warning(f"无法显示交易记录: {e}")
            
    except Exception as e:
        st.error(f"获取状态时出错: {e}")
        st.exception(e)

else:
    # 未运行时显示说明
    st.header("📝 使用说明")
    
    st.markdown("""
    ### 🎯 核心特点
    - **真实价格数据**：使用您的MEXC API获取实时价格
    - **模拟交易执行**：所有订单都是虚拟的，不会真正下单
    - **1,000,000 USDT虚拟资金**：大额资金模拟环境
    - **完整网格策略**：与真实交易使用相同的算法
    
    ### 🚀 快速开始
    1. 在左侧配置网格参数
    2. 确保价格区间合理覆盖当前价格
    3. 点击"开始模拟"启动
    4. 实时观察真实价格变化和模拟交易执行
    
    ### ⚙️ 参数建议
    - **价格区间**：建议以当前价格为中心，上下各留20-30%空间
    - **网格数量**：建议15-30个，平衡收益和交易频率
    - **网格模式**：等比模式适合波动较大的币种
    
    ### 💡 优势对比
    
    | 特性 | 真实价格模拟 🎯 | 完全虚拟模拟 |
    |------|----------------|-------------|
    | 价格数据 | ✅ 真实价格 | 🔄 模拟价格 |
    | 交易执行 | 🔄 模拟订单 | 🔄 模拟订单 |
    | 网络需求 | ✅ 需要 | ❌ 不需要 |
    | 真实性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
    | 学习价值 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
    
    ### ⚠️ 重要提醒
    - 这是模拟交易，不会产生真实的盈亏
    - 所有订单都是虚拟的，不会影响您的实际账户
    - 建议先在此模拟环境中验证策略效果
    """)
    
    # 显示历史记录
    import os
    if os.path.exists("real_price_grid_state.json"):
        st.info("💾 检测到历史模拟数据，开始新模拟将从上次状态恢复")

# 页脚
st.markdown("---")
st.markdown("*真实价格网格交易模拟器 - 使用真实价格数据的安全模拟环境*") 