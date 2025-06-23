# real_price_app.py – 使用真实价格的模拟交易GUI界面
import streamlit as st
import time
import threading
import pandas as pd
from decimal import Decimal
from streamlit_autorefresh import st_autorefresh
from grid_config import GridConfig
from real_price_grid_engine import RealPriceGridEngine
import plotly.graph_objects as go

# ===== K线加载与绘图（必须放在最前，供侧边栏使用） =====
@st.cache_data(ttl=60, show_spinner=False)
def load_klines(symbol: str, interval: str = "1h", limit: int = 200):
    """从 MEXC 获取 K 线数据并返回 DataFrame，如遇无效周期自动降级。"""
    from api_wrapper import MexcRest
    api = MexcRest(symbol)

    tried = []
    for iv in [interval, "60m", "30m", "15m", "1m"]:
        if iv in tried:
            continue
        tried.append(iv)
        try:
            raw = api.get_klines(iv, limit)
            if raw:
                interval = iv  # 使用成功的周期
                break
        except Exception as e:
            # 可能是无效 interval，继续尝试
            raw = []
    if not raw:
        return pd.DataFrame()

    # 动态列名适配（MEXC 部分市场仅返回 8 列）
    col_8  = ["openTime", "open", "high", "low", "close", "volume", "closeTime", "quoteAssetVolume"]
    col_12 = col_8 + ["numTrades", "takerBaseVol", "takerQuoteVol", "ignore"]
    cols   = col_12 if len(raw[0]) == 12 else col_8

    df = pd.DataFrame(raw, columns=cols)
    df["openTime"] = pd.to_datetime(df["openTime"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df.attrs["interval"] = interval
    return df


def render_kline_with_grid(container, symbol: str, cfg: GridConfig):
    """绘制带网格线的 K 线图"""
    df = load_klines(symbol)
    if df.empty:
        container.error("无法获取K线数据")
        return
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["openTime"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Kline",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            )
        ]
    )
    for p in cfg.grid_lines:
        fig.add_shape(type="line", x0=df["openTime"].iloc[0], x1=df["openTime"].iloc[-1], y0=float(p), y1=float(p), line=dict(color="rgba(0,0,255,0.2)", width=1, dash="dot"), layer="below")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=300, showlegend=False, xaxis_rangeslider_visible=False)
    container.plotly_chart(fig, use_container_width=True)

# ===========================================================
# UI 渲染函数 —— 必须放在最前，供后续逻辑调用
# ===========================================================

def _render_overview_tab(tab, stats: dict, symbol: str, runtime: float):
    col1, col2, col3, col4, col5, col6 = tab.columns(6)
    with col1:
        tab.metric("当前价格", f"{stats['current_price']:.6f}")
    with col2:
        tab.metric("总价值(USDT)", f"{stats['total_value']:.2f}", f"{stats['total_value'] - 10000:+.2f}")
    with col3:
        pct = ((stats['total_value'] / 10000) - 1) * 100
        tab.metric("收益率", f"{pct:.4f}%", f"{stats['roi']:.2%} 年化")
    with col4:
        tab.metric("完成对数", stats['pairs'])
    with col5:
        tab.metric("挂单数", stats['open'])
    with col6:
        tab.metric("当前网格", stats.get('current_grid', 'N/A'))

def _render_orders_tab(tab, engine):
    orders = engine.get_open_orders()
    if not orders:
        tab.info("暂无挂单")
        return
    data = []
    for oid, (idx, side) in orders.items():
        valid = 0 <= idx <= engine.cfg.N
        price = float(engine.cfg.grid_price(idx)) if valid else 0
        data.append({
            "订单ID": oid,
            "网格": idx if valid else f"{idx}(越界)",
            "方向": side,
            "价格": f"{price:.6f}" if valid else "错误",
            "数量": f"{engine.cfg.qty:.6f}",
            "价值": f"{price*float(engine.cfg.qty):.2f}" if valid else "错误",
        })
    def _clr(r):
        return ["background-color:#90EE90" if r["方向"]=="BUY" else "background-color:#FFB6C1"]*len(r)
    tab.dataframe(pd.DataFrame(data).style.apply(_clr, axis=1), use_container_width=True, height=380)

def _render_grid_tab(tab, engine, stats):
    cur_price = stats['current_price']
    cur_idx = engine.cfg.price_to_grid_index(cur_price)
    if cur_idx < 0 or cur_idx >= engine.cfg.N:
        tab.warning("当前价格超出网格范围！")
        return
    orders = engine.get_open_orders()
    rows = []
    for i in range(engine.cfg.N):
        has_buy = any(idx == i and s == 'BUY' for idx, s in orders.values())
        has_sell = any(idx == i and s == 'SELL' for idx, s in orders.values())
        rows.append({
            "网格": i,
            "买入价": f"{float(engine.cfg.get_buy_price(i)):.6f}",
            "卖出价": f"{float(engine.cfg.get_sell_price(i)):.6f}",
            "利润/格": f"{float(engine.cfg.profit_per_grid(i)):.4f}",
            "状态": ("🟢买单 " if has_buy else "") + ("🔴卖单" if has_sell else "") or "⚪空闲",
            "当前位置": "🎯" if i == cur_idx else "",
        })
    df = pd.DataFrame(rows)
    tab.dataframe(df, use_container_width=True, height=450)

def _render_trades_tab(tab, engine):
    trades = engine.get_trade_history()
    if not trades:
        tab.info("暂无交易记录")
        return
    recent = trades[-10:]
    rows = [{
        "时间": time.strftime("%H:%M:%S", time.localtime(t.get('timestamp', time.time()))),
        "方向": t['side'],
        "价格": f"{float(t['price']):.6f}",
        "数量": f"{float(t['quantity']):.6f}",
        "网格": t.get('grid_index', 'N/A'),
    } for t in recent]
    df = pd.DataFrame(rows)
    tab.dataframe(df, use_container_width=True, height=300)

def render_running_ui(engine, stats, symbol, runtime):
    t1, t2, t3, t4 = st.tabs(["📊 概览", "📋 挂单", "🎯 网格", "📈 交易记录"])
    _render_overview_tab(t1, stats, symbol, runtime)
    _render_orders_tab(t2, engine)
    _render_grid_tab(t3, engine, stats)
    _render_trades_tab(t4, engine)

# ===========================================================
# 以上为 UI 函数
# ===========================================================

# 页面配置
st.set_page_config(
    page_title="真实价格网格交易模拟器",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自动刷新：仅在模拟运行时启用，并放宽到 5 秒，避免配置阶段的频繁刷新造成卡顿
if st.session_state.get("real_simulation_engine"):
    st_autorefresh(interval=5000, key="refresh")  # 5 秒刷新一次

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

    # 网格模式 & 手续费率（提前定义，供后续利润检查）
    mode = st.selectbox("网格模式", ["geometric", "arithmetic"], disabled=running)
    fee_rate = st.number_input("手续费率(%)", value=0.05, min_value=0.0, step=0.01, disabled=running) / 100

    # ==== 新增：根据手续费检查单格利润是否为正，并给出建议 ====
    def recommend_grids(current_n:int):
        for n in range(current_n, 4, -1):
            test_cfg = GridConfig(
                symbol=symbol,
                lower_price=lower_price,
                upper_price=upper_price,
                grids=n,
                mode=mode,
                fee_rate=fee_rate,
                qty=1  # 数量设1，利润正负与数量无关
            )
            if min(test_cfg.profits) > 0:
                return n
        return 5

    test_cfg = GridConfig(symbol, lower_price, upper_price, grids, mode, fee_rate, qty=1)
    min_profit = float(min(test_cfg.profits))
    if min_profit <= 0:
        rec_n = recommend_grids(grids)
        st.error(f"❌ 当前网格过密，最小单格净利润 {min_profit:.4f} USDT ≤ 0，建议将网格数量降低至 ≤ {rec_n}")
    else:
        st.info(f"✅ 最小单格净利润 {min_profit:.4f} USDT (已扣手续费)")

    # ==== 原密度检查保留 ====
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
    
    st.markdown("---")
    st.info("💡 网格范围已根据当前价格自动建议，你可以自定义修改")
    
    # 显示价格范围状态
    current_price = st.session_state.get('current_price', 0.2)
    if current_price < lower_price or current_price > upper_price:
        st.warning("⚠️ 当前价格超出网格范围，建议调整!")
    else:
        st.success("✅ 当前价格在网格范围内")

    # ===== K 线预览 =====
    st.markdown("---")
    st.markdown("### 🕯️ 价格走势图 & 网格预览")
    preview_cfg = GridConfig(
        symbol=symbol,
        lower_price=lower_price,
        upper_price=upper_price,
        grids=grids,
        mode=mode,
        fee_rate=fee_rate,
        qty=10.0,
    )
    render_kline_with_grid(st, symbol, preview_cfg)

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
    # 立即刷新，避免等待自动刷新才能看到运行界面
    st.experimental_rerun()

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
        runtime = time.time() - st.session_state.get("real_start_time", time.time())
        # 计算当前所在网格索引
        stats['current_grid'] = engine.cfg.price_to_grid_index(stats['current_price'])

        # === 新 UI 布局 ===
        render_running_ui(engine, stats, symbol, runtime)
        st.stop()
        
    except Exception as e:
        st.error(f"获取状态时出错: {e}")
        st.exception(e)

else:
    # 未运行时显示精简说明
    with st.expander("ℹ️ 使用说明（点击展开）", expanded=False):
        st.markdown("""
        **核心特点**
        * 真实价格 · 模拟成交 · 安全无风险
        * 网格策略与实盘一致，可快速验证参数
        
        **快速开始**
        1. 左侧输入区间 & 网格数
        2. 查看K线+网格预览确认
        3. 点击 **开始模拟**
        
        **提示**  
        - 绿色提示代表参数合理  
        - 红色提示表示需要调整
        """)

# ===========================================================
# ⭐ UI 辅助函数（提前定义，保证调用时已存在）
# ===========================================================

# ------------------------------------------------------------
# 以上函数定义完毕
# ------------------------------------------------------------

# 页脚
st.markdown("---")
st.markdown("*真实价格网格交易模拟器 - 使用真实价格数据的安全模拟环境*") 