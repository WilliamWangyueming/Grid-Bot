"""
real_price_grid_engine.py - 使用真实价格进行模拟交易的网格引擎
价格来自您的真实API，但交易是模拟的，使用1000 USDT虚拟资金
修复版本：修正网格交易逻辑错误
"""
import json, os, time, threading
from decimal import Decimal
from real_price_simulation import RealPriceSimulationWrapper, RealPriceSimulationWebSocket
from grid_config import GridConfig

REAL_GRID_STATE_FILE = "real_price_grid_state.json"

class RealPriceGridEngine:
    def __init__(self, cfg: GridConfig, invest_usdt: Decimal = Decimal("10000"), required_base: Decimal = Decimal("0")):
        self.cfg = cfg
        self.api = RealPriceSimulationWrapper(cfg.symbol, float(invest_usdt))
        self.invest_usdt = invest_usdt
        self.req_base = required_base

        self.open_orders = {}          # orderId -> (grid_idx, side)
        self.pnl_real = Decimal(0)
        self.pairs = 0
        self.start_ts = time.time()
        self.lock = threading.Lock()

        self._load_state()
        self.ws = RealPriceSimulationWebSocket(self.api, self._on_deal)
        self.ws.run()
        threading.Thread(target=self._autosave, daemon=True).start()

    # ---------------- 启动 - 修复版 ----------------
    def bootstrap(self):
        if self.open_orders:            # 恢复挂单
            print("[REAL_SIM_BOOT] 恢复挂单成功"); return

        current_price = self.api.get_price()
        print(f"[REAL_SIM_BOOT] 当前真实价格: {current_price}")
        
        # 修复：使用新的网格索引方法
        current_grid_idx = self.cfg.price_to_grid_index(float(current_price))
        print(f"[REAL_SIM_BOOT] 当前价格所在网格: {current_grid_idx}")
        print(f"[REAL_SIM_BOOT] 网格范围: {self.cfg.lo} - {self.cfg.hi}")
        
        # 检查当前价格是否在网格范围内
        if current_price < self.cfg.lo or current_price >= self.cfg.hi:
            print(f"[REAL_SIM_BOOT] 警告: 当前价格 {current_price} 超出网格范围 [{self.cfg.lo}, {self.cfg.hi}]")
            print("[REAL_SIM_BOOT] 建议调整网格范围以包含当前价格")
            return
        
        # 根据投资金额和网格数量计算每格的USDT投入
        grid_n = self.cfg.N
        usdt_per_grid = max(self.invest_usdt / grid_n, Decimal("100"))
        # 重新计算每格的数量（基于当前价格）
        self.cfg.qty = float(usdt_per_grid / current_price)
        print(f"[REAL_SIM_BOOT] 每网格投入: {usdt_per_grid:.2f} USDT")
        print(f"[REAL_SIM_BOOT] 每单数量: {self.cfg.qty:.6f}")

        have_u = self.api.balance_of("USDT")
        base_asset = self.cfg.symbol.replace("USDT", "")
        have_a = self.api.balance_of(base_asset)
        print(f"[REAL_SIM_BOOT] USDT余额: {have_u:.2f}")
        print(f"[REAL_SIM_BOOT] {base_asset}余额: {have_a:.6f}")
        
        if have_u < self.invest_usdt:
            print("[REAL_SIM_BOOT] USDT 不足"); return

        orders_placed = 0
        
        # 修复：使用正确的网格交易策略
        print(f"[REAL_SIM_BOOT] 开始设置网格订单...")
        
        # 限制订单数量避免过多订单
        max_orders_per_side = min(15, grid_n // 3)  # 每边最多15个订单
        
        # 1. 在当前网格及以下设置买单（需要USDT）
        buy_orders_placed = 0
        usdt_available = have_u
        
        # 从当前网格开始向下设置买单
        for i in range(max_orders_per_side):
            grid_idx = current_grid_idx - i
            if grid_idx < 0:  # 超出网格下界
                break
                
            buy_price = self.cfg.get_buy_price(grid_idx)
            required_usdt = buy_price * Decimal(self.cfg.qty)
            
            # 确保买单价格低于当前价格（或接近当前价格）
            if buy_price >= current_price and grid_idx != current_grid_idx:
                print(f"[REAL_SIM_BOOT] 跳过买单网格{grid_idx}，价格{buy_price:.6f}不低于当前价格{current_price:.6f}")
                continue
            
            if usdt_available >= required_usdt:
                self._place("BUY", buy_price, grid_idx)
                usdt_available -= required_usdt
                buy_orders_placed += 1
                orders_placed += 1
                print(f"[REAL_SIM_BOOT] 设置买单: 网格{grid_idx} @ {buy_price:.6f}")
            else:
                print(f"[REAL_SIM_BOOT] USDT余额不足，停止设置买单")
                break
        
        # 2. 在当前网格以上设置卖单（需要基础币）
        sell_orders_placed = 0
        
        if have_a > 0:  # 如果有基础币，可以设置卖单
            base_available = have_a
            
            # 从当前网格的上一个网格开始设置卖单
            for i in range(1, max_orders_per_side + 1):
                grid_idx = current_grid_idx + i
                if grid_idx >= self.cfg.N:  # 超出网格上界
                    break
                    
                sell_price = self.cfg.get_sell_price(grid_idx)
                required_base = Decimal(self.cfg.qty)
                
                # 确保卖单价格高于当前价格
                if sell_price <= current_price:
                    print(f"[REAL_SIM_BOOT] 跳过卖单网格{grid_idx}，价格{sell_price:.6f}不高于当前价格{current_price:.6f}")
                    continue
                
                if base_available >= required_base:
                    self._place("SELL", sell_price, grid_idx)
                    base_available -= required_base
                    sell_orders_placed += 1
                    orders_placed += 1
                    print(f"[REAL_SIM_BOOT] 设置卖单: 网格{grid_idx} @ {sell_price:.6f}")
                else:
                    print(f"[REAL_SIM_BOOT] {base_asset}余额不足，停止设置卖单")
                    break
        else:
            print(f"[REAL_SIM_BOOT] 无{base_asset}余额，仅设置买单")
        
        # 3. 如果只有USDT且订单数量不足，扩大买单范围
        if have_a == 0 and buy_orders_placed < 5:
            print(f"[REAL_SIM_BOOT] 扩大买单覆盖范围...")
            step = max(1, grid_n // 20)  # 每隔几个网格设置一个买单
            
            for i in range(step, min(grid_n//2, current_grid_idx), step):
                grid_idx = current_grid_idx - i
                if grid_idx < 0:
                    break
                    
                buy_price = self.cfg.get_buy_price(grid_idx)
                required_usdt = buy_price * Decimal(self.cfg.qty)
                
                if usdt_available >= required_usdt:
                    self._place("BUY", buy_price, grid_idx)
                    usdt_available -= required_usdt
                    orders_placed += 1
                    print(f"[REAL_SIM_BOOT] 扩展买单: 网格{grid_idx} @ {buy_price:.6f}")
                else:
                    break
        
        # 总结
        if orders_placed == 0:
            print("[REAL_SIM_BOOT] 警告：没有设置任何订单！")
            print("[REAL_SIM_BOOT] 可能原因：")
            print("  1. 网格范围不包含当前价格")
            print("  2. 余额不足")
            print("  3. 网格过于密集")
        else:
            print(f"[REAL_SIM_BOOT] 启动完成！")
            print(f"[REAL_SIM_BOOT] 总订单数: {orders_placed}")
            print(f"[REAL_SIM_BOOT] 买单数: {buy_orders_placed}")
            print(f"[REAL_SIM_BOOT] 卖单数: {sell_orders_placed}")
            print(f"[REAL_SIM_BOOT] 当前价格网格: {current_grid_idx}")
            
        # 强制保存状态
        self._save_state()

    # ---------------- 下单 - 修复版 ----------------
    def _place(self, side, price: Decimal, grid_idx: int):
        # 确保网格索引有效
        if not self.cfg.is_valid_grid_index(grid_idx):
            print(f"[REAL_SIM_ORDER] 网格索引 {grid_idx} 无效，跳过下单")
            return
            
        # 确保下单价值合理
        order_value = price * Decimal(self.cfg.qty)
        min_order_value = Decimal("50")  # 最小订单价值50 USDT
        
        if order_value < min_order_value:
            # 调整数量以满足最小订单价值
            self.cfg.qty = float(min_order_value / price)
            order_value = min_order_value
            print(f"[REAL_SIM_ORDER] 调整订单数量为 {self.cfg.qty:.6f} 以满足最小价值要求")
            
        print(f"[REAL_SIM_ORDER] 准备下单: {side} {self.cfg.qty:.6f} @ {price:.6f}, 价值: {order_value:.2f} USDT (网格 {grid_idx})")
        
        # 检查余额
        if side == "BUY":
            available_usdt = self.api.balance_of("USDT")
            if available_usdt < order_value:
                print(f"[REAL_SIM_ORDER] USDT余额不足: {available_usdt:.2f} < {order_value:.2f}")
                return
        else:  # SELL
            base_asset = self.cfg.symbol.replace("USDT", "")
            available_base = self.api.balance_of(base_asset)
            if available_base < Decimal(self.cfg.qty):
                print(f"[REAL_SIM_ORDER] {base_asset}余额不足: {available_base:.6f} < {self.cfg.qty:.6f}")
                return
        
        try:
            r = self.api.place_limit(side, float(price), self.cfg.qty, grid_idx)
            order_id = str(r["orderId"])
            self.open_orders[order_id] = (grid_idx, side)
            print(f"[REAL_SIM_ORDER] 订单成功: {order_id} - {side} {self.cfg.qty:.6f} @ {price:.6f} (网格 {grid_idx})")
            self._save_state()
        except Exception as e:
            print("[REAL_SIM_ORDER‑ERR]", e)

    # ---------------- 成交 - 修复版 ----------------
    def _on_deal(self, d: dict):
        with self.lock:
            oid = str(d["orderId"])
            if oid not in self.open_orders: 
                return
            
            grid_idx, old_side = self.open_orders.pop(oid)
            deal_side = d["side"]
            deal_price = Decimal(str(d["price"]))
            
            print(f"[REAL_SIM_TRADE] {deal_side} 成交: 网格{grid_idx} @ {deal_price:.6f}")
            
            # 计算这一网格的利润
            try:
                grid_profit = self.cfg.profit_per_grid(grid_idx)
                self.pnl_real += grid_profit
                
                if deal_side == "SELL": 
                    self.pairs += 1
                    
                print(f"[REAL_SIM_TRADE] 网格利润: {grid_profit:.6f} USDT，累计: {self.pnl_real:.6f} USDT，完成对数: {self.pairs}")
            except Exception as e:
                print(f"[REAL_SIM_TRADE] 利润计算错误: {e}")

            # 修复：下相反方向的订单，使用正确的网格逻辑
            try:
                if deal_side == "BUY":
                    # 买单成交，在同一网格设置卖单
                    sell_price = self.cfg.get_sell_price(grid_idx)
                    self._place("SELL", sell_price, grid_idx)
                    print(f"[REAL_SIM_TRADE] 买单成交后设置卖单: 网格{grid_idx} @ {sell_price:.6f}")
                    
                elif deal_side == "SELL":
                    # 卖单成交，在下方网格设置买单
                    lower_grid = self.cfg.get_adjacent_grid(grid_idx, "down")
                    if lower_grid >= 0:
                        buy_price = self.cfg.get_buy_price(lower_grid)
                        self._place("BUY", buy_price, lower_grid)
                        print(f"[REAL_SIM_TRADE] 卖单成交后设置买单: 网格{lower_grid} @ {buy_price:.6f}")
                    else:
                        print(f"[REAL_SIM_TRADE] 卖单成交但无法在更低网格设置买单（已达底部）")
                        
            except Exception as e:
                print(f"[REAL_SIM_TRADE] 设置反向订单失败: {e}")
            
            self._save_state()

    # ---------------- 状态文件 ----------------
    def _save_state(self):
        data = dict(
            cfg=self.cfg.__dict__,
            pnl=str(self.pnl_real),
            pairs=self.pairs,
            orders=self.open_orders,
            start_ts=self.start_ts,
            invest_usdt=str(self.invest_usdt),
            req_base=str(self.req_base)
        )
        try:
            with open(REAL_GRID_STATE_FILE, "w") as f:
                json.dump(data, f, default=str, indent=2)
            # 同时保存模拟API状态
            self.api.save_state()
        except Exception as e:
            print(f"[REAL_SIM_SAVE_ERR] {e}")

    def _load_state(self):
        if not os.path.isfile(REAL_GRID_STATE_FILE): 
            return
        try:
            with open(REAL_GRID_STATE_FILE, "r") as f:
                d = json.load(f)
            self.pnl_real = Decimal(d.get("pnl", "0"))
            self.pairs = d.get("pairs", 0)
            self.start_ts = d.get("start_ts", time.time())
            self.invest_usdt = Decimal(d.get("invest_usdt", "10000"))
            self.req_base = Decimal(d.get("req_base", "0"))
            self.open_orders = {k: tuple(v) for k, v in d.get("orders", {}).items()}
            print(f"[REAL_SIM_STATE] 恢复 {len(self.open_orders)} 个挂单")
        except Exception as e:
            print("[REAL_SIM_STATE‑ERR]", e)
            self.open_orders.clear()

    # ---------------- Autosave ----------------
    def _autosave(self):
        while True:
            time.sleep(30)
            self._save_state()

    # ---------------- 平仓 ----------------
    def close_all(self):
        self.api.cancel_all()
        base = self.cfg.symbol.replace("USDT", "")
        qty = self.api.balance_of(base)
        if qty > 0: 
            self.api.place_market_sell(qty)
        self.open_orders.clear()
        self._save_state()
        print("[REAL_SIM_CLOSE‑ALL] 完成")

    # ---------------- 指标 ----------------
    def status(self):
        days = max((time.time() - self.start_ts) / 86400, 1e-6)
        roi = (self.pnl_real / self.invest_usdt) / Decimal(days) * Decimal(365)
        
        # 获取模拟API的统计信息
        sim_stats = self.api.get_statistics()
        
        # 确保类型一致性
        sim_profit = Decimal(str(sim_stats['profit']))
        
        return dict(
            pnl=float(self.pnl_real),
            unreal=float(sim_profit - self.pnl_real),  # 未实现盈亏
            roi=float(roi),
            pairs=self.pairs,
            open=len(self.open_orders),
            total_value=sim_stats['total_value'],
            current_price=sim_stats['current_price'],
            total_trades=sim_stats['total_trades'],
            runtime_hours=sim_stats['runtime_hours'],
            trades_per_hour=sim_stats['trades_per_hour']
        )

    def get_detailed_stats(self):
        """获取详细的统计信息"""
        stats = self.status()
        
        # 获取模拟API的余额信息
        usdt_balance = self.api.balance_of("USDT")
        base_asset = self.cfg.symbol.replace("USDT", "")
        base_balance = self.api.balance_of(base_asset)
        
        stats.update({
            'usdt_balance': float(usdt_balance),
            'base_balance': float(base_balance)
        })
        
        return stats

    def get_open_orders(self):
        """获取当前挂单列表"""
        with self.lock:
            return dict(self.open_orders)
    
    def get_trade_history(self):
        """获取交易历史"""
        try:
            return self.api.get_trade_history()
        except AttributeError:
            # 如果API没有这个方法，返回空列表
            return []

    def stop(self):
        """停止网格引擎"""
        self.ws.stop()
        print("[REAL_SIM_STOP] 网格引擎已停止") 