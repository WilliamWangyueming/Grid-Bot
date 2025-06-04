"""
real_price_simulation.py - 使用真实价格进行模拟交易的引擎
使用您现有的API获取真实价格，但所有交易都是模拟的
"""
import json
import os
import time
import threading
from decimal import Decimal
from typing import Dict, List
import pickle
from api_wrapper import MexcRest
from grid_config import GridConfig

REAL_SIM_STATE_FILE = "real_price_simulation_state.json"
REAL_SIM_BALANCE_FILE = "real_price_simulation_balance.pkl"

class RealPriceSimulationState:
    """使用真实价格的模拟交易状态"""
    def __init__(self, initial_usdt: Decimal = Decimal("10000")):
        self.usdt_balance = initial_usdt
        self.base_balance = Decimal("0")
        self.open_orders: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        self.start_time = time.time()
        self.order_id_counter = 10000

class RealPriceSimulationAPI:
    """使用真实价格的模拟API包装器"""
    
    def __init__(self, symbol="ALEOUSDT", initial_usdt=10000):
        self.symbol = symbol.upper()
        self.base_asset = symbol.replace("USDT", "")
        
        # 使用您现有的真实API获取价格
        self.real_api = MexcRest(symbol)
        
        # 模拟交易状态
        self.sim_state = RealPriceSimulationState(Decimal(str(initial_usdt)))
        
        # 保存初始资金用于利润计算
        self.initial_usdt = Decimal(str(initial_usdt))
        
        # 线程锁
        self.lock = threading.Lock()
        
        # 订单检查线程
        self.running = True
        self.order_check_thread = threading.Thread(target=self._check_orders, daemon=True)
        self.order_check_thread.start()
        
        # 成交回调
        self._deal_callback = None
        
        # 加载之前的状态
        self.load_state()
    
    def _check_orders(self):
        """检查订单是否应该成交 - 修复版"""
        while self.running:
            try:
                current_price = self.get_price()
                orders_to_fill = []
                
                with self.lock:
                    for order_id, order in self.sim_state.open_orders.items():
                        order_price = Decimal(str(order['price']))
                        side = order['side']
                        
                        # 修复：使用更精确的成交条件
                        should_fill = False
                        if side == "BUY":
                            # 买单成交条件：当前价格 <= 买单价格（允许0.1%的滑点容忍）
                            tolerance = order_price * Decimal("0.001")  # 0.1%容忍度
                            if current_price <= (order_price + tolerance):
                                should_fill = True
                                fill_price = order_price  # 以订单价格成交
                        elif side == "SELL":
                            # 卖单成交条件：当前价格 >= 卖单价格（允许0.1%的滑点容忍）
                            tolerance = order_price * Decimal("0.001")  # 0.1%容忍度
                            if current_price >= (order_price - tolerance):
                                should_fill = True
                                fill_price = order_price  # 以订单价格成交
                        
                        if should_fill:
                            orders_to_fill.append((order_id, fill_price))
                            print(f"[ORDER_FILL] {side} 订单 {order_id} 触发成交")
                            print(f"[ORDER_FILL] 订单价格: {order_price:.6f}, 当前价格: {current_price:.6f}, 成交价格: {fill_price:.6f}")
                
                # 执行成交（限制每次成交数量避免过于频繁）
                for order_id, fill_price in orders_to_fill[:3]:  # 每次最多处理3个成交
                    self._fill_order(order_id, fill_price)
                
                # 显示当前状态（每30秒）
                if int(time.time()) % 30 == 0:
                    with self.lock:
                        if self.sim_state.open_orders:
                            print(f"[ORDER_STATUS] 当前价格: {current_price:.6f}, 挂单数: {len(self.sim_state.open_orders)}")
                            # 按价格排序显示订单
                            buy_orders = []
                            sell_orders = []
                            for order_id, order in self.sim_state.open_orders.items():
                                if order['side'] == 'BUY':
                                    buy_orders.append((order['price'], order_id, order.get('grid_index', '?')))
                                else:
                                    sell_orders.append((order['price'], order_id, order.get('grid_index', '?')))
                            
                            # 显示最高的3个买单
                            buy_orders.sort(reverse=True)
                            if buy_orders:
                                print("  最高买单:")
                                for price, oid, grid in buy_orders[:3]:
                                    print(f"    BUY @ {price:.6f} (网格{grid})")
                            
                            # 显示最低的3个卖单
                            sell_orders.sort()
                            if sell_orders:
                                print("  最低卖单:")
                                for price, oid, grid in sell_orders[:3]:
                                    print(f"    SELL @ {price:.6f} (网格{grid})")
                
                time.sleep(1)  # 每1秒检查一次（降低频率减少日志）
                
            except Exception as e:
                print(f"[ORDER_CHECK_ERROR] {e}")
                time.sleep(2)
    
    def _fill_order(self, order_id: str, fill_price: Decimal):
        """执行订单成交"""
        with self.lock:
            if order_id not in self.sim_state.open_orders:
                return
                
            order = self.sim_state.open_orders[order_id]
            side = order['side']
            qty = Decimal(str(order['quantity']))
            
            if side == "BUY":
                cost = fill_price * qty
                if self.sim_state.usdt_balance >= cost:
                    self.sim_state.usdt_balance -= cost
                    self.sim_state.base_balance += qty
                    
                    # 记录交易
                    trade = {
                        "orderId": order_id,
                        "side": side,
                        "price": str(fill_price),
                        "quantity": str(qty),
                        "timestamp": time.time(),
                        "grid_index": order.get('grid_index', 0)
                    }
                    self.sim_state.trade_history.append(trade)
                    
                    # 移除已成交订单
                    del self.sim_state.open_orders[order_id]
                    
                    print(f"[SIM_TRADE] BUY {qty} @ {fill_price}, USDT余额: {self.sim_state.usdt_balance:.2f}")
                    
                    # 触发回调
                    if self._deal_callback:
                        self._deal_callback(trade)
                        
            elif side == "SELL":
                if self.sim_state.base_balance >= qty:
                    self.sim_state.base_balance -= qty
                    revenue = fill_price * qty
                    self.sim_state.usdt_balance += revenue
                    
                    # 记录交易
                    trade = {
                        "orderId": order_id,
                        "side": side,
                        "price": str(fill_price),
                        "quantity": str(qty),
                        "timestamp": time.time(),
                        "grid_index": order.get('grid_index', 0)
                    }
                    self.sim_state.trade_history.append(trade)
                    
                    # 移除已成交订单
                    del self.sim_state.open_orders[order_id]
                    
                    print(f"[SIM_TRADE] SELL {qty} @ {fill_price}, USDT余额: {self.sim_state.usdt_balance:.2f}")
                    
                    # 触发回调
                    if self._deal_callback:
                        self._deal_callback(trade)
    
    # 模拟API接口，保持与真实API一致
    def get_price(self) -> Decimal:
        """获取真实价格"""
        return self.real_api.get_price()
    
    def balance_of(self, asset: str) -> Decimal:
        """获取模拟余额"""
        with self.lock:
            if asset == "USDT":
                return self.sim_state.usdt_balance
            elif asset == self.base_asset:
                return self.sim_state.base_balance
            else:
                return Decimal("0")
    
    def get_balances(self) -> Dict[str, Decimal]:
        """获取所有模拟余额"""
        with self.lock:
            return {
                "USDT": self.sim_state.usdt_balance,
                self.base_asset: self.sim_state.base_balance
            }
    
    def place_limit(self, side: str, price: float, qty: float, idx: int) -> Dict:
        """下模拟限价单"""
        with self.lock:
            order_id = str(self.sim_state.order_id_counter)
            self.sim_state.order_id_counter += 1
            
            order = {
                "orderId": order_id,
                "side": side,
                "price": price,
                "quantity": qty,
                "grid_index": idx,
                "timestamp": time.time()
            }
            
            self.sim_state.open_orders[order_id] = order
            print(f"[SIM_ORDER] {side} {qty} @ {price} (网格 {idx})")
            
            return {"orderId": order_id}
    
    def place_market_sell(self, qty: Decimal):
        """模拟市价卖出"""
        with self.lock:
            if self.sim_state.base_balance >= qty:
                current_price = self.get_price()
                self.sim_state.base_balance -= qty
                revenue = current_price * qty
                self.sim_state.usdt_balance += revenue
                print(f"[SIM_MARKET_SELL] {qty} @ {current_price}")
    
    def cancel_all(self):
        """取消所有模拟订单"""
        with self.lock:
            canceled_count = len(self.sim_state.open_orders)
            self.sim_state.open_orders.clear()
            print(f"[SIM_CANCEL_ALL] 取消了 {canceled_count} 个订单")
    
    def open_orders(self) -> List[Dict]:
        """获取开放的模拟订单"""
        with self.lock:
            return list(self.sim_state.open_orders.values())
    
    def start_user_stream(self):
        """模拟用户数据流"""
        return "simulation_stream"
    
    def set_deal_callback(self, callback):
        """设置成交回调"""
        self._deal_callback = callback
    
    def get_trade_history(self) -> List[Dict]:
        """获取交易历史"""
        with self.lock:
            return list(self.sim_state.trade_history)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        with self.lock:
            current_price = self.get_price()
            total_value = self.sim_state.usdt_balance + (self.sim_state.base_balance * current_price)
            runtime_hours = (time.time() - self.sim_state.start_time) / 3600
            profit = total_value - self.initial_usdt
            
            return {
                'current_price': float(current_price),
                'total_value': float(total_value),
                'profit': float(profit),
                'usdt_balance': float(self.sim_state.usdt_balance),
                'base_balance': float(self.sim_state.base_balance),
                'total_trades': len(self.sim_state.trade_history),
                'runtime_hours': runtime_hours,
                'trades_per_hour': len(self.sim_state.trade_history) / max(runtime_hours, 0.01)
            }
    
    def save_state(self):
        """保存模拟状态"""
        with self.lock:
            # 保存JSON格式的基本数据
            data = {
                "usdt_balance": str(self.sim_state.usdt_balance),
                "base_balance": str(self.sim_state.base_balance),
                "open_orders": self.sim_state.open_orders,
                "trade_history": self.sim_state.trade_history,
                "start_time": self.sim_state.start_time,
                "order_id_counter": self.sim_state.order_id_counter
            }
            
            try:
                with open(REAL_SIM_STATE_FILE, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # 也保存到pickle文件作为备份
                with open(REAL_SIM_BALANCE_FILE, 'wb') as f:
                    pickle.dump(self.sim_state, f)
                    
                print(f"[SIM_SAVE] 状态已保存")
            except Exception as e:
                print(f"[SIM_SAVE_ERROR] {e}")
    
    def load_state(self):
        """加载模拟状态"""
        if os.path.exists(REAL_SIM_STATE_FILE):
            try:
                with open(REAL_SIM_STATE_FILE, 'r') as f:
                    data = json.load(f)
                
                self.sim_state.usdt_balance = Decimal(data.get("usdt_balance", "10000"))
                self.sim_state.base_balance = Decimal(data.get("base_balance", "0"))
                self.sim_state.open_orders = data.get("open_orders", {})
                self.sim_state.trade_history = data.get("trade_history", [])
                self.sim_state.start_time = data.get("start_time", time.time())
                self.sim_state.order_id_counter = data.get("order_id_counter", 10000)
                
                print(f"[SIM_LOAD] 状态已恢复")
                print(f"[SIM_LOAD] USDT余额: {self.sim_state.usdt_balance}")
                print(f"[SIM_LOAD] {self.base_asset}余额: {self.sim_state.base_balance}")
                print(f"[SIM_LOAD] 开放订单: {len(self.sim_state.open_orders)}")
                
            except Exception as e:
                print(f"[SIM_LOAD_ERROR] 无法加载状态: {e}")
    
    def stop(self):
        """停止模拟"""
        self.running = False
        self.save_state()
        print("[SIM] 真实价格模拟已停止")

class RealPriceSimulationWrapper:
    """包装器，使其与原GridEngine兼容"""
    
    def __init__(self, symbol="ALEOUSDT", initial_usdt=10000):
        self.sim_api = RealPriceSimulationAPI(symbol, initial_usdt)
        
    def __getattr__(self, name):
        return getattr(self.sim_api, name)

class RealPriceSimulationWebSocket:
    """模拟WebSocket，用于兼容性"""
    
    def __init__(self, rest_api, on_deal):
        self.rest_api = rest_api
        self.on_deal = on_deal
        self.running = True
        
        # 设置成交回调
        if hasattr(rest_api, 'sim_api'):
            rest_api.sim_api.set_deal_callback(on_deal)
        elif hasattr(rest_api, 'set_deal_callback'):
            rest_api.set_deal_callback(on_deal)
        else:
            print("[WS] 警告：无法设置成交回调")
    
    def run(self):
        """启动WebSocket（实际是空操作，因为订单检查在API中进行）"""
        print("[WS] 模拟WebSocket已启动")
    
    def stop(self):
        """停止WebSocket"""
        self.running = False
        print("[WS] 模拟WebSocket已停止") 