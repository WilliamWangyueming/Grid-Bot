# grid_config.py
# -----------------------------------------------------------
# 负责网格价格表、利润公式、网格平移等纯计算逻辑
# 修复版本：明确网格概念，区分网格线和交易价格
# -----------------------------------------------------------

from decimal import Decimal, getcontext
import math
from typing import List, Tuple

# 增加小数精度，防止浮点累积误差
getcontext().prec = 16

class GridConfig:
    """
    网格交易配置类 - 修复版
    
    核心概念：
    - 网格线(grid_lines): N+1条价格线，将价格区间分成N个网格
    - 网格区间: 第i个网格对应价格区间 [grid_lines[i], grid_lines[i+1]]
    - 交易逻辑: 在第i个网格的下边界买入，在上边界卖出
    
    symbol        : 交易对，如 "ALEOUSDT"
    lower_price   : 区间下界 P_low
    upper_price   : 区间上界 P_high
    grids         : 网格数量 N
    mode          : "arithmetic" 或 "geometric"
    fee_rate      : 手续费率 (一次下单，单边)
    qty           : 每条挂单的数量 (base coin 计)
    trailing_k    : 连续突破顶 N 格后触发上移
    """
    def __init__(
        self,
        symbol: str,
        lower_price: float,
        upper_price: float,
        grids: int,
        mode: str = "geometric",
        fee_rate: float = 0.0005,  # 0.05% 更合理的手续费率
        qty: float = 10.0,
        trailing_k: int = 4,
    ):
        assert lower_price < upper_price, "lower_price 必须小于 upper_price"
        assert grids > 0, "grids 必须为正整数"
        self.symbol = symbol.upper()
        self.lo     = Decimal(str(lower_price))
        self.hi     = Decimal(str(upper_price))
        self.N      = int(grids)
        self.mode   = mode.lower()
        self.fee    = Decimal(str(fee_rate))
        self.qty    = Decimal(str(qty))
        self.k      = trailing_k

        # 生成网格线（N+1 条），并预计算每格利润
        self.grid_lines: List[Decimal] = self._build_grid()
        self.profits   : List[Decimal] = self._calc_grid_profits()

    # =========================================================
    # 修复后的公开 API
    # =========================================================
    def grid_price(self, idx: int) -> Decimal:
        """第 idx 条网格线价格（0…N  inclusive）"""
        if idx < 0 or idx > self.N:
            raise IndexError(f"网格线索引 {idx} 超出范围 [0, {self.N}]")
        return self.grid_lines[idx]

    def get_buy_price(self, grid_idx: int) -> Decimal:
        """获取第 grid_idx 个网格的买入价格（网格下边界）"""
        if grid_idx < 0 or grid_idx >= self.N:
            raise IndexError(f"网格索引 {grid_idx} 超出范围 [0, {self.N-1}]")
        return self.grid_lines[grid_idx]

    def get_sell_price(self, grid_idx: int) -> Decimal:
        """获取第 grid_idx 个网格的卖出价格（网格上边界）"""
        if grid_idx < 0 or grid_idx >= self.N:
            raise IndexError(f"网格索引 {grid_idx} 超出范围 [0, {self.N-1}]")
        return self.grid_lines[grid_idx + 1]

    def price_to_grid_index(self, price: float) -> int:
        """
        将价格映射到所在的网格索引
        返回: 网格索引 (0 到 N-1)，如果超出范围返回 -1 或 N
        """
        p = Decimal(str(price))
        if p < self.lo:
            return -1
        if p >= self.hi:
            return self.N
        
        # 找到价格所在的网格区间
        for i in range(self.N):
            if self.grid_lines[i] <= p < self.grid_lines[i + 1]:
                return i
        return self.N - 1  # 边界情况

    def is_valid_grid_index(self, grid_idx: int) -> bool:
        """检查网格索引是否有效"""
        return 0 <= grid_idx < self.N

    def get_grid_range(self, grid_idx: int) -> Tuple[Decimal, Decimal]:
        """获取指定网格的价格范围"""
        if not self.is_valid_grid_index(grid_idx):
            raise IndexError(f"网格索引 {grid_idx} 无效")
        return self.grid_lines[grid_idx], self.grid_lines[grid_idx + 1]

    def profit_per_grid(self, grid_idx: int) -> Decimal:
        """该网格的理论净利润 (扣除双边手续费)"""
        if not self.is_valid_grid_index(grid_idx):
            raise IndexError(f"网格索引 {grid_idx} 无效")
        return self.profits[grid_idx]

    def get_adjacent_grid(self, grid_idx: int, direction: str) -> int:
        """
        获取相邻网格索引
        direction: "up" 或 "down"
        返回: 相邻网格索引，如果超出范围返回 -1
        """
        if direction == "up":
            next_idx = grid_idx + 1
            return next_idx if next_idx < self.N else -1
        elif direction == "down":
            next_idx = grid_idx - 1
            return next_idx if next_idx >= 0 else -1
        else:
            raise ValueError("direction 必须是 'up' 或 'down'")

    # =========================================================
    # 兼容性方法（保持向后兼容）
    # =========================================================
    def opposite_price(self, idx: int, side: str) -> Decimal:
        """
        成交一单后，在相邻网格挂反向单的价格
        已修复：使用正确的网格逻辑
        """
        side = side.upper()
        if side == "BUY":      
            # 买单成交后，在同一网格设置卖单
            return self.get_sell_price(idx)
        elif side == "SELL":   
            # 卖单成交后，在下方网格设置买单
            lower_grid = self.get_adjacent_grid(idx, "down")
            if lower_grid >= 0:
                return self.get_buy_price(lower_grid)
            else:
                raise ValueError(f"卖单成交后无法在更低网格设置买单，当前网格: {idx}")
        else:
            raise ValueError("side must be BUY or SELL")

    def shift_up(self, steps: int = 1):
        """网格向上平移 steps 格，保持网格间距不变"""
        if steps <= 0:
            return
        
        if steps >= self.N:
            print(f"[Trailing] 警告：平移步数 {steps} 超过网格数量 {self.N}")
            steps = min(steps, self.N - 1)
        
        shift_val = (self.grid_lines[steps] - self.grid_lines[0])
        self.grid_lines = [p + shift_val for p in self.grid_lines]
        self.lo += shift_val
        self.hi += shift_val
        
        # 重新计算利润（价格变化后利润可能变化）
        self.profits = self._calc_grid_profits()
        
        print(f"[Trailing] 网格上移 {steps} 格，新区间: {self.lo:.6f} - {self.hi:.6f}")

    def top_trigger(self) -> Decimal:
        """
        触发上移的价格阈值
        当价格超过该值时，说明已突破顶部 trailing_k 格
        """
        trigger_line_idx = self.N - self.k
        if trigger_line_idx < 0:
            trigger_line_idx = 0
        return self.grid_lines[trigger_line_idx]

    # =========================================================
    # 内部函数
    # =========================================================
    def _build_grid(self) -> List[Decimal]:
        """构建网格线"""
        if self.mode == "arithmetic":
            delta = (self.hi - self.lo) / self.N
            return [self.lo + i * delta for i in range(self.N + 1)]
        elif self.mode == "geometric":
            ratio = (self.hi / self.lo) ** (Decimal("1") / self.N)
            return [self.lo * (ratio ** i) for i in range(self.N + 1)]
        else:
            raise ValueError("mode must be 'arithmetic' or 'geometric'")

    def _calc_grid_profits(self) -> List[Decimal]:
        """
        计算每个网格的理论利润
        Profit = (Sell - Buy) * qty - 2 * fee_rate * Sell * qty
        使用卖价计算双边手续费（更保守）
        """
        profits = []
        for i in range(self.N):
            buy_price  = self.get_buy_price(i)
            sell_price = self.get_sell_price(i)
            gross = (sell_price - buy_price) * self.qty
            fee   = self.fee * sell_price * self.qty * Decimal("2")
            net_profit = gross - fee
            profits.append(net_profit)
        return profits
    
    # =========================================================
    # 调试和展示方法
    # =========================================================
    def print_grid_info(self):
        """打印网格详细信息，用于调试"""
        print(f"网格配置: {self.symbol}")
        print(f"价格范围: {self.lo:.6f} - {self.hi:.6f}")
        print(f"网格数量: {self.N}")
        print(f"网格模式: {self.mode}")
        print(f"每格数量: {self.qty}")
        print(f"手续费率: {self.fee}")
        print("-" * 50)
        print("网格详情:")
        for i in range(min(5, self.N)):  # 只显示前5个网格
            buy_p = self.get_buy_price(i)
            sell_p = self.get_sell_price(i)
            profit = self.profit_per_grid(i)
            print(f"网格 {i}: 买入 {buy_p:.6f} -> 卖出 {sell_p:.6f}, 利润 {profit:.6f}")
        if self.N > 5:
            print(f"... 还有 {self.N - 5} 个网格")

    # 可选：返回 DataFrame 供 GUI 表格展示
    def to_dataframe(self):
        import pandas as pd
        data = {
            "grid_index": list(range(self.N)),
            "buy_price": [float(self.get_buy_price(i)) for i in range(self.N)],
            "sell_price": [float(self.get_sell_price(i)) for i in range(self.N)],
            "profit": [float(self.profit_per_grid(i)) for i in range(self.N)],
        }
        return pd.DataFrame(data)

    def update_qty(self, new_qty):
        """动态更新每单数量并重新计算每格利润"""
        self.qty = Decimal(str(new_qty))
        # 数量变化后需要重新计算每格利润
        self.profits = self._calc_grid_profits()

# -------------------- 快速自测 --------------------
if __name__ == "__main__":
    cfg = GridConfig(
        symbol="ALEOUSDT",
        lower_price=0.08,
        upper_price=0.12,
        grids=10,
        mode="geometric",
        fee_rate=0.001,
        qty=50,
    )
    
    print("=== 网格配置测试 ===")
    cfg.print_grid_info()
    
    print("\n=== 价格映射测试 ===")
    test_prices = [0.07, 0.085, 0.10, 0.115, 0.13]
    for price in test_prices:
        grid_idx = cfg.price_to_grid_index(price)
        print(f"价格 {price:.6f} -> 网格 {grid_idx}")
    
    print("\n=== 触发价格测试 ===")
    print(f"触发上移价格: {cfg.top_trigger():.6f}")
    
    print("\n=== 相邻网格测试 ===")
    print(f"网格3上方: {cfg.get_adjacent_grid(3, 'up')}")
    print(f"网格3下方: {cfg.get_adjacent_grid(3, 'down')}")
