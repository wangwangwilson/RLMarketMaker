#!/usr/bin/env python3
"""
生成模拟的加密货币数据用于测试

在没有真实Tardis数据的情况下，生成符合格式的测试数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import argparse


def generate_crypto_market_data(
    start_time: datetime,
    duration_hours: int = 1,
    tick_freq_ms: int = 100,
    initial_price: float = 40000.0,
    volatility: float = 0.02,
    spread_bps: float = 5.0,
    output_path: str = None
) -> pd.DataFrame:
    """
    生成模拟的加密货币市场数据
    
    Args:
        start_time: 开始时间
        duration_hours: 持续时长（小时）
        tick_freq_ms: tick频率（毫秒）
        initial_price: 初始价格
        volatility: 波动率
        spread_bps: 价差（基点）
        output_path: 输出文件路径
    
    Returns:
        生成的DataFrame
    """
    print(f"生成加密货币市场数据...")
    print(f"  初始价格: ${initial_price:.2f}")
    print(f"  时长: {duration_hours} 小时")
    print(f"  频率: {tick_freq_ms}ms")
    
    # 计算tick数量
    n_ticks = int(duration_hours * 3600 * 1000 / tick_freq_ms)
    print(f"  总tick数: {n_ticks}")
    
    # 时间序列
    timestamps = [start_time + timedelta(milliseconds=i * tick_freq_ms) 
                  for i in range(n_ticks)]
    
    # 生成价格路径（几何布朗运动）
    dt = tick_freq_ms / 1000.0 / 3600.0  # 转换为小时
    returns = np.random.normal(0, volatility * np.sqrt(dt), n_ticks)
    
    # 添加一些周期性模式（模拟市场微观结构）
    cycle = np.sin(np.arange(n_ticks) * 2 * np.pi / 1000) * 0.0001
    returns += cycle
    
    # 累积收益得到价格
    price_path = initial_price * np.exp(np.cumsum(returns))
    
    # 生成价差（动态，与波动率相关）
    rolling_vol = pd.Series(returns).rolling(window=100, min_periods=1).std()
    spread_base = initial_price * spread_bps / 10000.0
    spread_path = spread_base * (1 + rolling_vol * 10)
    spread_path = spread_path.fillna(spread_base).values
    
    # 计算买卖价
    best_bid = price_path - spread_path / 2
    best_ask = price_path + spread_path / 2
    midprice = (best_bid + best_ask) / 2
    
    # 生成订单簿深度（对数正态分布）
    bid_size = np.random.lognormal(mean=5.0, sigma=1.0, size=n_ticks)
    ask_size = np.random.lognormal(mean=5.0, sigma=1.0, size=n_ticks)
    
    # 生成成交数据（泊松过程）
    trades_per_tick = np.random.poisson(lam=0.5, size=n_ticks)
    trade_volume = np.random.exponential(scale=1.0, size=n_ticks) * trades_per_tick
    
    # 计算收益率和波动率
    ret = np.concatenate([[0], np.diff(np.log(midprice))])
    volatility_est = pd.Series(ret).rolling(window=20, min_periods=1).std() * np.sqrt(365 * 24)
    
    # 计算订单簿不平衡
    imbalance = (bid_size - ask_size) / (bid_size + ask_size + 1e-8)
    
    # 构造DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'midprice': midprice,
        'spread': spread_path,
        'best_bid': best_bid,
        'best_ask': best_ask,
        'bid_size': bid_size,
        'ask_size': ask_size,
        'trades': trades_per_tick,
        'amount': trade_volume,
        'value': trade_volume * midprice,
        'ret': ret,
        'volatility': volatility_est.fillna(volatility).values,
        'imbalance': imbalance
    })
    
    # 保存
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"\n✓ 数据已保存: {output_path}")
        print(f"  数据形状: {df.shape}")
        print(f"  时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
        print(f"  价格范围: ${df['midprice'].min():.2f} - ${df['midprice'].max():.2f}")
        print(f"  平均价差: ${df['spread'].mean():.4f} ({df['spread'].mean() / df['midprice'].mean() * 10000:.2f} bps)")
        print(f"  平均波动率: {df['volatility'].mean():.4f}")
    
    return df


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='生成模拟加密货币数据')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='交易对')
    parser.add_argument('--date', type=str, default='2024-01-15',
                        help='日期 (YYYY-MM-DD)')
    parser.add_argument('--hours', type=int, default=2,
                        help='持续时长（小时）')
    parser.add_argument('--price', type=float, default=40000.0,
                        help='初始价格')
    parser.add_argument('--output', type=str,
                        help='输出文件路径')
    
    args = parser.parse_args()
    
    # 解析日期
    start_time = datetime.strptime(args.date, "%Y-%m-%d")
    
    # 默认输出路径
    if not args.output:
        args.output = f"data/replay/test_{args.symbol}_{args.date}_replay.parquet"
    
    # 生成数据
    generate_crypto_market_data(
        start_time=start_time,
        duration_hours=args.hours,
        tick_freq_ms=100,  # 100ms ticks
        initial_price=args.price,
        volatility=0.02,  # 2% 日波动率（crypto较高）
        spread_bps=5.0,   # 5个基点
        output_path=args.output
    )
    
    print(f"\n可以使用此数据文件进行测试:")
    print(f"  python3 scripts/test_tardis_integration.py --skip-download")


if __name__ == '__main__':
    main()
