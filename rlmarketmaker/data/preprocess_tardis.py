#!/usr/bin/env python3
"""
预处理Tardis加密货币数据以用于市场回放。

将原始的trades和orderbook数据转换为统一的MarketTick格式。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import gzip
import glob


def process_trades_data(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    处理trades数据
    
    Tardis trades格式:
    - timestamp: 微秒级时间戳
    - symbol: 交易对
    - exchange: 交易所
    - amount: 数量
    - price: 价格
    - side: buy/sell
    """
    # 转换时间戳
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'], unit='us')
    
    # 计算成交量加权价格(VWAP)
    trades_df['value'] = trades_df['amount'] * trades_df['price']
    
    return trades_df


def process_orderbook_data(book_df: pd.DataFrame) -> pd.DataFrame:
    """
    处理orderbook snapshot数据
    
    Tardis book_snapshot格式:
    - timestamp: 微秒级时间戳
    - symbol: 交易对
    - exchange: 交易所
    - asks: [[price, amount], ...]
    - bids: [[price, amount], ...]
    """
    # 转换时间戳
    book_df['timestamp'] = pd.to_datetime(book_df['timestamp'], unit='us')
    
    # 提取最佳买卖价和深度
    def extract_best(row, side):
        """提取最佳买卖价"""
        if side == 'bid':
            levels = row.get('bids', [])
        else:
            levels = row.get('asks', [])
        
        if len(levels) > 0 and len(levels[0]) >= 2:
            return float(levels[0][0]), float(levels[0][1])
        return np.nan, np.nan
    
    # 提取bid/ask信息
    book_df[['best_bid', 'bid_size']] = book_df.apply(
        lambda row: pd.Series(extract_best(row, 'bid')), axis=1
    )
    book_df[['best_ask', 'ask_size']] = book_df.apply(
        lambda row: pd.Series(extract_best(row, 'ask')), axis=1
    )
    
    return book_df


def merge_trades_and_book(
    trades_df: pd.DataFrame,
    book_df: pd.DataFrame,
    resample_freq: str = '100ms'
) -> pd.DataFrame:
    """
    合并trades和orderbook数据，生成统一的tick数据
    
    Args:
        trades_df: 处理后的trades数据
        book_df: 处理后的orderbook数据
        resample_freq: 重采样频率 (e.g. '100ms', '1s')
    
    Returns:
        合并后的DataFrame
    """
    # 设置时间索引
    trades_df = trades_df.set_index('timestamp')
    book_df = book_df.set_index('timestamp')
    
    # 对trades进行重采样
    trades_resampled = trades_df.resample(resample_freq).agg({
        'price': 'last',
        'amount': 'sum',
        'value': 'sum'
    })
    
    # 对orderbook进行重采样 (前向填充)
    book_resampled = book_df.resample(resample_freq).ffill()
    
    # 合并数据
    merged = pd.merge(
        book_resampled[['best_bid', 'best_ask', 'bid_size', 'ask_size']],
        trades_resampled[['price', 'amount', 'value']],
        left_index=True,
        right_index=True,
        how='outer'
    )
    
    # 前向填充缺失的orderbook数据
    merged[['best_bid', 'best_ask', 'bid_size', 'ask_size']] = \
        merged[['best_bid', 'best_ask', 'bid_size', 'ask_size']].ffill()
    
    # 计算中间价和价差
    merged['midprice'] = (merged['best_bid'] + merged['best_ask']) / 2
    merged['spread'] = merged['best_ask'] - merged['best_bid']
    
    # 填充成交数据（无成交时为0）
    merged['amount'] = merged['amount'].fillna(0)
    merged['value'] = merged['value'].fillna(0)
    merged['trades'] = (merged['amount'] > 0).astype(int)
    
    # 使用VWAP作为价格，若无成交则使用midprice
    merged['price'] = np.where(
        merged['amount'] > 0,
        merged['value'] / merged['amount'],
        merged['midprice']
    )
    
    # 计算收益率和滚动波动率
    merged['ret'] = merged['midprice'].pct_change()
    merged['volatility'] = merged['ret'].rolling(window=20, min_periods=1).std() * np.sqrt(252 * 24 * 3600)
    
    # 计算订单簿不平衡度
    merged['imbalance'] = (merged['bid_size'] - merged['ask_size']) / \
                          (merged['bid_size'] + merged['ask_size'] + 1e-8)
    
    # 删除NaN行
    merged = merged.dropna(subset=['midprice', 'spread'])
    
    # 选择最终列
    output_columns = [
        'midprice', 'spread', 'best_bid', 'best_ask',
        'bid_size', 'ask_size', 'trades', 'amount', 'value',
        'ret', 'volatility', 'imbalance'
    ]
    
    return merged[output_columns].reset_index()


def preprocess_tardis_data(
    trades_file: str,
    book_file: str,
    output_path: str,
    resample_freq: str = '100ms',
    max_rows: Optional[int] = None
) -> None:
    """
    预处理Tardis数据
    
    Args:
        trades_file: trades数据文件路径 (.csv.gz)
        book_file: orderbook数据文件路径 (.csv.gz)
        output_path: 输出parquet文件路径
        resample_freq: 重采样频率
        max_rows: 最大读取行数（用于测试）
    """
    print(f"处理Tardis数据...")
    print(f"  Trades文件: {trades_file}")
    print(f"  Book文件: {book_file}")
    
    # 读取trades数据
    print("读取trades数据...")
    with gzip.open(trades_file, 'rt') as f:
        trades_df = pd.read_csv(f, nrows=max_rows)
    print(f"  加载 {len(trades_df)} 条trades记录")
    
    # 读取orderbook数据
    print("读取orderbook数据...")
    with gzip.open(book_file, 'rt') as f:
        # Tardis book snapshot是JSON格式
        book_df = pd.read_csv(f, nrows=max_rows)
    print(f"  加载 {len(book_df)} 条book记录")
    
    # 处理数据
    print("处理trades数据...")
    trades_df = process_trades_data(trades_df)
    
    print("处理orderbook数据...")
    book_df = process_orderbook_data(book_df)
    
    # 合并数据
    print(f"合并数据 (重采样频率: {resample_freq})...")
    merged_df = merge_trades_and_book(trades_df, book_df, resample_freq)
    
    # 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(output_path, index=False)
    
    print(f"\n✓ 处理完成!")
    print(f"  输出文件: {output_path}")
    print(f"  数据形状: {merged_df.shape}")
    print(f"  时间范围: {merged_df['timestamp'].min()} 至 {merged_df['timestamp'].max()}")
    print(f"  平均价差: {merged_df['spread'].mean():.6f}")
    print(f"  平均波动率: {merged_df['volatility'].mean():.4f}")
    print(f"\n数据列: {merged_df.columns.tolist()}")
    print(f"\n数据预览:")
    print(merged_df.head())


def batch_preprocess_tardis(
    input_dir: str,
    output_dir: str,
    exchange: str,
    symbol: str,
    resample_freq: str = '100ms',
    max_rows: Optional[int] = None
) -> List[str]:
    """
    批量处理Tardis数据
    
    Args:
        input_dir: 输入目录（Tardis下载目录）
        output_dir: 输出目录
        exchange: 交易所
        symbol: 交易对
        resample_freq: 重采样频率
        max_rows: 最大读取行数
    
    Returns:
        处理后的文件列表
    """
    # 查找trades和book文件
    trades_pattern = f"{input_dir}/{exchange}/{symbol}/trades/*.csv.gz"
    book_pattern = f"{input_dir}/{exchange}/{symbol}/book_snapshot_*/*.csv.gz"
    
    trades_files = sorted(glob.glob(trades_pattern))
    book_files = sorted(glob.glob(book_pattern))
    
    if not trades_files:
        print(f"未找到trades文件: {trades_pattern}")
        return []
    
    if not book_files:
        print(f"未找到book文件: {book_pattern}")
        return []
    
    print(f"找到 {len(trades_files)} 个trades文件")
    print(f"找到 {len(book_files)} 个book文件")
    
    processed_files = []
    
    # 配对处理（假设文件名格式匹配）
    for trades_file in trades_files:
        # 提取日期
        trades_path = Path(trades_file)
        # 文件名格式: SYMBOL_trades_YYYY-MM-DD.csv.gz
        date_str = trades_path.stem.split('_')[-1].replace('.csv', '')
        
        # 查找对应的book文件
        matching_book = None
        for book_file in book_files:
            if date_str in book_file:
                matching_book = book_file
                break
        
        if not matching_book:
            print(f"警告: 未找到与 {trades_file} 匹配的book文件")
            continue
        
        # 输出文件
        output_file = f"{output_dir}/{exchange}_{symbol}_{date_str}_replay.parquet"
        
        # 处理
        try:
            preprocess_tardis_data(
                trades_file=trades_file,
                book_file=matching_book,
                output_path=output_file,
                resample_freq=resample_freq,
                max_rows=max_rows
            )
            processed_files.append(output_file)
        except Exception as e:
            print(f"处理失败 {trades_file}: {e}")
            continue
    
    return processed_files


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='预处理Tardis加密货币数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个日期
  python preprocess_tardis.py \\
      --trades-file data/tardis/binance/BTCUSDT/trades/BTCUSDT_trades_2024-01-15.csv.gz \\
      --book-file data/tardis/binance/BTCUSDT/book_snapshot_5/BTCUSDT_book_snapshot_5_2024-01-15.csv.gz \\
      --output data/replay/binance_BTCUSDT_2024-01-15_replay.parquet
  
  # 批量处理
  python preprocess_tardis.py --batch \\
      --input-dir data/tardis \\
      --output-dir data/replay \\
      --exchange binance \\
      --symbol BTCUSDT
        """
    )
    
    parser.add_argument('--trades-file', type=str,
                        help='Trades文件路径')
    parser.add_argument('--book-file', type=str,
                        help='Orderbook文件路径')
    parser.add_argument('--output', type=str,
                        help='输出文件路径')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理模式')
    parser.add_argument('--input-dir', type=str,
                        help='输入目录（批量模式）')
    parser.add_argument('--output-dir', type=str,
                        help='输出目录（批量模式）')
    parser.add_argument('--exchange', type=str,
                        help='交易所（批量模式）')
    parser.add_argument('--symbol', type=str,
                        help='交易对（批量模式）')
    parser.add_argument('--resample-freq', type=str, default='100ms',
                        help='重采样频率 (默认: 100ms)')
    parser.add_argument('--max-rows', type=int,
                        help='最大读取行数（用于测试）')
    
    args = parser.parse_args()
    
    if args.batch:
        # 批量处理模式
        if not all([args.input_dir, args.output_dir, args.exchange, args.symbol]):
            print("批量模式需要: --input-dir, --output-dir, --exchange, --symbol")
            return
        
        files = batch_preprocess_tardis(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            exchange=args.exchange,
            symbol=args.symbol,
            resample_freq=args.resample_freq,
            max_rows=args.max_rows
        )
        print(f"\n总共处理 {len(files)} 个文件")
    else:
        # 单文件处理模式
        if not all([args.trades_file, args.book_file, args.output]):
            print("单文件模式需要: --trades-file, --book-file, --output")
            return
        
        preprocess_tardis_data(
            trades_file=args.trades_file,
            book_file=args.book_file,
            output_path=args.output,
            resample_freq=args.resample_freq,
            max_rows=args.max_rows
        )


if __name__ == '__main__':
    main()
