#!/usr/bin/env python3
"""
测试Tardis数据集成

验证Tardis数据下载、预处理和回放的完整流程
"""

import os
import sys
from pathlib import Path
import yaml
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlmarketmaker.data.download_tardis import TardisDownloader
from rlmarketmaker.data.preprocess_tardis import preprocess_tardis_data
from rlmarketmaker.data.feeds import TardisReplayFeed
from rlmarketmaker.env.replay_market_env import ReplayMarketMakerEnv


def test_download(api_key: str, exchange: str, symbol: str, date: str, output_dir: str):
    """测试数据下载"""
    print("=" * 60)
    print("步骤 1: 测试Tardis数据下载")
    print("=" * 60)
    
    downloader = TardisDownloader(api_key=api_key)
    
    # 下载trades数据
    print(f"\n下载 {exchange}/{symbol} trades 数据 ({date})...")
    trades_file = downloader.download_dataset(
        exchange=exchange,
        symbol=symbol,
        data_type="trades",
        date=date,
        output_dir=output_dir
    )
    
    # 下载orderbook数据
    print(f"\n下载 {exchange}/{symbol} book_snapshot_5 数据 ({date})...")
    book_file = downloader.download_dataset(
        exchange=exchange,
        symbol=symbol,
        data_type="book_snapshot_5",
        date=date,
        output_dir=output_dir
    )
    
    if trades_file and book_file:
        print(f"\n✓ 数据下载成功!")
        print(f"  Trades: {trades_file}")
        print(f"  Book: {book_file}")
        return trades_file, book_file
    else:
        print(f"\n✗ 数据下载失败")
        return None, None


def test_preprocess(trades_file: str, book_file: str, output_file: str):
    """测试数据预处理"""
    print("\n" + "=" * 60)
    print("步骤 2: 测试数据预处理")
    print("=" * 60)
    
    try:
        preprocess_tardis_data(
            trades_file=trades_file,
            book_file=book_file,
            output_path=output_file,
            resample_freq='100ms',  # 100ms采样
            max_rows=100000  # 只处理前10万行用于测试
        )
        print(f"\n✓ 数据预处理成功: {output_file}")
        return True
    except Exception as e:
        print(f"\n✗ 数据预处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_replay_feed(replay_file: str):
    """测试数据回放"""
    print("\n" + "=" * 60)
    print("步骤 3: 测试TardisReplayFeed")
    print("=" * 60)
    
    try:
        # 创建feed
        feed = TardisReplayFeed(seed=42)
        
        # 配置
        config = {
            'data_path': replay_file,
            'episode_length': 100,  # 只测试100个tick
            'warmup_steps': 10,
            'volatility_multiplier': [0.9, 1.1],
            'spread_multiplier': [0.8, 1.2]
        }
        
        # 获取数据流
        print(f"\n测试数据回放 (前100个tick)...")
        tick_count = 0
        for tick in feed.get_env_feed(config):
            tick_count += 1
            if tick_count <= 5:
                print(f"  Tick {tick_count}: price={tick.midprice:.2f}, spread={tick.spread:.4f}, "
                      f"bid_size={tick.bid_size:.2f}, ask_size={tick.ask_size:.2f}")
            if tick_count >= 100:
                break
        
        print(f"\n✓ 数据回放成功: 读取了 {tick_count} 个ticks")
        return True
    except Exception as e:
        print(f"\n✗ 数据回放失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment(replay_file: str, config_file: str):
    """测试环境集成"""
    print("\n" + "=" * 60)
    print("步骤 4: 测试环境集成")
    print("=" * 60)
    
    try:
        # 加载配置
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # 更新数据路径
        config['feed']['data_path'] = replay_file
        
        # 创建环境
        print("\n创建市场环境...")
        feed = TardisReplayFeed(seed=42)
        env = ReplayMarketMakerEnv(
            feed=feed,
            config={**config['env'], **config['feed']},
            seed=42
        )
        
        # 重置环境
        print("重置环境...")
        obs, info = env.reset()
        print(f"  初始观察: {obs}")
        print(f"  初始信息: inventory={info['inventory']}, pnl={info['total_pnl']:.2f}, "
              f"midprice={info['midprice']:.2f}")
        
        # 运行几步
        print("\n运行环境步骤...")
        for step in range(10):
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if step < 3:
                print(f"  Step {step}: reward={reward:.4f}, inventory={info['inventory']:.2f}, "
                      f"pnl={info['total_pnl']:.2f}, midprice={info['midprice']:.2f}")
            
            if terminated or truncated:
                print(f"  Episode ended at step {step}")
                break
        
        print(f"\n✓ 环境测试成功!")
        return True
    except Exception as e:
        print(f"\n✗ 环境测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试流程"""
    parser = argparse.ArgumentParser(description='测试Tardis集成')
    parser.add_argument('--config', type=str, default='configs/api_keys.yaml',
                        help='API配置文件')
    parser.add_argument('--env-config', type=str, default='configs/tardis_replay.yaml',
                        help='环境配置文件')
    parser.add_argument('--exchange', type=str, default='binance',
                        help='交易所')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                        help='交易对')
    parser.add_argument('--date', type=str, default='2024-01-15',
                        help='测试日期 (YYYY-MM-DD)')
    parser.add_argument('--skip-download', action='store_true',
                        help='跳过下载步骤（如果已有数据）')
    
    args = parser.parse_args()
    
    # 加载API配置
    print("加载配置...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    api_key = config['tardis']['api_key']
    output_dir = config['tardis']['output_dir']
    
    print(f"\n配置信息:")
    print(f"  交易所: {args.exchange}")
    print(f"  交易对: {args.symbol}")
    print(f"  日期: {args.date}")
    print(f"  输出目录: {output_dir}")
    
    # 文件路径
    trades_file = f"{output_dir}/{args.exchange}/{args.symbol}/trades/{args.symbol}_trades_{args.date}.csv.gz"
    book_file = f"{output_dir}/{args.exchange}/{args.symbol}/book_snapshot_5/{args.symbol}_book_snapshot_5_{args.date}.csv.gz"
    replay_file = f"data/replay/{args.exchange}_{args.symbol}_{args.date}_replay.parquet"
    
    # 步骤1: 下载数据
    if not args.skip_download:
        trades_file, book_file = test_download(
            api_key=api_key,
            exchange=args.exchange,
            symbol=args.symbol,
            date=args.date,
            output_dir=output_dir
        )
        
        if not trades_file or not book_file:
            print("\n测试失败: 数据下载失败")
            return
    else:
        print(f"\n跳过下载步骤，使用现有文件:")
        print(f"  Trades: {trades_file}")
        print(f"  Book: {book_file}")
        
        if not (Path(trades_file).exists() and Path(book_file).exists()):
            print("\n错误: 指定的文件不存在")
            return
    
    # 步骤2: 预处理数据
    if not test_preprocess(trades_file, book_file, replay_file):
        print("\n测试失败: 数据预处理失败")
        return
    
    # 步骤3: 测试回放
    if not test_replay_feed(replay_file):
        print("\n测试失败: 数据回放失败")
        return
    
    # 步骤4: 测试环境
    if not test_environment(replay_file, args.env_config):
        print("\n测试失败: 环境集成失败")
        return
    
    # 全部成功
    print("\n" + "=" * 60)
    print("✓ 所有测试通过!")
    print("=" * 60)
    print("\nTardis数据集成验证完成，可以开始训练了！")
    print(f"\n下一步:")
    print(f"  1. 训练模型:")
    print(f"     python scripts/training/train_min.py --config {args.env_config}")
    print(f"  2. 评估模型:")
    print(f"     python scripts/evaluation/evaluate_replay.py --config {args.env_config} --checkpoint logs/checkpoints/policy.pt")


if __name__ == '__main__':
    main()
