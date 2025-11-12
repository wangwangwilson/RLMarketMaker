#!/usr/bin/env python3
"""
简化的加密货币数据集成测试

使用生成的测试数据验证完整流程
"""

import os
import sys
from pathlib import Path
import yaml

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from rlmarketmaker.data.feeds import TardisReplayFeed
from rlmarketmaker.env.replay_market_env import ReplayMarketMakerEnv


def test_replay_feed(replay_file: str):
    """测试数据回放"""
    print("=" * 60)
    print("测试 1: TardisReplayFeed 数据回放")
    print("=" * 60)
    
    try:
        # 创建feed
        feed = TardisReplayFeed(seed=42)
        
        # 配置
        config = {
            'data_path': replay_file,
            'episode_length': 100,
            'warmup_steps': 10,
            'volatility_multiplier': [0.9, 1.1],
            'spread_multiplier': [0.8, 1.2]
        }
        
        # 获取数据流
        print(f"\n读取数据: {replay_file}")
        print(f"测试前100个ticks...\n")
        
        tick_count = 0
        price_sum = 0
        spread_sum = 0
        
        for tick in feed.get_env_feed(config):
            tick_count += 1
            price_sum += tick.midprice
            spread_sum += tick.spread
            
            if tick_count <= 5:
                print(f"  Tick {tick_count}:")
                print(f"    价格: ${tick.midprice:.2f}")
                print(f"    价差: ${tick.spread:.4f}")
                print(f"    买单深度: {tick.bid_size:.2f}")
                print(f"    卖单深度: {tick.ask_size:.2f}")
                print(f"    成交数: {tick.trades}")
            
            if tick_count >= 100:
                break
        
        avg_price = price_sum / tick_count
        avg_spread = spread_sum / tick_count
        
        print(f"\n统计信息:")
        print(f"  总tick数: {tick_count}")
        print(f"  平均价格: ${avg_price:.2f}")
        print(f"  平均价差: ${avg_spread:.4f} ({avg_spread/avg_price*10000:.2f} bps)")
        
        print(f"\n✓ 数据回放测试通过!")
        return True
    except Exception as e:
        print(f"\n✗ 数据回放测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment(replay_file: str):
    """测试环境集成"""
    print("\n" + "=" * 60)
    print("测试 2: 市场环境集成")
    print("=" * 60)
    
    try:
        # 配置
        config = {
            'data_path': replay_file,
            'episode_length': 500,
            'max_inventory': 10.0,
            'tick_size': 0.01,
            'fee_bps': 2.0,
            'latency_ticks': 1,
            'fill_alpha': 0.85,
            'fill_beta': 0.4,
            'warmup_steps': 50,
            'volatility_multiplier': [0.9, 1.1],
            'spread_multiplier': [0.8, 1.2]
        }
        
        # 创建环境
        print("\n创建ReplayMarketMakerEnv...")
        feed = TardisReplayFeed(seed=42)
        env = ReplayMarketMakerEnv(
            feed=feed,
            config=config,
            seed=42
        )
        
        print(f"  观察空间: {env.observation_space}")
        print(f"  动作空间: {env.action_space}")
        
        # 重置环境
        print("\n重置环境...")
        obs, info = env.reset()
        print(f"  初始观察: {obs}")
        print(f"  初始状态:")
        print(f"    库存: {info['inventory']:.2f}")
        print(f"    PnL: ${info['total_pnl']:.2f}")
        print(f"    中间价: ${info['midprice']:.2f}")
        print(f"    价差: ${info['spread']:.4f}")
        
        # 运行环境
        print("\n运行环境步骤...")
        total_reward = 0
        fill_count = 0
        
        for step in range(50):
            # 随机动作
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            if info.get('filled_bid', 0) or info.get('filled_ask', 0):
                fill_count += 1
            
            if step < 5:
                print(f"\n  Step {step+1}:")
                print(f"    动作: {action}")
                print(f"    奖励: {reward:.4f}")
                print(f"    库存: {info['inventory']:.2f}")
                print(f"    PnL: ${info['total_pnl']:.2f}")
                print(f"    价格: ${info['midprice']:.2f}")
            
            if terminated or truncated:
                print(f"\n  Episode在第{step+1}步结束")
                break
        
        print(f"\n统计信息 (50步):")
        print(f"  总奖励: {total_reward:.4f}")
        print(f"  成交次数: {fill_count}")
        print(f"  最终库存: {info['inventory']:.2f}")
        print(f"  最终PnL: ${info['total_pnl']:.2f}")
        
        print(f"\n✓ 环境集成测试通过!")
        return True
    except Exception as e:
        print(f"\n✗ 环境集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_episodes(replay_file: str, n_episodes: int = 3):
    """测试多个episode"""
    print("\n" + "=" * 60)
    print(f"测试 3: 多Episode稳定性 ({n_episodes} episodes)")
    print("=" * 60)
    
    try:
        config = {
            'data_path': replay_file,
            'episode_length': 200,
            'max_inventory': 10.0,
            'tick_size': 0.01,
            'fee_bps': 2.0,
            'latency_ticks': 1,
            'fill_alpha': 0.85,
            'fill_beta': 0.4,
            'warmup_steps': 50
        }
        
        feed = TardisReplayFeed(seed=42)
        env = ReplayMarketMakerEnv(feed=feed, config=config, seed=42)
        
        print(f"\n运行 {n_episodes} 个episodes...")
        
        episode_rewards = []
        episode_pnls = []
        
        for ep in range(n_episodes):
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            
            for step in range(200):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(total_reward)
            episode_pnls.append(info['total_pnl'])
            
            print(f"  Episode {ep+1}: "
                  f"steps={steps}, "
                  f"reward={total_reward:.2f}, "
                  f"pnl=${info['total_pnl']:.2f}, "
                  f"inventory={info['inventory']:.2f}")
        
        print(f"\n统计:")
        print(f"  平均奖励: {sum(episode_rewards)/len(episode_rewards):.2f}")
        print(f"  平均PnL: ${sum(episode_pnls)/len(episode_pnls):.2f}")
        
        print(f"\n✓ 多Episode测试通过!")
        return True
    except Exception as e:
        print(f"\n✗ 多Episode测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试流程"""
    # 使用生成的测试数据
    replay_file = "data/replay/test_BTCUSDT_2024-01-15_replay.parquet"
    
    print("=" * 60)
    print("加密货币数据集成测试")
    print("=" * 60)
    print(f"\n数据文件: {replay_file}")
    
    # 检查文件是否存在
    if not Path(replay_file).exists():
        print(f"\n错误: 数据文件不存在: {replay_file}")
        print(f"请先运行: python3 scripts/generate_test_crypto_data.py")
        return False
    
    # 运行测试
    results = []
    
    # 测试1: 数据回放
    results.append(test_replay_feed(replay_file))
    
    # 测试2: 环境集成
    results.append(test_environment(replay_file))
    
    # 测试3: 多Episode稳定性
    results.append(test_multiple_episodes(replay_file, n_episodes=3))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n通过: {passed}/{total}")
    
    if all(results):
        print("\n✓ 所有测试通过!")
        print("\n加密货币数据集成验证完成！")
        print("\n下一步:")
        print("  1. 下载真实Tardis数据（可选）")
        print("  2. 使用此配置训练模型:")
        print("     python3 scripts/training/train_min.py --config configs/tardis_replay.yaml")
        return True
    else:
        print("\n✗ 部分测试失败")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
