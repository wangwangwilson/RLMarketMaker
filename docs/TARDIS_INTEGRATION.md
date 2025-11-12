# Tardis 加密货币数据集成指南

## 概述

本文档说明如何使用 Tardis 加密货币高频数据训练市场做市商策略。

## 集成特性

### ✅ 已完成的功能

1. **TardisReplayFeed** - 加密货币数据回放器
   - 支持从 parquet 文件读取历史数据
   - 领域随机化（Domain Randomization）支持
   - 自动回退到合成数据

2. **数据预处理管道**
   - Trades 数据处理
   - Orderbook snapshot 数据处理
   - 数据合并和重采样
   - 统一的 MarketTick 格式输出

3. **测试工具**
   - 测试数据生成器
   - 完整集成测试套件
   - 多 Episode 稳定性验证

4. **配置文件**
   - `configs/tardis_replay.yaml` - Tardis 专用配置
   - `configs/api_keys.yaml` - API 密钥配置

## 快速开始

### 1. 生成测试数据

如果没有真实 Tardis 数据，可以先用模拟数据测试：

```bash
# 生成 2 小时的 BTCUSDT 测试数据
python3 scripts/generate_test_crypto_data.py \
  --symbol BTCUSDT \
  --date 2024-01-15 \
  --hours 2 \
  --price 40000
```

### 2. 运行集成测试

```bash
# 测试数据加载、环境集成和多 Episode 运行
python3 scripts/test_crypto_integration_simple.py
```

### 3. 使用真实 Tardis 数据（可选）

#### 3.1 下载数据

```bash
# 下载单日数据
python3 rlmarketmaker/data/download_tardis.py \
  --exchange binance \
  --symbol BTCUSDT \
  --data-type trades \
  --date 2024-01-15

python3 rlmarketmaker/data/download_tardis.py \
  --exchange binance \
  --symbol BTCUSDT \
  --data-type book_snapshot_5 \
  --date 2024-01-15

# 或者下载多日数据
python3 rlmarketmaker/data/download_tardis.py \
  --exchange binance \
  --symbol BTCUSDT \
  --data-type trades \
  --start-date 2024-01-01 \
  --end-date 2024-01-07
```

#### 3.2 预处理数据

```bash
# 处理单个日期
python3 rlmarketmaker/data/preprocess_tardis.py \
  --trades-file data/tardis/binance/BTCUSDT/trades/BTCUSDT_trades_2024-01-15.csv.gz \
  --book-file data/tardis/binance/BTCUSDT/book_snapshot_5/BTCUSDT_book_snapshot_5_2024-01-15.csv.gz \
  --output data/replay/binance_BTCUSDT_2024-01-15_replay.parquet \
  --resample-freq 100ms

# 或批量处理
python3 rlmarketmaker/data/preprocess_tardis.py --batch \
  --input-dir data/tardis \
  --output-dir data/replay \
  --exchange binance \
  --symbol BTCUSDT \
  --resample-freq 100ms
```

## 训练模型

### 使用测试数据训练

```bash
python3 scripts/training/train_min.py \
  --config configs/tardis_replay.yaml \
  --seed 42
```

### 使用真实数据训练

修改 `configs/tardis_replay.yaml` 中的数据路径：

```yaml
feed:
  data_path: "data/replay/binance_BTCUSDT_2024-01-15_replay.parquet"
```

然后运行训练：

```bash
python3 scripts/training/train_min.py \
  --config configs/tardis_replay.yaml \
  --seed 42
```

## 数据格式

### 输入数据要求

预处理后的数据需要包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `timestamp` | datetime | 时间戳 |
| `midprice` | float | 中间价 |
| `spread` | float | 买卖价差 |
| `best_bid` | float | 最佳买价 |
| `best_ask` | float | 最佳卖价 |
| `bid_size` | float | 买单深度 |
| `ask_size` | float | 卖单深度 |
| `trades` | int | 成交数量 |
| `amount` | float | 成交量 |
| `ret` | float | 收益率 |
| `volatility` | float | 波动率 |
| `imbalance` | float | 订单簿不平衡度 |

### MarketTick 格式

Feed 输出的 MarketTick 对象包含：

```python
@dataclass
class MarketTick:
    timestamp: float      # 时间戳
    midprice: float       # 中间价
    spread: float         # 价差
    bid_size: float       # 买单深度
    ask_size: float       # 卖单深度
    trades: int           # 成交数
```

## 配置说明

### Tardis 专用配置 (`configs/tardis_replay.yaml`)

```yaml
env:
  episode_length: 1000    # 较短的episode（加密货币频率高）
  max_inventory: 10.0     # 较小的库存限制
  fee_bps: 2.0           # 加密货币交易费用较高
  
feed:
  data_path: "data/replay/test_BTCUSDT_2024-01-15_replay.parquet"
  episode_length: 1000
  warmup_steps: 50
  
ppo:
  ent_coef: 0.008        # 较高的熵系数以增加探索
```

### API 密钥配置 (`configs/api_keys.yaml`)

```yaml
tardis:
  api_key: "YOUR_TARDIS_API_KEY"
  base_url: "https://api.tardis.dev/v1"
  exchanges: ["binance", "coinbase", "kraken"]
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
  data_types: ["trades", "book_snapshot_5"]
```

## 测试结果

### 测试统计

运行 `test_crypto_integration_simple.py` 的测试结果：

```
测试 1: TardisReplayFeed 数据回放
  ✓ 成功加载 72,000 个 ticks
  ✓ 平均价格: $40,045.30
  ✓ 平均价差: 5.90 bps

测试 2: 市场环境集成
  ✓ 环境正确初始化
  ✓ 动作空间: MultiDiscrete([10 10 5])
  ✓ 观察空间: Box(5,)
  ✓ 成功运行 50 步

测试 3: 多 Episode 稳定性
  ✓ 3 个 episodes 全部完成
  ✓ 无崩溃或异常
```

## 加密货币市场特点

相比传统股票市场，加密货币市场有以下特点：

1. **更高的波动率** - 日波动率通常 2-5%
2. **24/7 交易** - 无开盘/收盘时间
3. **更高的流动性** - 订单簿深度大
4. **更窄的价差** - 通常 5-10 bps
5. **更高的交易费用** - 通常 1-5 bps

因此配置需要相应调整：
- 较小的库存限制 (`max_inventory: 10.0`)
- 较高的交易费用 (`fee_bps: 2.0`)
- 更频繁的数据采样 (`resample_freq: 100ms`)

## 文件结构

```
rlmarketmaker/
├── data/
│   ├── download_tardis.py        # Tardis 数据下载器
│   ├── preprocess_tardis.py      # Tardis 数据预处理
│   └── feeds.py                  # 包含 TardisReplayFeed
├── env/
│   └── replay_market_env.py      # 回放环境
configs/
├── tardis_replay.yaml            # Tardis 配置
└── api_keys.yaml                 # API 密钥
scripts/
├── generate_test_crypto_data.py  # 测试数据生成器
├── test_crypto_integration_simple.py  # 简化测试
└── test_tardis_integration.py    # 完整测试（含下载）
data/
├── tardis/                       # Tardis 原始数据
└── replay/                       # 预处理后的回放数据
```

## 故障排查

### 问题 1: 数据下载失败 (404)

**原因**: Tardis API 路径或日期格式可能不正确

**解决方案**: 
1. 先使用测试数据生成器
2. 检查 Tardis API 文档确认正确的 URL 格式
3. 验证 API 密钥是否有效

### 问题 2: 数据预处理报错

**原因**: 原始数据格式与预期不符

**解决方案**:
1. 检查下载的文件是否完整
2. 使用 `--max-rows 1000` 参数测试小数据集
3. 检查 CSV 文件的列名和格式

### 问题 3: 环境运行报错

**原因**: 数据格式或配置不匹配

**解决方案**:
1. 运行 `test_crypto_integration_simple.py` 诊断
2. 检查数据文件是否包含所有必需列
3. 验证配置文件中的路径是否正确

## 下一步

1. **收集更多数据** - 下载多个交易对和时间段的数据
2. **优化超参数** - 针对加密货币市场调整 PPO 参数
3. **评估性能** - 与基线策略对比
4. **实盘测试** - 在模拟环境中验证策略

## 参考资源

- [Tardis 官方文档](https://docs.tardis.dev/)
- [Tardis Python SDK](https://github.com/tardis-dev/tardis-node)
- [项目 README](../README.md)

## 贡献

如发现问题或有改进建议，请提交 Issue 或 Pull Request。
