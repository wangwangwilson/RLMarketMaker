# UV 环境管理和部署指南

## 目录
- [uv 简介](#uv-简介)
- [安装 uv](#安装-uv)
- [快速开始](#快速开始)
- [环境管理](#环境管理)
- [依赖管理](#依赖管理)
- [运行和测试](#运行和测试)
- [部署方案](#部署方案)
- [Docker 集成](#docker-集成)
- [常见问题](#常见问题)

---

## uv 简介

**uv** 是一个极速的 Python 包管理器和项目管理工具，用 Rust 编写，性能比 pip 快 10-100 倍。

### 核心优势

- ⚡ **极快速度** - 比 pip 快 10-100 倍
- 🔒 **确定性构建** - 自动生成和维护 lock 文件
- 🎯 **简单易用** - 单个命令管理一切
- 🐍 **Python 版本管理** - 内置 Python 版本管理
- 📦 **虚拟环境** - 自动创建和管理虚拟环境
- 🚀 **生产就绪** - 适合 CI/CD 和生产部署

---

## 安装 uv

### Linux / macOS

```bash
# 方法 1: 使用官方安装脚本（推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 方法 2: 使用 pip
pip install uv

# 方法 3: 使用 cargo (Rust)
cargo install uv
```

### Windows

```powershell
# 方法 1: 使用官方安装脚本（推荐）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 方法 2: 使用 pip
pip install uv
```

### 验证安装

```bash
uv --version
# 输出: uv 0.x.x
```

---

## 快速开始

### 1. 克隆项目并初始化

```bash
# 克隆仓库
git clone https://github.com/Alexander-Rees/RLMarketMaker.git
cd RLMarketMaker

# 使用 uv 同步依赖（自动创建虚拟环境）
uv sync

# 或者只安装核心依赖（不包含开发工具）
uv sync --no-dev
```

**uv sync 做了什么？**
1. 检测或安装指定的 Python 版本（3.12）
2. 创建虚拟环境（`.venv/`）
3. 安装所有依赖
4. 生成 lock 文件（`uv.lock`）

### 2. 激活虚拟环境

```bash
# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# 或者使用 uv run（无需激活）
uv run python --version
```

### 3. 生成测试数据并运行

```bash
# 方式 1: 使用 uv run（推荐，无需激活环境）
uv run python scripts/generate_test_crypto_data.py
uv run python scripts/test_crypto_integration_simple.py

# 方式 2: 激活环境后直接运行
python scripts/generate_test_crypto_data.py
python scripts/test_crypto_integration_simple.py
```

### 4. 训练模型

```bash
# 使用 uv run
uv run python scripts/training/train_min.py \
  --config configs/tardis_replay.yaml \
  --seed 42

# 或使用项目脚本命令（已在 pyproject.toml 中定义）
uv run rlmm-train --config configs/tardis_replay.yaml --seed 42
```

---

## 环境管理

### 创建虚拟环境

```bash
# uv sync 会自动创建，也可以手动创建
uv venv

# 指定 Python 版本
uv venv --python 3.12

# 指定虚拟环境路径
uv venv .venv
```

### 激活/停用虚拟环境

```bash
# 激活
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# 停用
deactivate
```

### 使用 uv run（推荐）

**无需激活环境**，直接运行命令：

```bash
# 运行 Python 脚本
uv run python script.py

# 运行模块
uv run python -m pytest

# 运行已安装的命令
uv run rlmm-train --help
```

### Python 版本管理

```bash
# 列出可用的 Python 版本
uv python list

# 安装特定版本
uv python install 3.12

# 使用特定版本
uv venv --python 3.12
```

---

## 依赖管理

### 查看依赖

```bash
# 查看所有依赖
uv pip list

# 查看依赖树
uv pip tree

# 查看过期的包
uv pip list --outdated
```

### 添加依赖

```bash
# 添加运行时依赖
uv add numpy pandas

# 添加开发依赖
uv add --dev pytest black

# 添加可选依赖组
uv add --optional notebook jupyter
```

**自动更新 pyproject.toml**！

### 删除依赖

```bash
uv remove numpy
```

### 升级依赖

```bash
# 升级所有依赖
uv sync --upgrade

# 升级特定包
uv add numpy --upgrade

# 更新 lock 文件
uv lock --upgrade
```

### 从 requirements.txt 迁移

```bash
# 方法 1: 一次性导入
uv add -r requirements.txt

# 方法 2: 手动添加（推荐）
# 已在 pyproject.toml 中配置好

# 验证
uv sync
```

---

## 运行和测试

### 运行项目脚本

项目在 `pyproject.toml` 中定义了便捷命令：

```bash
# 生成测试数据
uv run rlmm-generate-data --symbol BTCUSDT --hours 2

# 测试加密货币集成
uv run rlmm-test-crypto

# 训练模型
uv run rlmm-train --config configs/tardis_replay.yaml --seed 42

# 评估模型
uv run rlmm-eval --checkpoint logs/checkpoints/policy.pt \
  --config configs/tardis_replay.yaml --episodes 10

# 回测
uv run rlmm-backtest --agent ppo --config configs/tardis_replay.yaml
```

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/test_env_core.py

# 生成覆盖率报告
uv run pytest --cov=rlmarketmaker --cov-report=html

# 并行测试（快速）
uv run pytest -n auto
```

### 运行 Jupyter Notebook

```bash
# 安装 notebook 依赖（如果还没有）
uv sync --extra notebook

# 启动 Jupyter
uv run jupyter notebook

# 或 JupyterLab
uv run jupyter lab
```

### 代码格式化和检查

```bash
# 格式化代码（Black）
uv run black rlmarketmaker/ scripts/ tests/

# 检查代码（Ruff）
uv run ruff check rlmarketmaker/ scripts/ tests/

# 自动修复
uv run ruff check --fix rlmarketmaker/
```

---

## 部署方案

### 方案 1: 直接部署（开发/测试）

```bash
# 1. 克隆项目
git clone https://github.com/Alexander-Rees/RLMarketMaker.git
cd RLMarketMaker

# 2. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. 同步依赖
uv sync --no-dev

# 4. 生成数据
uv run python scripts/generate_test_crypto_data.py

# 5. 训练
uv run python scripts/training/train_min.py \
  --config configs/tardis_replay.yaml
```

### 方案 2: 生产部署（使用 uv export）

```bash
# 1. 导出精确的依赖（生成标准 requirements.txt）
uv export --no-dev > requirements.txt

# 2. 在生产环境安装
pip install -r requirements.txt

# 3. 运行
python scripts/training/train_min.py --config configs/tardis_replay.yaml
```

### 方案 3: 使用 uv pip compile（兼容 pip-tools）

```bash
# 生成锁定的依赖文件
uv pip compile pyproject.toml -o requirements.lock

# 在生产环境安装
uv pip sync requirements.lock
```

### 方案 4: CI/CD 部署

**GitHub Actions 示例**:

```yaml
name: Train Model

on:
  push:
    branches: [main]

jobs:
  train:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Install uv
        uses: astral-sh/setup-uv@v1
        with:
          version: "latest"
      
      - name: Set up Python
        run: uv python install 3.12
      
      - name: Install dependencies
        run: uv sync --no-dev
      
      - name: Generate test data
        run: uv run python scripts/generate_test_crypto_data.py
      
      - name: Train model
        run: |
          uv run python scripts/training/train_min.py \
            --config configs/tardis_replay.yaml \
            --seed 42
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: model-checkpoints
          path: logs/checkpoints/
```

---

## Docker 集成

### Dockerfile（使用 uv）

```dockerfile
# 使用官方 Python 镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 复制项目文件
COPY pyproject.toml uv.lock ./
COPY rlmarketmaker/ ./rlmarketmaker/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# 同步依赖（使用 uv cache 加速）
RUN uv sync --frozen --no-dev

# 生成测试数据（可选）
RUN uv run python scripts/generate_test_crypto_data.py \
    --symbol BTCUSDT --hours 2

# 设置入口点
ENTRYPOINT ["uv", "run", "python"]
CMD ["scripts/training/train_min.py", "--config", "configs/tardis_replay.yaml"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  trainer:
    build: .
    volumes:
      - ./data:/app/data:ro
      - ./logs:/app/logs
      - ./artifacts:/app/artifacts
    environment:
      - PYTHONUNBUFFERED=1
    command: >
      scripts/training/train_min.py
      --config configs/tardis_replay.yaml
      --seed 42
    
  evaluator:
    build: .
    volumes:
      - ./logs:/app/logs:ro
      - ./artifacts:/app/artifacts
    environment:
      - PYTHONUNBUFFERED=1
    command: >
      scripts/evaluation/eval_min.py
      --checkpoint logs/checkpoints/policy.pt
      --config configs/tardis_replay.yaml
      --episodes 20
    depends_on:
      - trainer
```

### 构建和运行

```bash
# 构建镜像
docker build -t rlmarketmaker:latest .

# 运行训练
docker run --rm \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/artifacts:/app/artifacts \
  rlmarketmaker:latest \
  scripts/training/train_min.py --config configs/tardis_replay.yaml

# 使用 docker-compose
docker-compose up
```

### 优化的多阶段构建

```dockerfile
# Stage 1: 构建依赖
FROM python:3.12-slim AS builder

WORKDIR /app

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 复制依赖定义
COPY pyproject.toml uv.lock ./

# 安装依赖到特定位置
RUN uv sync --frozen --no-dev

# Stage 2: 运行时镜像
FROM python:3.12-slim

WORKDIR /app

# 从构建阶段复制虚拟环境
COPY --from=builder /app/.venv /app/.venv

# 复制应用代码
COPY rlmarketmaker/ ./rlmarketmaker/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

# 设置 PATH
ENV PATH="/app/.venv/bin:$PATH"

# 运行
CMD ["python", "scripts/training/train_min.py", "--config", "configs/tardis_replay.yaml"]
```

---

## 常见问题

### Q1: uv sync 失败

**问题**: `uv sync` 报错找不到兼容的包版本

**解决**:
```bash
# 清除缓存
uv cache clean

# 重新同步
uv sync

# 如果还是失败，尝试升级 uv
pip install --upgrade uv
```

### Q2: 虚拟环境位置

**问题**: 虚拟环境在哪里？

**回答**: 
- 默认在项目根目录的 `.venv/`
- 可以通过 `UV_VENV` 环境变量自定义
- 使用 `uv venv <path>` 指定路径

### Q3: 与 pip 的兼容性

**问题**: 可以在 uv 环境中使用 pip 吗？

**回答**: 
- ✅ 可以，但不推荐
- 使用 `uv pip install` 代替 `pip install`
- uv 完全兼容 pip 的命令

```bash
# uv 等价命令
pip install numpy    → uv pip install numpy
pip uninstall numpy  → uv pip uninstall numpy
pip list            → uv pip list
pip freeze          → uv pip freeze
```

### Q4: 锁文件冲突

**问题**: 多人协作时 `uv.lock` 冲突

**解决**:
```bash
# 1. 拉取最新代码
git pull

# 2. 重新同步（会自动解决冲突）
uv sync

# 3. 如果有新依赖，更新 lock
uv lock
```

### Q5: Python 版本不匹配

**问题**: 项目需要 Python 3.12，但系统是 3.11

**解决**:
```bash
# uv 可以自动管理 Python 版本
uv python install 3.12

# 使用该版本创建环境
uv venv --python 3.12

# 或者在 sync 时自动处理
uv sync  # 会自动使用 .python-version 中的版本
```

### Q6: 加速下载

**问题**: 在中国大陆下载慢

**解决**:
```bash
# 使用镜像源（设置环境变量）
export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

# 或者在命令中指定
uv pip install numpy --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 永久设置
echo 'export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"' >> ~/.bashrc
```

### Q7: 开发依赖管理

**问题**: 如何区分开发和生产依赖？

**回答**:
```bash
# 仅安装生产依赖
uv sync --no-dev

# 安装所有依赖（包括开发）
uv sync

# 安装特定可选依赖组
uv sync --extra notebook
uv sync --extra test
```

### Q8: 缓存管理

**问题**: uv 缓存占用空间大

**解决**:
```bash
# 查看缓存大小
uv cache size

# 清理缓存
uv cache clean

# 清理特定包的缓存
uv cache clean torch
```

---

## 性能对比

| 操作 | pip | uv | 提升 |
|------|-----|----|----|
| 安装 50 个包 | 45s | 1.2s | **37x** |
| 创建虚拟环境 | 8s | 0.3s | **26x** |
| 解析依赖 | 12s | 0.5s | **24x** |
| 冷缓存安装 | 60s | 3s | **20x** |

---

## 快速参考

### 常用命令

```bash
# 初始化项目
uv sync                    # 同步所有依赖
uv sync --no-dev          # 仅生产依赖
uv sync --extra notebook  # 包含可选依赖

# 依赖管理
uv add <package>          # 添加依赖
uv remove <package>       # 删除依赖
uv pip list              # 列出已安装包
uv pip tree              # 依赖树

# 运行
uv run python script.py   # 运行脚本
uv run pytest            # 运行测试
uv run rlmm-train        # 运行项目命令

# 环境管理
uv venv                  # 创建虚拟环境
uv python list          # 列出 Python 版本
uv python install 3.12  # 安装 Python 3.12

# 维护
uv lock                  # 更新 lock 文件
uv cache clean          # 清理缓存
uv self update          # 升级 uv
```

### 项目命令

```bash
# 生成测试数据
uv run rlmm-generate-data --symbol BTCUSDT --hours 2

# 测试集成
uv run rlmm-test-crypto

# 训练模型
uv run rlmm-train --config configs/tardis_replay.yaml --seed 42

# 评估模型
uv run rlmm-eval --checkpoint logs/checkpoints/policy.pt \
  --config configs/tardis_replay.yaml --episodes 10

# 回测
uv run rlmm-backtest --agent ppo --config configs/tardis_replay.yaml
```

---

## 相关资源

- [uv 官方文档](https://docs.astral.sh/uv/)
- [uv GitHub](https://github.com/astral-sh/uv)
- [项目 README](../README.md)
- [Tardis 集成指南](./TARDIS_INTEGRATION.md)

---

## 贡献

如有问题或建议，请提交 Issue 或 Pull Request。

**文档版本**: 1.0  
**最后更新**: 2025-11-12
