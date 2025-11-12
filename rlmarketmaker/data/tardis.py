"""Tardis 数据下载与回放支持。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Iterator, List, Optional

import numpy as np
import pandas as pd
from tardis_dev import datasets

from .feeds import Feed, MarketTick


class TardisDataProcessor:
    """下载并处理 Tardis 报价数据。"""

    def __init__(
        self,
        exchange: str = "binance",
        symbol: str = "btcusdt",
        raw_dir: str = "data/tardis_raw",
        processed_dir: str = "data/tardis",
    ):
        self.exchange = exchange.lower()
        self.symbol = symbol.lower()
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_quotes(
        self,
        start_date: str, end_date: str, *, api_key: Optional[str] = None, force: bool = False
    ) -> None:
        """下载指定日期区间的 quotes 数据。"""
        key = api_key or os.getenv("TARDIS_API_KEY")
        if not key:
            raise RuntimeError("TARDIS_API_KEY not provided. Set env var or pass api_key.")

        expected_files = list(self._expected_files(start_date, end_date))
        if not force:
            missing = [path for path in expected_files if not path.exists()]
            if not missing:
                return

        datasets.download(
            exchange=self.exchange,
            data_types=["quotes"],
            symbols=[self.symbol],
            from_date=start_date,
            to_date=end_date,
            format="csv",
            api_key=key,
            download_dir=str(self.raw_dir),
        )

    def build_market_ticks(
        self,
        start_date: str,
        end_date: str,
        *, resample_rule: Optional[str] = "250ms", downsample: int = 1, max_rows: Optional[int] = None, output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """生成符合 MarketTick 字段的 DataFrame。"""
        if downsample < 1:
            raise ValueError("downsample must be >= 1")

        frames: List[pd.DataFrame] = []
        for raw_file in self._expected_files(start_date, end_date):
            if not raw_file.exists():
                raise FileNotFoundError(f"Missing raw quotes file: {raw_file}")
            df = pd.read_csv(raw_file)
            if df.empty:
                continue

            df["datetime"] = pd.to_datetime(df["timestamp"], unit="us", utc=True)
            df["midprice"] = (df["ask_price"] + df["bid_price"]) / 2.0
            df["spread"] = (df["ask_price"] - df["bid_price"]).clip(lower=0.0)
            df["bid_size"] = df["bid_amount"].clip(lower=0.0)
            df["ask_size"] = df["ask_amount"].clip(lower=0.0)

            if resample_rule:
                df = (
                    df.set_index("datetime")[["midprice", "spread", "bid_size", "ask_size"]]
                    .resample(resample_rule)
                    .last()
                    .dropna()
                    .reset_index()
                )
            else:
                df = df[["datetime", "midprice", "spread", "bid_size", "ask_size"]]

            if df.empty:
                continue

            df["timestamp"] = df["datetime"].astype("int64") / 1e9
            df["trades"] = 0  # Quotes dataset has no trade counts
            frames.append(df[["timestamp", "midprice", "spread", "bid_size", "ask_size", "trades"]])

        if not frames:
            raise RuntimeError("No data loaded from raw files. Check date range and downloads.")

        combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

        if downsample > 1:
            combined = combined.iloc[::downsample].reset_index(drop=True)

        if max_rows is not None:
            combined = combined.head(max_rows)

        if output_path:
            output = Path(output_path)
            if not output.is_absolute():
                output = self.processed_dir / output
            output.parent.mkdir(parents=True, exist_ok=True)
            combined.to_parquet(output, index=False)

        return combined
    def _expected_files(self, start_date: str, end_date: str) -> Iterator[Path]:
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        if end <= start:
            raise ValueError("end_date must be after start_date")

        dates = pd.date_range(start=start, end=end - pd.Timedelta(days=1), freq="D")
        symbol = self.symbol.replace("/", "-").replace(":", "-").upper()

        for date in dates:
            filename = f"{self.exchange}_quotes_{date.strftime('%Y-%m-%d')}_{symbol}.csv.gz"
            yield self.raw_dir / filename


class TardisReplayFeed(Feed):
    """Replay feed built on top of preprocessed Tardis data."""

    def __init__(self, data_path: Optional[str] = None, seed: Optional[int] = None):
        self.data_path = data_path
        self.rng = np.random.RandomState(seed)
        self._cached_df: Optional[pd.DataFrame] = None
        self._cached_path: Optional[Path] = None

    def get_env_feed(self, config: Dict[str, Any]) -> Iterator[MarketTick]:
        """Replay preprocessed quote data with light domain randomization."""
        data_path = config.get("data_path") or self.data_path
        if data_path is None:
            raise ValueError("TardisReplayFeed requires 'data_path' in config or constructor.")

        df = self._load_data(Path(data_path))
        if df.empty:
            raise RuntimeError(f"No data found in {data_path}")

        episode_length = config.get("episode_length", min(3600, len(df)))
        warmup_steps = config.get("warmup_steps", min(100, len(df) // 10))

        # Domain randomization multipliers
        vol_low, vol_high = config.get("volatility_multiplier", [0.9, 1.1])
        spr_low, spr_high = config.get("spread_multiplier", [0.8, 1.2])
        vol_mult = float(self.rng.uniform(vol_low, vol_high))
        spread_mult = float(self.rng.uniform(spr_low, spr_high))

        max_start = max(1, len(df) - episode_length)
        start_idx = 0 if max_start <= warmup_steps else int(self.rng.randint(warmup_steps, max_start))
        subset = df.iloc[start_idx : start_idx + episode_length].copy()
        subset["midprice"] *= vol_mult
        subset["spread"] *= spread_mult

        for row in subset.itertuples(index=False):
            yield MarketTick(
                timestamp=float(row.timestamp),
                midprice=float(row.midprice),
                spread=float(row.spread),
                bid_size=float(row.bid_size),
                ask_size=float(row.ask_size),
                trades=int(row.trades),
            )

    def _load_data(self, path: Path) -> pd.DataFrame:
        if self._cached_df is not None and self._cached_path == path:
            return self._cached_df
        if not path.exists():
            raise FileNotFoundError(f"Tardis data file not found: {path}")
        df = pd.read_parquet(path)
        required = {"timestamp", "midprice", "spread", "bid_size", "ask_size", "trades"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Tardis data missing required columns: {missing}")
        self._cached_df = df.reset_index(drop=True)
        self._cached_path = path
        return self._cached_df

