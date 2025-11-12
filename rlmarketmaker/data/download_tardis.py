#!/usr/bin/env python3
"""
Download crypto market data from Tardis.

使用Tardis API下载历史高频数据文件。
支持下载 trades, orderbook snapshots, book depth 等数据。
"""

import os
import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import argparse
import time


class TardisDownloader:
    """Tardis数据下载器"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.tardis.dev/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def download_dataset(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        date: str,
        output_dir: str
    ) -> Optional[str]:
        """
        下载单个数据集文件
        
        Args:
            exchange: 交易所名称 (e.g. "binance")
            symbol: 交易对 (e.g. "BTCUSDT")
            data_type: 数据类型 (e.g. "trades", "book_snapshot_5")
            date: 日期 (YYYY-MM-DD)
            output_dir: 输出目录
            
        Returns:
            下载的文件路径，失败返回 None
        """
        # 构造下载URL
        url = f"{self.base_url}/datasets/{exchange}/{data_type}/{date}/{symbol}.csv.gz"
        
        # 创建输出目录
        output_path = Path(output_dir) / exchange / symbol / data_type
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 输出文件名
        filename = f"{symbol}_{data_type}_{date}.csv.gz"
        output_file = output_path / filename
        
        # 检查文件是否已存在
        if output_file.exists():
            print(f"文件已存在: {output_file}")
            return str(output_file)
        
        print(f"下载数据: {exchange}/{symbol}/{data_type}/{date}")
        
        try:
            response = requests.get(url, headers=self.headers, stream=True, timeout=300)
            response.raise_for_status()
            
            # 流式下载文件
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = output_file.stat().st_size / 1024 / 1024  # MB
            print(f"✓ 下载完成: {output_file} ({file_size:.2f} MB)")
            return str(output_file)
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"✗ 数据不存在: {url}")
            else:
                print(f"✗ HTTP错误: {e}")
            return None
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            return None
    
    def download_multiple_days(
        self,
        exchange: str,
        symbol: str,
        data_type: str,
        start_date: str,
        end_date: str,
        output_dir: str
    ) -> List[str]:
        """
        下载多天的数据
        
        Args:
            exchange: 交易所名称
            symbol: 交易对
            data_type: 数据类型
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            output_dir: 输出目录
            
        Returns:
            成功下载的文件列表
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        downloaded_files = []
        current = start
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            file_path = self.download_dataset(
                exchange=exchange,
                symbol=symbol,
                data_type=data_type,
                date=date_str,
                output_dir=output_dir
            )
            
            if file_path:
                downloaded_files.append(file_path)
            
            # 避免API限流
            time.sleep(0.5)
            
            current += timedelta(days=1)
        
        return downloaded_files
    
    def get_available_exchanges(self) -> List[str]:
        """获取可用的交易所列表"""
        url = f"{self.base_url}/exchanges"
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            return [ex['id'] for ex in data]
        except Exception as e:
            print(f"获取交易所列表失败: {e}")
            return []


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='从Tardis下载加密货币历史数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载单日数据
  python download_tardis.py --exchange binance --symbol BTCUSDT \\
      --data-type trades --date 2024-01-15 --api-key YOUR_KEY
  
  # 下载多日数据
  python download_tardis.py --exchange binance --symbol ETHUSDT \\
      --data-type book_snapshot_5 --start-date 2024-01-01 \\
      --end-date 2024-01-07 --api-key YOUR_KEY
        """
    )
    
    parser.add_argument('--exchange', type=str, required=True,
                        help='交易所名称 (e.g. binance, coinbase)')
    parser.add_argument('--symbol', type=str, required=True,
                        help='交易对 (e.g. BTCUSDT, ETHUSDT)')
    parser.add_argument('--data-type', type=str, required=True,
                        help='数据类型 (e.g. trades, book_snapshot_5)')
    parser.add_argument('--date', type=str,
                        help='日期 (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str,
                        help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                        help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--api-key', type=str,
                        help='Tardis API密钥')
    parser.add_argument('--config', type=str, default='configs/api_keys.yaml',
                        help='配置文件路径')
    parser.add_argument('--output-dir', type=str, default='data/tardis',
                        help='输出目录')
    
    args = parser.parse_args()
    
    # 获取API密钥
    api_key = args.api_key
    if not api_key:
        # 从配置文件读取
        import yaml
        try:
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
                api_key = config['tardis']['api_key']
        except Exception as e:
            print(f"无法从配置文件读取API密钥: {e}")
            print("请使用 --api-key 参数提供API密钥")
            return
    
    # 创建下载器
    downloader = TardisDownloader(api_key=api_key)
    
    # 下载数据
    if args.date:
        # 下载单日
        downloader.download_dataset(
            exchange=args.exchange,
            symbol=args.symbol,
            data_type=args.data_type,
            date=args.date,
            output_dir=args.output_dir
        )
    elif args.start_date and args.end_date:
        # 下载多日
        files = downloader.download_multiple_days(
            exchange=args.exchange,
            symbol=args.symbol,
            data_type=args.data_type,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir
        )
        print(f"\n总共下载 {len(files)} 个文件")
    else:
        print("错误: 请提供 --date 或 --start-date/--end-date")


if __name__ == '__main__':
    main()
