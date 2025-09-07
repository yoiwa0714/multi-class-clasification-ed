#!/usr/bin/env python3
"""
multi_ed_v020.py
純正ED法（Error Diffusion Learning Algorithm）Python実装 v0.2.0
Original C implementation by Isamu Kaneko (1999)
"""

import numpy as np
import random
import math
import time
import argparse
import os
import datetime
from typing import List, Tuple, Optional
from tqdm import tqdm

# ED-Genuine モジュールインポート
from modules.ed_core import EDGenuine
from modules.network_mnist import EDNetworkMNIST
from modules.visualization import RealtimeLearningVisualizer, RealtimeConfusionMatrixVisualizer
from modules.data_loader import MiniBatchDataLoader
from modules.performance import TrainingProfiler, LearningResultsBuffer

# ハイパーパラメータ管理クラス（ed_genuine.prompt.md準拠）
class HyperParams:
    """
    ED法ハイパーパラメータ管理クラス
    金子勇氏オリジナル仕様のデフォルト値を保持し、実行時引数での変更を可能にする
    """
    
    def __init__(self):
        """デフォルト値設定（C実装準拠）"""
        # ED法関連パラメータ（金子勇氏オリジナル値）
        self.learning_rate = 0.8      # 学習率 (alpha)
        self.initial_amine = 0.3      # 初期アミン濃度 (beta) - ed_genuine.prompt.md準拠値
        self.diffusion_rate = 1.0     # アミン拡散係数 (u1)
        self.sigmoid_threshold = 0.4  # シグモイド閾値 (u0)
        self.initial_weight_1 = 1.0   # 重み初期値1
        self.initial_weight_2 = 1.0   # 重み初期値2
        
        # 実行時パラメータ
        self.train_samples = 100      # 訓練データ数
        self.test_samples = 100       # テストデータ数
        self.epochs = 3               # エポック数（デフォルト3に変更）
        self.hidden_neurons = 128     # 隠れ層ニューロン数
        self.batch_size = 32          # ミニバッチサイズ（新機能：金子勇氏理論拡張）
        self.random_seed = None       # ランダムシード（Noneはランダム）
        self.enable_visualization = False  # 精度/誤差可視化
        self.enable_profiling = False # 詳細プロファイリング（パフォーマンス分析用）
        self.verbose = False          # 詳細表示
        self.quiet_mode = False       # 簡潔出力モード（グリッドサーチ用）
        self.force_cpu = False        # CPU強制実行モード
        self.fashion_mnist = False    # Fashion-MNISTデータセット使用
        self.save_fig = None          # 図表保存ディレクトリ (None: 無効, str: ディレクトリ指定)
    
    def parse_args(self, args=None):
        """
        argparseによるハイパーパラメータ解析
        ed_genuine.prompt.md準拠: アルゴリズムの完全性を保持
        """
        parser = argparse.ArgumentParser(
            description='純正ED法（Error Diffusion Learning Algorithm）実行 v0.1.8',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
ED法ハイパーパラメータ説明:
  学習率(alpha): ニューロンの学習強度を制御
  アミン濃度(beta): 初期誤差信号の強度
  拡散係数(u1): アミン（誤差信号）の拡散率
  シグモイド閾値(u0): 活性化関数の感度
  
Original Algorithm: 金子勇 (1999)
Implementation: Python with ed_genuine.prompt.md compliance
            """
        )
        
        # ED法関連パラメータ群（機能順配置）
        ed_group = parser.add_argument_group('ED法アルゴリズムパラメータ')
        ed_group.add_argument('--learning_rate', '--lr', type=float, default=self.learning_rate,
                             help=f'学習率 alpha (デフォルト: {self.learning_rate})')
        ed_group.add_argument('--amine', '--ami', type=float, default=self.initial_amine,
                             help=f'初期アミン濃度 beta (デフォルト: {self.initial_amine})')
        ed_group.add_argument('--diffusion', '--dif', type=float, default=self.diffusion_rate,
                             help=f'アミン拡散係数 u1 (デフォルト: {self.diffusion_rate})')
        ed_group.add_argument('--sigmoid', '--sig', type=float, default=self.sigmoid_threshold,
                             help=f'シグモイド閾値 u0 (デフォルト: {self.sigmoid_threshold})')
        ed_group.add_argument('--weight1', '--w1', type=float, default=self.initial_weight_1,
                             help=f'重み初期値1 (デフォルト: {self.initial_weight_1})')
        ed_group.add_argument('--weight2', '--w2', type=float, default=self.initial_weight_2,
                             help=f'重み初期値2 (デフォルト: {self.initial_weight_2})')
        
        # 実行時パラメータ群（機能順配置）
        exec_group = parser.add_argument_group('実行時設定パラメータ')
        exec_group.add_argument('--train_samples', '--train', type=int, default=self.train_samples,
                               help=f'訓練データ数 (デフォルト: {self.train_samples})')
        exec_group.add_argument('--test_samples', '--test', type=int, default=self.test_samples,
                               help=f'テストデータ数 (デフォルト: {self.test_samples})')
        exec_group.add_argument('--epochs', '--epo', type=int, default=self.epochs,
                               help=f'エポック数 (デフォルト: {self.epochs})')
        exec_group.add_argument('--hidden', '--hid', type=int, default=self.hidden_neurons,
                               help=f'隠れ層ニューロン数 (デフォルト: {self.hidden_neurons})')
        exec_group.add_argument('--batch_size', '--batch', type=int, default=self.batch_size,
                               help=f'ミニバッチサイズ (デフォルト: {self.batch_size}) - 金子勇氏理論拡張')
        exec_group.add_argument('--seed', type=int, default=self.random_seed,
                               help=f'ランダムシード (デフォルト: ランダム)')
        exec_group.add_argument('--viz', action='store_true', default=self.enable_visualization,
                               help='リアルタイム可視化を有効化 (デフォルト: 無効)')
        exec_group.add_argument('--verbose', '--v', action='store_true', default=self.verbose,
                               help='詳細表示を有効化 (デフォルト: 無効)')
        exec_group.add_argument('--quiet', '--q', action='store_true', default=False,
                               help='簡潔出力モード - グリッドサーチ用 (デフォルト: 無効)')
        exec_group.add_argument('--profile', '--p', action='store_true', default=False,
                               help='訓練時間詳細プロファイリング有効化 (デフォルト: 無効)')
        exec_group.add_argument('--cpu', action='store_true', default=self.force_cpu,
                               help='CPU強制実行モード (GPU無効化、デフォルト: 無効)')
        exec_group.add_argument('--fashion', action='store_true', default=False,
                               help='Fashion-MNISTデータセット使用 (デフォルト: 通常MNIST)')
        exec_group.add_argument('--save_fig', nargs='?', const='images', default=None,
                               help='図表保存を有効化 (引数なし: ./images, 引数あり: 指定ディレクトリ)')
        
        # 引数解析
        parsed_args = parser.parse_args(args)
        
        # パラメータ値の更新
        self.learning_rate = parsed_args.learning_rate
        self.initial_amine = parsed_args.amine
        self.diffusion_rate = parsed_args.diffusion
        self.sigmoid_threshold = parsed_args.sigmoid
        self.initial_weight_1 = parsed_args.weight1
        self.initial_weight_2 = parsed_args.weight2
        
        self.train_samples = parsed_args.train_samples
        self.test_samples = parsed_args.test_samples
        self.epochs = parsed_args.epochs
        self.hidden_neurons = parsed_args.hidden
        self.batch_size = parsed_args.batch_size
        self.random_seed = parsed_args.seed
        self.enable_visualization = parsed_args.viz
        self.verbose = parsed_args.verbose
        self.quiet_mode = parsed_args.quiet
        self.enable_profiling = parsed_args.profile
        self.force_cpu = parsed_args.cpu
        self.fashion_mnist = parsed_args.fashion
        self.save_fig = getattr(parsed_args, 'save_fig', None)
        
        return parsed_args
    
    def set_random_seed(self):
        """
        ランダムシード設定（再現性確保）
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            random.seed(self.random_seed)
            # NOTE: mathモジュールはシード設定をサポートしていない
            if self.verbose:
                print(f"ランダムシード設定: {self.random_seed}")
        else:
            if self.verbose:
                print("ランダムシード: 未設定（ランダム）")
    
    def validate_params(self):
        """
        パラメータ妥当性検証（ed_genuine.prompt.md準拠）
        生物学的制約とアルゴリズム制約のチェック
        """
        errors = []
        
        # ED法パラメータ制約
        if self.learning_rate <= 0:
            errors.append("学習率は正の値である必要があります")
        if self.initial_amine <= 0:
            errors.append("初期アミン濃度は正の値である必要があります")
        if self.diffusion_rate <= 0:
            errors.append("アミン拡散係数は正の値である必要があります")
        if self.sigmoid_threshold <= 0:
            errors.append("シグモイド閾値は正の値である必要があります")
        
        # 実行時パラメータ制約
        if self.train_samples <= 0:
            errors.append("訓練データ数は正の整数である必要があります")
        if self.test_samples <= 0:
            errors.append("テストデータ数は正の整数である必要があります")
        if self.epochs <= 0:
            errors.append("エポック数は正の整数である必要があります")
        if self.hidden_neurons <= 0:
            errors.append("隠れ層ニューロン数は正の整数である必要があります")
            
        # 実用的制約（メモリ・計算量）
        if self.train_samples > 10000:
            errors.append("訓練データ数は10000以下を推奨します")
        if self.test_samples > 10000:
            errors.append("テストデータ数は10000以下を推奨します")
        if self.hidden_neurons > 1000:
            errors.append("隠れ層ニューロン数は1000以下を推奨します")
        
        # 可視化オプション制約チェック
        if self.enable_visualization and self.epochs < 3:
            print("⚠️ --vizオプションは3エポック以上でないと使用できません。")
            print("   可視化オプションを無効にして実行を継続します。")
            self.enable_visualization = False
            
        if errors:
            raise ValueError("パラメータエラー:\n" + "\n".join(f"  - {error}" for error in errors))
    
    def display_config(self):
        """設定パラメータの表示"""
        print("=" * 60)
        print("ED法実行設定")
        print("=" * 60)
        print("【ED法アルゴリズムパラメータ】")
        print(f"  学習率 (alpha):         {self.learning_rate:.3f}")
        print(f"  初期アミン濃度 (beta):  {self.initial_amine:.3f}")
        print(f"  アミン拡散係数 (u1):    {self.diffusion_rate:.3f}")
        print(f"  シグモイド閾値 (u0):    {self.sigmoid_threshold:.3f}")
        print(f"  重み初期値1:            {self.initial_weight_1:.3f}")
        print(f"  重み初期値2:            {self.initial_weight_2:.3f}")
        print()
        print("【実行時設定パラメータ】")
        print(f"  データセット:           {'Fashion-MNIST' if self.fashion_mnist else 'MNIST'}")
        print(f"  訓練データ数:           {self.train_samples}")
        print(f"  テストデータ数:         {self.test_samples}")
        print(f"  エポック数:             {self.epochs}")
        print(f"  隠れ層ニューロン数:     {self.hidden_neurons}")
        print(f"  ミニバッチサイズ:       {self.batch_size} {'(逐次処理)' if self.batch_size == 1 else '(ミニバッチ)'}")
        print(f"  リアルタイム可視化:     {'ON' if self.enable_visualization else 'OFF'}")
        print(f"  詳細表示:               {'ON' if self.verbose else 'OFF'}")
        print(f"  図表保存:               {'ON -> ' + self.save_fig if self.save_fig else 'OFF'}")
        print("=" * 60)

# 可視化ライブラリ - 日本語フォント対応
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
import matplotlib.font_manager as fm

# 日本語フォント設定（ed_genuine.prompt.md準拠 - 最適化版）
def setup_japanese_font():
    """
    利用可能な日本語フォントを自動検出して設定
    ed_genuine.prompt.md仕様: 日本語化Linuxの標準フォント使用
    """
    try:
        # システム内の利用可能フォント一覧を取得
        available_fonts = set([f.name for f in fm.fontManager.ttflist])
        
        # 日本語フォント候補（優先度順）
        japanese_font_candidates = [
            'Noto Sans CJK JP',   # Ubuntu/Debian標準
            'Noto Sans JP',       # Ubuntu/Debian代替
            'DejaVu Sans',        # 一般的なLinux
            'Liberation Sans',    # Red Hat系標準
            'TakaoGothic',        # CentOS/RHEL（存在時のみ）
            'VL Gothic',          # その他日本語（存在時のみ）
        ]
        
        # 実際に利用可能な日本語フォントを選択
        selected_font = None
        for font in japanese_font_candidates:
            if font in available_fonts:
                selected_font = font
                break
        
        # フォント設定（存在するフォントのみ）
        if selected_font:
            rcParams['font.family'] = [selected_font, 'sans-serif']
            print(f"✅ 日本語フォント検出・設定完了: {selected_font}")
        else:
            rcParams['font.family'] = ['sans-serif']
            print("⚠️ 日本語フォント未検出: デフォルトフォント使用")
        
        rcParams['axes.unicode_minus'] = False
        
        # matplotlib警告を最小化
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
        
    except Exception as e:
        print(f"フォント設定エラー: {e}")
        rcParams['font.family'] = ['sans-serif']
        rcParams['axes.unicode_minus'] = False

# フォント設定実行
setup_japanese_font()

# MNIST データセット読み込み用
try:
    import torch
    import torchvision
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
    print("torchvision検出: MNISTデータセット利用可能")
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("torchvision未インストール: MNISTデータセット利用不可")

# GPU基盤実装（Phase GPU-1）
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy検出: GPU高速化機能利用可能")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy未インストール: CPU版のみ利用可能")

# 可視化クラスはmodules/visualization.pyに移動
# データローダークラスはmodules/data_loader.pyに移動

def run_classification(hyperparams=None):
    """
    MNIST/Fashion-MNIST分類実行関数 - ハイパーパラメータ対応版
    Args:
        hyperparams: HyperParamsインスタンス（Noneの場合はデフォルト使用）
    """
    if hyperparams is None:
        hyperparams = HyperParams()
    
    # ランダムシード設定（再現性確保）
    hyperparams.set_random_seed()
    
    # ネットワーク作成（ハイパーパラメータ渡し）
    network = EDNetworkMNIST(hyperparams)
    
    # 分類実行 - ハイパーパラメータから設定取得
    results = network.run_classification(
        random_state=42  # ランダムシードは固定
    )
    
    print(f"\n最終結果: 精度 {results['final_accuracy']:.3f}")
    return results


def main():
    """
    メイン実行関数 - MNIST/Fashion-MNIST分類専用版
    
    【v0.1.8実行仕様】
    - MNIST/Fashion-MNISTデータセット対応
    - 28×28画像パターン（784次元）、10クラス分類
    - ハイパーパラメータコマンドライン制御対応
    - 混同行列可視化機能完全対応
    - 今後の開発ベースファイルとして最適化
    
    【ed_genuine.prompt.md準拠実装】
    - 独立出力ニューロンアーキテクチャ保持
    - アミン拡散学習制御継承
    - 金子勇氏オリジナル仕様完全準拠
    """
    pass  # メインロジックはif __name__ == "__main__"で実行


if __name__ == "__main__":
    # ハイパーパラメータ解析
    hyperparams = HyperParams()
    
    try:
        # コマンドライン引数解析
        args = hyperparams.parse_args()
        
        # パラメータ妥当性検証
        hyperparams.validate_params()
        
        # 設定表示（quietモード以外）
        if not hyperparams.quiet_mode:
            hyperparams.display_config()
        
        # 分類実行（引数指定可能）
        if TORCHVISION_AVAILABLE:
            results = run_classification(hyperparams)
            
            # 実行結果表示
            if hyperparams.verbose:
                print("\n実行完了 - ハイパーパラメータによる柔軟な設定対応")
                print(f"使用パラメータ: lr={hyperparams.learning_rate}, "
                      f"epochs={hyperparams.epochs}, "
                      f"hidden={hyperparams.hidden_neurons}, "
                      f"dataset={'Fashion-MNIST' if hyperparams.fashion_mnist else 'MNIST'}")
        else:
            print("❌ 分類テストにはtorchvisionが必要です:")
            print("   pip install torchvision")
            exit(1)
            
    except ValueError as e:
        print(f"❌ パラメータエラー: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ 実行が中断されました")
        exit(0)



if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        exit(1)
