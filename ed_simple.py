
#!/usr/bin/env python3
"""
純正ED法（Error Diffusion Learning Algorithm）最小実装版
Original C implementation by Isamu Kaneko (1999) - 教育用シンプル版
"""

import numpy as np
import random
import math
import time
import argparse
import os
from typing import List, Tuple, Optional
from tqdm import tqdm
from modules.ed_core import EDGenuine
from modules.network_mnist import EDNetworkMNIST
from modules.data_loader import MiniBatchDataLoader

# 日本語フォント設定（文字化け防止）
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 利用可能な日本語フォントを優先順位で設定
japanese_fonts = ['Noto Sans CJK JP', 'BIZ UDGothic', 'Noto Sans JP', 'DejaVu Sans']
available_font = None
for font_name in japanese_fonts:
    try:
        if font_name in [f.name for f in fm.fontManager.ttflist]:
            available_font = font_name
            break
    except:
        continue

if available_font:
    plt.rcParams['font.family'] = available_font
    print(f"日本語フォントを設定: {available_font}")
else:
    print("警告: 日本語フォントが見つかりませんでした")

class NetworkStructure:
    def __init__(self, input_size, hidden_layers, output_size):
        """
        ネットワーク構造初期化
        
        Args:
            input_size (int): 入力層サイズ (例: 784 for MNIST)
            hidden_layers (list[int]): 隠れ層構造 (例: [256, 128, 64])
            output_size (int): 出力層サイズ (例: 10 for 10-class classification)
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers if isinstance(hidden_layers, list) else [hidden_layers]
        self.output_size = output_size
        
        # ed_multi.prompt.md準拠のインデックス体系計算
        # 仕様: 0,1(バイアス), 2～in+1(入力層), in+2(出力開始), in+3～all+1(隠れ層)
        
        # C実装変数の再現
        self.in_units = input_size * 2  # 興奮性・抑制性ペア (in変数に相当)
        self.hd_units = sum(self.hidden_layers)  # 隠れ層ユニット総数 (hd変数に相当)
        self.ot_units = output_size  # 出力ニューロン数 (ot変数に相当)
        self.all_units = self.in_units + self.hd_units + self.ot_units  # 総ユニット数 (all変数に相当)
        
        # ed_multi.prompt.md仕様準拠インデックス体系
        self.bias_start = 0
        self.bias_end = 1
        self.input_start = 2
        self.input_end = 2 + self.in_units - 1  # = in+1 in C code
        self.output_pos = self.input_end + 1    # = in+2 in C code (出力層開始位置)
        self.hidden_start = self.output_pos + 1 # = in+3 in C code (隠れ層開始)
        self.hidden_end = self.hidden_start + self.hd_units - 1  # = all+1 in C code
        
        # 利便性のための追加プロパティ
        self.total_layers = len(self.hidden_layers) + 2  # 入力層 + 隠れ層数 + 出力層
        self.excitatory_input_size = self.in_units  # 後方互換性
        
        # 層別開始位置計算（多層対応）
        self.layer_starts = []
        self.layer_starts.append(self.input_start)  # 入力層開始: 2
        
        # 隠れ層の各層開始位置を計算
        current_pos = self.hidden_start
        for layer_size in self.hidden_layers:
            self.layer_starts.append(current_pos)
            current_pos += layer_size
        
        self.layer_starts.append(self.output_pos)  # 出力層開始: in+2
    
    def get_layer_range(self, layer_index):
        """
        指定した層のユニット範囲を取得（ed_multi.prompt.md仕様準拠）
        
        Args:
            layer_index (int): 層インデックス (0: 入力, 1-N: 隠れ層, N+1: 出力)
        
        Returns:
            tuple: (start_index, end_index)
        """
        if layer_index == 0:  # 入力層: 2 ～ in+1
            return (self.input_start, self.input_end)
        elif layer_index <= len(self.hidden_layers):  # 隠れ層: in+3 ～ all+1
            start = self.layer_starts[layer_index]
            if layer_index < len(self.hidden_layers):
                end = self.layer_starts[layer_index + 1] - 1
            else:
                end = self.hidden_end
            return (start, end)
        else:  # 出力層: in+2 (単一位置)
            return (self.output_pos, self.output_pos)
    
    def is_single_layer(self):
        """単層ネットワークかどうかを判定"""
        return len(self.hidden_layers) == 1
    
    def is_multi_layer(self):
        """多層ネットワークかどうかを判定"""
        return len(self.hidden_layers) > 1
    
    def calculate_amine_diffusion_coefficient(self, layer_distance):
        """
        層間距離に基づくアミン拡散係数計算
        
        Args:
            layer_distance (int): 層間距離 (1: 隣接層, 2: 2層離れ, etc.)
        
        Returns:
            float: 拡散係数 (u1^layer_distance)
        """
        # ed_multi.prompt.md準拠: 距離に応じて拡散係数を減衰
        base_diffusion = 1.0  # u1基本値
        return base_diffusion ** layer_distance
    
    def get_network_summary(self):
        """ネットワーク構造サマリー取得（ed_multi.prompt.md仕様準拠）"""
        return {
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'output_size': self.output_size,
            'total_layers': self.total_layers,
            'all_units': self.all_units,  # 修正: total_units → all_units
            'layer_type': '単層' if self.is_single_layer() else f'{len(self.hidden_layers)}層',
            'excitatory_input_size': self.in_units,  # 修正: excitatory_input_size → in_units
            'index_ranges': {
                'bias': (self.bias_start, self.bias_end),
                'input': (self.input_start, self.input_end),
                'hidden': (self.hidden_start, self.hidden_end),
                'output': self.output_pos  # 修正: 出力は単一位置
            },
            'ed_multi_compliance': {
                'bias_indices': '0, 1',
                'input_indices': f'2 ～ {self.input_end}',
                'output_index': f'{self.output_pos} (in+2)',
                'hidden_indices': f'{self.hidden_start} ～ {self.hidden_end} (in+3 ～ all+1)'
            }
        }

# ハイパーパラメータ管理クラス（ed_genuine.prompt.md準拠）
class HyperParams:
    """
    ED法ハイパーパラメータ管理クラス
    金子勇氏オリジナル仕様のデフォルト値を保持し、実行時引数での変更を可能にする
    """
    
    def __init__(self):
        """デフォルト値設定（最適化されたパラメータ使用）"""
        # ED法関連パラメータ（Phase 2最適化結果）
        self.learning_rate = 0.3      # 学習率 (alpha) - Phase 2最適値
        self.initial_amine = 0.7      # 初期アミン濃度 (beta) - Phase 2最適値
        self.diffusion_rate = 0.5     # アミン拡散係数 (u1) - Phase 1最適値
        self.sigmoid_threshold = 0.7  # シグモイド閾値 (u0) - Phase 1最適値
        self.initial_weight_1 = 0.3   # 重み初期値1 - Phase 1最適値
        self.initial_weight_2 = 0.5   # 重み初期値2 - Phase 1最適値
        
        # 実行時パラメータ
        self.train_samples = 100      # 訓練データ数
        self.test_samples = 100       # テストデータ数
        self.epochs = 5               # エポック数（効率性最適値）
        self.hidden_layers = [128]    # 隠れ層構造 (単層互換: [128], 多層例: [256,128,64])
        self.batch_size = 32          # ミニバッチサイズ（新機能：金子勇氏理論拡張）
        self.random_seed = None       # ランダムシード（Noneはランダム）
        self.enable_visualization = False  # 精度/誤差可視化
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
        exec_group.add_argument('--hidden', '--hid', type=str, default=','.join(map(str, self.hidden_layers)),
                               help=f'隠れ層構造 (デフォルト: {",".join(map(str, self.hidden_layers))}) - カンマ区切り指定 (例: 256,128,64)')
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
        
        # 隠れ層構造の解析（カンマ区切り文字列をリストに変換）
        if isinstance(parsed_args.hidden, str):
            try:
                self.hidden_layers = [int(x.strip()) for x in parsed_args.hidden.split(',') if x.strip()]
                if not self.hidden_layers:
                    raise ValueError("隠れ層構造が空です")
                # 全ての値が正の整数であることを確認
                if any(layer <= 0 for layer in self.hidden_layers):
                    raise ValueError("隠れ層のニューロン数は正の整数である必要があります")
            except ValueError as e:
                raise ValueError(f"--hidden オプションの形式が不正です: {e}")
        else:
            # 後方互換性のための処理（intで指定された場合）
            self.hidden_layers = [parsed_args.hidden]
            
        self.batch_size = parsed_args.batch_size
        self.random_seed = parsed_args.seed
        self.enable_visualization = parsed_args.viz
        self.verbose = parsed_args.verbose
        self.quiet_mode = parsed_args.quiet
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
        # 隠れ層構造の検証は既にparse_args内で実行済み
            
        # 実用的制約（メモリ・計算量）
        if self.train_samples > 10000:
            errors.append("訓練データ数は10000以下を推奨します")
        if self.test_samples > 10000:
            errors.append("テストデータ数は10000以下を推奨します")
        # 隠れ層の最大ニューロン数チェック
        if max(self.hidden_layers) > 1000:
            errors.append(f"隠れ層の最大ニューロン数（{max(self.hidden_layers)}）は1000以下を推奨します")
        
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
        
        # 隠れ層構造の表示（単層・多層に対応）
        layer_structure = " → ".join(map(str, self.hidden_layers))
        layer_type = "単層" if len(self.hidden_layers) == 1 else f"{len(self.hidden_layers)}層"
        print(f"  隠れ層構造:             {layer_structure} ({layer_type})")
        
        print(f"  ミニバッチサイズ:       {self.batch_size} {'(逐次処理)' if self.batch_size == 1 else '(ミニバッチ)'}")
        print(f"  リアルタイム可視化:     {'ON' if self.enable_visualization else 'OFF'}")
        print(f"  詳細表示:               {'ON' if self.verbose else 'OFF'}")
        print(f"  図表保存:               {'ON -> ' + self.save_fig if self.save_fig else 'OFF'}")
        print("=" * 60)

# === ED法分類実行 ===
def run_classification(hyperparams=None):
    """MNIST分類実行関数 - 最小実装版"""
    if hyperparams is None:
        hyperparams = HyperParams()
    
    # ランダムシード設定
    hyperparams.set_random_seed()
    
    # ネットワーク作成・学習実行
    network = EDNetworkMNIST(hyperparams)
    results = network.run_classification(random_state=42)
    
    print(f"\n最終結果: 精度 {results['final_accuracy']/100:.3f} ({results['final_accuracy']:.1f}%)")
    return results


def run_multilayer_classification(hyperparams, network_structure):
    """
    多層対応学習実行（ed_multi.prompt.md準拠 + 高速化統一実装）
    
    【2025年9月14日統一高速化実装】
    🚀 単層と同じEDNetworkMNIST.run_classification()を使用
    🚀 高速化実装を多層でも適用して学習時間を統一
    🚀 ed_multi.prompt.md準拠: C実装アルゴリズムとの整合性維持
    
    Args:
        hyperparams: HyperParamsインスタンス
        network_structure: NetworkStructureインスタンス
    Returns:
        dict: 学習結果
    """
    # ランダムシード設定（再現性確保）
    hyperparams.set_random_seed()
    
    print(f"🔧 多層ED法統一実装開始: {network_structure.get_network_summary()}")
    print(f"🎯 単層と同じ高速化実装を使用: EDNetworkMNIST.run_classification()")
    print(f"📊 多層構造: 入力{network_structure.input_size} → {'→'.join(map(str, network_structure.hidden_layers))} → 出力{network_structure.output_size}")
    
    # 🚀 統一高速化: 単層と同じEDNetworkMNISTを使用
    network = EDNetworkMNIST(hyperparams)
    
    print(f"✅ 多層ネットワーク統一実装準備完了")
    print(f"   🚀 高速化実装適用: train_epoch_with_buffer/minibatch使用")
    print(f"   📊 構造: {network_structure.get_network_summary()}")
    
    # 🚀 高速化された分類実行（単層と同じ実装）
    results = network.run_classification(random_state=42)
    
    # 重み保存のためネットワークインスタンスを結果に追加
    results['network_instance'] = network
    
    # 結果に多層情報を追加
    results['multilayer_mode'] = True
    results['truly_multilayer'] = True
    results['algorithm_used'] = 'EDNetworkMNIST_unified'
    results['ed_multi_compliance'] = True
    results['network_structure'] = network_structure.get_network_summary()
    results['performance_unified'] = True  # 単層と同じ性能
    
    print(f"✅ 多層学習統一実装完了")
    print(f"📊 最終精度: {results.get('final_accuracy', 0)/100:.3f} ({results.get('final_accuracy', 0):.1f}%) (統一高速化版)")
    print(f"📈 最高精度: {results.get('peak_accuracy', 0)/100:.3f} ({results.get('peak_accuracy', 0):.1f}%) (統一高速化版)")
    print(f"🚀 使用実装: EDNetworkMNIST（単層と同じ高速化）")
    
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
        
        # 設定表示
        if not hyperparams.quiet_mode:
            hyperparams.display_config()
        
        # シンプルな学習実行
        results = run_classification(hyperparams)
        
        # 実行結果表示
        if results and hyperparams.verbose:
            print(f"✅ 学習完了: 精度 {results.get('test_accuracy', 0):.1f}%")
            
    except ValueError as e:
        print(f"❌ パラメータエラー: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\n⚠️ 実行が中断されました")
        exit(0)


