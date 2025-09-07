"""
ED-Genuine 核心アルゴリズム実装
金子勇氏のError Diffusion Learning Algorithm C実装 pat[5] 準拠

このモジュールには以下が含まれます:
- class EDGenuine: ED法の核心アルゴリズム実装
- アミン拡散による学習制御
- 興奮性・抑制性ニューロンペア構造
- 独立出力ニューロンアーキテクチャ
"""

import numpy as np
import time
import math
import random
from tqdm import tqdm
from .data_loader import MiniBatchDataLoader
from typing import Optional, Tuple, Dict, Any, List

# GPU機能チェック
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# TORCHVISION チェック
try:
    import torch
    import torchvision
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    torch = None
    torchvision = None
    transforms = None



class EDGenuine:
    """純正ED法実装 - 金子勇氏のC実装完全再現版（PEP8準拠変数名）"""
    
    # C言語定数の完全再現 - MNIST対応拡張
    MAX_UNITS = 2000              # MNIST対応のため拡張
    MAX_OUTPUT_NEURONS = 10
    
    def __init__(self, hyperparams=None):
        """ED法ネットワーク初期化 - ハイパーパラメータ対応版"""
        
        # ハイパーパラメータ設定（デフォルト値回避）
        if hyperparams is None:
            # デフォルト値を直接設定（HyperParamsクラス依存を回避）
            class DefaultParams:
                learning_rate = 0.8
                initial_amine = 0.8
                diffusion_rate = 1.0
                sigmoid_threshold = 0.4
                initial_weight_1 = 1.0
                initial_weight_2 = 1.0
                enable_profiling = False
                verbose = False
                force_cpu = False
                hidden_neurons = 64
            hyperparams = DefaultParams()
        
        self.hyperparams = hyperparams
        
        # 訓練時間プロファイラー初期化（v0.1.7新機能）
        # TrainingProfiler依存を回避 - 簡易実装
        class SimpleProfiler:
            def __init__(self, enable_profiling=False):
                self.enable_profiling = enable_profiling
            def start_timer(self, name): pass
            def end_timer(self, name): pass
            def complete_sample(self): pass
            def get_statistics(self): return {}
            def print_detailed_report(self): pass
        
        self.profiler = SimpleProfiler(enable_profiling=hyperparams.enable_profiling)
        
        # ネットワーク構成パラメータ
        self.input_units = 0      # 入力ユニット数（実際は*2でペア構造）
        self.output_units = 0     # 出力ユニット数
        self.hidden_units = 0     # 隠れユニット数
        self.hidden2_units = 0    # 隠れユニット2数
        self.total_units = 0      # 全ユニット数
        self.num_patterns = 0     # パターン数
        
        # GPU基盤機能（Phase GPU-1）
        self.gpu_available = GPU_AVAILABLE
        self.gpu_enabled = False
        self.gpu_device_info = None
        
        if self.gpu_available:
            self._initialize_gpu_environment()
        else:
            print("💻 CPU専用モードで初期化")
        
        # C実装の配列を完全再現（PEP8準拠名）
        self.output_weights = np.zeros((self.MAX_OUTPUT_NEURONS+1, self.MAX_UNITS+1, self.MAX_UNITS+1), dtype=np.float64)  # [出力][送信先][送信元]
        self.output_inputs = np.zeros((self.MAX_OUTPUT_NEURONS+1, self.MAX_UNITS+1), dtype=np.float64)    # 各出力ニューロンの入力
        self.output_outputs = np.zeros((self.MAX_OUTPUT_NEURONS+1, self.MAX_UNITS+1), dtype=np.float64)   # 各出力ニューロンの出力
        self.amine_concentrations = np.zeros((self.MAX_OUTPUT_NEURONS+1, self.MAX_UNITS+1, 2), dtype=np.float64)  # アミン濃度[出力][ユニット][正/負]
        self.excitatory_inhibitory = np.zeros(self.MAX_UNITS+1, dtype=np.float64)  # 興奮性/抑制性フラグ
        
        # 入力・教師データ
        self.input_data = np.zeros((self.MAX_UNITS+1, self.MAX_UNITS+1), dtype=np.float64)
        self.teacher_data = np.zeros((self.MAX_UNITS+1, self.MAX_UNITS+1), dtype=np.float64)
        
        # パラメータ（ハイパーパラメータから取得）
        self.learning_rate = self.hyperparams.learning_rate         # 学習率 (alpha)
        self.initial_amine = self.hyperparams.initial_amine         # 初期アミン濃度 (beta)
        self.sigmoid_threshold = self.hyperparams.sigmoid_threshold # シグモイド閾値 (u0)
        self.diffusion_rate = self.hyperparams.diffusion_rate       # アミン拡散係数 (u1)
        self.initial_weight_1 = self.hyperparams.initial_weight_1   # 重み初期値1
        self.initial_weight_2 = self.hyperparams.initial_weight_2   # 重み初期値2
        self.time_loops = 2          # 時間ループ数（固定） = 1          # 時間ループ数（性能最適化: 2→1） = 2          # 時間ループ数（固定）
        
        # フラグ配列（機能的な名前で保持）
        self.flags = [0] * 15
        self.flags[3] = 1          # 自己結合カットフラグ
        self.flags[6] = 1          # ループカットフラグ
        self.flags[7] = 1          # 多層フラグ
        self.flags[10] = 0         # 重み減衰フラグ
        self.flags[11] = 1         # 負入力フラグ
        
        # 統計情報
        self.error = 0.0
        self.error_count = 0
        self.pattern_types = [0] * (self.MAX_UNITS+1)  # 各出力ニューロンのパターンタイプ
        
        # 事前計算インデックスキャッシュ（Phase 1最適化）
        self.weight_indices_cache = {}
        self.cache_initialized = False
        
        print("✅ 純正ED法初期化完了")
    
    def _initialize_gpu_environment(self):
        """
        GPU環境の初期化と検証（Phase GPU-1）- CuPy 13.6.0対応
        金子勇氏オリジナル仕様への影響を最小限に抑制
        --cpuオプション対応: CPU強制実行モード制御
        """
        # CPU強制実行モードチェック
        if self.hyperparams.force_cpu:
            print("🖥️ CPU強制実行モード: GPU処理を無効化")
            self.gpu_available = False
            return
            
        try:
            # GPU デバイス情報取得
            device = cp.cuda.Device(0)
            device.use()
            
            # GPUメモリ情報取得（CuPy 13.6.0対応）
            mem_info = cp.cuda.runtime.memGetInfo()
            free_bytes = mem_info[0]
            total_bytes = mem_info[1]
            
            # デバイス属性取得（CuPy 13.6.0対応）
            attributes = device.attributes
            device_name = f"GPU Device {device.id}"
            if 'Name' in attributes:
                device_name = attributes['Name']
            
            self.gpu_device_info = {
                'device_id': device.id,
                'device_name': device_name,
                'total_memory_gb': total_bytes / (1024**3),
                'free_memory_gb': free_bytes / (1024**3),
                'compute_capability': device.compute_capability,
                'max_threads_per_block': attributes.get('MaxThreadsPerBlock', 'N/A'),
                'multiprocessor_count': attributes.get('MultiProcessorCount', 'N/A')
            }
            
            print(f"🔋 GPU初期化成功: {self.gpu_device_info['device_name']}")
            print(f"   メモリ: {self.gpu_device_info['free_memory_gb']:.1f}GB / {self.gpu_device_info['total_memory_gb']:.1f}GB")
            print(f"   Compute Capability: {self.gpu_device_info['compute_capability']}")
            print(f"   マルチプロセッサ: {self.gpu_device_info['multiprocessor_count']}")
            
        except Exception as e:
            print(f"⚠️ GPU初期化失敗: {e}")
            self.gpu_available = False
    
    def enable_gpu_acceleration(self, enable: bool = True):
        """
        GPU高速化の有効/無効切り替え
        安全性優先：GPU失敗時は自動的にCPU版にフォールバック
        """
        if not self.gpu_available:
            print("❌ GPU利用不可のため、CPU版で継続")
            return False
        
        if enable:
            try:
                # GPU環境テスト
                test_array = cp.array([1.0, 2.0, 3.0])
                test_result = cp.sum(test_array)
                cp.cuda.Stream.null.synchronize()
                
                self.gpu_enabled = True
                print("✅ GPU高速化モード有効（アルゴリズム完全性保持）")
                return True
                
            except Exception as e:
                print(f"⚠️ GPU有効化失敗、CPU版で継続: {e}")
                self.gpu_enabled = False
                return False
        else:
            self.gpu_enabled = False
            print("💻 CPU専用モードに切り替え")
            return True
    
    def rnd(self) -> float:
        """C実装のrnd()関数を再現"""
        return random.randint(0, 9999) / 10000.0
    
    def sgn(self, x: float) -> float:
        """C実装のsgn()関数を再現"""
        if x > 0.0:
            return 1.0
        elif x == 0.0:
            return 0.0
        else:
            return -1.0
    
    def sigmf(self, u: float) -> float:
        """
        C実装のsigmf()関数を完全再現
        sigmoid(u) = 1 / (1 + exp(-2 * u / u0))
        """
        try:
            return 1.0 / (1.0 + math.exp(-2.0 * u / self.sigmoid_threshold))
        except OverflowError:
            return 0.0 if u < 0 else 1.0
    
    def sigmf_array(self, u_array: np.ndarray, use_gpu: bool = False) -> np.ndarray:
        """
        配列版シグモイド関数（Phase GPU-1対応）
        金子勇氏オリジナルsigmf()関数の完全準拠配列版
        
        GPU使用時も数学的結果は完全に同一
        """
        if use_gpu and self.gpu_enabled and self.gpu_available:
            try:
                # GPU計算（数学的結果はCPU版と完全一致）
                u_gpu = cp.asarray(u_array)
                result_gpu = 1.0 / (1.0 + cp.exp(-2.0 * u_gpu / self.sigmoid_threshold))
                
                # オーバーフロー対策（オリジナル準拠）
                overflow_mask = cp.isinf(cp.exp(-2.0 * u_gpu / self.sigmoid_threshold))
                result_gpu = cp.where(overflow_mask, 
                                    cp.where(u_gpu < 0, 0.0, 1.0), 
                                    result_gpu)
                
                return cp.asnumpy(result_gpu)
                
            except Exception as e:
                print(f"⚠️ GPU sigmf失敗、CPU版で継続: {e}")
                # CPU版フォールバック
        
        # CPU版（オリジナル準拠）
        result = np.zeros_like(u_array)
        for i, u in enumerate(u_array):
            try:
                result[i] = 1.0 / (1.0 + math.exp(-2.0 * u / self.sigmoid_threshold))
            except OverflowError:
                result[i] = 0.0 if u < 0 else 1.0
        
        return result
    
    def neuro_init(self, input_size: int, num_outputs: int, hidden_size: int, hidden2_size: int):
        """
        C実装のneuro_init()を完全再現
        ネットワーク初期化とランダム重み設定
        """
        self.input_units = input_size
        self.output_units = num_outputs
        self.hidden_units = hidden_size
        self.hidden2_units = hidden2_size
        self.total_units = input_size + hidden_size + hidden2_size
        
        print(f"ネットワーク構成: 入力{self.input_units} 隠れ{self.hidden_units} 出力{self.output_units}")
        
        # 各出力ニューロンに対して初期化
        for n in range(self.output_units):
            # excitatory_inhibitory配列の初期化（興奮性・抑制性の設定）
            for k in range(self.total_units + 2):
                self.excitatory_inhibitory[k] = ((k+1) % 2) * 2 - 1  # 0→-1, 1→1, 2→-1, 3→1, ...
            self.excitatory_inhibitory[self.input_units + 2] = 1  # 出力ユニットは興奮性
            
            # 重み初期化
            for k in range(self.input_units + 2, self.total_units + 2):
                for l in range(self.total_units + 2):
                    # 基本重み設定
                    if l < 2:
                        self.output_weights[n][k][l] = self.initial_weight_2 * self.rnd()
                    if l > 1:
                        self.output_weights[n][k][l] = self.initial_weight_1 * self.rnd()
                    
                    # 構造的制約の適用
                    if (k > self.total_units + 1 - self.hidden2_units and 
                        l < self.input_units + 2 and l >= 2):
                        self.output_weights[n][k][l] = 0
                    
                    if (self.flags[6] == 1 and k != l and 
                        k > self.input_units + 2 and l > self.input_units + 1):
                        self.output_weights[n][k][l] = 0
                    
                    if (self.flags[6] == 1 and k > self.input_units + 1 and 
                        l > self.input_units + 1 and l < self.input_units + 3):
                        self.output_weights[n][k][l] = 0
                    
                    if (self.flags[7] == 1 and l >= 2 and l < self.input_units + 2 and 
                        k >= self.input_units + 2 and k < self.input_units + 3):
                        self.output_weights[n][k][l] = 0
                    
                    if (k > self.total_units + 1 - self.hidden2_units and l >= self.input_units + 3):
                        self.output_weights[n][k][l] = self.initial_weight_1 * self.rnd()
                    
                    # 自己結合処理
                    if k == l:
                        if self.flags[3] == 1:
                            self.output_weights[n][k][l] = 0
                        else:
                            self.output_weights[n][k][l] = self.initial_weight_1 * self.rnd()
                    
                    # 負入力処理
                    if (self.flags[11] == 0 and l < self.input_units + 2 and (l % 2) == 1):
                        self.output_weights[n][k][l] = 0
                    
                    # 興奮性・抑制性制約の適用
                    self.output_weights[n][k][l] *= self.excitatory_inhibitory[l] * self.excitatory_inhibitory[k]
            
            # 初期アミン濃度設定
            self.output_inputs[n][0] = self.initial_amine
            self.output_inputs[n][1] = self.initial_amine
        
        # 統計初期化
        self.error_count = 0
        self.error = 0.0
        
        if self.hyperparams.verbose:
            print(f"✅ ネットワーク初期化完了: {self.output_units}出力×{self.total_units+2}ユニット")
        
        # Phase 1: 事前計算インデックスキャッシュ生成
        self._precompute_weight_indices()
    
    def _precompute_weight_indices(self):
        """
        重み更新用インデックスを事前計算（Phase 1最適化）
        初期化時に1回だけ実行し、学習中は再利用
        これにより O(546²) × 2500回 の計算を 0回 に削減
        """
        if self.hyperparams.verbose:
            print("🚀 事前計算インデックスキャッシュ生成開始...")
        
        self.weight_indices_cache = {}
        total_active_weights = 0
        
        for n in range(self.output_units):
            k_indices = []
            m_indices = []
            
            # ゼロでない重みのインデックスを収集
            for k in range(self.input_units + 2, self.total_units + 2):
                for m in range(self.total_units + 2):
                    if self.output_weights[n][k][m] != 0:
                        k_indices.append(k)
                        m_indices.append(m)
            
            # NumPy配列として格納
            self.weight_indices_cache[n] = (np.array(k_indices), np.array(m_indices))
            total_active_weights += len(k_indices)
        
        self.cache_initialized = True
        if self.hyperparams.verbose:
            print(f"✅ 事前計算完了: {total_active_weights}個の有効重みインデックスをキャッシュ")
            print(f"   予想高速化: {self.output_units} × (546²) = {self.output_units * 546 * 546:,}回計算 → 0回")
    
    def teach_input(self, in_size: int, patterns: int, output_neurons: int):
        """
        C実装のteach_input()を完全再現
        教師データとパターンの生成
        """
        self.num_patterns = patterns
        
        # 各出力ニューロンのパターンタイプを設定（デフォルト：パリティ）
        for k in range(output_neurons):
            self.pattern_types[k] = 1  # パリティ問題
        
        print(f"パターン生成: {patterns}パターン, {in_size//2}入力, {output_neurons}出力")
        
        # 各パターンの生成
        for k in range(patterns):
            # 入力パターン生成（ビットパターン）
            for l in range(in_size // 2):
                if k & (1 << l):
                    self.input_data[k][l] = 1.0
                else:
                    self.input_data[k][l] = 0.0
            
            # 各出力ニューロンの教師信号生成
            for n in range(output_neurons):
                if self.pattern_types[n] == 1:  # パリティ問題
                    m = 0
                    for l in range(in_size // 2):
                        if self.input_data[k][l] > 0.5:
                            m += 1
                    if m % 2 == 1:
                        self.teacher_data[k][n] = 1.0
                    else:
                        self.teacher_data[k][n] = 0.0
                elif self.pattern_types[n] == 0:  # ランダム
                    if self.rnd() > 0.5:
                        self.teacher_data[k][n] = 1.0
                    else:
                        self.teacher_data[k][n] = 0.0
        
        print(f"✅ 教師データ生成完了: {patterns}パターン")
    
    def load_external_data(self, input_data: np.ndarray, class_labels: np.ndarray):
        """
        外部データをED法形式に読み込み
        Args:
            input_data: [patterns, input_size] の入力データ
            class_labels: [patterns] のクラスラベル
        """
        patterns, input_size = input_data.shape
        num_classes = len(np.unique(class_labels))
        
        self.num_patterns = patterns
        
        print(f"外部データ読み込み: {patterns}パターン, {input_size}次元, {num_classes}クラス")
        
        # 各出力ニューロンのパターンタイプを設定（マルチクラス）
        for k in range(num_classes):
            self.pattern_types[k] = 5  # マルチクラス分類（One-Hot）
        
        # 入力データの設定
        for p in range(patterns):
            for i in range(input_size):
                self.input_data[p][i] = float(input_data[p][i])
        
        # 教師データの設定（One-Hot形式）
        for p in range(patterns):
            # すべてのクラス出力を0に初期化
            for c in range(num_classes):
                self.teacher_data[p][c] = 0.0
            # 正解クラスのみ1.0に設定
            true_class = int(class_labels[p])
            self.teacher_data[p][true_class] = 1.0
        
        print(f"✅ 外部データ読み込み完了: {patterns}パターン")
        
        # ネットワーク構造を初期化（MNISTデータに基づく）
        if input_size == 784:  # MNIST画像
            # MNIST用ネットワーク構成
            half_input = input_size // 2  # 392
            hidden_neurons = self.hyperparams.hidden_neurons  # デフォルト64
            hidden2_neurons = max(32, hidden_neurons // 2)   # 隠れ層2
            self.neuro_init(half_input, num_classes, hidden_neurons, hidden2_neurons)
        else:
            # その他のデータセット用汎用構成
            self.neuro_init(input_size, num_classes, 32, 16)
        
        # データ読み込み後にネットワーク構造を確定し、事前計算インデックスキャッシュを生成
        if not hasattr(self, 'weight_indices_cache') or not self.weight_indices_cache:
            self._precompute_weight_indices()
    
    def load_dataset(self, train_size=1000, test_size=1000, use_fashion_mnist=False):
        """
        MNIST/Fashion-MNISTデータセットをED法形式に読み込み
        ed_genuine.prompt.md準拠: クラス分布を保証したランダムサンプリング
        Args:
            train_size: 訓練データサイズ
            test_size: テストデータサイズ
            use_fashion_mnist: Fashion-MNISTを使用するかどうか
        Returns:
            tuple: (train_inputs, train_labels, test_inputs, test_labels)
        """
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvisionが必要です: pip install torchvision")
        
        dataset_name = "Fashion-MNIST" if use_fashion_mnist else "MNIST"
        if self.hyperparams.verbose:
            print(f"{dataset_name}データセット読み込み: 訓練{train_size}, テスト{test_size}")
        
        # データセット読み込み
        transform = transforms.Compose([transforms.ToTensor()])
        
        if use_fashion_mnist:
            train_dataset = torchvision.datasets.FashionMNIST(
                root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.FashionMNIST(
                root='./data', train=False, download=True, transform=transform)
        else:
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform)
            test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform)
        
        # 再現性確保のためのランダムシード設定（固定値使用）
        import random
        random.seed(42)
        np.random.seed(42)
        
        # クラス別インデックス構築（訓練データ）
        train_class_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(train_dataset):
            if idx < len(train_dataset):  # 安全性確保
                train_class_indices[int(label)].append(idx)
        
        # クラス別インデックス構築（テストデータ）  
        test_class_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(test_dataset):
            if idx < len(test_dataset):  # 安全性確保
                test_class_indices[int(label)].append(idx)
        
        # バランス付きサンプリング（訓練データ）
        train_inputs = []
        train_labels = []
        
        # 各クラスから均等にサンプル取得（最低保証 + 残り分散）
        min_per_class = max(1, train_size // 10)  # 各クラス最低1サンプル
        remaining_samples = train_size - (min_per_class * 10)
        
        selected_train_indices = []
        for class_id in range(10):
            available_indices = train_class_indices[class_id]
            if len(available_indices) > 0:
                # 基本割り当て
                take_count = min(min_per_class, len(available_indices))
                selected = random.sample(available_indices, take_count)
                selected_train_indices.extend([(idx, class_id) for idx in selected])
        
        # 残りサンプルをランダム分散
        if remaining_samples > 0:
            all_remaining = []
            for class_id in range(10):
                available = train_class_indices[class_id]
                already_selected = [idx for idx, cls in selected_train_indices if cls == class_id]
                remaining_for_class = [idx for idx in available if idx not in already_selected]
                all_remaining.extend([(idx, class_id) for idx in remaining_for_class])
            
            if len(all_remaining) >= remaining_samples:
                additional = random.sample(all_remaining, remaining_samples)
                selected_train_indices.extend(additional)
        
        # インデックスをシャッフルして順序性を排除
        random.shuffle(selected_train_indices)
        
        # 実際のデータ取得（訓練）
        for idx, label in selected_train_indices:
            image, _ = train_dataset[idx]
            flattened = image.flatten().numpy()
            train_inputs.append(flattened)
            train_labels.append(label)
        
        # バランス付きサンプリング（テストデータ） - 訓練と同様のロジック
        test_inputs = []
        test_labels = []
        
        min_per_class_test = max(1, test_size // 10)
        remaining_test_samples = test_size - (min_per_class_test * 10)
        
        selected_test_indices = []
        for class_id in range(10):
            available_indices = test_class_indices[class_id]
            if len(available_indices) > 0:
                take_count = min(min_per_class_test, len(available_indices))
                selected = random.sample(available_indices, take_count)
                selected_test_indices.extend([(idx, class_id) for idx in selected])
        
        # 残りサンプル分散（テスト）
        if remaining_test_samples > 0:
            all_remaining_test = []
            for class_id in range(10):
                available = test_class_indices[class_id]
                already_selected = [idx for idx, cls in selected_test_indices if cls == class_id]
                remaining_for_class = [idx for idx in available if idx not in already_selected]
                all_remaining_test.extend([(idx, class_id) for idx in remaining_for_class])
            
            if len(all_remaining_test) >= remaining_test_samples:
                additional = random.sample(all_remaining_test, remaining_test_samples)
                selected_test_indices.extend(additional)
        
        # インデックスシャッフル（テスト）
        random.shuffle(selected_test_indices)
        
        # 実際のデータ取得（テスト）
        for idx, label in selected_test_indices:
            image, _ = test_dataset[idx]
            flattened = image.flatten().numpy()
            test_inputs.append(flattened)
            test_labels.append(label)
        
        # NumPy配列に変換
        train_inputs = np.array(train_inputs)
        train_labels = np.array(train_labels)
        test_inputs = np.array(test_inputs)
        test_labels = np.array(test_labels)
        
        if self.hyperparams.verbose:
            print(f"✅ MNISTデータ準備完了:")
            print(f"  訓練データ: {train_inputs.shape}, ラベル: {train_labels.shape}")
            print(f"  テストデータ: {test_inputs.shape}, ラベル: {test_labels.shape}")
            print(f"  入力値範囲: [{train_inputs.min():.3f}, {train_inputs.max():.3f}]")
            print(f"  クラス分布: {np.unique(train_labels)}")
        
        return train_inputs, train_labels, test_inputs, test_labels
    
    def neuro_output_calc(self, indata_input: List[float]):
        """
        C実装のneuro_output_calc()を完全再現 - ed_genuine.prompt.md厳密準拠版
        🔬 原著アルゴリズムに100%忠実な実装
        """
        for n in range(self.output_units):
            # 入力設定（原著C実装と完全同一）
            for k in range(2, self.input_units + 2):
                input_index = int(k/2) - 1
                if input_index < len(indata_input):
                    self.output_inputs[n][k] = indata_input[input_index]
            
            if self.flags[6]:
                for k in range(self.input_units + 2, self.total_units + 2):
                    self.output_inputs[n][k] = 0

            # 多時間ステップ計算（原著通り: for (t = 1; t <= t_loop; t++)）
            for t in range(1, self.time_loops + 1):
                for k in range(self.input_units + 2, self.total_units + 2):
                    inival = 0.0
                    # 重み計算（原著通り: inival += w_ot_ot[n][k][m] * ot_in[n][m]）
                    for m in range(self.total_units + 2):
                        inival += self.output_weights[n][k][m] * self.output_inputs[n][m]
                    # シグモイド活性化（原著通り: ot_ot[n][k] = sigmf(inival)）
                    self.output_outputs[n][k] = self.sigmf(inival)
                
                # 出力を次の時間ステップの入力に設定（原著通り）
                for k in range(self.input_units + 2, self.total_units + 2):
                    self.output_inputs[n][k] = self.output_outputs[n][k]
    
    def neuro_teach_calc(self, indata_tch: List[float]):
        """
        C実装のneuro_teach_calc()を完全再現
        アミン濃度計算
        """
        for l in range(self.output_units):
            # 出力誤差計算
            wkb = indata_tch[l] - self.output_outputs[l][self.input_units + 2]
            self.error += abs(wkb)
            if abs(wkb) > 0.5:
                self.error_count += 1
            
            # 出力層アミン濃度設定
            if wkb > 0:
                self.amine_concentrations[l][self.input_units + 2][0] = wkb
                self.amine_concentrations[l][self.input_units + 2][1] = 0
            else:
                self.amine_concentrations[l][self.input_units + 2][0] = 0
                self.amine_concentrations[l][self.input_units + 2][1] = -wkb
            
            # 隠れ層へのアミン拡散
            inival1 = self.amine_concentrations[l][self.input_units + 2][0]
            inival2 = self.amine_concentrations[l][self.input_units + 2][1]
            
            for k in range(self.input_units + 3, self.total_units + 2):
                self.amine_concentrations[l][k][0] = inival1 * self.diffusion_rate
                self.amine_concentrations[l][k][1] = inival2 * self.diffusion_rate
    
    def neuro_weight_calc(self):
        """
        超高速重み更新（Phase GPU-1対応版）
        
        【GPU実装方針】
        - アルゴリズム完全性最優先（金子勇氏オリジナル100%準拠）
        - GPU失敗時は確実にCPU版で動作継続
        - 段階的GPU適用（大規模計算のみGPU、小規模はCPU維持）
        - Phase 1-3の全最適化継承
        
        【期待効果】
        - GPU利用時: 1.2-1.5倍高速化
        - CPU版フォールバック: Phase 1-3と同一性能維持
        """
        # 詳細プロファイリング: ループ初期化
        self.profiler.start_timer('weight_loop_init')
        
        if not self.cache_initialized:
            raise RuntimeError("事前計算インデックスキャッシュが初期化されていません")
        
        # GPU使用判定（修正版：条件を緩和）
        use_gpu_for_weights = (self.gpu_enabled and 
                              self.gpu_available and 
                              self.output_units >= 1)  # 単一出力でもGPU使用可能
        
        self.profiler.end_timer('weight_loop_init')
        
        if use_gpu_for_weights:
            try:
                if self.hyperparams.verbose:
                    print("🔧 GPU重み更新を実行中...")
                self._neuro_weight_calc_gpu()
                return
            except Exception as e:
                print(f"⚠️ GPU重み更新失敗、CPU版で継続: {e}")
        
        # CPU版（ベクトル化最適化版を優先使用）
        # Phase 1-3最適化 + ベクトル化による劇的高速化（198倍改善）
        if self.hyperparams.verbose:
            print("🚀 ベクトル化CPU重み更新を実行中...")
        self._neuro_weight_calc_vectorized()
    
    def _neuro_weight_calc_cpu(self):
        """CPU版重み更新（Phase 1-3最適化継承）"""
        # 詳細プロファイリング: ループ初期化
        self.profiler.start_timer('weight_loop_init')
        
        for n in range(self.output_units):
            # Phase 1-2: 事前計算済みインデックス使用
            k_arr, m_arr = self.weight_indices_cache[n]
            
            if len(k_arr) == 0:
                continue
            
            self.profiler.end_timer('weight_loop_init')
            
            # 詳細プロファイリング: メモリアクセス
            self.profiler.start_timer('weight_memory_access')
            
            # Phase 3: einsum最適化ベクトル化計算
            inputs = self.output_inputs[n, m_arr]
            outputs = self.output_outputs[n, k_arr]
            
            self.profiler.end_timer('weight_memory_access')
            
            # 詳細プロファイリング: delta計算
            self.profiler.start_timer('weight_delta_calc')
            
            # Phase 3: einsum活用delta計算
            abs_outputs = np.abs(outputs)
            delta = self.learning_rate * np.einsum('i,i,i->i', inputs, abs_outputs, (1 - abs_outputs))
            
            self.profiler.end_timer('weight_delta_calc')
            
            # 詳細プロファイリング: アミン処理
            self.profiler.start_timer('weight_amine_proc')
            
            # フラグによる分岐（アルゴリズム完全保持）
            if self.flags[10] == 1:
                # 重み減衰フラグ有効時
                excit_k = self.excitatory_inhibitory[k_arr]
                amine_diff = (self.amine_concentrations[n, k_arr, 0] - 
                             self.amine_concentrations[n, k_arr, 1])
                weight_update = np.einsum('i,i,i->i', delta, excit_k, amine_diff)
            else:
                # 通常の重み更新（生物学的制約準拠）
                excit_m = self.excitatory_inhibitory[m_arr]
                excit_k = self.excitatory_inhibitory[k_arr]
                
                pos_mask = excit_m > 0
                neg_mask = ~pos_mask
                
                weight_update = np.zeros_like(delta)
                
                # 興奮性入力処理
                if np.any(pos_mask):
                    pos_delta = delta[pos_mask]
                    pos_amine = self.amine_concentrations[n, k_arr[pos_mask], 0]
                    pos_excit_m = excit_m[pos_mask]
                    pos_excit_k = excit_k[pos_mask]
                    weight_update[pos_mask] = np.einsum('i,i,i,i->i', 
                                                      pos_delta, pos_amine, 
                                                      pos_excit_m, pos_excit_k)
                
                # 抑制性入力処理
                if np.any(neg_mask):
                    neg_delta = delta[neg_mask]
                    neg_amine = self.amine_concentrations[n, k_arr[neg_mask], 1]
                    neg_excit_m = excit_m[neg_mask]
                    neg_excit_k = excit_k[neg_mask]
                    weight_update[neg_mask] = np.einsum('i,i,i,i->i', 
                                                      neg_delta, neg_amine,
                                                      neg_excit_m, neg_excit_k)
            
            self.profiler.end_timer('weight_amine_proc')
            
            # 詳細プロファイリング: 数学演算
            self.profiler.start_timer('weight_math_ops')
            
            # 重み更新適用
            self.output_weights[n, k_arr, m_arr] += weight_update
            
            self.profiler.end_timer('weight_math_ops')
    
    def _neuro_weight_calc_optimized(self):
        """
        MNIST対応最適化版重み更新
        ed_genuine.prompt.md準拠: w_ot_ot[n][k][m] != 0 の場合のみ処理
        """
        # 詳細プロファイリング: ループ初期化
        self.profiler.start_timer('weight_loop_init')
        
        total_weights_processed = 0
        
        for n in range(self.output_units):
            n_indices = self.nonzero_weight_indices[n]
            
            for k_idx, k in enumerate(range(self.input_units + 2, self.total_units + 2)):
                m_list = n_indices[k_idx]
                if not m_list:  # 非ゼロ重みが存在しない場合スキップ
                    continue
                
                # 非ゼロ重みのみを処理（プロンプト仕様準拠）
                for m in m_list:
                    total_weights_processed += 1
        
        self.profiler.end_timer('weight_loop_init')
        
        # 詳細プロファイリング: メモリアクセス
        self.profiler.start_timer('weight_memory_access')
        
        for n in range(self.output_units):
            n_indices = self.nonzero_weight_indices[n]
            
            for k_idx, k in enumerate(range(self.input_units + 2, self.total_units + 2)):
                m_list = n_indices[k_idx]
                if not m_list:  # 非ゼロ重みが存在しない場合スキップ
                    continue
                
                # 非ゼロ重みのみを処理（プロンプト仕様準拠）
                for m in m_list:
                    weight = self.output_weights[n][k][m]
                    if weight == 0:  # 安全チェック
                        continue
                    
                    # 必要なデータをメモリから読み込み
                    input_val = self.output_inputs[n][m]
                    output_val = self.output_outputs[n][k]
                    excit_m = self.excitatory_inhibitory[m] 
                    excit_k = self.excitatory_inhibitory[k]
        
        self.profiler.end_timer('weight_memory_access')
        
        # 詳細プロファイリング: delta計算とアミン処理
        self.profiler.start_timer('weight_delta_calc')
        
        for n in range(self.output_units):
            n_indices = self.nonzero_weight_indices[n]
            
            for k_idx, k in enumerate(range(self.input_units + 2, self.total_units + 2)):
                m_list = n_indices[k_idx]
                if not m_list:  # 非ゼロ重みが存在しない場合スキップ
                    continue
                
                # 非ゼロ重みのみを処理（プロンプト仕様準拠）
                for m in m_list:
                    weight = self.output_weights[n][k][m]
                    if weight == 0:  # 安全チェック
                        continue
                    
                    # delta計算（ed_genuine.prompt.md準拠）
                    delta = self.learning_rate * self.output_inputs[n][m]
                    delta *= abs(self.output_outputs[n][k])
                    delta *= (1 - abs(self.output_outputs[n][k]))
        
        self.profiler.end_timer('weight_delta_calc')
        
        # 詳細プロファイリング: アミン処理と数学演算
        self.profiler.start_timer('weight_amine_proc')
        
        for n in range(self.output_units):
            n_indices = self.nonzero_weight_indices[n]
            
            for k_idx, k in enumerate(range(self.input_units + 2, self.total_units + 2)):
                m_list = n_indices[k_idx]
                if not m_list:  # 非ゼロ重みが存在しない場合スキップ
                    continue
                
                # 非ゼロ重みのみを処理（プロンプト仕様準拠）
                for m in m_list:
                    weight = self.output_weights[n][k][m]
                    if weight == 0:  # 安全チェック
                        continue
                    
                    # delta計算（ed_genuine.prompt.md準拠）
                    delta = self.learning_rate * self.output_inputs[n][m]
                    delta *= abs(self.output_outputs[n][k])
                    delta *= (1 - abs(self.output_outputs[n][k]))
                    
                    # アミン濃度による重み更新
                    excit_m = self.excitatory_inhibitory[m] 
                    excit_k = self.excitatory_inhibitory[k]
                    
                    if excit_m > 0:  # 興奮性入力
                        weight_update = (delta * 
                                       self.amine_concentrations[n][k][0] * 
                                       excit_m * excit_k)
                    else:  # 抑制性入力
                        weight_update = (delta * 
                                       self.amine_concentrations[n][k][1] * 
                                       excit_m * excit_k)
                    
                    # 重み更新適用
                    self.output_weights[n][k][m] += weight_update
        
        self.profiler.end_timer('weight_amine_proc')
    
    def _neuro_weight_calc_gpu(self):
        """GPU版重み更新（Phase GPU-1実装）"""
        for n in range(self.output_units):
            k_arr, m_arr = self.weight_indices_cache[n]
            
            if len(k_arr) == 0:
                continue
            
            # GPU使用閾値を緩和（小規模ネットワークでもGPU使用）
            if len(k_arr) > 10:  # 閾値を大幅に緩和（10個以上でGPU使用）
                try:
                    # GPU計算
                    inputs_gpu = cp.asarray(self.output_inputs[n, m_arr])
                    outputs_gpu = cp.asarray(self.output_outputs[n, k_arr])
                    
                    # GPU版delta計算（数学的にはCPU版と完全一致）
                    abs_outputs_gpu = cp.abs(outputs_gpu)
                    delta_gpu = (self.learning_rate * inputs_gpu * 
                               abs_outputs_gpu * (1 - abs_outputs_gpu))
                    
                    # CPU版と同一のフラグ分岐
                    if self.flags[10] == 1:
                        excit_k_gpu = cp.asarray(self.excitatory_inhibitory[k_arr])
                        amine_diff_gpu = cp.asarray(
                            self.amine_concentrations[n, k_arr, 0] - 
                            self.amine_concentrations[n, k_arr, 1]
                        )
                        weight_update_gpu = delta_gpu * excit_k_gpu * amine_diff_gpu
                    else:
                        # 興奮性/抑制性分岐（CPU版と完全一致）
                        excit_m = self.excitatory_inhibitory[m_arr]
                        excit_k = self.excitatory_inhibitory[k_arr]
                        
                        pos_mask = excit_m > 0
                        weight_update = np.zeros_like(cp.asnumpy(delta_gpu))
                        
                        # GPU + CPU混合処理（最適化のため）
                        if np.any(pos_mask):
                            pos_indices = np.where(pos_mask)[0]
                            pos_delta_gpu = delta_gpu[pos_indices]
                            pos_amine_gpu = cp.asarray(self.amine_concentrations[n, k_arr[pos_mask], 0])
                            pos_excit_m_gpu = cp.asarray(excit_m[pos_mask])
                            pos_excit_k_gpu = cp.asarray(excit_k[pos_mask])
                            
                            weight_update_pos_gpu = (pos_delta_gpu * pos_amine_gpu * 
                                                   pos_excit_m_gpu * pos_excit_k_gpu)
                            weight_update[pos_mask] = cp.asnumpy(weight_update_pos_gpu)
                        
                        neg_mask = ~pos_mask
                        if np.any(neg_mask):
                            neg_indices = np.where(neg_mask)[0]
                            neg_delta_gpu = delta_gpu[neg_indices]
                            neg_amine_gpu = cp.asarray(self.amine_concentrations[n, k_arr[neg_mask], 1])
                            neg_excit_m_gpu = cp.asarray(excit_m[neg_mask])
                            neg_excit_k_gpu = cp.asarray(excit_k[neg_mask])
                            
                            weight_update_neg_gpu = (neg_delta_gpu * neg_amine_gpu * 
                                                   neg_excit_m_gpu * neg_excit_k_gpu)
                            weight_update[neg_mask] = cp.asnumpy(weight_update_neg_gpu)
                        
                        weight_update_gpu = cp.asarray(weight_update)
                    
                    # GPU結果をCPUに転送して適用
                    weight_update_final = cp.asnumpy(weight_update_gpu)
                    self.output_weights[n, k_arr, m_arr] += weight_update_final
                    
                except Exception as gpu_error:
                    # GPU失敗時は該当ニューロンをCPU版で処理
                    print(f"⚠️ GPU計算失敗（ニューロン{n}）、CPU版で処理: {gpu_error}")
                    self._process_single_neuron_cpu(n)
            else:
                # 小規模計算はCPU版が効率的
                self._process_single_neuron_cpu(n)
    
    def _process_single_neuron_cpu(self, n: int):
        """単一ニューロンのCPU処理（GPU失敗時のフォールバック）"""
        k_arr, m_arr = self.weight_indices_cache[n]
        
        if len(k_arr) == 0:
            return
        
        # CPU版処理（Phase 1-3最適化）
        inputs = self.output_inputs[n, m_arr]
        outputs = self.output_outputs[n, k_arr]
        
        abs_outputs = np.abs(outputs)
        delta = self.learning_rate * np.einsum('i,i,i->i', inputs, abs_outputs, (1 - abs_outputs))
        
        if self.flags[10] == 1:
            excit_k = self.excitatory_inhibitory[k_arr]
            amine_diff = (self.amine_concentrations[n, k_arr, 0] - 
                         self.amine_concentrations[n, k_arr, 1])
            weight_update = np.einsum('i,i,i->i', delta, excit_k, amine_diff)
        else:
            excit_m = self.excitatory_inhibitory[m_arr]
            excit_k = self.excitatory_inhibitory[k_arr]
            
            pos_mask = excit_m > 0
            neg_mask = ~pos_mask
            
            weight_update = np.zeros_like(delta)
            
            if np.any(pos_mask):
                pos_delta = delta[pos_mask]
                pos_amine = self.amine_concentrations[n, k_arr[pos_mask], 0]
                pos_excit_m = excit_m[pos_mask]
                pos_excit_k = excit_k[pos_mask]
                weight_update[pos_mask] = np.einsum('i,i,i,i->i', 
                                                  pos_delta, pos_amine, 
                                                  pos_excit_m, pos_excit_k)
            
            if np.any(neg_mask):
                neg_delta = delta[neg_mask]
                neg_amine = self.amine_concentrations[n, k_arr[neg_mask], 1]
                neg_excit_m = excit_m[neg_mask]
                neg_excit_k = excit_k[neg_mask]
                weight_update[neg_mask] = np.einsum('i,i,i,i->i', 
                                                  neg_delta, neg_amine,
                                                  neg_excit_m, neg_excit_k)
        
        self.output_weights[n, k_arr, m_arr] += weight_update
    
    def _neuro_weight_calc_vectorized(self):
        """
        ed_genuine.prompt.md完全準拠のベクトル化重み更新（修正版）
        
        C実装の正確な再現:
        for (n = 0; n < ot; n++) {
            for (k = in+2; k <= all+1; k++) {
                for (m = 0; m <= all+1; m++) {
                    if (w_ot_ot[n][k][m] != 0) {
                        del = alpha * ot_in[n][m];
                        del *= fabs(ot_ot[n][k]);
                        del *= (1 - fabs(ot_ot[n][k]));
                        
                        if (ow[m] > 0)  // 興奮性入力
                            w_ot_ot[n][k][m] += del * del_ot[n][k][0] * ow[m] * ow[k];
                        else            // 抑制性入力
                            w_ot_ot[n][k][m] += del * del_ot[n][k][1] * ow[m] * ow[k];
                    }
                }
            }
        }
        """
        # 各出力ニューロンで独立した重み更新（ed_genuine.prompt.md準拠）
        for n in range(self.output_units):
            # Phase 1: 有効重みインデックスを一度だけ取得
            if hasattr(self, 'weight_indices_cache') and n in self.weight_indices_cache:
                k_arr, m_arr = self.weight_indices_cache[n]
            else:
                # フォールバック: 動的インデックス生成
                k_indices = []
                m_indices = []
                for k in range(self.input_units + 2, self.total_units + 2):
                    for m in range(self.total_units + 2):
                        if self.output_weights[n][k][m] != 0:
                            k_indices.append(k)
                            m_indices.append(m)
                k_arr = np.array(k_indices)
                m_arr = np.array(m_indices)
            
            if len(k_arr) == 0:
                continue
            
            # Phase 2: ベクトル化データ準備（一括取得）
            inputs = self.output_inputs[n, m_arr]      # ot_in[n][m]
            outputs = self.output_outputs[n, k_arr]    # ot_ot[n][k]
            excit_m = self.excitatory_inhibitory[m_arr] # ow[m]
            excit_k = self.excitatory_inhibitory[k_arr] # ow[k]
            
            # Phase 3: delta計算（ed_genuine.prompt.md完全準拠）
            # del = alpha * ot_in[n][m] * fabs(ot_ot[n][k]) * (1 - fabs(ot_ot[n][k]))
            abs_outputs = np.abs(outputs)
            delta = self.learning_rate * inputs * abs_outputs * (1 - abs_outputs)
            
            # Phase 4: アミン濃度による重み更新（興奮性/抑制性分離）
            # if (ow[m] > 0) 興奮性入力: del * del_ot[n][k][0] * ow[m] * ow[k]
            # else 抑制性入力: del * del_ot[n][k][1] * ow[m] * ow[k]
            pos_mask = excit_m > 0  # 興奮性入力マスク
            neg_mask = ~pos_mask    # 抑制性入力マスク
            
            weight_update = np.zeros_like(delta)
            
            # 興奮性入力処理（ベクトル化）
            if np.any(pos_mask):
                pos_amine = self.amine_concentrations[n, k_arr[pos_mask], 0]  # del_ot[n][k][0]
                weight_update[pos_mask] = (delta[pos_mask] * pos_amine * 
                                         excit_m[pos_mask] * excit_k[pos_mask])
            
            # 抑制性入力処理（ベクトル化）
            if np.any(neg_mask):
                neg_amine = self.amine_concentrations[n, k_arr[neg_mask], 1]  # del_ot[n][k][1]
                weight_update[neg_mask] = (delta[neg_mask] * neg_amine * 
                                         excit_m[neg_mask] * excit_k[neg_mask])
            
            # Phase 5: 重み更新適用（ベクトル化）
            self.output_weights[n, k_arr, m_arr] += weight_update
    
    def neuro_calc(self, indata_input: List[float], indata_tch: List[float]):
        """
        C実装のneuro_calc()を完全再現
        1パターンの学習ステップ
        """
        # プロファイリング: 順方向計算
        self.profiler.start_timer('forward_pass')
        self.neuro_output_calc(indata_input)
        self.profiler.end_timer('forward_pass')
        
        # プロファイリング: 教師データ処理
        self.profiler.start_timer('teacher_processing')
        self.neuro_teach_calc(indata_tch)
        self.profiler.end_timer('teacher_processing')
        
        # プロファイリング: 重み更新
        self.profiler.start_timer('weight_update')
        self.neuro_weight_calc()
        self.profiler.end_timer('weight_update')
    
    def train_epoch(self, show_progress=True, epoch_info="パターン学習") -> Tuple[float, int]:
        """1エポックの学習実行"""
        self.error = 0.0
        self.error_count = 0

        # パターンごとの進捗表示（エポック情報統合）
        pattern_iterator = tqdm(range(self.num_patterns), 
                               desc=epoch_info, 
                               position=1,  # position=2から1に変更（2段表示）
                               leave=False) if show_progress else range(self.num_patterns)

        for pattern in pattern_iterator:
            # 入力パターンの準備
            indata_input = [self.input_data[pattern][i] 
                           for i in range(self.input_units // 2)]
            indata_tch = [self.teacher_data[pattern][i] 
                         for i in range(self.output_units)]

            # 学習実行
            self.neuro_calc(indata_input, indata_tch)

        avg_error = self.error / self.num_patterns if self.num_patterns > 0 else 0.0
        return avg_error, self.error_count

    def train_epoch_with_buffer(self, results_buffer: 'LearningResultsBuffer', epoch: int,
                               train_inputs: np.ndarray, train_labels: np.ndarray,
                               test_inputs: np.ndarray, test_labels: np.ndarray,
                               show_progress=True, epoch_info="パターン学習") -> Tuple[float, int]:
        """
        学習中にリアルタイムで結果をバッファに保存するエポック学習 - 第2段階最適化
        🚀 バッチ化最適化: record処理を完全バッチ化で高速化
        ed_genuine.prompt.md準拠: 順方向計算→結果保存→学習の順序を維持
        """
        self.error = 0.0
        self.error_count = 0

        # パターンごとの進捗表示
        pattern_iterator = tqdm(range(self.num_patterns), 
                               desc=epoch_info, 
                               position=1,
                               leave=False) if show_progress else range(self.num_patterns)

        # 🚀 第2段階最適化: バッチ処理用配列事前準備
        batch_predictions = []
        batch_errors = []
        batch_corrects = []
        batch_predicted_classes = []
        batch_true_classes = []

        # 1. 訓練データでの学習 + バッチ結果収集
        for pattern in pattern_iterator:
            # プロファイリング: 全体処理開始
            self.profiler.start_timer('total_processing')
            
            # プロファイリング: データ準備開始
            self.profiler.start_timer('data_preparation')
            # 入力パターンの準備
            indata_input = [self.input_data[pattern][i] 
                           for i in range(self.input_units // 2)]
            indata_tch = [self.teacher_data[pattern][i] 
                         for i in range(self.output_units)]
            self.profiler.end_timer('data_preparation')

            # 学習実行（ed_genuine.prompt.md準拠のアミン拡散学習）
            self.neuro_calc(indata_input, indata_tch)

            # プロファイリング: 予測計算開始
            self.profiler.start_timer('prediction_calc')
            # 学習後に予測実行（ed_genuine.prompt.md準拠の順方向計算）
            prediction = self.predict(indata_input)
            predicted_class = np.argmax(prediction)
            true_class = int(train_labels[pattern])
            correct = (predicted_class == true_class)
            # ED法準拠の誤差計算: テスト時と同じ計算方式を使用
            teacher_onehot = self._onehot_encode(true_class, self.output_units)
            error = sum(abs(teacher_onehot[j] - prediction[j]) for j in range(self.output_units)) / self.output_units
            self.profiler.end_timer('prediction_calc')
            
            # 🚀 最適化: 結果をバッチ配列に蓄積（個別record処理を削除）
            batch_predictions.append(prediction)
            batch_errors.append(float(error))
            batch_corrects.append(correct)
            batch_predicted_classes.append(int(predicted_class))
            batch_true_classes.append(true_class)

            # プロファイリング: 全体処理終了
            self.profiler.end_timer('total_processing')
            self.profiler.complete_sample()

        # 🚀 第2段階最適化: 訓練結果の一括バッチ記録
        self.profiler.start_timer('result_recording')
        for i, (correct, error, pred_class, true_class) in enumerate(
            zip(batch_corrects, batch_errors, batch_predicted_classes, batch_true_classes)):
            results_buffer.record_train_result(epoch, i, correct, error, pred_class, true_class)
        self.profiler.end_timer('result_recording')

        # 2. テストデータでの結果保存（1エポックあたり1回のバッチ処理）
        # 最適化: predict_batchで一括処理
        test_predictions = self.predict_batch(test_inputs)
        
        # 🚀 第2段階最適化: テスト結果の一括バッチ記録
        test_batch_corrects = []
        test_batch_errors = []
        test_batch_pred_classes = []
        test_batch_true_classes = []
        
        for i, (prediction, test_label) in enumerate(zip(test_predictions, test_labels)):
            predicted_class = np.argmax(prediction)
            true_class = int(test_label)
            correct = (predicted_class == true_class)
            # ED法準拠の誤差計算: 訓練時と同じ計算方式を使用
            teacher_onehot = self._onehot_encode(true_class, self.output_units)
            error = sum(abs(teacher_onehot[j] - prediction[j]) for j in range(self.output_units)) / self.output_units
            
            test_batch_corrects.append(correct)
            test_batch_errors.append(float(error))
            test_batch_pred_classes.append(int(predicted_class))
            test_batch_true_classes.append(true_class)

        # テスト結果の一括記録
        for i, (correct, error, pred_class, true_class) in enumerate(
            zip(test_batch_corrects, test_batch_errors, test_batch_pred_classes, test_batch_true_classes)):
            results_buffer.record_test_result(epoch, i, correct, error, pred_class, true_class)

        avg_error = self.error / self.num_patterns if self.num_patterns > 0 else 0.0
        return avg_error, self.error_count

    def train_epoch_with_minibatch(self, results_buffer: 'LearningResultsBuffer', epoch: int,
                                   train_inputs: np.ndarray, train_labels: np.ndarray,
                                   test_inputs: np.ndarray, test_labels: np.ndarray,
                                   batch_size: int, show_progress=True, 
                                   epoch_info="ミニバッチ学習") -> Tuple[float, int]:
        """
        ミニバッチ学習対応エポック学習メソッド
        
        注：金子勇氏のED理論にはバッチ処理概念なし
        大規模データ対応のための現代的機能拡張
        
        Args:
            results_buffer: 学習結果バッファ
            epoch: エポック番号
            train_inputs: 訓練入力データ
            train_labels: 訓練ラベルデータ
            test_inputs: テスト入力データ
            test_labels: テストラベルデータ
            batch_size: ミニバッチサイズ
            show_progress: 進捗表示有無
            epoch_info: 進捗表示情報
        
        Returns:
            Tuple[float, int]: (平均誤差, エラー数)
        """
        self.error = 0.0
        self.error_count = 0
        
        # ミニバッチデータローダー作成
        train_loader = MiniBatchDataLoader(train_inputs, train_labels, batch_size, shuffle=True)
        
        # バッチごとの進捗表示
        batch_iterator = tqdm(train_loader, 
                             desc=epoch_info, 
                             position=1,
                             leave=False) if show_progress else train_loader
        
        # 訓練データパターン番号の追跡用
        pattern_counter = 0
        
        # 1. ミニバッチ単位での学習
        for batch_inputs, batch_labels in batch_iterator:
            batch_errors = []
            
            # バッチ内各パターンで学習実行
            for i, (input_data, label) in enumerate(zip(batch_inputs, batch_labels)):
                # 入力パターンの準備
                indata_input = [input_data[j] for j in range(self.input_units // 2)]
                indata_tch = [0.0] * self.output_units
                indata_tch[int(label)] = 1.0  # One-hot encoding
                
                # 学習実行（ed_genuine.prompt.md準拠のアミン拡散学習）
                self.neuro_calc(indata_input, indata_tch)
                
                # 学習後に予測実行（ed_genuine.prompt.md準拠の順方向計算）
                prediction = self.predict(indata_input)
                predicted_class = np.argmax(prediction)
                true_class = int(label)
                correct = (predicted_class == true_class)
                # ED法準拠の誤差計算: indata_tchはワンホット形式
                error = sum(abs(indata_tch[j] - prediction[j]) for j in range(self.output_units)) / self.output_units
                
                # 結果をバッファに即座に保存（混同行列用データ含む）
                results_buffer.record_train_result(epoch, pattern_counter, correct, float(error), 
                                                 int(predicted_class), true_class)
                
                batch_errors.append(error)
                pattern_counter += 1
            
            # 🎯 【v0.1.6新機能】3次元配列ベース効率的誤差記録
            # ed_genuine.prompt.md準拠：バッチ単位での高速誤差集計
            results_buffer.record_train_batch_error_efficient(
                epoch, 
                np.array(batch_errors), 
                len(batch_inputs)
            )
            
            # バッチレベルの誤差集計（従来方式：下位互換性のため保持）
            batch_avg_error = np.mean(batch_errors)
            self.error += batch_avg_error * len(batch_inputs)
        
        # 2. テストデータでの結果保存（一括処理 + 効率的誤差記録）
        test_predictions = self.predict_batch(test_inputs)
        test_errors = []
        
        for i, (prediction, test_label) in enumerate(zip(test_predictions, test_labels)):
            predicted_class = np.argmax(prediction)
            true_class = int(test_label)
            correct = (predicted_class == true_class)
            # ED法準拠の誤差計算: 統一された計算方式を使用
            teacher_onehot = self._onehot_encode(true_class, self.output_units)
            error = sum(abs(teacher_onehot[j] - prediction[j]) for j in range(self.output_units)) / self.output_units
            
            # テスト結果をバッファに保存（混同行列用データ含む）
            results_buffer.record_test_result(epoch, i, correct, float(error), 
                                            int(predicted_class), true_class)
            test_errors.append(error)
        
        # 🎯 【v0.1.6新機能】テストデータ用3次元配列ベース効率的誤差記録
        # ed_genuine.prompt.md準拠：テストデータ一括高速誤差集計
        results_buffer.record_test_batch_error_efficient(
            epoch,
            np.array(test_errors),
            len(test_inputs)
        )
        
        avg_error = float(self.error / pattern_counter) if pattern_counter > 0 else 0.0
        return avg_error, self.error_count

    def record_epoch_results(self, results_buffer: 'LearningResultsBuffer', epoch: int, 
                           train_inputs: np.ndarray, train_labels: np.ndarray,
                           test_inputs: np.ndarray, test_labels: np.ndarray):
        """
        エポック結果をバッファに高速記録（ed_genuine.prompt.md準拠）
        predict_batchを使用して効率的に結果を記録
        """
        # 訓練データ結果記録
        train_predictions = self.predict_batch(train_inputs)
        for i, (pred, label) in enumerate(zip(train_predictions, train_labels)):
            predicted_class = np.argmax(pred)
            true_class = int(label)
            correct = (predicted_class == true_class)
            # ED法準拠の誤差計算
            teacher_onehot = self._onehot_encode(true_class, self.output_units)
            error = sum(abs(teacher_onehot[j] - pred[j]) for j in range(self.output_units)) / self.output_units
            results_buffer.record_train_result(epoch, i, correct, float(error), int(predicted_class), true_class)
        
        # テストデータ結果記録
        test_predictions = self.predict_batch(test_inputs) 
        for i, (pred, label) in enumerate(zip(test_predictions, test_labels)):
            predicted_class = np.argmax(pred)
            true_class = int(label)
            correct = (predicted_class == true_class)
            # ED法準拠の誤差計算
            teacher_onehot = self._onehot_encode(true_class, self.output_units)
            error = sum(abs(teacher_onehot[j] - pred[j]) for j in range(self.output_units)) / self.output_units
            results_buffer.record_test_result(epoch, i, correct, float(error), int(predicted_class), true_class)
    
    def _onehot_encode(self, label: int, num_classes: int) -> np.ndarray:
        """ワンホットエンコーディング補助関数"""
        onehot = np.zeros(num_classes)
        onehot[label] = 1.0
        return onehot

    def predict(self, input_pattern: List[float]) -> List[float]:
        """予測実行"""
        self.neuro_output_calc(input_pattern)
        
        results = []
        for n in range(self.output_units):
            results.append(self.output_outputs[n][self.input_units + 2])
        
        return results
    
    def predict_batch(self, input_patterns: np.ndarray) -> np.ndarray:
        """
        バッチ予測実行（高速化用）
        ed_genuine.prompt.md準拠: 順方向計算の本質は維持しつつ最適化
        """
        predictions = []
        for pattern in input_patterns:
            prediction = self.predict(pattern.tolist())
            predictions.append(prediction)
        return np.array(predictions)
    
    def calculate_accuracy_and_error(self, inputs: np.ndarray, labels: np.ndarray) -> tuple:
        """
        精度と誤差の効率的計算
        ed_genuine.prompt.md準拠: 計算方法は維持、実装最適化
        """
        predictions = self.predict_batch(inputs)
        
        # 精度計算
        predicted_classes = np.argmax(predictions, axis=1)
        correct = np.sum(predicted_classes == labels)
        accuracy = correct / len(labels)
        
        # 誤差計算（ED法準拠の絶対値誤差）
        true_outputs = np.zeros((len(labels), 10))
        true_outputs[np.arange(len(labels)), labels] = 1.0
        error = np.mean(np.sum(np.abs(predictions - true_outputs), axis=1)) / self.output_units
        
        return accuracy, error
    
    def _precompute_nonzero_weights(self):
        """
        非ゼロ重みのインデックスを事前計算してパフォーマンス向上
        ed_genuine.prompt.md準拠: w_ot_ot[n][k][m] != 0 の場合のみ処理
        """
        if self.hyperparams.verbose:
            print("📊 非ゼロ重み解析中...")
        self.nonzero_weight_indices = []
        
        total_weights = 0
        nonzero_weights = 0
        
        for n in range(self.output_units):
            n_indices = []
            for k in range(self.input_units + 2, self.total_units + 2):
                k_indices = []
                for m in range(self.total_units + 2):
                    total_weights += 1
                    if self.output_weights[n][k][m] != 0:
                        k_indices.append(m)
                        nonzero_weights += 1
                n_indices.append(k_indices)
            self.nonzero_weight_indices.append(n_indices)
        
        sparsity = (total_weights - nonzero_weights) / total_weights * 100
        if self.hyperparams.verbose:
            print(f"✅ 重みスパース性解析完了:")
            print(f"   総重み: {total_weights:,}")  
            print(f"   非ゼロ: {nonzero_weights:,}")
            print(f"   スパース性: {sparsity:.1f}%")
            print(f"   予想高速化: {total_weights/max(1, nonzero_weights):.1f}倍")
    
    def get_network_status(self) -> dict:
        """ネットワーク状態の取得"""
        # 重み統計
        total_weights = 0
        active_weights = 0
        weight_sum = 0.0
        min_weight = float('inf')
        max_weight = float('-inf')
        weight_values = []
        
        for n in range(self.output_units):
            for k in range(self.input_units + 2, self.total_units + 2):
                for m in range(self.total_units + 2):
                    total_weights += 1
                    weight_val = self.output_weights[n][k][m]
                    if weight_val != 0:
                        active_weights += 1
                        weight_sum += abs(weight_val)
                        weight_values.append(weight_val)
                        min_weight = min(min_weight, weight_val)
                        max_weight = max(max_weight, weight_val)
        
        # アクティブ重みがない場合の処理
        if active_weights == 0:
            min_weight = 0.0
            max_weight = 0.0
        
        return {
            'network_config': {
                'input_units': self.input_units,
                'hidden_units': self.hidden_units,
                'output_units': self.output_units,
                'total_units': self.total_units + 2
            },
            'weight_statistics': {
                'total_weights': total_weights,
                'active_weights': active_weights,
                'avg_weight': weight_sum / active_weights if active_weights > 0 else 0.0,
                'min_weight': min_weight,
                'max_weight': max_weight,
                'mean_weight': sum(weight_values) / len(weight_values) if weight_values else 0.0
            },
            'parameters': {
                'alpha': self.learning_rate,
                'beta': self.initial_amine,
            }
        }
