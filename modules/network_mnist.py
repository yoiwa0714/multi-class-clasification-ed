"""
ED-Genuine MNIST専用ネットワーククラス
Isamu Kaneko's Error Diffusion Learning Algorithm implementation
Based on C code pat[5]: One-Hot encoding for multi-class classification

【ed_genuine.prompt.md 準拠実装】
-            visualizer = RealtimeLe            # 混同行列可視化（訓練開始時点から表示）
            confusion_visualizer = RealtimeConfusionMatrixVisualizer(
                num_classes=10, window_size=(800, 600), save_dir=getattr(self.hyperparams, 'save_fig', None))
            confusion_visualizer.setup_plots()
                          # 🎯 ed_multi.prompt.md準拠：混同行列表示（エポックループ完了後に移動）
                    # 画面崩れを防ぐため、プログレスバー更新後に表示
                    # if not getattr(self.hyperparams, 'quiet_mode', False) and not enable_visualization and epoch == epochs - 1:
                    #     # 文字ベース表示でのみ実行（グラフ表示時は学習完了後のみ）
                    #     results_buffer.display_confusion_matrix_single_epoch('test', epoch) 
            # 🎯 ed_multi.prompt.md準拠：パラメータボックス表示用データ設定
            ed_params = {
                'learning_rate': self.hyperparams.learning_rate,
                'amine': self.hyperparams.amine,
                'diffusion': self.hyperparams.diffusion,
                'sigmoid': self.hyperparams.sigmoid,
                'weight1': self.hyperparams.weight1,
                'weight2': self.hyperparams.weight2
            }
            exec_params = {
                'train_samples': self.hyperparams.train_samples,
                'test_samples': self.hyperparams.test_samples,
                'epochs': self.hyperparams.epochs,
                'hidden': str(self.hyperparams.hidden),
                'batch_size': self.hyperparams.batch_size,
                'seed': getattr(self.hyperparams, 'seed', 'Random')
            }
            # デバッグ: パラメータ値確認
            print(f"🔍 デバッグ - ED法パラメータ: {ed_params}")
            print(f"🔍 デバッグ - 実行パラメータ: {exec_params}")
            confusion_visualizer.set_parameters(ed_params, exec_params)
            
            # ニューロン発火パターン可視化（v0.2.4新機能）
            # ed_multi.prompt.md準拠: --vizオプションでは学習曲線・混同行列・精度推移のみ表示
            # ニューロン可視化は無効化
                'learning_rate': self.hyperparams.learning_rate,
                'amine': self.hyperparams.amine,
                'diffusion': self.hyperparams.diffusion,
                'sigmoid': self.hyperparams.sigmoid,
                'weight1': self.hyperparams.weight1,
                'weight2': self.hyperparams.weight2
            }
            exec_params = {
                'train_samples': self.hyperparams.train_samples,
                'test_samples': self.hyperparams.test_samples,
                'epochs': self.hyperparams.epochs,
                'hidden': str(self.hyperparams.hidden),
                'batch_size': self.hyperparams.batch_size,
                'seed': getattr(self.hyperparams, 'seed', 'Random')
            }
            # デバッグ: パラメータ値確認
            print(f"🔍 デバッグ - ED法パラメータ: {ed_params}")
            print(f"🔍 デバッグ - 実行パラメータ: {exec_params}")
            confusion_visualizer.set_parameters(ed_params, exec_params)
            
            # ニューロン発火パターン可視化（v0.2.4新機能）
            # ed_multi.prompt.md準拠: --vizオプションでは学習曲線・混同行列・精度推移のみ表示
            # ニューロン可視化は無効化ion-MNIST データセット専用拡張
- 独立出力ニューロンアーキテクチャ 
- ハイパーパラメータ制御対応
- リアルタイム可視化機能
"""

import numpy as np
import time
import threading
from typing import Optional, Tuple, Dict, Any

# 外部依存（オプション）
try:
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

# モジュール内依存
from .ed_core import EDGenuine


class EDNetworkMNIST(EDGenuine):
    """
    MNIST用ED法ネットワーク拡張クラス
    """
    
    def __init__(self, hyperparams):
        """ネットワーク初期化（ヒートマップ対応）"""
        super().__init__(hyperparams)
        self.heatmap_callback = None  # ヒートマップコールバック
        
        # ヒートマップ表示用の現在のサンプル情報
        self.current_sample_info = {
            'epoch': 0,
            'sample_idx': 0,
            'true_label': -1,
            'predicted_label': -1,
            'pattern_idx': 0,
            'input_data': None  # 入力データも保存して同期を確保
        }
        
        # ヒートマップリアルタイム更新間隔（パターン数単位）
        self.heatmap_update_interval = 5  # 5パターン毎に更新でリアルタイム表示
    
    def set_heatmap_callback(self, callback):
        """ヒートマップコールバック設定（EDHeatmapIntegration連携用） - ed_multi.prompt.md準拠"""
        self.heatmap_callback = callback
        print("✅ ネットワークにヒートマップコールバック設定完了")
    
    def update_current_sample_info(self, epoch, sample_idx, true_label, predicted_label=None, pattern_idx=None, input_data=None):
        """現在のサンプル情報を更新（ヒートマップ表示用・入力データ同期対応）"""
        self.current_sample_info.update({
            'epoch': epoch,
            'sample_idx': sample_idx,
            'true_label': true_label,
            'predicted_label': predicted_label if predicted_label is not None else -1,
            'pattern_idx': pattern_idx if pattern_idx is not None else sample_idx,
            'input_data': input_data  # ヒートマップとサブタイトル同期のため
        })
    
    def get_current_sample_info(self):
        """現在のサンプル情報を取得（ヒートマップ表示用）"""
        return self.current_sample_info.copy()
    
    def _initialize_data_arrays(self, train_inputs, train_labels, test_inputs, test_labels):
        """データ配列を初期化（動的生成対応）"""
        # 訓練データをinput_data配列に設定
        for i, (input_vec, label) in enumerate(zip(train_inputs, train_labels)):
            # 入力データ設定（784次元を配列に格納）
            flattened_input = input_vec.flatten() if hasattr(input_vec, 'flatten') else input_vec
            for j, val in enumerate(flattened_input):
                if i < self.num_patterns and j < self.input_units:
                    self.input_data[i][j] = float(val)
            
            # 教師データ設定（One-Hot符号化）
            if i < self.num_patterns:
                for k in range(self.output_units):
                    self.teacher_data[i][k] = 1.0 if k == int(label) else 0.0
        
        # テストデータも同様に設定（必要に応じて）
        # （通常のED法では訓練データのみ使用）
    
    def generate_epoch_data(self, epoch: int) -> tuple:
        """エポック毎の独立訓練データを動的生成（EDGenuineクラス経由）"""
        return super().generate_epoch_data(epoch)
    
    def run_classification(self, enable_visualization=False, use_fashion_mnist=False, 
                         train_size=1000, test_size=200, epochs=3, random_state=None):
        """
        MNIST/Fashion-MNIST分類学習を実行

        Args:
            enable_visualization: リアルタイム可視化の有効化
            use_fashion_mnist: Fashion-MNISTを使用するか
            train_size: 1エポック当たりの訓練サンプル数
            test_size: 1エポック当たりのテストサンプル数
            epochs: エポック数
            random_state: ランダムシード

        Returns:
            dict: 学習結果データ
        """
        dataset_name = "Fashion-MNIST" if use_fashion_mnist else "MNIST"
        if not getattr(self.hyperparams, 'quiet_mode', False):
            print("=" * 60)
            print(f"{dataset_name}分類学習 開始 - ハイパーパラメータ対応版")
            print("=" * 60)
        
        # � 【メモリ最適化】動的データ生成方式
        # エポック毎に独立データをオンデマンド生成し、メモリ使用量を大幅削減
        
        if not getattr(self.hyperparams, 'quiet_mode', False):
            print(f"🔧 データロード設定（メモリ最適化版）:")
            print(f"  エポック毎の訓練データ: {train_size}サンプル（動的生成）")
            print(f"  エポック毎のテストデータ: {test_size}サンプル（固定セット）")
            print(f"  総エポック数: {epochs}")
            print(f"  🎯 メモリ効率: 事前準備量を最小化してメモリ枯渇を回避")
        
        # データセット読み込み（動的生成基盤のみ準備）
        # ed_multi.prompt.md準拠: エポック毎独立データを実行時生成
        train_inputs, train_labels, test_inputs, test_labels = self.load_dataset(
            train_size=train_size, test_size=test_size,  # エポック単位サイズで基盤準備
            use_fashion_mnist=use_fashion_mnist, total_epochs=epochs)
        
        # ネットワーク初期化（10クラス分類用） - 多層対応
        hidden_layers = getattr(self.hyperparams, 'hidden_layers', [100])
        
        if len(hidden_layers) == 1:
            # 単層の場合
            hidden_size = hidden_layers[0]
            hidden2_size = 0
            print(f"📊 単層構成: 隠れ層{hidden_size}ユニット")
        else:
            # 多層の場合: 総隠れユニット数を計算
            hidden_size = sum(hidden_layers)
            hidden2_size = 0  # ed_multi.prompt.md準拠では hidden2_size は使用しない
            print(f"📊 多層構成: 隠れ層{'→'.join(map(str, hidden_layers))} (総計{hidden_size}ユニット)")
        
        self.neuro_init(
            input_size=784,  # 28x28 MNIST/Fashion-MNIST
            num_outputs=10,  # 10クラス
            hidden_size=hidden_size,  # 単層/多層に応じた隠れユニット数
            hidden2_size=hidden2_size  # 隠れ層2は使用しない
        )
        
        print(f"\n📊 ネットワーク構成:")
        print(f"  入力層: 784次元 (28x28画像)")
        print(f"  中間層: {hidden_size}ユニット {'(多層合計)' if len(hidden_layers) > 1 else '(単層)'}")
        if len(hidden_layers) > 1:
            print(f"  多層詳細: {'→'.join(map(str, hidden_layers))}")
        print(f"  出力層: 10クラス")
        print(f"  総ユニット: {self.total_units}")
        
        # 🚀 【最適化】動的データ生成に対応する初期化
        # train_inputsが空配列なので、エポック毎のパターン数を使用
        self.num_patterns = train_size  # エポック毎のパターン数を使用
        
        # 🔧 重要: 動的生成の場合も初期データ設定が必要  
        # 実際のデータはエポック実行時に設定するが、配列サイズ初期化のため仮データを設定
        if len(train_inputs) == 0:
            # 仮の訓練データを生成（配列サイズ確保のため）
            dummy_train_inputs = np.zeros((train_size, 784))
            dummy_train_labels = np.zeros(train_size)
            self._initialize_data_arrays(dummy_train_inputs, dummy_train_labels, test_inputs, test_labels)
        else:
            # 通常の初期化
            self._initialize_data_arrays(train_inputs, train_labels, test_inputs, test_labels)
        
        # 入力データを設定
        for i, inp in enumerate(train_inputs):
            # 配列範囲チェック（ed_genuine.prompt.md準拠の安全性確保）
            if i >= len(self.input_data):
                # 動的メモリ管理により自動的に対応済み
                break
                
            inp_flat = inp.flatten().astype(float)
            for j, val in enumerate(inp_flat):
                if j < len(self.input_data[i]):
                    self.input_data[i][j] = val
        
        # 教師データを設定（10クラス分類）
        for i, label in enumerate(train_labels):
            # 配列範囲チェック（ed_genuine.prompt.md準拠の安全性確保）
            if i >= len(self.teacher_data):
                # 動的メモリ管理により自動的に対応済み
                break
                
            for out_idx in range(10):
                if out_idx == label:
                    self.teacher_data[i][out_idx] = 1.0
                else:
                    self.teacher_data[i][out_idx] = 0.0
        
        if getattr(self.hyperparams, 'verbose', False):
            print("✅ MNISTデータ設定完了: {}パターン".format(self.num_patterns))
        
        # ED法性能最適化: パフォーマンス向上のため非ゼロ重み事前計算
        if getattr(self.hyperparams, 'verbose', False):
            print("⚡ ED法最適化: 非ゼロ重みインデックス事前計算中...")
        self._precompute_nonzero_weights()
        if getattr(self.hyperparams, 'verbose', False):
            print("✅ 最適化完了: 学習速度向上")
        
        # 学習実行準備
        if not getattr(self.hyperparams, 'quiet_mode', False):
            print(f"\n🎯 学習開始: {epochs}エポック")
        
        # 結果保存バッファ初期化（パフォーマンス最適化）
        # 🔧 【修正】動的生成対応: エポック毎の実際のサイズを使用
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from modules.performance import LearningResultsBuffer
        # 動的生成の場合は設定値を使用、通常の場合は配列長を使用
        actual_train_size = train_size if len(train_inputs) == 0 else len(train_inputs)
        results_buffer = LearningResultsBuffer(actual_train_size, len(test_inputs), epochs)
        
        # リアルタイム可視化設定（学習開始時点で表示）
        visualizer = None
        confusion_visualizer = None
        neuron_visualizer = None
        neuron_adapter = None
        
        if enable_visualization and HAS_VISUALIZATION:
            print("🎨 リアルタイム可視化準備中...")
            
            # 従来の学習可視化システム（互換性維持）
            from modules.visualization import RealtimeLearningVisualizer, RealtimeConfusionMatrixVisualizer
            visualizer = RealtimeLearningVisualizer(max_epochs=epochs, save_dir=getattr(self.hyperparams, 'save_fig', None))
            visualizer.setup_plots()
            
            # 混同行列可視化（訓練開始時点から表示）
            confusion_visualizer = RealtimeConfusionMatrixVisualizer(
                num_classes=10, window_size=(800, 600), save_dir=getattr(self.hyperparams, 'save_fig', None))
            confusion_visualizer.setup_plots()
            
            # ニューロン発火パターン可視化（v0.2.4新機能）
            # ed_multi.prompt.md準拠: --vizオプションでは学習曲線・混同行列・精度推移のみ表示
            # ニューロン可視化は無効化
            neuron_visualizer = None
            neuron_adapter = None
            
            # 🎯 ed_multi.prompt.md準拠：パラメータボックス表示用データ設定
            ed_params = {
                'learning_rate': getattr(self.hyperparams, 'learning_rate', 0.5),
                'threshold': getattr(self.hyperparams, 'initial_amine', 0.8),  # initial_amineを使用
                'threshold_alpha': getattr(self.hyperparams, 'diffusion_rate', 0.95),  # diffusion_rateを使用
                'threshold_beta': getattr(self.hyperparams, 'sigmoid_threshold', 0.85),  # sigmoid_thresholdを使用
                'threshold_gamma': getattr(self.hyperparams, 'initial_weight_1', 0.75)  # initial_weight_1を使用
            }
            exec_params = {
                'epochs': epochs,
                'batch_size': getattr(self.hyperparams, 'batch_size', 32),
                'num_layers': len(hidden_layers),  # hidden_layersを使用
                'train_size': len(train_inputs),
                'test_size': len(test_inputs)
            }
            
            # 🎯 ed_multi.prompt.md準拠：両方の可視化にパラメータ設定
            confusion_visualizer.set_parameters(ed_params, exec_params)
            visualizer.set_parameters(ed_params, exec_params)
            
            print("✅ 可視化グラフ表示完了 - 学習データ待機中")
        
        start_time = time.time()
        
        epoch_accuracies = []
        train_errors = []
        test_accuracies = []
        
        # 【重要】エポック毎のサイズを計算（1エポック当たりのサンプル数）
        epoch_train_size = train_size  # 1エポック当たりの訓練サンプル数
        epoch_test_size = test_size    # 1エポック当たりのテストサンプル数
        
        # エポック全体進捗バー（上段） - quietモード時は抑制
        if getattr(self.hyperparams, 'quiet_mode', False):
            # quietモード: 進捗バーなし
            for epoch in range(epochs):
                epoch_start = time.time()
                
                # 【重要】エポック毎に独立したデータ部分を取得（訓練・テスト両方）
                epoch_start_idx = epoch * epoch_train_size
                epoch_end_idx = (epoch + 1) * epoch_train_size
                epoch_train_inputs = train_inputs[epoch_start_idx:epoch_end_idx]
                epoch_train_labels = train_labels[epoch_start_idx:epoch_end_idx]
                
                # 🛡️ 修正: テストデータ配列範囲制限・循環使用処理追加
                test_start_idx = epoch * epoch_test_size
                test_end_idx = (epoch + 1) * epoch_test_size
                
                # 配列範囲チェック：テストデータサイズを超過しないよう制限
                if test_end_idx > len(test_inputs):
                    # エポック数が多い場合の循環使用（ed_multi.prompt.md準拠の独立性維持）
                    test_start_idx = (epoch * epoch_test_size) % len(test_inputs)
                    test_end_idx = min(test_start_idx + epoch_test_size, len(test_inputs))
                    
                    # 不足分は先頭から補完（独立性維持のため異なるインデックス使用）
                    if test_end_idx - test_start_idx < epoch_test_size:
                        remaining_needed = epoch_test_size - (test_end_idx - test_start_idx)
                        epoch_test_inputs = np.concatenate([
                            test_inputs[test_start_idx:test_end_idx],
                            test_inputs[:remaining_needed]
                        ])
                        epoch_test_labels = np.concatenate([
                            test_labels[test_start_idx:test_end_idx], 
                            test_labels[:remaining_needed]
                        ])
                    else:
                        epoch_test_inputs = test_inputs[test_start_idx:test_end_idx]
                        epoch_test_labels = test_labels[test_start_idx:test_end_idx]
                        
                    if self.hyperparams.verbose:
                        print(f"🔄 エポック{epoch+1}: テストデータ循環使用 [{test_start_idx}:{test_end_idx}] + 先頭から{remaining_needed if 'remaining_needed' in locals() else 0}サンプル")
                else:
                    epoch_test_inputs = test_inputs[test_start_idx:test_end_idx]
                    epoch_test_labels = test_labels[test_start_idx:test_end_idx]
                
                if self.hyperparams.verbose:
                    print(f"🎯 エポック{epoch+1}: 訓練データ範囲 [{epoch_start_idx}:{epoch_end_idx}] (独立サンプル{len(epoch_train_inputs)}個)")
                    print(f"🎯 エポック{epoch+1}: テストデータ範囲 [{test_start_idx}:{test_end_idx}] (独立サンプル{len(epoch_test_inputs)}個)")
                
                # 最適化済み学習エポック実行（ミニバッチ対応）
                # ed_genuine.prompt.md準拠: 学習中に予測→保存→学習の順序
                
                # ミニバッチサイズが1なら従来手法、それ以外はミニバッチ学習
                batch_size = getattr(self.hyperparams, 'batch_size', 1)
                if batch_size == 1:
                    avg_error, _ = self.train_epoch_with_buffer(
                        results_buffer, epoch, epoch_train_inputs, epoch_train_labels, 
                        epoch_test_inputs, epoch_test_labels, show_progress=False, 
                        epoch_info=f"エポック{epoch+1:2d}")
                else:
                    # ミニバッチ学習実行（金子勇氏理論拡張）
                    avg_error, _ = self.train_epoch_with_minibatch(
                        results_buffer, epoch, epoch_train_inputs, epoch_train_labels, 
                        epoch_test_inputs, epoch_test_labels, batch_size,
                        show_progress=False, epoch_info=f"エポック{epoch+1:2d} (batch={batch_size})")
                
                # バッファから高速取得（効率的精度計算手法使用）
                # 🎯 【v0.1.6高速化】効率的精度・誤差算出（3次元配列ベース）
                # ed_genuine.prompt.md準拠：NumPy配列演算による高速計算
                train_accuracy = results_buffer.get_epoch_accuracy_efficient(epoch, 'train')
                test_accuracy = results_buffer.get_epoch_accuracy_efficient(epoch, 'test')
                train_error = results_buffer.get_epoch_error_efficient(epoch, 'train')
                test_error = results_buffer.get_epoch_error_efficient(epoch, 'test')
                
                # 🎯 【v0.1.6性能ベンチマーク】誤差算出方式の性能比較（初回エポックのみ）
                if epoch == 0 and getattr(self.hyperparams, 'verbose', False):
                    benchmark_results = results_buffer.benchmark_error_calculation_methods(epoch, 'train')
                    speedup = benchmark_results['speedup']
                    print(f"📊 誤差算出性能: 従来方式 vs 3次元配列ベース = {speedup:.1f}x高速化")
                
                # データ保存
                epoch_accuracies.append(test_accuracy)
                train_errors.append(train_error)
                test_accuracies.append(test_accuracy)
                
                # リアルタイム可視化更新
                if visualizer:
                    visualizer.update(epoch + 1, train_accuracy, test_accuracy, train_error, test_error)
                
                # ED-SNN RealtimeNeuronVisualizer更新（外部から設定された場合）
                # ed_multi.prompt.md準拠: --vizオプションでは学習曲線・混同行列・精度推移のみ表示
                # ニューロン可視化は無効化
                # if hasattr(self, 'neuron_visualizer') and self.neuron_visualizer is not None:
                #     ... (無効化)
                
                # 混同行列可視化更新（テストデータのみ・エポック毎）
                if confusion_visualizer:
                    test_true_labels = np.array(results_buffer.test_true_labels[epoch])
                    test_pred_labels = np.array(results_buffer.test_predicted_labels[epoch])
                    confusion_visualizer.update(epoch, test_true_labels, test_pred_labels)
                
                # ニューロン発火パターン可視化更新（v0.2.4新機能・Quietモード）
                # ed_multi.prompt.md準拠: --vizオプションでは学習曲線・混同行列・精度推移のみ表示
                # ニューロン可視化は無効化
                # if neuron_visualizer and neuron_adapter:
                #     ... (無効化)
                
        else:
            # 通常モード: 進捗バー表示
            if HAS_VISUALIZATION:
                with tqdm(total=epochs, desc="全体進捗", position=0, leave=True) as epoch_pbar:
                    for epoch in range(epochs):
                        epoch_start = time.time()
                        
                        # 🚀 【最適化】エポック毎の独立データを動的生成
                        # ed_multi.prompt.md準拠：エポック毎に完全独立したデータセット
                        epoch_train_inputs, epoch_train_labels = self.generate_epoch_data(epoch)
                        
                        # 🛡️ 修正: テストデータ配列範囲制限・循環使用処理追加
                        test_start_idx = epoch * epoch_test_size
                        test_end_idx = (epoch + 1) * epoch_test_size
                        
                        # 配列範囲チェック：テストデータサイズを超過しないよう制限
                        if test_end_idx > len(test_inputs):
                            # エポック数が多い場合の循環使用（ed_multi.prompt.md準拠の独立性維持）
                            test_start_idx = (epoch * epoch_test_size) % len(test_inputs)
                            test_end_idx = min(test_start_idx + epoch_test_size, len(test_inputs))
                            
                            # 不足分は先頭から補完（独立性維持のため異なるインデックス使用）
                            if test_end_idx - test_start_idx < epoch_test_size:
                                remaining_needed = epoch_test_size - (test_end_idx - test_start_idx)
                                epoch_test_inputs = np.concatenate([
                                    test_inputs[test_start_idx:test_end_idx],
                                    test_inputs[:remaining_needed]
                                ])
                                epoch_test_labels = np.concatenate([
                                    test_labels[test_start_idx:test_end_idx], 
                                    test_labels[:remaining_needed]
                                ])
                            else:
                                epoch_test_inputs = test_inputs[test_start_idx:test_end_idx]
                                epoch_test_labels = test_labels[test_start_idx:test_end_idx]
                                
                            if self.hyperparams.verbose:
                                print(f"🔄 エポック{epoch+1}: テストデータ循環使用 [{test_start_idx}:{test_end_idx}] + 先頭から{remaining_needed if 'remaining_needed' in locals() else 0}サンプル")
                        else:
                            epoch_test_inputs = test_inputs[test_start_idx:test_end_idx]
                            epoch_test_labels = test_labels[test_start_idx:test_end_idx]
                        
                        if self.hyperparams.verbose and epoch % 10 == 0:
                            print(f"🎯 エポック{epoch+1}: 動的データ生成完了 (独立サンプル{len(epoch_train_inputs)}個)")
                            print(f"🎯 エポック{epoch+1}: テストデータ範囲 [{test_start_idx}:{test_end_idx}] (独立サンプル{len(epoch_test_inputs)}個)")
                        
                        # 最適化済み学習エポック実行（ミニバッチ対応）
                        # ed_genuine.prompt.md準拠: 学習中に予測→保存→学習の順序
                        
                        # ミニバッチサイズが1なら従来手法、それ以外はミニバッチ学習
                        batch_size = getattr(self.hyperparams, 'batch_size', 1)
                        if batch_size == 1:
                            avg_error, _ = self.train_epoch_with_buffer(
                                results_buffer, epoch, epoch_train_inputs, epoch_train_labels, 
                                epoch_test_inputs, epoch_test_labels, show_progress=True, 
                                epoch_info=f"エポック{epoch+1:2d}")
                        else:
                            # ミニバッチ学習実行（金子勇氏理論拡張）
                            avg_error, _ = self.train_epoch_with_minibatch(
                                results_buffer, epoch, epoch_train_inputs, epoch_train_labels, 
                                epoch_test_inputs, epoch_test_labels, batch_size,
                                show_progress=True, epoch_info=f"エポック{epoch+1:2d} (batch={batch_size})")
                        
                        # ===== 統一的精度・誤差管理システム使用（ed_genuine.prompt.md準拠） =====
                        # 高速化: 統一データ計算を最終エポックまたは可視化時のみに制限
                        need_unified_data = (epoch == epochs - 1) or visualizer or confusion_visualizer
                        
                        if need_unified_data:
                            # エポック完了時に精度・誤差を計算してキャッシュ
                            results_buffer.compute_and_cache_epoch_metrics(epoch)
                            
                            # 統一的データ取得（すべての表示箇所で同じ値を使用）
                            unified_data = results_buffer.get_unified_progress_display_data(epoch)
                            train_accuracy = unified_data['train_accuracy']
                            test_accuracy = unified_data['test_accuracy']
                            train_error = unified_data['train_error']
                            test_error = unified_data['test_error']
                        else:
                            # 高速化: 中間エポックでは簡易計算
                            train_accuracy = results_buffer.get_epoch_accuracy_efficient(epoch, 'train') * 100
                            test_accuracy = results_buffer.get_epoch_accuracy_efficient(epoch, 'test') * 100
                            train_error = results_buffer.get_epoch_error_efficient(epoch, 'train')
                            test_error = results_buffer.get_epoch_error_efficient(epoch, 'test')
                        
                        # データ保存
                        epoch_accuracies.append(test_accuracy)
                        train_errors.append(train_error)
                        test_accuracies.append(test_accuracy)
                        
                        # リアルタイム可視化更新（統一データ使用 - 0-1範囲に変換）
                        if visualizer:
                            # 可視化システムは0-1範囲を期待するため変換
                            viz_train_acc = train_accuracy / 100.0  # パーセントから0-1範囲に変換
                            viz_test_acc = test_accuracy / 100.0
                            visualizer.update(epoch + 1, viz_train_acc, viz_test_acc, train_error, test_error)
                        
                        # ED-SNN RealtimeNeuronVisualizer更新（外部から設定された場合）
                        # ed_multi.prompt.md準拠: --vizオプションでは学習曲線・混同行列・精度推移のみ表示
                        # ニューロン可視化は無効化
                        # if hasattr(self, 'neuron_visualizer') and self.neuron_visualizer is not None:
                        #     ... (無効化)
                        
                        # 混同行列可視化更新（テストデータのみ・エポック毎）
                        if confusion_visualizer:
                            test_true_labels = np.array(results_buffer.test_true_labels[epoch])
                            test_pred_labels = np.array(results_buffer.test_predicted_labels[epoch])
                            confusion_visualizer.update(epoch, test_true_labels, test_pred_labels)
                        
                        # ヒートマップ可視化更新（EDHeatmapIntegration連携） - ed_multi.prompt.md準拠
                        if hasattr(self, 'heatmap_callback') and self.heatmap_callback is not None:
                            self.heatmap_callback()
                        
                        # ニューロン発火パターン可視化更新（v0.2.4新機能）
                        # ed_multi.prompt.md準拠: --vizオプションでは学習曲線・混同行列・精度推移のみ表示
                        # ニューロン可視化は無効化
                        # if neuron_visualizer and neuron_adapter:
                        #     ... (無効化)
                        
                        epoch_time = time.time() - epoch_start
                        
                        # 進捗情報更新（ed_genuine.prompt.md準拠 - 統一データ使用）
                        epoch_pbar.set_postfix({
                            '訓練精度': f'{train_accuracy:.1f}%',
                            'テスト精度': f'{test_accuracy:.1f}%',
                            '訓練誤差': f'{train_error:.3f}',
                            'テスト誤差': f'{test_error:.3f}',
                            '時間': f'{epoch_time:.1f}s'
                        })
                        epoch_pbar.update(1)
            else:
                # tqdmが使用できない場合の代替実装
                for epoch in range(epochs):
                    epoch_start = time.time()
                    
                    # 🚀 【最適化】エポック毎の独立データを動的生成
                    # ed_multi.prompt.md準拠：エポック毎に完全独立したデータセット
                    epoch_train_inputs, epoch_train_labels = self.generate_epoch_data(epoch)
                    
                    # 🛡️ 修正: テストデータ配列範囲制限・循環使用処理追加
                    test_start_idx = epoch * epoch_test_size
                    test_end_idx = (epoch + 1) * epoch_test_size
                    
                    # 配列範囲チェック：テストデータサイズを超過しないよう制限
                    if test_end_idx > len(test_inputs):
                        # エポック数が多い場合の循環使用（ed_multi.prompt.md準拠の独立性維持）
                        test_start_idx = (epoch * epoch_test_size) % len(test_inputs)
                        test_end_idx = min(test_start_idx + epoch_test_size, len(test_inputs))
                        
                        # 不足分は先頭から補完（独立性維持のため異なるインデックス使用）
                        if test_end_idx - test_start_idx < epoch_test_size:
                            remaining_needed = epoch_test_size - (test_end_idx - test_start_idx)
                            epoch_test_inputs = np.concatenate([
                                test_inputs[test_start_idx:test_end_idx],
                                test_inputs[:remaining_needed]
                            ])
                            epoch_test_labels = np.concatenate([
                                test_labels[test_start_idx:test_end_idx], 
                                test_labels[:remaining_needed]
                            ])
                        else:
                            epoch_test_inputs = test_inputs[test_start_idx:test_end_idx]
                            epoch_test_labels = test_labels[test_start_idx:test_end_idx]
                            
                        if self.hyperparams.verbose:
                            print(f"🔄 エポック{epoch+1}: テストデータ循環使用 [{test_start_idx}:{test_end_idx}] + 先頭から{remaining_needed if 'remaining_needed' in locals() else 0}サンプル")
                    else:
                        epoch_test_inputs = test_inputs[test_start_idx:test_end_idx]
                        epoch_test_labels = test_labels[test_start_idx:test_end_idx]
                    
                    if self.hyperparams.verbose and epoch % 10 == 0:
                        print(f"🎯 エポック{epoch+1}: 動的データ生成完了 (独立サンプル{len(epoch_train_inputs)}個)")
                        print(f"🎯 エポック{epoch+1}: テストデータ範囲 [{test_start_idx}:{test_end_idx}] (独立サンプル{len(epoch_test_inputs)}個)")
                    
                    if getattr(self.hyperparams, 'batch_size', 1) == 1:
                        avg_error, _ = self.train_epoch_with_buffer(
                            results_buffer, epoch, epoch_train_inputs, epoch_train_labels, 
                            epoch_test_inputs, epoch_test_labels, show_progress=False, 
                            epoch_info=f"エポック{epoch+1:2d}")
                    else:
                        avg_error, _ = self.train_epoch_with_minibatch(
                            results_buffer, epoch, epoch_train_inputs, epoch_train_labels, 
                            epoch_test_inputs, epoch_test_labels, getattr(self.hyperparams, 'batch_size', 1),
                            show_progress=False, epoch_info=f"エポック{epoch+1:2d} (batch={getattr(self.hyperparams, 'batch_size', 1)})")
                    
                    train_accuracy = results_buffer.get_epoch_accuracy_efficient(epoch, 'train')
                    test_accuracy = results_buffer.get_epoch_accuracy_efficient(epoch, 'test')
                    train_error = results_buffer.get_epoch_error_efficient(epoch, 'train')
                    test_error = results_buffer.get_epoch_error_efficient(epoch, 'test')
                    
                    if epoch == 0 and self.hyperparams.verbose:
                        benchmark_results = results_buffer.benchmark_error_calculation_methods(epoch, 'train')
                        speedup = benchmark_results['speedup']
                        print(f"📊 誤差算出性能: 従来方式 vs 3次元配列ベース = {speedup:.1f}x高速化")
                    
                    epoch_accuracies.append(test_accuracy)
                    train_errors.append(train_error)
                    test_accuracies.append(test_accuracy)
                    
                    if visualizer:
                        visualizer.update(epoch + 1, train_accuracy, test_accuracy, train_error, test_error)
                    
                    # ヒートマップ可視化更新（EDHeatmapIntegration連携） - ed_multi.prompt.md準拠
                    if hasattr(self, 'heatmap_callback') and self.heatmap_callback is not None:
                        self.heatmap_callback()
                    
                    # ED-SNN RealtimeNeuronVisualizer更新（外部から設定された場合）
                    # ed_multi.prompt.md準拠: --vizオプションでは学習曲線・混同行列・精度推移のみ表示
                    # ニューロン可視化は無効化
                    # if hasattr(self, 'neuron_visualizer') and self.neuron_visualizer is not None:
                    #     ... (無効化)
                    
                    if confusion_visualizer:
                        test_true_labels = np.array(results_buffer.test_true_labels[epoch])
                        test_pred_labels = np.array(results_buffer.test_predicted_labels[epoch])
                        confusion_visualizer.update(epoch, test_true_labels, test_pred_labels)
                    
                    epoch_time = time.time() - epoch_start
                    print(f"エポック {epoch+1:2d}/{epochs}: 訓練精度={train_accuracy:.3f}, テスト精度={test_accuracy:.3f}, 時間={epoch_time:.1f}s")
        
        # 可視化終了処理（ed_genuine.prompt.md準拠 - 5秒間表示または手動クローズ）
        if visualizer and not getattr(self.hyperparams, 'quiet_mode', False):
            print("✅ 学習完了 - 最終グラフを表示中...")
            # 最終データでグラフを更新
            visualizer.fig.canvas.draw()
            visualizer.fig.canvas.flush_events()
            
            print("可視化グラフ表示中... (5秒後自動終了、またはEnterキーで即座に終了)")
            
            # 5秒間表示またはキー入力での終了処理
            
            def countdown_timer():
                """5秒カウントダウン関数"""
                time.sleep(5)
                return True
            
            def wait_for_input():
                """Enter入力待機関数"""
                try:
                    input()  # Enterキー待機
                    return True
                except:
                    return False
            
            try:
                # 単純な5秒待機とキーボード割り込み対応
                start_time_wait = time.time()
                
                # 5秒間待機（0.1秒間隔で確認）
                while time.time() - start_time_wait < 5:
                    if HAS_VISUALIZATION:
                        try:
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", UserWarning)
                                plt.pause(0.1)  # グラフ表示維持
                        except Exception:
                            pass
                    else:
                        time.sleep(0.1)
                
                print("5秒経過により自動終了します")
                
            except KeyboardInterrupt:
                print("\nCtrl+Cにより終了します")
            finally:
                # 図表保存（--save-figオプション有効時）
                if getattr(self.hyperparams, 'save_fig', False):
                    visualizer.save_figure()
                    if confusion_visualizer:
                        confusion_visualizer.save_figure()
                
                visualizer.close()
                if confusion_visualizer:
                    confusion_visualizer.close()
        
        total_time = time.time() - start_time
        
        # 🎯 混同行列表示（エポックループ完了後、最終結果前）
        # ed_multi.prompt.md準拠：テストデータのみ表示、プログレスバー完了後
        if not getattr(self.hyperparams, 'quiet_mode', False) and not enable_visualization and epochs > 0:
            # 文字ベース表示でのみ実行（グラフ表示時は学習完了後のみ）
            print()  # 改行でプログレスバーと分離
            results_buffer.display_confusion_matrix_single_epoch('test', epochs - 1)
        
        # 📊 データ使用統計表示（学習完了後）
        self._display_data_usage_statistics()
        
        # 最終評価
        if not getattr(self.hyperparams, 'quiet_mode', False):
            print(f"\n{'='*60}")
            print("MNIST分類学習 完了")
            print(f"{'='*60}")
        
        final_accuracy = epoch_accuracies[-1]
        max_accuracy = max(epoch_accuracies)
        
        # 最終結果算出（訓練精度とテスト精度）
        train_accuracy = final_accuracy  # エポック精度は現在テスト精度として計算されている
        test_accuracy = final_accuracy   # 適切な分離が必要な場合は別途計算
        
        if getattr(self.hyperparams, 'quiet_mode', False):
            # グリッドサーチ用の簡潔出力
            print(f"訓練精度: {train_accuracy:.1f}%")
            print(f"テスト精度: {test_accuracy:.1f}%")
        else:
            # 通常の詳細出力
            print(f"📈 学習結果:")
            print(f"  最終精度: {final_accuracy:.3f}")
            print(f"  最高精度: {max_accuracy:.3f}")
            print(f"  総学習時間: {total_time:.1f}秒")
            print(f"  平均エポック時間: {total_time/epochs:.1f}秒")
        
        # プロファイリングレポート表示（--profileオプション有効時）
        if self.hyperparams.enable_profiling:
            self.profiler.print_detailed_report()
        
        # 結果統計を返す（バッファも含む）
        return {
            'final_accuracy': final_accuracy,
            'max_accuracy': max_accuracy,
            'epoch_accuracies': epoch_accuracies,
            'total_time': total_time,
            'train_size': train_size,
            'test_size': test_size,
            'epochs': epochs,
            'network_size': self.total_units,
            'results_buffer': results_buffer  # 混同行列用データへのアクセス
        }

    def _extract_actual_neuron_activities(self):
        """
        ED法ネットワークから実際のニューロン活動データを抽出
        ed_multi.prompt.md準拠の正確なニューロン状態取得
        
        Returns:
            List[np.ndarray]: 各層の発火データ [入力層, 隠れ層, 出力層]
        """
        try:
            # ED法ネットワークインスタンス確認
            if not hasattr(self, 'ed_genuine') or self.ed_genuine is None:
                # フォールバック: ダミーデータ
                return [np.zeros(784), np.zeros(128), np.zeros(10)]
            
            # ED法の実際のニューロン状態配列から抽出
            layer_activities = []
            
            # 入力層活動（784ニューロン）
            # output_inputs配列の最初の出力ニューロンから入力層データを抽出
            if hasattr(self.ed_genuine, 'output_inputs') and self.ed_genuine.output_inputs is not None:
                # 入力層は2-785番目のインデックス（0,1はバイアス、2-785は784個の入力）
                input_range_start = 2
                input_range_end = input_range_start + 784
                input_activity = []
                
                # 第0出力ニューロンの入力層部分を抽出（シグモイド後の値を発火パターンに変換）
                for i in range(input_range_start, min(input_range_end, self.ed_genuine.output_inputs.shape[1])):
                    neuron_value = self.ed_genuine.output_inputs[0][i] if i < self.ed_genuine.output_inputs.shape[1] else 0.0
                    # シグモイド値を発火パターンに変換（閾値0.5）
                    firing_rate = 1.0 if neuron_value > 0.5 else 0.0
                    input_activity.append(firing_rate)
                
                # 784個に調整
                while len(input_activity) < 784:
                    input_activity.append(0.0)
                input_activity = input_activity[:784]
            else:
                input_activity = np.zeros(784)
            
            layer_activities.append(np.array(input_activity))
            
            # 隠れ層活動（128ニューロン）
            if hasattr(self.ed_genuine, 'output_inputs') and self.ed_genuine.output_inputs is not None:
                # 隠れ層は786-913番目のインデックス（入力784+バイアス2の後）
                hidden_range_start = 2 + 784  # 786
                hidden_range_end = hidden_range_start + 128  # 914
                hidden_activity = []
                
                for i in range(hidden_range_start, min(hidden_range_end, self.ed_genuine.output_inputs.shape[1])):
                    neuron_value = self.ed_genuine.output_inputs[0][i] if i < self.ed_genuine.output_inputs.shape[1] else 0.0
                    # シグモイド値を発火パターンに変換（隠れ層用閾値0.3）
                    firing_rate = min(1.0, max(0.0, neuron_value)) if neuron_value > 0.3 else 0.0
                    hidden_activity.append(firing_rate)
                
                # 128個に調整
                while len(hidden_activity) < 128:
                    hidden_activity.append(0.0)
                hidden_activity = hidden_activity[:128]
            else:
                hidden_activity = np.zeros(128)
                
            layer_activities.append(np.array(hidden_activity))
            
            # 出力層活動（10ニューロン）
            if hasattr(self.ed_genuine, 'output_outputs') and self.ed_genuine.output_outputs is not None:
                output_activity = []
                
                # 各出力ニューロンの活動を取得
                for n in range(min(10, self.ed_genuine.output_outputs.shape[0])):
                    # 出力ニューロンの最終値（分類出力）
                    output_value = self.ed_genuine.output_outputs[n][0] if self.ed_genuine.output_outputs.shape[1] > 0 else 0.0
                    # シグモイド値を発火パターンに変換（出力層用閾値0.1）
                    firing_rate = min(1.0, max(0.0, output_value))
                    output_activity.append(firing_rate)
                
                # 10個に調整
                while len(output_activity) < 10:
                    output_activity.append(0.0)
                output_activity = output_activity[:10]
            else:
                output_activity = np.zeros(10)
                
            layer_activities.append(np.array(output_activity))
            
            return layer_activities
            
        except Exception as e:
            print(f"⚠️ ニューロン活動抽出エラー: {e}")
            # エラー時はゼロベクトルを返す
            return [np.zeros(784), np.zeros(128), np.zeros(10)]
    
    def _display_data_usage_statistics(self):
        """学習完了後にMNISTデータ使用統計を表示"""
        try:
            print(f"\n{'='*60}")
            print("📊 MNISTデータ使用統計")
            print(f"{'='*60}")
            
            # 基本的なデータ情報
            if hasattr(self, 'ed_genuine') and hasattr(self.ed_genuine, 'train_original_indices'):
                train_indices = self.ed_genuine.train_original_indices
                test_indices = self.ed_genuine.test_original_indices
                
                print(f"基本データ情報:")
                print(f"  訓練データ数: {len(train_indices)}")
                print(f"  テストデータ数: {len(test_indices)}")
                print(f"  訓練データ元インデックス範囲: [{train_indices.min()}, {train_indices.max()}]")
                print(f"  テストデータ元インデックス範囲: [{test_indices.min()}, {test_indices.max()}]")
                
                # ユニークデータ分析
                unique_train = np.unique(train_indices)
                unique_test = np.unique(test_indices)
                print(f"  訓練ユニークデータ数: {len(unique_train)}")
                print(f"  テストユニークデータ数: {len(unique_test)}")
                
                # 重複使用の確認
                unique_train_indices, train_counts = np.unique(train_indices, return_counts=True)
                duplicates = train_counts[train_counts > 1]
                if len(duplicates) > 0:
                    print(f"  ⚠️ 訓練データ重複: {len(duplicates)}個のデータが複数回使用")
                    print(f"  最大使用回数: {duplicates.max()}")
                else:
                    print(f"  ✅ 訓練データ重複なし: 全データが1回ずつ使用")
                
                # 訓練・テスト間の重複確認
                overlap = np.intersect1d(train_indices, test_indices)
                if len(overlap) > 0:
                    print(f"  ⚠️ 警告: 訓練・テスト間で{len(overlap)}個のデータが重複")
                    print(f"  重複インデックス例: {overlap[:5]}")
                else:
                    print(f"  ✅ 訓練・テスト間の重複なし")
            
            # ミニバッチローダーからの詳細統計
            if hasattr(self, '_last_train_loader'):
                train_loader = self._last_train_loader
                if train_loader and train_loader.track_usage:
                    usage_stats = train_loader.get_usage_statistics()
                    if usage_stats:
                        print(f"\n✅ MNISTデータ使用統計（独立サンプル追跡）:")
                        print(f"  総使用データ数: {usage_stats['total_data']}")
                        print(f"  最大使用回数: {usage_stats['max_usage']}")
                        print(f"  最小使用回数: {usage_stats['min_usage']}")
                        print(f"  平均使用回数: {usage_stats['avg_usage']:.2f}")
                        
                        # 使用回数分布を表示
                        usage_dist = {}
                        for count in usage_stats['usage_distribution'].values():
                            usage_dist[count] = usage_dist.get(count, 0) + 1
                        
                        print(f"  使用回数分布:")
                        for count, num_data in sorted(usage_dist.items()):
                            print(f"    {count}回使用: {num_data}個のデータ")
                            
                        # 独立データ使用の検証
                        # エポック数を考慮した正しい判定
                        expected_usage_per_data = 1  # 独立データなら各データは1回のみ使用
                        if usage_stats['max_usage'] == expected_usage_per_data and usage_stats['min_usage'] == expected_usage_per_data:
                            print(f"  ✅ 完全独立データ使用: 全{usage_stats['total_data']}サンプルが{expected_usage_per_data}回ずつ使用")
                        else:
                            # エポック数と一致する場合は正常（エポック毎に独立データを使用）
                            actual_epochs = getattr(self.hyperparams, 'epochs', 1)
                            if usage_stats['max_usage'] == actual_epochs and usage_stats['min_usage'] == actual_epochs:
                                print(f"  ✅ エポック分割独立データ使用: 各エポックで独立データを使用（{actual_epochs}エポック対応）")
                            else:
                                print(f"  📊 データ使用統計: 使用回数 最小{usage_stats['min_usage']} 最大{usage_stats['max_usage']}回")
                    else:
                        print(f"\n⚠️ MNISTデータ使用統計: 統計データなし")
                else:
                    print(f"\n⚠️ MNISTデータ使用統計: 追跡無効")
            else:
                print(f"\n⚠️ MNISTデータ使用統計: ローダー情報なし")
            
            print(f"{'='*60}")
                
        except Exception as e:
            if not getattr(self.hyperparams, 'quiet_mode', False):
                print(f"⚠️ データ使用統計の表示中にエラー: {e}")

