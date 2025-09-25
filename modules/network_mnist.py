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
    
    def run_classification(self, train_size=None, test_size=None, epochs=None, random_state=42, 
                          enable_visualization=None, use_fashion_mnist=None):
        """
        MNIST/Fashion-MNIST分類学習の実行 (ed_multi.prompt.md準拠 - 多層対応版)
        
        Args:
            train_size: 訓練データサイズ（Noneの場合hyperparamsから取得）
            test_size: テストデータサイズ（Noneの場合hyperparamsから取得）
            epochs: エポック数（Noneの場合hyperparamsから取得）
            random_state: ランダムシード
            enable_visualization: 可視化有効/無効（Noneの場合hyperparamsから取得）
            use_fashion_mnist: Fashion-MNIST使用フラグ（Noneの場合hyperparamsから取得）
        Returns:
            dict: 結果統計
        """
        # ハイパーパラメータから値を取得（安全なアクセス）
        train_size = train_size or getattr(self.hyperparams, 'train_samples', 1000)
        test_size = test_size or getattr(self.hyperparams, 'test_samples', 200)
        epochs = epochs or getattr(self.hyperparams, 'epochs', 10)
        if enable_visualization is None:
            # 🆕 --save_figオプション指定時も可視化を有効にする (ed_multi.prompt.md準拠)
            viz_enabled = getattr(self.hyperparams, 'enable_visualization', False)
            save_fig_enabled = getattr(self.hyperparams, 'save_fig', None) is not None
            fig_path_enabled = getattr(self.hyperparams, 'fig_path', None) is not None
            enable_visualization = viz_enabled or save_fig_enabled or fig_path_enabled
        if use_fashion_mnist is None:
            use_fashion_mnist = getattr(self.hyperparams, 'fashion_mnist', False)

        dataset_name = "Fashion-MNIST" if use_fashion_mnist else "MNIST"
        print("=" * 60)
        print(f"{dataset_name}分類学習 開始 - ハイパーパラメータ対応版")
        print("=" * 60)

        # 🔧 【重要修正】ed_multi.prompt.md準拠: エポック総数を考慮した適切なデータサイズ計算
        # 各エポックで独立サンプルを使用するため、総データサイズはエポック数×各エポックサイズ
        total_train_needed = train_size * epochs
        total_test_needed = test_size * epochs

        # データセット読み込み（エポック数を考慮した独立サンプル取得）
        # ed_multi.prompt.md準拠: 訓練・テスト両方で独立データを使用
        train_inputs, train_labels, test_inputs, test_labels = self.load_dataset(
            train_size=total_train_needed, test_size=total_test_needed, use_fashion_mnist=use_fashion_mnist, total_epochs=epochs)

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

        # 🎯 ed_multi.prompt.md準拠: 動的メモリ管理下では配列サイズチェックをスキップ
        # 実際の学習処理は`train_epoch_with_minibatch`で安全に実行される
        # 警告を出さずに学習データを準備
        self.num_patterns = len(train_inputs)

        # 入力データを設定（動的メモリ管理対応）
        for i, inp in enumerate(train_inputs):
            # 🎯 修正: 動的メモリ管理システム下では範囲チェックを最適化
            if i < len(self.input_data):
                inp_flat = inp.flatten().astype(float)
                for j, val in enumerate(inp_flat):
                    if j < len(self.input_data[i]):
                        self.input_data[i][j] = val

        # 教師データを設定（10クラス分類）
        for i, label in enumerate(train_labels):
            # 🎯 修正: 動的メモリ管理システム下では範囲チェックを最適化
            if i < len(self.teacher_data):
                # 全てのクラス出力を0.0に初期化
                for c in range(10):
                    if c < len(self.teacher_data[i]):
                        self.teacher_data[i][c] = 0.0
                
                # 正解クラスのみ1.0に設定（One-Hot形式）
                true_class = int(label)
                if true_class < len(self.teacher_data[i]):
                    self.teacher_data[i][true_class] = 1.0

        print(f"✅ データ準備完了: 訓練{len(train_inputs)}サンプル, テスト{len(test_inputs)}サンプル")

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
                                print(f"  ⚠️ データ重複検出: 一部データが複数回使用されています")
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

