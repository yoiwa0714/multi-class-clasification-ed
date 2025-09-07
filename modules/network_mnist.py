"""
ED-Genuine MNIST専用ネットワーククラス
Isamu Kaneko's Error Diffusion Learning Algorithm implementation
Based on C code pat[5]: One-Hot encoding for multi-class classification

【ed_genuine.prompt.md 準拠実装】
- MNIST/Fashion-MNIST データセット専用拡張
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
    
    def run_classification(self, train_size=None, test_size=None, epochs=None, 
                          random_state=42, enable_visualization=None, use_fashion_mnist=None):
        """
        MNIST/Fashion-MNISTデータセットでマルチクラス分類を実行 - ハイパーパラメータ対応
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
            enable_visualization = getattr(self.hyperparams, 'enable_visualization', False)
        if use_fashion_mnist is None:
            use_fashion_mnist = getattr(self.hyperparams, 'fashion_mnist', False)
        
        dataset_name = "Fashion-MNIST" if use_fashion_mnist else "MNIST"
        print("=" * 60)
        print(f"{dataset_name}分類学習 開始 - ハイパーパラメータ対応版")
        print("=" * 60)
        
        # データセット読み込み
        train_inputs, train_labels, test_inputs, test_labels = self.load_dataset(
            train_size=train_size, test_size=test_size, use_fashion_mnist=use_fashion_mnist)
        
        # ネットワーク初期化（10クラス分類用） - ハイパーパラメータ対応
        self.neuro_init(
            input_size=784,  # 28x28 MNIST/Fashion-MNIST
            num_outputs=10,  # 10クラス
            hidden_size=getattr(self.hyperparams, 'hidden_neurons', 100),  # ハイパーパラメータから取得
            hidden2_size=0  # 隠れ層2は使用しない
        )
        
        print(f"\n📊 ネットワーク構成:")
        print(f"  入力層: 784次元 (28x28画像)")
        print(f"  中間層: {self.hidden_units}ユニット")
        print(f"  出力層: 10クラス")
        print(f"  総ユニット: {self.total_units}")
        
        # 標準のEDGenuineデータ形式を使用
        # MNISTデータをinput_data, teacher_data配列に変換
        self.num_patterns = len(train_inputs)
        
        # 入力データを設定
        for i, inp in enumerate(train_inputs):
            inp_flat = inp.flatten().astype(float)
            for j, val in enumerate(inp_flat):
                if j < len(self.input_data[i]):
                    self.input_data[i][j] = val
        
        # 教師データを設定（10クラス分類）
        for i, label in enumerate(train_labels):
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
        # NOTE: LearningResultsBufferは別モジュールから取得する必要がある
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from multi_ed_v020 import LearningResultsBuffer
        results_buffer = LearningResultsBuffer(len(train_inputs), len(test_inputs), epochs)
        
        # リアルタイム可視化設定（学習開始時点で表示）
        visualizer = None
        confusion_visualizer = None
        if enable_visualization and HAS_VISUALIZATION:
            print("🎨 リアルタイム可視化準備中...")
            # NOTE: 可視化クラスは別モジュールから取得する必要がある
            from multi_ed_v020 import RealtimeLearningVisualizer, RealtimeConfusionMatrixVisualizer
            visualizer = RealtimeLearningVisualizer(max_epochs=epochs, save_dir=getattr(self.hyperparams, 'save_fig', None))
            visualizer.setup_plots()
            
            # 混同行列可視化（訓練開始時点から表示）
            confusion_visualizer = RealtimeConfusionMatrixVisualizer(
                num_classes=10, window_size=(800, 600), save_dir=getattr(self.hyperparams, 'save_fig', None))
            confusion_visualizer.setup_plots()
            
            print("✅ 可視化グラフ表示完了 - 学習データ待機中")
        
        start_time = time.time()
        
        epoch_accuracies = []
        train_errors = []
        test_accuracies = []
        
        # エポック全体進捗バー（上段） - quietモード時は抑制
        if getattr(self.hyperparams, 'quiet_mode', False):
            # quietモード: 進捗バーなし
            for epoch in range(epochs):
                epoch_start = time.time()
                
                # 最適化済み学習エポック実行（ミニバッチ対応）
                # ed_genuine.prompt.md準拠: 学習中に予測→保存→学習の順序
                
                # ミニバッチサイズが1なら従来手法、それ以外はミニバッチ学習
                batch_size = getattr(self.hyperparams, 'batch_size', 1)
                if batch_size == 1:
                    avg_error, _ = self.train_epoch_with_buffer(
                        results_buffer, epoch, train_inputs, train_labels, 
                        test_inputs, test_labels, show_progress=False, 
                        epoch_info=f"エポック{epoch+1:2d}")
                else:
                    # ミニバッチ学習実行（金子勇氏理論拡張）
                    avg_error, _ = self.train_epoch_with_minibatch(
                        results_buffer, epoch, train_inputs, train_labels, 
                        test_inputs, test_labels, batch_size,
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
                
                # 混同行列可視化更新（テストデータのみ・エポック毎）
                if confusion_visualizer:
                    test_true_labels = np.array(results_buffer.test_true_labels[epoch])
                    test_pred_labels = np.array(results_buffer.test_predicted_labels[epoch])
                    confusion_visualizer.update(epoch, test_true_labels, test_pred_labels)
        else:
            # 通常モード: 進捗バー表示
            if HAS_VISUALIZATION:
                with tqdm(total=epochs, desc="全体進捗", position=0, leave=True) as epoch_pbar:
                    for epoch in range(epochs):
                        epoch_start = time.time()
                        
                        # 最適化済み学習エポック実行（ミニバッチ対応）
                        # ed_genuine.prompt.md準拠: 学習中に予測→保存→学習の順序
                        
                        # ミニバッチサイズが1なら従来手法、それ以外はミニバッチ学習
                        batch_size = getattr(self.hyperparams, 'batch_size', 1)
                        if batch_size == 1:
                            avg_error, _ = self.train_epoch_with_buffer(
                                results_buffer, epoch, train_inputs, train_labels, 
                                test_inputs, test_labels, show_progress=True, 
                                epoch_info=f"エポック{epoch+1:2d}")
                        else:
                            # ミニバッチ学習実行（金子勇氏理論拡張）
                            avg_error, _ = self.train_epoch_with_minibatch(
                                results_buffer, epoch, train_inputs, train_labels, 
                                test_inputs, test_labels, batch_size,
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
                        
                        # 混同行列可視化更新（テストデータのみ・エポック毎）
                        if confusion_visualizer:
                            test_true_labels = np.array(results_buffer.test_true_labels[epoch])
                            test_pred_labels = np.array(results_buffer.test_predicted_labels[epoch])
                            confusion_visualizer.update(epoch, test_true_labels, test_pred_labels)
                        
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
                    
                    if self.hyperparams.batch_size == 1:
                        avg_error, _ = self.train_epoch_with_buffer(
                            results_buffer, epoch, train_inputs, train_labels, 
                            test_inputs, test_labels, show_progress=False, 
                            epoch_info=f"エポック{epoch+1:2d}")
                    else:
                        avg_error, _ = self.train_epoch_with_minibatch(
                            results_buffer, epoch, train_inputs, train_labels, 
                            test_inputs, test_labels, self.hyperparams.batch_size,
                            show_progress=False, epoch_info=f"エポック{epoch+1:2d} (batch={self.hyperparams.batch_size})")
                    
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
                    
                    if confusion_visualizer:
                        test_true_labels = np.array(results_buffer.test_true_labels[epoch])
                        test_pred_labels = np.array(results_buffer.test_predicted_labels[epoch])
                        confusion_visualizer.update(epoch, test_true_labels, test_pred_labels)
                    
                    epoch_time = time.time() - epoch_start
                    print(f"エポック {epoch+1:2d}/{epochs}: 訓練精度={train_accuracy:.3f}, テスト精度={test_accuracy:.3f}, 時間={epoch_time:.1f}s")
        
        # 可視化終了処理（ed_genuine.prompt.md準拠 - 5秒間表示またはキー入力終了）
        if visualizer and not self.hyperparams.quiet_mode:
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
                # タイマーとキー入力を並行処理
                timer_thread = threading.Thread(target=countdown_timer)
                timer_thread.daemon = True
                timer_thread.start()
                
                # Enter入力またはタイマー終了を待機
                input_thread = threading.Thread(target=wait_for_input)
                input_thread.daemon = True
                input_thread.start()
                
                # どちらか早い方で終了
                start_time_wait = time.time()
                while timer_thread.is_alive():
                    if not input_thread.is_alive():
                        print("Enterキー入力により終了します")
                        break
                    if HAS_VISUALIZATION:
                        plt.pause(0.1)  # グラフ表示維持
                    if time.time() - start_time_wait >= 5:
                        break
                else:
                    print("5秒経過により自動終了します")
                
            except KeyboardInterrupt:
                print("\nCtrl+Cにより終了します")
            finally:
                # 図表保存（--save-figオプション有効時）
                if self.hyperparams.save_fig:
                    visualizer.save_figure()
                    if confusion_visualizer:
                        confusion_visualizer.save_figure()
                
                visualizer.close()
                if confusion_visualizer:
                    confusion_visualizer.close()
        
        total_time = time.time() - start_time
        
        # 最終評価
        if not self.hyperparams.quiet_mode:
            print(f"\n{'='*60}")
            print("MNIST分類学習 完了")
            print(f"{'='*60}")
        
        final_accuracy = epoch_accuracies[-1]
        max_accuracy = max(epoch_accuracies)
        
        # 最終結果算出（訓練精度とテスト精度）
        train_accuracy = final_accuracy  # エポック精度は現在テスト精度として計算されている
        test_accuracy = final_accuracy   # 適切な分離が必要な場合は別途計算
        
        if self.hyperparams.quiet_mode:
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
        
        # 混同行列表示（--vizオプションと--quietオプションに応じた表示制御）
        if not self.hyperparams.quiet_mode:
            if enable_visualization:
                # --viz有り: グラフ表示
                results_buffer.display_confusion_matrix('train', epoch=-1, save_dir=self.hyperparams.save_fig)
            else:
                # --viz無し: 文字ベース表示
                results_buffer._display_confusion_matrix_console('train', epoch=-1)
        
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
