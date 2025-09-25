"""
visualization.py - ED法リアルタイム可視化モジュール
ed_genuine.prompt.md準拠の実装
"""

import matplotlib
# バックエンドの明示的設定 - GUIディスプレイ用
try:
    # 対話的バックエンドを試行（優先順位）
    if matplotlib.get_backend() == 'agg':
        # Qt5Aggを最初に試行（最も安定）
        try:
            matplotlib.use('Qt5Agg', force=True)
        except Exception:
            try:
                # TkAggを次に試行
                matplotlib.use('TkAgg', force=True)
            except Exception:
                # 最後にaggのまま（保存専用）
                pass
except Exception:
    pass

import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import warnings
import threading
import time

# 日本語フォント設定（文字化け防止）
import matplotlib.font_manager as fm
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
    print(f"可視化モジュール: 日本語フォント設定 - {available_font}")
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'  # フォールバック


class RealtimeLearningVisualizer:
    """リアルタイム学習可視化クラス"""
    
    def __init__(self, max_epochs, window_size=(800, 600), save_dir=None):
        """
        可視化インスタンス初期化
        Args:
            max_epochs: 最大エポック数
            window_size: ウィンドウサイズ (width, height) - v0.1.3: 混同行列と同サイズに変更
            save_dir: 図表保存ディレクトリ (Noneなら保存しない)
        """
        self.max_epochs = max_epochs
        self.window_size = window_size
        self.save_dir = save_dir
        
        # データ保存用
        self.epochs = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.train_errors = []
        self.test_errors = []
        
        # グラフ初期化
        self.fig = None
        self.ax_acc = None
        self.ax_err = None
        self.lines = {}
        
        # 🎯 ed_multi.prompt.md準拠：パラメータボックス管理
        self.ed_params = {}
        self.exec_params = {}
        self.param_ax_ed = None
        self.param_ax_exec = None
        
    def setup_plots(self):
        """グラフの初期設定 - ed_genuine.prompt.md準拠, v0.1.3: サイズ800x600に拡大"""
        # ウィンドウサイズ設定（v0.1.3: 800x600に拡大 + パラメータボックス対応）
        dpi = 100
        figsize = (self.window_size[0]/dpi, self.window_size[1]/dpi)
        
        # 🎯 ed_multi.prompt.md準拠：パラメータボックス付きレイアウト
        # 4分割: パラメータボックス2つ（上段） + メイングラフ2つ（下段）
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        
        # パラメータボックス（上段）
        self.param_ax_ed = plt.subplot2grid((3, 2), (0, 0))
        self.param_ax_exec = plt.subplot2grid((3, 2), (0, 1))
        
        # メインの学習進捗グラフ（中段～下段）
        self.ax_acc = plt.subplot2grid((3, 2), (1, 0), rowspan=2)
        self.ax_err = plt.subplot2grid((3, 2), (1, 1), rowspan=2)
        
        # ウィンドウタイトル設定（安全な方法）
        try:
            if self.fig.canvas.manager:
                self.fig.canvas.manager.set_window_title("ED法学習進捗 - リアルタイム可視化")
        except:
            pass  # エラーがあっても継続
        
        # 左側グラフ：精度
        self.ax_acc.set_title("訓練・テスト精度", fontweight='bold')
        self.ax_acc.set_xlabel("エポック数")
        self.ax_acc.set_ylabel("精度")
        # ed_multi.prompt.md準拠: エポック数が1の場合も対応
        self.ax_acc.set_xlim(1, max(2, self.max_epochs))  # 最小幅を確保
        self.ax_acc.set_ylim(0, 1)
        self.ax_acc.grid(True, alpha=0.3)
        
        # 右側グラフ：誤差
        self.ax_err.set_title("訓練・テスト誤差", fontweight='bold')
        self.ax_err.set_xlabel("エポック数")
        self.ax_err.set_ylabel("誤差")
        # ed_multi.prompt.md準拠: エポック数が1の場合も対応
        self.ax_err.set_xlim(1, max(2, self.max_epochs))  # 最小幅を確保
        self.ax_err.set_ylim(0, 1.0)  # 🔧 縦軸スケールを0.0～1.0に修正
        self.ax_err.grid(True, alpha=0.3)
        
        # 線の初期化（空のデータで表示）
        self.lines['train_acc'], = self.ax_acc.plot([], [], 'b-', label='訓練精度', linewidth=2)
        self.lines['test_acc'], = self.ax_acc.plot([], [], 'r-', label='テスト精度', linewidth=2)
        self.lines['train_err'], = self.ax_err.plot([], [], 'b-', label='訓練誤差', linewidth=2)
        self.lines['test_err'], = self.ax_err.plot([], [], 'r-', label='テスト誤差', linewidth=2)
        
        # 凡例設定
        self.ax_acc.legend(loc='lower right', fontsize=10, framealpha=0.9)
        self.ax_err.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # 🎯 ed_multi.prompt.md準拠：パラメータボックス初期化
        self._setup_parameter_boxes()
        
        plt.tight_layout()
        
        # インタラクティブモード有効化と初期描画
        try:
            plt.ion()
            # 非インタラクティブ環境では警告が出るが動作には問題なし
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.show(block=False)  # 非ブロッキング表示
        except Exception:
            pass  # 表示失敗しても継続
        
        # 初期描画を強制実行
        try:
            if hasattr(self.fig, 'canvas') and self.fig.canvas:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
                # 非インタラクティブ環境での警告を抑制
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    plt.pause(0.05)
        except Exception:
            pass  # 描画失敗しても継続
    
    def update(self, epoch, train_acc, test_acc, train_err, test_err):
        """
        グラフデータ更新 - ed_genuine.prompt.md準拠のリアルタイム表示
        Args:
            epoch: 現在のエポック
            train_acc: 訓練精度
            test_acc: テスト精度  
            train_err: 訓練誤差
            test_err: テスト誤差
        """
        # データ追加
        self.epochs.append(epoch)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)
        self.train_errors.append(train_err)
        self.test_errors.append(test_err)
        
        # 線データ更新
        self.lines['train_acc'].set_data(self.epochs, self.train_accuracies)
        self.lines['test_acc'].set_data(self.epochs, self.test_accuracies)
        self.lines['train_err'].set_data(self.epochs, self.train_errors)
        self.lines['test_err'].set_data(self.epochs, self.test_errors)
        
        # 誤差グラフの縦軸再スケール処理
        if self.train_errors and self.test_errors:
            max_error = max(max(self.train_errors), max(self.test_errors))
            if max_error > 2.0:
                new_max = max_error * 1.2
                self.ax_err.set_ylim(0, new_max)
        
        # グラフ再描画（確実な表示更新）
        self.ax_acc.relim()
        self.ax_acc.autoscale_view()
        self.ax_err.relim()
        self.ax_err.autoscale_view()
        
        # キャンバス描画更新
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # 🎯 ed_multi.prompt.md準拠：パラメータボックス更新
        self._update_parameter_boxes()
        
        # 短時間の一時停止でリアルタイム表示（警告抑制）
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.pause(0.01)
        except Exception:
            pass
    
    def set_parameters(self, ed_params, exec_params):
        """パラメータ設定（ed_multi.prompt.md準拠）"""
        self.ed_params = ed_params.copy() if ed_params else {}
        self.exec_params = exec_params.copy() if exec_params else {}
        self._update_parameter_boxes()
    
    def _setup_parameter_boxes(self):
        """パラメータボックスの初期化（ed_multi.prompt.md準拠）"""
        if self.param_ax_ed is not None:
            self.param_ax_ed.axis('off')
        if self.param_ax_exec is not None:
            self.param_ax_exec.axis('off')
        self._update_parameter_boxes()
    
    def _update_parameter_boxes(self):
        """パラメータボックスの内容更新（ed_multi.prompt.md準拠）"""
        # ED法パラメータボックス更新
        if self.param_ax_ed is not None:
            self.param_ax_ed.clear()
            self.param_ax_ed.axis('off')
            
            ed_text = "ED法パラメータ設定\n"
            ed_text += f"学習率(α): {self.ed_params.get('learning_rate', 0.5)}\n"
            ed_text += f"初期アミン濃度(β): {self.ed_params.get('threshold', 0.8)}\n"
            ed_text += f"アミン拡散係数(u1): {self.ed_params.get('threshold_alpha', 1.0)}\n"
            ed_text += f"シグモイド閾値(u0): {self.ed_params.get('threshold_beta', 0.4)}\n"
            ed_text += f"重み初期値1: {self.ed_params.get('threshold_gamma', 1.0)}"
            
            self.param_ax_ed.text(0.5, 0.5, ed_text, transform=self.param_ax_ed.transAxes,
                                 fontsize=8, ha='center', va='center',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # 実行パラメータボックス更新
        if self.param_ax_exec is not None:
            self.param_ax_exec.clear()
            self.param_ax_exec.axis('off')
            
            exec_text = "実行パラメータ設定\n"
            exec_text += f"訓練データ数: {self.exec_params.get('train_size', 100)}\n"
            exec_text += f"テストデータ数: {self.exec_params.get('test_size', 100)}\n"
            exec_text += f"エポック数: {self.exec_params.get('epochs', 5)}\n"
            exec_text += f"隠れ層数: {self.exec_params.get('num_layers', 1)}\n"
            exec_text += f"ミニバッチサイズ: {self.exec_params.get('batch_size', 32)}"
            
            self.param_ax_exec.text(0.5, 0.5, exec_text, transform=self.param_ax_exec.transAxes,
                                   fontsize=8, ha='center', va='center',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    def close(self):
        """可視化ウィンドウを閉じる"""
        if self.fig:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    plt.close(self.fig)
            except Exception:
                # 警告抑制のエラーでも閉じる処理は継続
                plt.close(self.fig)
    
    def save_figure(self):
        """リアルタイム学習グラフを保存する"""
        if not self.save_dir or not self.fig:
            return
            
        # 🆕 save_dirが特定のファイルパス（.pngで終わる）かディレクトリかを判定
        if self.save_dir.endswith('.png'):
            # 特定のファイルパスとして保存（グリッドサーチ対応）
            filepath = self.save_dir
            # ディレクトリ部分を作成
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
        else:
            # ディレクトリとして扱い、タイムスタンプ付きファイル名を生成（従来動作）
            os.makedirs(self.save_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"realtime-{timestamp}.png"
            filepath = os.path.join(self.save_dir, filename)
        
        try:
            # グラフ保存
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"📊 リアルタイム学習グラフ保存完了: {filepath}")
        except Exception as e:
            print(f"❌ グラフ保存エラー: {e}")


class RealtimeConfusionMatrixVisualizer:
    """リアルタイム混同行列可視化クラス - ed_genuine.prompt.md準拠"""
    
    def __init__(self, num_classes=10, window_size=(640, 480), save_dir=None):
        """
        リアルタイム混同行列可視化インスタンス初期化
        Args:
            num_classes: クラス数（MNIST/Fashion-MNISTの場合は10）
            window_size: ウィンドウサイズ (width, height)
            save_dir: 図表保存ディレクトリ（Noneなら保存しない）
        """
        self.num_classes = num_classes
        self.window_size = window_size
        self.save_dir = save_dir

        # 累積混同行列データ（エポック毎に累積）
        self.cumulative_confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        self.total_samples = 0
        self.current_epoch = 0

        # グラフ関連
        self.fig = None
        self.ax_confusion = None
        self.ax_accuracy = None
        self.im = None  # ヒートマップオブジェクト
        self.colorbar = None

        # クラス精度履歴（エポック毎）
        self.epoch_accuracies = []  # 全体精度
        self.class_accuracies_history = []  # クラス別精度履歴
        
        # 🆕 エポック別混同行列履歴（最後のエポック用保存機能 - ed_multi.prompt.md準拠）
        self.epoch_confusion_matrices = []  # エポック毎の混同行列
        self.epoch_sample_counts = []  # エポック毎のサンプル数  # クラス別精度履歴
        
        # パラメータボックス関連
        self.param_ax_ed = None  # ED法パラメータボックス
        self.param_ax_exec = None  # 実行パラメータボックス
        self.ed_params = {}  # ED法パラメータ
        self.exec_params = {}  # 実行パラメータ
        
    def setup_plots(self):
        """混同行列グラフの初期設定（パラメータボックス対応版）"""
        # ウィンドウサイズ設定
        dpi = 100
        figsize = (self.window_size[0]/dpi, self.window_size[1]/dpi)
        
        try:
            # 🎯 ed_multi.prompt.md準拠：上部20％をパラメータボックス、下部80％をグラフ用にレイアウト変更
            self.fig = plt.figure(figsize=figsize, dpi=dpi)
            
            # パラメータボックス用（上部20％、幅調整版）
            # ED法パラメータボックス：幅2倍（2列分使用）
            self.param_ax_ed = self.fig.add_subplot(2, 6, (1, 2))  # 左側：ED法パラメータ（幅2倍）
            # 実行パラメータボックス：右側に移動
            self.param_ax_exec = self.fig.add_subplot(2, 6, 4)  # 右側：実行パラメータ
            
            # メイングラフ用（下部80％、左右分割）
            self.ax_confusion = self.fig.add_subplot(2, 2, 3)  # 混動行列
            self.ax_accuracy = self.fig.add_subplot(2, 2, 4)  # 精度グラフ
            
        except Exception as e:
            print(f"⚠️  可視化設定エラー: {e}")
            return
        
        # ウィンドウタイトル設定
        try:
            if hasattr(self.fig, 'canvas') and hasattr(self.fig.canvas, 'manager') and self.fig.canvas.manager:
                self.fig.canvas.manager.set_window_title("ED法リアルタイム混同行列 - 学習進捗監視")
        except Exception:
            pass
        
        # 左側：混同行列ヒートマップ
        initial_matrix = np.zeros((self.num_classes, self.num_classes))
        try:
            self.im = self.ax_confusion.imshow(initial_matrix, cmap='Blues', interpolation='nearest', 
                                            vmin=0, vmax=1)  # 初期値範囲設定
            if hasattr(self, '_debug_enabled') and self._debug_enabled:
                print(f"✅ ヒートマップオブジェクト初期化完了")
        except Exception as e:
            self.im = None
            print(f"⚠️  ヒートマップ初期化エラー: {e}")
            if hasattr(self, '_debug_enabled') and self._debug_enabled:
                print(f"   ax_confusion存在: {self.ax_confusion is not None}")
                import traceback
                traceback.print_exc()
        
        self.ax_confusion.set_title("混同行列（エポック単位）", fontweight='bold')
        self.ax_confusion.set_xlabel('予測クラス')
        self.ax_confusion.set_ylabel('正解クラス')
        
        # 軸ラベル設定
        self.ax_confusion.set_xticks(range(self.num_classes))
        self.ax_confusion.set_yticks(range(self.num_classes))
        self.ax_confusion.set_xticklabels(range(self.num_classes))
        self.ax_confusion.set_yticklabels(range(self.num_classes))
        
        # カラーバー追加
        try:
            if self.im is not None:
                self.colorbar = plt.colorbar(self.im, ax=self.ax_confusion, shrink=0.8)
        except Exception:
            self.colorbar = None
        
        # 右側：クラス別精度ヒストグラム（棒グラフ）
        self.ax_accuracy.set_title("クラス別精度分布", fontweight='bold')
        self.ax_accuracy.set_xlabel('クラス')
        self.ax_accuracy.set_ylabel('精度')
        self.ax_accuracy.set_xlim(-0.5, self.num_classes - 0.5)  # クラス範囲設定
        self.ax_accuracy.set_ylim(0, 1.0)
        self.ax_accuracy.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # パラメータボックスの初期化
        self._setup_parameter_boxes()
        
        # インタラクティブモード有効化（警告抑制）
        try:
            plt.ion()
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.show(block=False)
            
            # 初期描画
            if hasattr(self.fig, 'canvas') and self.fig.canvas:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                
                # 非インタラクティブ環境での警告を抑制
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    plt.pause(0.05)
        except Exception as e:
            print(f"⚠️  可視化初期化エラー: {e}")
            # 非インタラクティブ環境では表示失敗しても継続
        
    def update(self, epoch: int, true_labels: np.ndarray, predicted_labels: np.ndarray):
        """
        エポック終了時の混同行列更新
        Args:
            epoch: 現在のエポック（0から開始）
            true_labels: 実際のラベル配列
            predicted_labels: 予測ラベル配列
        """
        self.current_epoch = epoch + 1

        # エポック毎の混同行列計算
        epoch_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for true_label, pred_label in zip(true_labels, predicted_labels):
            if 0 <= true_label < self.num_classes and 0 <= pred_label < self.num_classes:
                epoch_matrix[true_label][pred_label] += 1

        # 🆕 エポック別混同行列履歴保存（最後のエポック用保存機能 - ed_multi.prompt.md準拠）
        self.epoch_confusion_matrices.append(epoch_matrix.copy())
        self.epoch_sample_counts.append(len(true_labels))

        # 累積混同行列更新
        self.cumulative_confusion_matrix += epoch_matrix
        self.total_samples += len(true_labels)

        # ed_multi.prompt.md準拠: エポック単位の混同行列を表示
        # 正規化混同行列計算（表示用）- エポック単位のデータを使用
        normalized_matrix = self._normalize_matrix(epoch_matrix)

        # ヒートマップ更新（エポック単位データを使用）
        if self.im is not None:
            try:
                self.im.set_array(normalized_matrix)
                # カラースケール動的調整
                max_val = np.max(normalized_matrix) if np.max(normalized_matrix) > 0 else 1.0
                self.im.set_clim(0, max_val)
            except Exception as e:
                print(f"⚠️  ヒートマップ更新エラー: {e}")

        # 数値表示更新（エポック単位データを使用）
        self._update_text_annotations(epoch_matrix, normalized_matrix)

        # クラス別精度計算と履歴更新（累積データから計算）
        class_accuracies = self._calculate_class_accuracies(self.cumulative_confusion_matrix)
        overall_accuracy = np.trace(self.cumulative_confusion_matrix) / max(1, self.total_samples)

        self.class_accuracies_history.append(class_accuracies)
        self.epoch_accuracies.append(overall_accuracy)

        # 右側グラフ更新
        self._update_accuracy_plot()

        # パラメータボックス更新（ed_multi.prompt.md準拠）
        if hasattr(self, 'ed_params') and hasattr(self, 'exec_params'):
            self._update_parameter_boxes()

        # タイトル更新（エポック単位情報を表示）
        try:
            # エポック単位の精度を計算
            epoch_accuracy = np.trace(epoch_matrix) / max(1, len(true_labels))
            title_text = f"混同行列（エポック単位）\nエポック{self.current_epoch}\n精度: {epoch_accuracy:.3f}, サンプル: {len(true_labels)}"
            self.ax_confusion.set_title(title_text)
        except Exception:
            pass

        # グラフ再描画（警告抑制）
        try:
            if hasattr(self.fig, 'canvas') and self.fig.canvas:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            
            # 非インタラクティブ環境での警告を抑制
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                plt.pause(0.01)
        except Exception as e:
            print(f"⚠️  グラフ描画エラー: {e}")
        
    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """混動行列の正規化（行毎の正規化）"""
        normalized = np.zeros_like(matrix, dtype=float)
        for i in range(matrix.shape[0]):
            row_sum = np.sum(matrix[i])
            if row_sum > 0:
                normalized[i] = matrix[i] / row_sum
        return normalized
        
    def _update_text_annotations(self, raw_matrix: np.ndarray, normalized_matrix: np.ndarray):
        """混同行列の数値表示更新"""
        try:
            # 既存の文字を削除
            if hasattr(self.ax_confusion, 'texts'):
                for txt in self.ax_confusion.texts:
                    txt.remove()
        except Exception:
            pass
        
        try:
            # 新しい文字を追加
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    raw_value = raw_matrix[i, j]
                    norm_value = normalized_matrix[i, j]
                    if raw_value > 0:
                        # 色の選択（背景の明度に応じて）
                        color = 'white' if norm_value > 0.5 else 'black'
                        # 生の数値を表示
                        self.ax_confusion.text(j, i, str(raw_value), ha='center', va='center',
                                              color=color, fontsize=8, weight='bold')
        except Exception as e:
            print(f"⚠️  数値表示更新エラー: {e}")
                                          
    def _calculate_class_accuracies(self, matrix: np.ndarray) -> np.ndarray:
        """クラス別精度計算"""
        accuracies = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            total_for_class = np.sum(matrix[i])
            if total_for_class > 0:
                accuracies[i] = matrix[i, i] / total_for_class
        return accuracies
        
    def _update_accuracy_plot(self):
        """クラス別精度ヒストグラム（棒グラフ）の更新 - ed_genuine.prompt.md準拠"""
        if not self.class_accuracies_history or not hasattr(self.ax_accuracy, 'clear'):
            return
            
        try:
            # グラフクリア
            self.ax_accuracy.clear()
            
            # 設定再適用
            self.ax_accuracy.set_title("クラス別精度分布", fontweight='bold')
            self.ax_accuracy.set_xlabel('クラス')
            self.ax_accuracy.set_ylabel('精度')
            self.ax_accuracy.set_xlim(-0.5, self.num_classes - 0.5)
            self.ax_accuracy.set_ylim(0, 1.0)
            self.ax_accuracy.grid(True, alpha=0.3)
            
            # 最新エポックのクラス別精度を取得
            if self.class_accuracies_history:
                latest_class_accuracies = self.class_accuracies_history[-1]
                
                # 棒グラフ作成
                x_pos = np.arange(self.num_classes)
                bars = self.ax_accuracy.bar(x_pos, latest_class_accuracies, 
                                          color='skyblue', alpha=0.7, 
                                          edgecolor='darkblue', linewidth=0.5)
                
                # X軸設定
                self.ax_accuracy.set_xticks(x_pos)
                self.ax_accuracy.set_xticklabels(range(self.num_classes))
                
                # 精度値を棒の上に表示
                for i, (bar, acc) in enumerate(zip(bars, latest_class_accuracies)):
                    if acc > 0:
                        self.ax_accuracy.text(bar.get_x() + bar.get_width()/2, 
                                            bar.get_height() + 0.02,
                                            f'{acc:.2f}', ha='center', va='bottom', 
                                            fontsize=8, weight='bold')
                
                # 全体精度を横線で表示
                if self.epoch_accuracies:
                    overall_accuracy = self.epoch_accuracies[-1]
                    self.ax_accuracy.axhline(y=overall_accuracy, color='red', 
                                           linestyle='--', alpha=0.8, linewidth=2,
                                           label=f'全体精度: {overall_accuracy:.3f}')
                    self.ax_accuracy.legend(loc='upper right', fontsize=8)
                
                # エポック情報表示
                epoch_text = f'エポック: {self.current_epoch}\nサンプル: {self.total_samples}'
                self.ax_accuracy.text(0.02, 0.98, epoch_text, transform=self.ax_accuracy.transAxes,
                                    verticalalignment='top', fontsize=8,
                                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
        except Exception as e:
            print(f"⚠️  クラス別ヒストグラム更新エラー: {e}")
        
    def close(self):
        """可視化ウィンドウを閉じる"""
        if self.fig:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    plt.close(self.fig)
            except Exception:
                # 警告抑制のエラーでも閉じる処理は継続
                plt.close(self.fig)
            
    def save_figure(self):
        """
        混同行列グラフを保存する (ed_multi.prompt.md準拠 - 最後のエポック重視)
        
        学習初期からの累積ではなく、最後の数エポック分のデータのみを使用。
        目安: 100/エポック以上のサンプル数になるまでのエポック数を使用。
        """
        if not self.save_dir or not self.fig:
            return
            
        # 🆕 save_dirが特定のファイルパス（.pngで終わる）かディレクトリかを判定
        if self.save_dir.endswith('.png'):
            # 特定のファイルパスの場合、混同行列用のファイル名を生成
            base_path = self.save_dir.replace('.png', '')
            directory = os.path.dirname(base_path)
            base_name = os.path.basename(base_path)
            os.makedirs(directory, exist_ok=True)
        else:
            # ディレクトリとして扱い、タイムスタンプ付きファイル名を生成（従来動作）
            directory = self.save_dir
            base_name = f"confusion-final-{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(directory, exist_ok=True)
        
        # 最後のエポックの混同行列計算（ed_multi.prompt.md準拠）
        recent_epochs_matrix, recent_sample_count = self._calculate_recent_epochs_matrix()
        
        # 元の累積データを一時保存
        original_cumulative = self.cumulative_confusion_matrix.copy()
        original_total_samples = self.total_samples
        
        try:
            # 最後のエポック分の混同行列で一時的に表示を更新
            self.cumulative_confusion_matrix = recent_epochs_matrix
            self.total_samples = recent_sample_count
            
            # 正規化混同行列計算
            normalized_matrix = self._normalize_matrix(recent_epochs_matrix)
            
            # ヒートマップ更新
            if self.im is not None:
                self.im.set_array(normalized_matrix)
                max_val = np.max(normalized_matrix) if np.max(normalized_matrix) > 0 else 1.0
                self.im.set_clim(0, max_val)
            
            # 数値表示更新
            self._update_text_annotations(recent_epochs_matrix, normalized_matrix)
            
            # タイトル更新（最後のエポック用）
            epochs_used = len([x for x in self.epoch_sample_counts[-10:] if sum(self.epoch_sample_counts[-10:]) >= 100])
            epochs_used = max(1, epochs_used)
            overall_accuracy = np.trace(recent_epochs_matrix) / max(1, recent_sample_count)
            
            title_text = f"混同行列（最後{epochs_used}エポック）\\n精度: {overall_accuracy:.3f}, サンプル: {recent_sample_count}"
            self.ax_confusion.set_title(title_text)
            
            # グラフ描画
            if hasattr(self.fig, 'canvas') and self.fig.canvas:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            
            # ファイル保存
            filename = f"{base_name}_confusion.png"
            filepath = os.path.join(directory, filename)
            
            # グラフ保存
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"📊 最終混同行列グラフ保存完了: {filepath}")
            print(f"   📈 対象: 最後{epochs_used}エポック分 (サンプル数: {recent_sample_count})")
            
        except Exception as e:
            print(f"❌ 混同行列グラフ保存エラー: {e}")
        finally:
            # 元の累積データに戻す
            self.cumulative_confusion_matrix = original_cumulative
            self.total_samples = original_total_samples
            
            # 表示を元に戻す（累積版）
            try:
                normalized_matrix = self._normalize_matrix(original_cumulative)
                if self.im is not None:
                    self.im.set_array(normalized_matrix)
                    max_val = np.max(normalized_matrix) if np.max(normalized_matrix) > 0 else 1.0
                    self.im.set_clim(0, max_val)
                self._update_text_annotations(original_cumulative, normalized_matrix)
                
                overall_accuracy = np.trace(original_cumulative) / max(1, original_total_samples)
                title_text = f"混同行列（エポック単位）\\nエポック{self.current_epoch}\\n精度: {overall_accuracy:.3f}, サンプル: {original_total_samples}"
                self.ax_confusion.set_title(title_text)
                
                if hasattr(self.fig, 'canvas') and self.fig.canvas:
                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
            except Exception:
                pass

    def _setup_parameter_boxes(self):
        """パラメータボックスの初期設定（ed_multi.prompt.md準拠）"""
        # ED法パラメータボックス（左側・薄い青色）
        if self.param_ax_ed is not None:
            self.param_ax_ed.axis('off')
            # 初期状態では空のボックスを表示
            self.param_ax_ed.text(0.5, 0.5, "ED法パラメータ設定\n（初期化中...）", 
                                 transform=self.param_ax_ed.transAxes,
                                 fontsize=9, ha='center', va='center',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # 実行パラメータボックス（右側・薄い緑色）
        if self.param_ax_exec is not None:
            self.param_ax_exec.axis('off')
            # 初期状態では空のボックスを表示
            self.param_ax_exec.text(0.5, 0.5, "実行パラメータ設定\n（初期化中...）", 
                                   transform=self.param_ax_exec.transAxes,
                                   fontsize=9, ha='center', va='center',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

    def set_parameters(self, ed_params: dict, exec_params: dict):
        """パラメータ設定とボックス更新（ed_multi.prompt.md準拠）"""
        self.ed_params = ed_params
        self.exec_params = exec_params
        self._update_parameter_boxes()

    def _update_parameter_boxes(self):
        """パラメータボックスの内容更新（ed_multi.prompt.md準拠）"""
        # ED法パラメータボックス更新
        if self.param_ax_ed is not None:
            self.param_ax_ed.clear()
            self.param_ax_ed.axis('off')
            
            ed_text = "ED法パラメータ設定\n"
            ed_text += f"学習率(α): {self.ed_params.get('learning_rate', 0.5)}\n"
            ed_text += f"初期アミン濃度(β): {self.ed_params.get('threshold', 0.8)}\n"
            ed_text += f"アミン拡散係数(u1): {self.ed_params.get('threshold_alpha', 1.0)}\n"
            ed_text += f"シグモイド閾値(u0): {self.ed_params.get('threshold_beta', 0.4)}\n"
            ed_text += f"重み初期値1: {self.ed_params.get('threshold_gamma', 1.0)}"
            
            self.param_ax_ed.text(0.5, 0.5, ed_text, transform=self.param_ax_ed.transAxes,
                                 fontsize=8, ha='center', va='center',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # 実行パラメータボックス更新
        if self.param_ax_exec is not None:
            self.param_ax_exec.clear()
            self.param_ax_exec.axis('off')
            
            exec_text = "実行パラメータ設定\n"
            exec_text += f"訓練データ数: {self.exec_params.get('train_size', 100)}\n"
            exec_text += f"テストデータ数: {self.exec_params.get('test_size', 100)}\n"
            exec_text += f"エポック数: {self.exec_params.get('epochs', 5)}\n"
            exec_text += f"隠れ層数: {self.exec_params.get('num_layers', 1)}\n"
            exec_text += f"ミニバッチサイズ: {self.exec_params.get('batch_size', 32)}"
            
            self.param_ax_exec.text(0.5, 0.5, exec_text, transform=self.param_ax_exec.transAxes,
                                   fontsize=8, ha='center', va='center',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))

    
    def _calculate_recent_epochs_matrix(self):
        """
        最後の数エポック分の混同行列を計算 (ed_multi.prompt.md準拠)
        
        Returns:
            tuple: (recent_epochs_matrix, recent_sample_count)
        """
        if not self.epoch_confusion_matrices or not self.epoch_sample_counts:
            # データがない場合は累積データを返す
            return self.cumulative_confusion_matrix.copy(), self.total_samples
        
        # 100/エポック以上になるまでのエポック数を決定
        target_samples = 100
        recent_sample_count = 0
        epochs_to_use = 0
        
        # 最後のエポックから遡って、100サンプル以上になるまでカウント
        for i in range(len(self.epoch_sample_counts) - 1, -1, -1):
            recent_sample_count += self.epoch_sample_counts[i]
            epochs_to_use += 1
            if recent_sample_count >= target_samples:
                break
        
        # 最低1エポックは使用
        epochs_to_use = max(1, epochs_to_use)
        
        # 最後のepochs_to_use分の混同行列を合計
        recent_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        start_idx = max(0, len(self.epoch_confusion_matrices) - epochs_to_use)
        
        actual_sample_count = 0
        for i in range(start_idx, len(self.epoch_confusion_matrices)):
            recent_matrix += self.epoch_confusion_matrices[i]
            actual_sample_count += self.epoch_sample_counts[i]
        
        return recent_matrix, actual_sample_count


class PredictionResultVisualizer:
    """
    学習完了後の予測結果可視化クラス
    入力画像と正解・予測クラスを一覧表示
    """
    
    def __init__(self, window_size=(800, 600), auto_close_seconds=5):
        """
        初期化
        Args:
            window_size: ウィンドウサイズ (width, height) - 精度/誤差グラフと同サイズ
            auto_close_seconds: 自動クローズまでの秒数（学習進捗グラフと同様）
        """
        self.window_size = window_size
        self.auto_close_seconds = auto_close_seconds
        self.fig = None
        self._close_timer = None
        
    def show_predictions(self, images, true_labels, predicted_labels, class_names=None):
        """
        予測結果を可視化表示
        
        Args:
            images: 入力画像データ (N, H, W) または (N, H, W, C)
            true_labels: 正解ラベル (N,)
            predicted_labels: 予測ラベル (N,)
            class_names: クラス名リスト（Noneなら数値表示）
        """
        if len(images) == 0:
            print("⚠️ 表示する画像データがありません")
            return
            
        # 既存のウィンドウがあれば閉じる（仕様7）
        if self.fig is not None:
            plt.close(self.fig)
            
        # 表示設定
        cols = 5  # 仕様4: 横5列固定
        rows = (len(images) + cols - 1) // cols  # 仕様5: (データ数÷5 + 1)行の実装
        
        # ウィンドウサイズ設定（仕様2）
        width_inch = self.window_size[0] / 100  # ピクセルをインチに変換
        height_inch = self.window_size[1] / 100
        
        # 図作成
        self.fig, axes = plt.subplots(rows, cols, figsize=(width_inch, height_inch))
        self.fig.suptitle('学習結果: 入力画像と予測クラス', fontsize=14, weight='bold')
        
        # 単一行の場合、axesを2次元にする
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
            
        # 各画像を表示
        for idx in range(len(images)):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # 画像表示（形状変換対応）
            image = images[idx]
            
            # 平坦化された画像データを28x28に変形（MNIST/Fashion-MNIST対応）
            if len(image.shape) == 1 and image.shape[0] == 784:
                image = image.reshape(28, 28)
            elif len(image.shape) == 3 and image.shape[0] == 1:
                image = image.squeeze(0)  # (1, H, W) -> (H, W)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                image = image.squeeze(2)  # (H, W, 1) -> (H, W)
                
            ax.imshow(image, cmap='gray')
            ax.axis('off')
            
            # ラベル表示（仕様8: 2行表示）
            true_label = true_labels[idx]
            pred_label = predicted_labels[idx]
            
            # クラス名またはラベル番号
            if class_names is not None:
                true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
                pred_name = class_names[pred_label] if pred_label < len(class_names) else str(pred_label)
            else:
                true_name = str(true_label)
                pred_name = str(pred_label)
            
            # 色設定（仕様9: 正解=青、誤答=赤）
            color = 'blue' if true_label == pred_label else 'red'
            
            # タイトル設定（仕様8: 2行表示）
            title_text = f'正解: {true_name}\n予測: {pred_name}'
            ax.set_title(title_text, fontsize=8, color=color, weight='bold')
            
        # 余った軸を非表示
        for idx in range(len(images), rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
            
        # レイアウト調整
        plt.tight_layout()
        
        # スクロール対応（仕様5）
        if rows > 10:  # 多すぎる場合はスクロールバー表示を促す
            print(f"📊 {len(images)}個の画像を{rows}行×{cols}列で表示中")
            print("💡 ウィンドウが大きい場合は、スクロールして全体をご確認ください")
            
        # ウィンドウ表示（仕様6: 自動クローズ機能付き）
        # interactiveモードでウィンドウを表示
        plt.ion()  # インタラクティブモード有効化
        plt.show(block=False)
        
        # 描画強制実行
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # 自動クローズタイマー開始（学習進捗グラフと同様）
        self._start_auto_close_timer()
        
        print("✅ 予測結果可視化ウィンドウを表示しました")
        print(f"⏰ ウィンドウは{self.auto_close_seconds}秒後に自動で閉じられます")
        print("💡 手動でクローズボタン[×]を押すことも可能です")
        
        print(f"✅ 予測結果可視化完了: {len(images)}個の画像を表示")
        print(f"📊 表示形式: {rows}行×{cols}列")
        
        correct_count = sum(1 for i in range(len(true_labels)) if true_labels[i] == predicted_labels[i])
        accuracy = correct_count / len(true_labels) * 100
        print(f"🎯 表示データ精度: {accuracy:.1f}% ({correct_count}/{len(true_labels)})")
        
    def _start_auto_close_timer(self):
        """自動クローズタイマーを開始（ed_multi.prompt.md準拠）"""
        def auto_close():
            time.sleep(self.auto_close_seconds)
            if self.fig is not None:
                try:
                    # メインスレッドで実行する必要があるため、after_idleを使用
                    self.fig.canvas.manager.window.after_idle(lambda: self.close())
                except Exception:
                    # フォールバック：直接クローズを試行
                    try:
                        self.close()
                    except Exception:
                        pass
        
        # タイマースレッドを開始
        self._close_timer = threading.Thread(target=auto_close, daemon=True)
        self._close_timer.start()
        
    def close(self):
        """ウィンドウを閉じる"""
        if self.fig is not None:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    plt.close(self.fig)
            except Exception:
                plt.close(self.fig)
            self.fig = None
