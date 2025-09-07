"""
visualization.py
純正ED法（Error Diffusion Learning Algorithm）Python実装 v0.2.0
Original C implementation by Isamu Kaneko (1999)
"""

import matplotlib.pyplot as plt
import numpy as np
import datetime
import os


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
        
    def setup_plots(self):
        """グラフの初期設定 - ed_genuine.prompt.md準拠, v0.1.3: サイズ800x600に拡大"""
        # ウィンドウサイズ設定（v0.1.3: 800x600に拡大）
        dpi = 100
        figsize = (self.window_size[0]/dpi, self.window_size[1]/dpi)
        
        self.fig, (self.ax_acc, self.ax_err) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
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
        self.ax_acc.set_xlim(1, self.max_epochs)  # 横軸を1から開始
        self.ax_acc.set_ylim(0, 1)
        self.ax_acc.grid(True, alpha=0.3)
        
        # 右側グラフ：誤差
        self.ax_err.set_title("訓練・テスト誤差", fontweight='bold')
        self.ax_err.set_xlabel("エポック数")
        self.ax_err.set_ylabel("誤差")
        self.ax_err.set_xlim(1, self.max_epochs)  # 横軸を1から開始
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
        
        plt.tight_layout()
        
        # インタラクティブモード有効化と初期描画
        plt.ion()
        plt.show(block=False)  # 非ブロッキング表示
        
        # 初期描画を強制実行
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # 非ブロッキング表示でUIを更新
        plt.pause(0.05)
    
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
        
        # 短時間の一時停止でリアルタイム表示
        plt.pause(0.01)
    
    def close(self):
        """可視化ウィンドウを閉じる"""
        if self.fig:
            plt.close(self.fig)
    
    def save_figure(self):
        """リアルタイム学習グラフを保存する"""
        if not self.save_dir or not self.fig:
            return
            
        # ディレクトリ作成
        os.makedirs(self.save_dir, exist_ok=True)
        
        # タイムスタンプ生成
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ファイル名生成
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
        
    def setup_plots(self):
        """混同行列グラフの初期設定"""
        # ウィンドウサイズ設定
        dpi = 100
        figsize = (self.window_size[0]/dpi, self.window_size[1]/dpi)
        
        try:
            self.fig, (self.ax_confusion, self.ax_accuracy) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
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
        
        self.ax_confusion.set_title("混同行列（累積）", fontweight='bold')
        self.ax_confusion.set_xlabel('予測クラス')
        self.ax_confusion.set_ylabel('実際クラス')
        
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
        
        # インタラクティブモード有効化
        try:
            plt.ion()
            plt.show(block=False)
            
            # 初期描画
            if hasattr(self.fig, 'canvas') and self.fig.canvas:
                self.fig.canvas.draw()
        except Exception as e:
            print(f"⚠️  グラフ表示エラー: {e}")
        
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
        
        # 累積混同行列更新
        self.cumulative_confusion_matrix += epoch_matrix
        self.total_samples += len(true_labels)
        
        # 正規化混同行列計算（表示用）
        normalized_matrix = self._normalize_matrix(self.cumulative_confusion_matrix)
        
        # ヒートマップ更新
        if self.im is not None:
            try:
                self.im.set_array(normalized_matrix)
                # カラースケール動的調整
                max_val = np.max(normalized_matrix) if np.max(normalized_matrix) > 0 else 1.0
                self.im.set_clim(0, max_val)
            except Exception as e:
                print(f"⚠️  ヒートマップ更新エラー: {e}")
        
        # 数値表示更新
        self._update_text_annotations(self.cumulative_confusion_matrix, normalized_matrix)
        
        # クラス別精度計算と履歴更新
        class_accuracies = self._calculate_class_accuracies(self.cumulative_confusion_matrix)
        overall_accuracy = np.trace(self.cumulative_confusion_matrix) / max(1, self.total_samples)
        
        self.class_accuracies_history.append(class_accuracies)
        self.epoch_accuracies.append(overall_accuracy)
        
        # 右側グラフ更新
        self._update_accuracy_plot()
        
        # タイトル更新（3行表示で見切れ防止）
        try:
            title_text = f"混同行列（累積）\nエポック{self.current_epoch}\n精度: {overall_accuracy:.3f}, サンプル: {self.total_samples}"
            self.ax_confusion.set_title(title_text)
        except Exception:
            pass
        
        # グラフ再描画
        try:
            if hasattr(self.fig, 'canvas') and self.fig.canvas:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
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
            plt.close(self.fig)
            
    def save_figure(self):
        """混同行列グラフを保存する"""
        if not self.save_dir or not self.fig:
            return
            
        # ディレクトリ作成
        os.makedirs(self.save_dir, exist_ok=True)
        
        # タイムスタンプ生成
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ファイル名生成
        filename = f"confusion-realtime-{timestamp}.png"
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            # グラフ保存
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"📊 リアルタイム混同行列グラフ保存完了: {filepath}")
        except Exception as e:
            print(f"❌ 混同行列グラフ保存エラー: {e}")
