#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ED-SNN ヒートマップリアルタイム表示クラス v4.2 (rainbow・正方形ブロック対応版)

新仕様:
1. 全体で8層(入力層1、隠れ層6、出力層1)までは、最大2行4列で表示
2. 9層以上の場合、8個のヒートマップを2行4列で表示し、中間の層は省略
3. 省略アルゴリズム: 
   - 上段: 入力層 + 隠れ層1-3 (4個)
   - 下段: 出力層の3つ前の隠れ層 + 出力層 (4個)
4. タイトル・パラメータを右上に統合表示
5. rainbowカラーマップ・正方形ブロック対応
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import warnings
import time
import threading

# 日本語フォント設定
def setup_japanese_font():
    """標準でインストールされている日本語フォントを設定"""
    try:
        # Linux環境での一般的な日本語フォント候補
        japanese_fonts = [
            'Noto Sans CJK JP',
            'DejaVu Sans', 
            'Liberation Sans',
            'TakaoPGothic',
            'IPAexGothic',
            'sans-serif'
        ]
        
        for font_name in japanese_fonts:
            try:
                plt.rcParams['font.family'] = font_name
                # テスト描画で確認
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.text(0.5, 0.5, 'テスト', fontsize=10)
                plt.close(fig)
                print(f"✅ 日本語フォント設定: {font_name}")
                return True
            except:
                continue
        
        # フォールバック: デフォルト設定
        plt.rcParams['font.family'] = 'sans-serif'
        print("⚠️ 日本語フォント設定: デフォルト使用")
        return False
        
    except Exception as e:
        print(f"❌ 日本語フォント設定エラー: {e}")
        return False

# 初期化時にフォント設定を実行
setup_japanese_font()


class LearningResultsTracker:
    """ED法学習結果追跡システム - ed_multi.prompt.md準拠"""
    
    def __init__(self):
        """学習結果データの初期化"""
        self.latest_results = {
            'epoch': 0,
            'sample_idx': 0,
            'true_label': -1,
            'predicted_label': -1,
            'train_accuracy': 0.0,
            'test_accuracy': 0.0,
            'train_error': 0.0,
            'test_error': 0.0,
            'learning_time': 0.0,
            'timestamp': time.time()
        }
    
    def update_learning_results(self, results_data):
        """ED法ネットワークから学習結果を更新"""
        if results_data:
            self.latest_results.update(results_data)
            self.latest_results['timestamp'] = time.time()
    
    def get_subtitle_text(self):
        """サブタイトル用テキストを生成"""
        epoch = self.latest_results['epoch']
        true_label = self.latest_results['true_label']
        predicted_label = self.latest_results['predicted_label']
        
        true_text = str(true_label) if true_label >= 0 else '-'
        pred_text = str(predicted_label) if predicted_label >= 0 else '-'
        
        return f"エポック#: {epoch}    正解クラス: {true_text}    予測クラス: {pred_text}"


class DisplayTimingController:
    """表示タイミング制御システム - ed_multi.prompt.md準拠"""
    
    def __init__(self, interval=0.3):
        """タイミング制御の初期化
        
        Args:
            interval: 更新間隔（秒）（デフォルト0.3秒でリアルタイム表示）
        """
        self.interval = interval
        self.last_update_time = 0
        
    def should_update(self):
        """指定間隔での更新判定
        
        Returns:
            bool: 更新すべきかどうか
        """
        current_time = time.time()
        if current_time - self.last_update_time >= self.interval:
            self.last_update_time = current_time
            return True
        return False
    
    def set_interval(self, interval):
        """更新間隔を変更"""
        self.interval = interval


class IntervalDisplaySystem:
    """インターバル表示システム - ED法理論準拠"""
    
    def __init__(self, visualizer, interval=0.3):
        """インターバル表示システムの初期化
        
        Args:
            visualizer: HeatmapRealtimeVisualizerインスタンス
            interval: 更新間隔（秒）（デフォルト0.3秒でリアルタイム表示）
        """
        self.visualizer = visualizer
        self.interval = interval
        self.running = False
        self.thread = None
        self.last_activity_data = None
        
    def start_interval_updates(self):
        """定期更新開始（メインスレッド専用モード）"""
        if not self.running:
            self.running = True
            # threadingを使用せず、メインスレッド専用モードで動作
            print(f"🎯 インターバル表示システム開始: メインスレッド専用モード")
            print(f"🎯 更新はコールバックベースで実行されます")
            
    def stop_interval_updates(self):
        """定期更新停止（メインスレッド専用モード）"""
        if self.running:
            self.running = False
            print(f"🎯 インターバル表示システム停止")
        
    def set_activity_data(self, layer_activations):
        """最新の活動データを設定 - ed_multi.prompt.md準拠"""
        # データ設定前後の状態を記録
        old_data = self.last_activity_data
        self.last_activity_data = layer_activations
        
        # コールバックベース更新：メインスレッドで安全に実行
        if self.running:
            self.update_display_callback()
    
    def update_display_callback(self):
        """コールバックベース表示更新（メインスレッド専用） - ed_multi.prompt.md準拠"""
        try:
            # 可視化システムが初期化されており、データがある場合のみ更新
            is_initialized = self.visualizer.is_initialized
            has_data = self.last_activity_data is not None
            has_fig = self.visualizer.fig is not None
            fig_exists = plt.fignum_exists(self.visualizer.fig.number) if has_fig else False
            
            if (is_initialized and has_data and has_fig and fig_exists):
                # メインスレッド専用更新：シンプルな描画更新のみ
                self.visualizer.fig.canvas.draw_idle()
                self.visualizer.fig.canvas.flush_events()
            
                
        except Exception as e:
            print(f"⚠️ コールバック更新エラー: {e}")
        
    def _update_loop(self):
        """コールバック方式に移行したため非使用 - ed_multi.prompt.md準拠"""
        # この方法は非推奨：matplotlib GUIスレッド問題を回避するため
        # 今はupdate_display_callbackのコールバック方式を使用
        print(f"⚠️ [DEBUG] 古い_update_loopは使用されません（コールバック方式に移行済み）")
        return
        
        # 以下は無効化されたコード（参考用）:
        update_count = 0
        while self.running:
            try:
                update_count += 1
                
                # 可視化システムが初期化されており、データがある場合のみ更新
                is_initialized = self.visualizer.is_initialized
                has_data = self.last_activity_data is not None
                has_fig = self.visualizer.fig is not None
                fig_exists = plt.fignum_exists(self.visualizer.fig.number) if has_fig else False
                
                if (is_initialized and has_data and has_fig and fig_exists):
                    # メインスレッド専用更新：threadingを使わずに直接更新
                    try:
                        # シンプルなdraw()とflush_events()のみ使用
                        self.visualizer.fig.canvas.draw_idle()
                        self.visualizer.fig.canvas.flush_events()
                    except Exception as e:
                        print(f"⚠️ メインスレッド更新エラー: {e}")
                else:
                    pass  # 更新条件未満のためスキップ
                    
                # 0.3秒間隔でリアルタイム感を向上
                time.sleep(0.3)
            except Exception as e:
                print(f"⚠️ インターバル更新エラー: {e}")
                break
        
        print("🎯 インターバル更新ループ終了")


class HeatmapRealtimeVisualizer:
    """ED-SNN ヒートマップリアルタイム表示クラス v4.2 (rainbow・正方形ブロック対応版)"""
    
    def __init__(self, 
                 layer_shapes: List[Tuple[int, int]], 
                 show_parameters: bool = True,
                 update_interval: float = 0.1,
                 colormap: str = 'rainbow',
                 ed_params: Optional[Dict] = None,
                 exec_params: Optional[Dict] = None):
        """
        初期化
        
        Args:
            layer_shapes: 各層の形状 [(height, width), ...]
            show_parameters: パラメータ表示するかどうか
            update_interval: 更新間隔（秒）
            colormap: カラーマップ
            ed_params: ED法アルゴリズムパラメータ
            exec_params: 実行時設定パラメータ
        """
        self.layer_shapes = layer_shapes
        self.show_parameters = show_parameters
        self.update_interval = update_interval
        self.colormap = colormap
        
        # 状態管理
        self.fig = None
        self.axes = {}  # {layer_index: ax}
        self.title_ax = None
        self.param_ax_lif = None
        self.param_ax_ed = None
        self.heatmap_objects = {}  # ヒートマップオブジェクト
        self.colorbar_objects = {}  # カラーバーオブジェクト
        self.is_initialized = False
        
        # パラメータ保存
        self.lif_params = {}  # 現在は未使用（下位互換性のため保持）
        self.ed_params = ed_params or {}
        self.exec_params = exec_params or {}
        
        # 学習結果データ取得システム（フェーズ1）
        self.learning_results_tracker = LearningResultsTracker()
        
        # 表示タイミング制御システム（フェーズ2用）
        self.timing_controller = DisplayTimingController(interval=update_interval)
        
        # インターバル表示システム（フェーズ3）
        self.interval_system = IntervalDisplaySystem(self, interval=0.3)
        self.training_info = {}
    
    def _calculate_layout(self, num_layers: int) -> Dict[str, Any]:
        """
        新仕様: 最大2行4列、8層超過時は省略アルゴリズム適用
        
        Args:
            num_layers: 総層数
            
        Returns:
            レイアウト情報の辞書
        """
        # 固定: 最大2行4列のヒートマップ表示
        max_heatmaps = 8
        
        if num_layers <= max_heatmaps:
            # 8層以下の場合：全層を表示
            selected_layers = list(range(num_layers))
            layout_type = f"full_{num_layers}_layers"
            
            # 実際の配置計算（8層以下でも2行4列に収める）
            if num_layers <= 4:
                actual_rows = 1
                actual_cols = num_layers
            else:
                actual_rows = 2
                actual_cols = 4
                
        else:
            # 9層以上の場合：省略アルゴリズム適用
            # 上段: 入力層(0) + 隠れ層1-3 (インデックス1,2,3) = 4個
            upper_layers = [0, 1, 2, 3]
            
            # 下段: 出力層の3つ前から出力層まで (4個)
            output_layer_idx = num_layers - 1  # 出力層のインデックス
            
            # 出力層の3つ前の隠れ層から開始
            lower_start = output_layer_idx - 3
            lower_layers = list(range(lower_start, output_layer_idx + 1))
            
            selected_layers = upper_layers + lower_layers
            layout_type = f"abbreviated_{num_layers}_to_8"
            actual_rows = 2
            actual_cols = 4
        
        return {
            'rows': actual_rows,
            'cols': actual_cols,
            'selected_layers': selected_layers,
            'layout_type': layout_type,
            'total_original_layers': num_layers,
            'is_abbreviated': num_layers > max_heatmaps,
            'max_display_count': len(selected_layers)
        }
    
    def _get_layer_label(self, layer_idx: int, total_layers: int, is_abbreviated: bool) -> str:
        """層のラベルを生成"""
        if layer_idx == 0:
            return "入力層"
        elif layer_idx == total_layers - 1:
            return "出力層"
        else:
            return f"隠れ層{layer_idx}"
    
    def setup_visualization(self, layer_activations: List[np.ndarray]):
        """
        可視化ウィンドウの初期化（新仕様レイアウト）
        
        Args:
            layer_activations: 各層の活性化データ
        """
        if self.is_initialized:
            return
        
        try:
            # レイアウト計算
            num_layers = len(layer_activations)
            layout = self._calculate_layout(num_layers)
            
            # 図のサイズ計算（新仕様: 全体統合レイアウト）
            fig_width = 10.67  # 固定幅の2/3（ヒートマップ + 右上パラメータエリア）
            fig_height = 6.67  # 固定高さの2/3（2行4列ヒートマップ + 上部余白）
            
            # 図作成（新統合レイアウト）
            self.fig = plt.figure(figsize=(fig_width, fig_height))
            
            # メインレイアウト: 全体を1つのGridSpecで管理
            # 上部: タイトル・パラメータエリア (高さ比2.5 - 位置を上に移動)
            # 下部: ヒートマップエリア (高さ比10)  
            gs_main = gridspec.GridSpec(2, 1, figure=self.fig, height_ratios=[2.5, 10], hspace=0.1)
            
            # 上部エリア: タイトル・パラメータを右上に配置
            gs_upper = gridspec.GridSpecFromSubplotSpec(
                1, 2, gs_main[0, 0], width_ratios=[3, 4.032], wspace=0.1
            )
            
            # 上部左: メインタイトル専用エリア
            self.title_ax = self.fig.add_subplot(gs_upper[0, 0])
            self.title_ax.axis('off')  # 軸非表示
            
            # 上部右: パラメータ表示エリア（左上原点座標系で指定）
            if self.show_parameters:
                # 左上原点座標系: [left_from_left, top_from_top, width, height]
                # 共通設定：幅を統一
                box_width = 0.52
                
                # ED法パラメータ設定ボックス（上段）
                left, top, width, height = 0.4, 0.01, box_width, 0.12
                self.param_ax_lif = self.fig.add_axes([left, 1-top-height, width, height])
                
                # 実行パラメータ設定ボックス（下段）
                left, top, width, height = 0.4, 0.14, box_width, 0.12
                self.param_ax_ed = self.fig.add_axes([left, 1-top-height, width, height])
                
                self.param_ax_lif.axis('off')
                self.param_ax_ed.axis('off')
            else:
                self.param_ax_lif = None
                self.param_ax_ed = None
            
            # 下部エリア: ヒートマップ専用（2行4列）
            gs_heatmap = gridspec.GridSpecFromSubplotSpec(
                layout['rows'], layout['cols'], 
                gs_main[1, 0],
                hspace=0.3, wspace=0.3
            )
            
            # ヒートマップ用のサブプロットを作成
            self.axes = {}
            self.heatmap_objects = {}
            self.colorbar_objects = {}
            
            for i, layer_idx in enumerate(layout['selected_layers']):
                row = i // layout['cols']
                col = i % layout['cols']
                
                if row < layout['rows'] and col < layout['cols']:
                    ax = self.fig.add_subplot(gs_heatmap[row, col])
                    self.axes[layer_idx] = ax
                    self.heatmap_objects[layer_idx] = None
                    self.colorbar_objects[layer_idx] = None
                    
                    # タイトル設定
                    label = self._get_layer_label(layer_idx, num_layers, layout['is_abbreviated'])
                    shape_info = f"({self.layer_shapes[layer_idx][0]}×{self.layer_shapes[layer_idx][1]})"
                    ax.set_title(f"{label}\n{shape_info}", fontsize=10, pad=5)
            
            self.is_initialized = True
            
            # ウィンドウの準備（インタラクティブモードONのみ、表示は学習開始時）
            plt.ion()  # インタラクティブモードON
            
        except Exception as e:
            print(f"❌ ヒートマップ初期化エラー: {e}")
            raise
    
    def update_parameters(self, lif_params: Optional[Dict] = None, 
                         ed_params: Optional[Dict] = None,
                         training_info: Optional[Dict] = None):
        """パラメータ情報を更新"""
        if lif_params:
            self.lif_params.update(lif_params)
        if ed_params:
            self.ed_params.update(ed_params)
        if training_info:
            self.training_info.update(training_info)
    
    def update_learning_results(self, results_data: Dict):
        """学習結果データを更新（フェーズ1）
        
        Args:
            results_data: 学習結果データ
                - epoch: エポック番号
                - sample_idx: サンプルインデックス
                - true_label: 正解ラベル
                - predicted_label: 予測ラベル
                - train_accuracy: 訓練精度
                - test_accuracy: テスト精度
                - train_error: 訓練誤差
                - test_error: テスト誤差
        """
        self.learning_results_tracker.update_learning_results(results_data)
        
        # サブタイトル更新は update_display でのヒートマップ更新時にのみ実行
        # (同期のため、ここでは更新しない)
    
    def _update_subtitle(self):
        """サブタイトルを更新 - ed_multi.prompt.md準拠"""
        
        if hasattr(self, 'title_ax') and self.title_ax:
            subtitle_text = self.learning_results_tracker.get_subtitle_text()
            
            # タイトルエリアをクリア
            self.title_ax.clear()
            self.title_ax.axis('off')
            
            # メインタイトル（フォントサイズ: 14→18pt）
            self.title_ax.text(0.5, 0.7, "ED-Genuine ヒートマップリアルタイム表示", 
                              transform=self.title_ax.transAxes,
                              fontsize=18, fontweight='bold', ha='center')
            
            # サブタイトル（フォントサイズ: 11→15pt、色分け対応）
            subtitle_color = self._get_subtitle_color()
            self.title_ax.text(0.5, 0.3, subtitle_text, 
                              transform=self.title_ax.transAxes,
                              fontsize=15, ha='center', color=subtitle_color)
        
    
    def _get_subtitle_color(self):
        """サブタイトルの色を決定: 正解=予測なら青、不一致なら赤"""
        try:
            # 学習結果データから正解・予測クラスを取得
            data = self.learning_results_tracker.latest_results
            if data:
                true_label = data.get('true_label', -1)
                predicted_label = data.get('predicted_label', -1)
                
                # 両方が有効な値（-1以外）で一致するかチェック
                if true_label != -1 and predicted_label != -1:
                    if true_label == predicted_label:
                        return 'blue'  # 正解時は青色
                    else:
                        return 'red'   # 不正解時は赤色
            
            # 初期状態や無効値の場合は黒色
            return 'black'
        except Exception as e:
            return 'black'
    
    def start_interval_display(self):
        """インターバル表示システム開始（フェーズ3）"""
        if self.interval_system is None:
            self.interval_system = IntervalDisplaySystem(self, interval=0.3)
        self.interval_system.start_interval_updates()
    
    def stop_interval_display(self):
        """インターバル表示システム停止（フェーズ3）"""
        if self.interval_system:
            self.interval_system.stop_interval_updates()
    
    def force_update_display(self):
        """強制的に表示を更新（リアルタイム表示システム - ed_multi.prompt.md準拠）"""
        
        if self.is_initialized and self.interval_system and self.interval_system.last_activity_data:
            try:
                # 1. 既存のヒートマップとカラーバーを削除（remove()使用）
                self._clear_all_heatmaps()
                
                # 2. パラメータボックスのみ更新（サブタイトルはupdate_displayで同期更新）
                self._draw_parameter_boxes()
                
                # 3. 新しいヒートマップデータで描画
                layer_activations = self.interval_system.last_activity_data
                layout = self._calculate_layout(len(layer_activations))
                
                for layer_idx in self.axes.keys():
                    if layer_idx < len(layer_activations):
                        ax = self.axes[layer_idx]
                        data = layer_activations[layer_idx]
                        
                        # データを2Dに変換
                        if data.ndim == 1:
                            height, width = self.layer_shapes[layer_idx]
                            if len(data) == height * width:
                                data_2d = data.reshape(height, width)
                            else:
                                sqrt_size = int(np.sqrt(len(data)))
                                if sqrt_size * sqrt_size < len(data):
                                    sqrt_size += 1
                                padded_data = np.zeros(sqrt_size * sqrt_size)
                                padded_data[:len(data)] = data
                                data_2d = padded_data.reshape(sqrt_size, sqrt_size)
                        elif data.ndim == 2:
                            data_2d = data
                        else:
                            data_2d = data.reshape(data.shape[0], -1)
                        
                        # 新しいヒートマップ描画
                        import matplotlib.cm as cm
                        cmap = cm.get_cmap(self.colormap)
                        cmap.set_bad(color='#404040')
                        
                        im = ax.imshow(data_2d, cmap=cmap, aspect='equal', interpolation='nearest')
                        self.heatmap_objects[layer_idx] = im
                        
                        # タイトル設定
                        label = self._get_layer_label(layer_idx, len(layer_activations), layout['is_abbreviated'])
                        shape_info = f"({data_2d.shape[0]}×{data_2d.shape[1]})"
                        ax.set_title(f"{label}\n{shape_info}", fontsize=10, pad=5)
                        
                        ax.set_xticks([])
                        ax.set_yticks([])
                        
                        # カラーバー作成
                        try:
                            cb = plt.colorbar(im, ax=ax, shrink=0.6)
                            self.colorbar_objects[layer_idx] = cb
                        except Exception:
                            self.colorbar_objects[layer_idx] = None
                
                # 4. 描画を確定（pause()使用でリアルタイム表示）
                if self.fig:
                    plt.figure(self.fig.number)
                    plt.draw()
                    plt.pause(0.01)  # pause()でリアルタイム表示
                    
            except Exception as e:
                print(f"⚠️ 強制更新エラー: {e}")
    
    
    def _draw_parameter_boxes(self):
        """パラメータボックスを右上に描画"""
        if not self.show_parameters:
            return
        
        # ED法アルゴリズムパラメータボックス（上段）
        if self.param_ax_lif is not None:
            self.param_ax_lif.clear()
            self.param_ax_lif.axis('off')
            
            # ED法アルゴリズムパラメータテキスト（左右2列配置）
            ed_algo_text = "ED法パラメータ設定\n"
            # 左列: learning_rate, amine, diffusion
            ed_algo_text += f"学習率(alpha): {self.ed_params.get('learning_rate', '0.8')}         "
            ed_algo_text += f"重み初期値1: {self.ed_params.get('weight1', '1.0')}\n"
            ed_algo_text += f"初期アミン濃度(beta): {self.ed_params.get('amine', '0.3')}     "
            ed_algo_text += f"重み初期値2: {self.ed_params.get('weight2', '1.0')}\n"
            ed_algo_text += f"アミン拡散係数(u1): {self.ed_params.get('diffusion', '1.0')}   "
            ed_algo_text += f"シグモイド閾値(u0): {self.ed_params.get('sigmoid', '0.4')}"
            
            self.param_ax_lif.text(0.5, 0.5, ed_algo_text, transform=self.param_ax_lif.transAxes,
                                  fontsize=9, verticalalignment='center',
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # 実行時設定パラメータボックス（下段）
        if self.param_ax_ed is not None:
            self.param_ax_ed.clear()
            self.param_ax_ed.axis('off')
            
            # 実行時設定パラメータテキスト（左右2列配置・絞り込み版）
            exec_text = "実行パラメータ設定\n"
            # 左列: 基本設定パラメータ
            exec_text += f"訓練データ数: {self.exec_params.get('train_samples', '32')}              "
            exec_text += f"エポック数: {self.exec_params.get('epochs', '5')}\n"
            exec_text += f"テストデータ数: {self.exec_params.get('test_samples', '100')}           "
            exec_text += f"隠れ層構造: {self.exec_params.get('hidden', '128')}\n"
            exec_text += f"ミニバッチサイズ: {self.exec_params.get('batch_size', '32')}           "
            exec_text += f"ランダムシード: {self.exec_params.get('seed', 'Random')}\n"
            
            # Fashion-MNISTオプション表示（使用時のみ）
            if self.exec_params.get('fashion', False):
                exec_text += "Fashion-MNIST: 有効"
            
            self.param_ax_ed.text(0.5, 0.5, exec_text, transform=self.param_ax_ed.transAxes,
                                 fontsize=9, verticalalignment='center',
                                 bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    def update_display(self, layer_activations: List[np.ndarray], 
                      epoch: int, sample_idx: int, 
                      true_label: int = -1, predicted_label: int = -1):
        """
        ヒートマップ表示を更新（学習結果統合版） - ed_multi.prompt.md準拠
        
        Args:
            layer_activations: 各層の活性化データ
            epoch: 現在のエポック
            sample_idx: 現在のサンプルインデックス
            true_label: 正解ラベル
            predicted_label: 予測ラベル
        """
        if not self.is_initialized:
            self.setup_visualization(layer_activations)
        
        # 学習結果データを更新（フェーズ1）
        results_data = {
            'epoch': epoch,
            'sample_idx': sample_idx,
            'true_label': true_label,
            'predicted_label': predicted_label
        }
        self.update_learning_results(results_data)
        
        # インターバル表示システムに活動データを設定（フェーズ3）
        if self.interval_system:
            self.interval_system.set_activity_data(layer_activations)
        
        # ☆ 表示タイミング制御を最初に判定（同期のため）
        should_update = self.timing_controller.should_update()
        if not should_update:
            return  # 指定間隔に達していない場合は両方スキップ
        
        # ☆ タイミング制御OKの場合のみ、サブタイトルとヒートマップを同期更新
        self._update_subtitle()
        
        try:
            # 1. 既存のヒートマップとカラーバーを全て削除
            self._clear_all_heatmaps()
            
            # 2. パラメータボックス描画（サブタイトルは上で更新済み）
            self._draw_parameter_boxes()
            
            layout = self._calculate_layout(len(layer_activations))
            
            for layer_idx in self.axes.keys():
                if layer_idx < len(layer_activations):
                    ax = self.axes[layer_idx]
                    data = layer_activations[layer_idx]
                    
                    # データを2Dに変換（正方形ブロック用）
                    if data.ndim == 1:
                        # 1次元データを正方形に近い形状に変換
                        height, width = self.layer_shapes[layer_idx]
                        if len(data) == height * width:
                            data_2d = data.reshape(height, width)
                        else:
                            # サイズが合わない場合は正方形に近い形状で表示
                            sqrt_size = int(np.sqrt(len(data)))
                            if sqrt_size * sqrt_size < len(data):
                                sqrt_size += 1
                            # パディングして正方形に
                            padded_data = np.zeros(sqrt_size * sqrt_size)
                            padded_data[:len(data)] = data
                            data_2d = padded_data.reshape(sqrt_size, sqrt_size)
                    elif data.ndim == 2:
                        data_2d = data
                    else:
                        # 3次元以上の場合は最初の2次元を取得
                        data_2d = data.reshape(data.shape[0], -1)
                    
                    # 新しいヒートマップ描画（正方形アスペクト比）
                    # カラーマップを取得し、NaN値（非活動セル）の色を濃い灰色に設定
                    import matplotlib.cm as cm
                    cmap = cm.get_cmap(self.colormap)
                    cmap.set_bad(color='#404040')  # 濃い灰色 (RGB: 64, 64, 64)
                    
                    im = ax.imshow(data_2d, cmap=cmap, aspect='equal', interpolation='nearest')
                    self.heatmap_objects[layer_idx] = im
                    
                    # タイトル設定
                    label = self._get_layer_label(layer_idx, len(layer_activations), layout['is_abbreviated'])
                    shape_info = f"({data_2d.shape[0]}×{data_2d.shape[1]})"
                    ax.set_title(f"{label}\n{shape_info}", fontsize=10, pad=5)
                    
                    # 軸ラベル非表示
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # カラーバー作成（エラー回避）
                    try:
                        cb = plt.colorbar(im, ax=ax, shrink=0.6)
                        self.colorbar_objects[layer_idx] = cb
                    except Exception as cb_error:
                        self.colorbar_objects[layer_idx] = None
            
            # 5. 描画完了後に一時停止（カラーバー増殖回避）
            plt.draw()
            plt.pause(0.1)  # 0.1秒停止
            
        except Exception as e:
            print(f"❌ ヒートマップ更新エラー: {e}")
    
    def _clear_all_heatmaps(self):
        """全てのヒートマップとカラーバーを削除（remove()使用 - ed_multi.prompt.md準拠）"""
        try:
            # カラーバーを削除（remove()使用）
            for layer_idx, cb in list(self.colorbar_objects.items()):
                if cb is not None:
                    try:
                        cb.remove()
                    except Exception:
                        pass
            
            # ヒートマップオブジェクトを削除（remove()使用）
            for layer_idx, im in list(self.heatmap_objects.items()):
                if im is not None:
                    try:
                        im.remove()
                    except Exception:
                        pass
            
            # 各軸をクリア
            for layer_idx, ax in self.axes.items():
                if ax is not None:
                    ax.clear()
            
            # オブジェクト辞書をリセット
            for layer_idx in self.heatmap_objects.keys():
                self.heatmap_objects[layer_idx] = None
                self.colorbar_objects[layer_idx] = None
                
        except Exception as e:
            print(f"⚠️ ヒートマップクリアエラー: {e}")
    
    def close(self):
        """リソースの解放"""
        # インターバル表示システムを停止（フェーズ3）
        self.stop_interval_display()
        
        if hasattr(self, 'fig') and self.fig is not None:
            # カラーバーを削除
            for layer_idx, cb in self.colorbar_objects.items():
                if cb is not None:
                    try:
                        cb.remove()
                    except:
                        pass
            
            # すべての軸をクリア
            for ax in self.axes.values():
                if ax is not None:
                    ax.clear()
            
            # パラメータ軸をクリア
            if hasattr(self, 'param_ax_lif') and self.param_ax_lif is not None:
                self.param_ax_lif.clear()
            if hasattr(self, 'param_ax_ed') and self.param_ax_ed is not None:
                self.param_ax_ed.clear()
            if hasattr(self, 'title_ax') and self.title_ax is not None:
                self.title_ax.clear()
            
            plt.close(self.fig)
            self.fig = None
            self.is_initialized = False
            print("✅ ヒートマップ可視化ウィンドウを閉じました")
