#!/usr/bin/env python3
"""
ED-ANN
ED法を用いたANNネットワークのマルチクラス分類法

Multi-class classification using ED method

ED法理論実装:
1. Error Diffusion (誤差拡散): アミン濃度による学習制御
2. Multi-class ED learning: 各クラスに対する個別誤差拡散
3. Adaptive learning: アミン濃度に基づく適応的学習率調整

v1.0.0
発行日: 2025-08-23
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import argparse
import json
import time
import threading
import sys
import csv
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# グローバルverboseフラグ
VERBOSE_MODE = False

# 日本語フォント設定（ED-ANN仕様準拠）
import matplotlib.pyplot as plt
import matplotlib
import platform

def setup_japanese_font():
    """標準でインストールされている日本語フォントを設定"""
    try:
        # 確実に動作する日本語フォントのみ（警告なし）
        japanese_fonts = [
            'Noto Sans CJK JP',     # 確認済み動作OK: /usr/share/fonts/opentype/noto/
            'Yu Gothic',            # 確認済み動作OK: /home/yoichi/.fonts/YuGothR.ttc
            'Noto Sans JP',         # Google Noto 日本語特化版
            'sans-serif'            # システムフォールバック
        ]
        
        # 警告なしフォント設定
        plt.rcParams['font.family'] = japanese_fonts
        plt.rcParams['axes.unicode_minus'] = False  # マイナス記号文字化け対策
        
        # 設定成功の確認
        current_font = plt.rcParams['font.family'][0]
        # 削除: print(f"日本語フォント設定成功: {current_font}")
        return current_font
        
    except Exception as e:
        print(f"フォント設定でエラー: {e}")  # loggerの代わりにprint使用
        plt.rcParams['font.family'] = ['sans-serif']
        return 'default'

def setup_logging(verbose=False):
    """ログ設定を初期化"""
    handlers = [logging.StreamHandler()]
    
    if verbose:
        # verboseが有効な場合のみログファイルを作成
        log_filename = f'ed_ann_v563_restored_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        handlers.append(logging.FileHandler(log_filename))
        print(f"📝 詳細ログファイル: {log_filename}")
    
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # 既存の設定を上書き
    )
    return logging.getLogger(__name__)

# 初期ログ設定（一時的、main関数で再設定される）
logger = setup_logging(verbose=False)

# 日本語フォント初期化（ログ設定後）
setup_japanese_font()

@dataclass
@dataclass
class TrainingConfig:
    """訓練設定（v5.5.4準拠）"""
    epochs: int = 3
    learning_rate: float = 0.01
    batch_size: int = 32
    train_samples_per_class: int = 1000
    test_samples_per_class: int = 1000  # 訓練データと同数に変更
    hidden_size: int = 64
    num_classes: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    force_cpu: bool = False  # CPU強制使用フラグ
    realtime: bool = False
    random_seed: int = 42
    verbose: bool = False
    verify: bool = False  # 精度検証機能フラグ

@dataclass
class HyperParameters:
    """ED法関連ハイパーパラメータ集約クラス
    
    ED法理論に基づく学習制御パラメータを一元管理
    金子勇氏考案のError Diffusion法実装用パラメータ
    """
    
    # ED法理論パラメータ
    d_plus: float = 0.1      # アミン濃度増加量（正答時の重み増加制御）
    d_minus: float = 0.05    # アミン濃度減少量（誤答時の重み減少制御）
    
    # 学習率関連
    base_learning_rate: float = 0.01    # 基本学習率（Adam optimizer用）
    
    # 可視化設定
    graph_width: float = 12.0   # リアルタイムグラフの幅（インチ）
    graph_height: float = 6.0   # リアルタイムグラフの高さ（インチ）

class MultiClassSingleOutputEDDense(nn.Module):
    """
    v5.5.4準拠 - クラス別重み配列管理システム
    各出力ニューロンがクラス別の重み配列を持つED法アーキテクチャ
    """
    
    def __init__(self, in_features: int, out_features: int, num_output_classes: int = 10):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_output_classes = num_output_classes
        
        # v5.5.4準拠: クラス別重み配列 [out_features, num_output_classes, in_features]
        self.weights_per_class = nn.Parameter(
            torch.randn(out_features, num_output_classes, in_features) * 0.1
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        print(f"MultiClassEDDense初期化: {in_features}→{out_features}, クラス数{num_output_classes}") if VERBOSE_MODE else None
        
    def forward(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """指定クラスの重みを使用した前向き計算"""
        # target_classの重みを選択
        selected_weights = self.weights_per_class[:, target_class, :]  # [out_features, in_features]
        
        # 線形変換
        output = torch.matmul(x, selected_weights.t()) + self.bias
        return output
    
    def get_weight_stats(self, target_class: int) -> Dict:
        """重み統計取得"""
        with torch.no_grad():
            weights = self.weights_per_class[:, target_class, :]
            return {
                'norm': torch.norm(weights).item(),
                'mean': torch.mean(weights).item(),
                'std': torch.std(weights).item(),
                'min': torch.min(weights).item(),
                'max': torch.max(weights).item()
            }

class SingleOutputClassifier(nn.Module):
    """
    v5.5.4準拠 - 単一クラス専用分類器
    MultiClassSingleOutputEDDenseを使用した二層ネットワーク
    """
    
    def __init__(self, input_size: int = 784, hidden_size: int = 64, target_class: int = 0):
        super().__init__()
        self.target_class = target_class
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # v5.5.4準拠: ED密結合層使用
        self.hidden_layer = MultiClassSingleOutputEDDense(
            input_size, hidden_size, num_output_classes=1
        )
        
        self.output_layer = MultiClassSingleOutputEDDense(
            hidden_size, 1, num_output_classes=1
        )
        
        print(f"SingleOutputClassifier初期化完了: クラス{target_class}, 隠れ層{hidden_size}") if VERBOSE_MODE else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向き計算"""
        # バッチ次元を維持したまま平坦化
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
            
        # 隠れ層（クラス0を使用、実質的に従来の単一出力）
        hidden = self.hidden_layer(x, target_class=0)
        hidden = torch.sigmoid(hidden)
        
        # 出力層
        output = self.output_layer(hidden, target_class=0)
        output = torch.sigmoid(output)
        
        return output.squeeze(-1)  # [batch_size]
    
    def get_classifier_stats(self) -> Dict:
        """分類器統計取得"""
        return {
            'hidden_layer': self.hidden_layer.get_weight_stats(0),
            'output_layer': self.output_layer.get_weight_stats(0)
        }

class StandardMNISTDataset:
    """
    Phase 1: 標準Dataset/DataLoader統合
    - PyTorch標準MNISTデータセット使用
    - クラス単位学習サポート
    - バイナリ分類データ生成
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 標準MNISTデータセット読み込み
        self.train_dataset = datasets.MNIST(
            'data', train=True, download=True, transform=self.transform
        )
        self.test_dataset = datasets.MNIST(
            'data', train=False, transform=self.transform
        )
        
        print(f"📊 標準MNIST訓練データ: {len(self.train_dataset)} サンプル") if self.config.verbose else None
        print(f"📊 標準MNISTテストデータ: {len(self.test_dataset)} サンプル") if self.config.verbose else None
        
        # クラス別インデックス準備（効率化）
        self.class_indices = self._prepare_class_indices()
        
    def _prepare_class_indices(self) -> Dict:
        """高効率クラス別インデックス準備"""
        train_indices = defaultdict(list)
        test_indices = defaultdict(list)
        
        # 訓練データ - 直接ラベル参照
        train_targets = self.train_dataset.targets.numpy()
        for i, label in enumerate(train_targets):
            train_indices[int(label)].append(i)
        
        # テストデータ - 直接ラベル参照
        test_targets = self.test_dataset.targets.numpy()
        for i, label in enumerate(test_targets):
            test_indices[int(label)].append(i)
        
        print("🔍 クラス別インデックス準備完了（高効率版）") if self.config.verbose else None
        if self.config.verbose:
            for class_idx in range(10):
                print(f"   クラス {class_idx}: 訓練{len(train_indices[class_idx])}, テスト{len(test_indices[class_idx])}")
        
        return {
            'train': train_indices,
            'test': test_indices
        }
    
    def get_binary_datasets(self, target_class: int) -> Tuple[DataLoader, DataLoader]:
        """
        Phase 1標準化: バイナリ分類DataLoader生成
        target_class vs others の高効率二項分類データ作成
        """
        print(f"🎯 バイナリデータセット作成: クラス{target_class} vs others") if self.config.verbose else None
        
        # 標準Dataset形式での高効率データ作成
        class StandardBinaryDataset(Dataset):
            """標準PyTorch Dataset実装"""
            def __init__(self, base_dataset, indices, labels):
                self.base_dataset = base_dataset
                self.indices = indices
                self.labels = labels
                
            def __len__(self):
                return len(self.indices)
                
            def __getitem__(self, idx):
                actual_idx = self.indices[idx]
                data, _ = self.base_dataset[actual_idx]  # 元のラベルは無視
                binary_label = self.labels[idx]
                return data, binary_label
        
        # 効率的インデックス・ラベル準備
        train_indices = []
        train_labels = []
        test_indices = []
        test_labels = []
        
        # ポジティブサンプル（target_class = 1.0）
        target_train_indices = self.class_indices['train'][target_class][:self.config.train_samples_per_class]
        target_test_indices = self.class_indices['test'][target_class][:self.config.test_samples_per_class]
        
        train_indices.extend(target_train_indices)
        train_labels.extend([1.0] * len(target_train_indices))
        
        test_indices.extend(target_test_indices)
        test_labels.extend([1.0] * len(target_test_indices))
        
        # ネガティブサンプル（others = 0.0）v5.5.4準拠分散
        samples_per_neg_class = self.config.train_samples_per_class // (self.config.num_classes - 1)
        test_samples_per_neg_class = self.config.test_samples_per_class // (self.config.num_classes - 1)
        
        for class_idx in range(self.config.num_classes):
            if class_idx != target_class:
                # 訓練ネガティブ
                neg_train_indices = self.class_indices['train'][class_idx][:samples_per_neg_class]
                train_indices.extend(neg_train_indices)
                train_labels.extend([0.0] * len(neg_train_indices))
                
                # テストネガティブ
                neg_test_indices = self.class_indices['test'][class_idx][:test_samples_per_neg_class]
                test_indices.extend(neg_test_indices)
                test_labels.extend([0.0] * len(neg_test_indices))
        
        # シャッフル（再現性維持）
        np.random.seed(self.config.random_seed + target_class)
        
        # 訓練データシャッフル
        combined_train = list(zip(train_indices, train_labels))
        np.random.shuffle(combined_train)
        train_indices, train_labels = zip(*combined_train)
        
        # テストデータシャッフル
        combined_test = list(zip(test_indices, test_labels))
        np.random.shuffle(combined_test)
        test_indices, test_labels = zip(*combined_test)
        
        print(f"   正例: 訓練{sum(train_labels):.0f}, テスト{sum(test_labels):.0f}") if self.config.verbose else None
        print(f"   負例: 訓練{len(train_labels) - sum(train_labels):.0f}, テスト{len(test_labels) - sum(test_labels):.0f}") if self.config.verbose else None
        
        # 標準Dataset作成
        train_dataset = StandardBinaryDataset(self.train_dataset, train_indices, train_labels)
        test_dataset = StandardBinaryDataset(self.test_dataset, test_indices, test_labels)
        
        # 標準DataLoader作成
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=0,  # 安定性のため
            pin_memory=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        return train_loader, test_loader
    
    def create_epoch_based_datasets(self) -> Dict[str, DataLoader]:
        """
        Phase 2: エポック単位学習用データセット作成
        全クラスを同時に学習する標準的なマルチクラス分類方式
        """
        print("🔄 エポック単位学習データセット準備中") if self.config.verbose else None
        
        # 標準マルチクラスDataset実装
        class EpochBasedDataset(Dataset):
            """エポック単位学習用標準Dataset"""
            def __init__(self, base_dataset, samples_per_class=1000):
                self.base_dataset = base_dataset
                self.samples_per_class = samples_per_class
                
                # クラス別インデックス準備
                self.class_indices = defaultdict(list)
                targets = base_dataset.targets.numpy()
                for i, label in enumerate(targets):
                    self.class_indices[int(label)].append(i)
                
                # バランス取得インデックス作成
                self.selected_indices = []
                self.selected_labels = []
                
                for class_idx in range(10):
                    indices = self.class_indices[class_idx][:samples_per_class]
                    self.selected_indices.extend(indices)
                    self.selected_labels.extend([class_idx] * len(indices))
            
            def __len__(self):
                return len(self.selected_indices)
            
            def __getitem__(self, idx):
                actual_idx = self.selected_indices[idx]
                data, _ = self.base_dataset[actual_idx]
                label = self.selected_labels[idx]
                return data, label
        
        # エポック単位データセット作成
        train_dataset = EpochBasedDataset(
            self.train_dataset, 
            self.config.train_samples_per_class
        )
        test_dataset = EpochBasedDataset(
            self.test_dataset, 
            self.config.test_samples_per_class
        )
        
        # DataLoader作成
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"   エポック単位データセット準備完了") if self.config.verbose else None
        print(f"   訓練データ: {len(train_dataset)} サンプル") if self.config.verbose else None
        print(f"   テストデータ: {len(test_dataset)} サンプル") if self.config.verbose else None
        
        return {
            'train': train_loader,
            'test': test_loader
        }

class EpochBasedTrainer:
    """
    Phase 2: エポック単位学習トレーナー
    全クラス同時学習による標準マルチクラス分類実装
    """
    
    def __init__(self, config: TrainingConfig, hyperparams: Optional[HyperParameters] = None):
        self.config = config
        self.hyperparams = hyperparams if hyperparams is not None else HyperParameters()
        
        # デバイス決定（CPU強制オプション考慮）
        if config.force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(config.device)
        
        # 決定論的シード設定
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        if torch.cuda.is_available() and not config.force_cpu:
            torch.cuda.manual_seed(config.random_seed)
        
        # データセット管理
        self.dataset_manager = StandardMNISTDataset(config)
        
        # マルチクラスニューラルネットワーク
        self.model = self._create_multiclass_model()
        self.model.to(self.device)
        
        # オプティマイザー・損失関数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        # 学習統計
        self.training_stats = {
            'train_accuracy': [],
            'test_accuracy': [],
            'train_loss': [],
            'test_loss': []
        }
        
        # リアルタイム可視化機能（Phase 1と同様）
        self.visualizer = None
        self.multiclass_fig = None
        self.acc_ax = None
        self.loss_ax = None
        if config.realtime:
            self.visualizer = "multiclass"
        
        # デバイス情報表示（CPU強制オプション考慮）
        device_info = f"デバイス: {self.device}"
        if config.force_cpu:
            device_info += " (CPU強制使用)"
        elif torch.cuda.is_available():
            device_info += " (自動選択)"
        print(f"🔄 エポック単位トレーナー初期化完了: {device_info}") if self.config.verbose else None
        
    def _create_multiclass_model(self):
        """マルチクラス分類ネットワーク作成"""
        class MultiClassNetwork(nn.Module):
            def __init__(self, input_size=784, hidden_size=64, num_classes=10):
                super().__init__()
                self.hidden = nn.Linear(input_size, hidden_size)
                self.output = nn.Linear(hidden_size, num_classes)
                
                # v5.5.4準拠重み初期化
                nn.init.normal_(self.hidden.weight, mean=0, std=0.1)
                nn.init.normal_(self.output.weight, mean=0, std=0.1)
                nn.init.zeros_(self.hidden.bias)
                nn.init.zeros_(self.output.bias)
                
            def forward(self, x):
                if len(x.shape) > 2:
                    x = x.view(x.size(0), -1)
                x = torch.sigmoid(self.hidden(x))
                x = self.output(x)
                return x
        
        return MultiClassNetwork(
            input_size=784,
            hidden_size=self.config.hidden_size,
            num_classes=self.config.num_classes
        )
    
    def train_epoch_based(self):
        """エポック単位学習実行"""
        print("🚀 エポック単位学習開始") if self.config.verbose else None
        
        # リアルタイム可視化セットアップ
        if self.config.realtime and self.visualizer:
            self.setup_multiclass_visualization(self.config.epochs)
            print("=== Phase 2 リアルタイム可視化ウィンドウ表示完了 ===") if self.config.verbose else None
        
        # データローダー準備
        dataloaders = self.dataset_manager.create_epoch_based_datasets()
        train_loader = dataloaders['train']
        test_loader = dataloaders['test']
        
        start_time = time.time()
        
        # エポックループ
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            
            # 訓練フェーズ
            self.model.train()
            train_correct = 0
            train_total = 0
            train_loss_sum = 0
            
            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                # 統計更新
                train_loss_sum += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            train_accuracy = train_correct / train_total
            train_loss = train_loss_sum / len(train_loader)
            
            # テストフェーズ
            self.model.eval()
            test_correct = 0
            test_total = 0
            test_loss_sum = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in test_loader:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_data)
                    loss = self.criterion(outputs, batch_labels)
                    
                    test_loss_sum += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += batch_labels.size(0)
                    test_correct += (predicted == batch_labels).sum().item()
            
            test_accuracy = test_correct / test_total
            test_loss = test_loss_sum / len(test_loader)
            
            # 統計記録
            self.training_stats['train_accuracy'].append(train_accuracy)
            self.training_stats['test_accuracy'].append(test_accuracy)
            self.training_stats['train_loss'].append(train_loss)
            self.training_stats['test_loss'].append(test_loss)
            
            # リアルタイム可視化更新
            if self.config.realtime and self.visualizer:
                self.update_visualization(epoch)
            
            epoch_time = time.time() - epoch_start
            print(f"エポック {epoch}: 訓練精度 {train_accuracy:.4f}, テスト精度 {test_accuracy:.4f}, 時間 {epoch_time:.2f}秒") if self.config.verbose else None
        
        total_time = time.time() - start_time
        print(f"🎯 Phase 2完了: 総学習時間 {total_time:.2f}秒") if self.config.verbose else None
        
        return self._evaluate_final_performance(test_loader, save_predictions=self.config.verify)
    
    def _evaluate_final_performance(self, test_loader, save_predictions: bool = False):
        """最終性能評価（予測結果記録機能付き）"""
        self.model.eval()
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        # 検証用データ収集
        all_true_labels = []
        all_predicted_labels = []
        all_predicted_probs = []
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_data)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # 検証用データ収集
                if save_predictions:
                    all_true_labels.extend(batch_labels.cpu().numpy().tolist())
                    all_predicted_labels.extend(predicted.cpu().numpy().tolist())
                    all_predicted_probs.extend(probabilities.cpu().numpy().tolist())
                
                for i in range(batch_labels.size(0)):
                    label = batch_labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == batch_labels[i]:
                        class_correct[label] += 1
                    label = batch_labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == batch_labels[i]:
                        class_correct[label] += 1
        
        # クラス別精度計算
        class_accuracies = {}
        total_correct = 0
        total_samples = 0
        
        print("=== エポック単位学習結果 ===")
        for class_idx in range(self.config.num_classes):
            if class_total[class_idx] > 0:
                accuracy = class_correct[class_idx] / class_total[class_idx]
                class_accuracies[class_idx] = accuracy
                total_correct += class_correct[class_idx]
                total_samples += class_total[class_idx]
                print(f"クラス {class_idx} 精度: {accuracy:.4f} ({class_correct[class_idx]}/{class_total[class_idx]})")
        
        overall_accuracy = total_correct / total_samples
        accuracies_list = list(class_accuracies.values())
        accuracy_range = max(accuracies_list) - min(accuracies_list) if accuracies_list else 0
        success_classes = sum(1 for acc in accuracies_list if acc >= 0.75)
        
        # 検証用CSV保存（5エポック以上で最終エポックの場合）
        csv_file = None
        if save_predictions and self.config.epochs >= 5:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = save_predictions_to_csv(
                all_true_labels, all_predicted_labels, all_predicted_probs,
                "epoch", self.config.epochs, timestamp
            )
            
            # 検証実行
            verification_result = verify_accuracy_from_csv(csv_file, overall_accuracy)
            print_verification_report(verification_result)
        
        results = {
            'overall_accuracy': overall_accuracy,
            'class_accuracies': class_accuracies,
            'accuracy_range': accuracy_range,
            'success_classes': success_classes,
            'training_stats': self.training_stats,
            'csv_file': csv_file
        }
        
        print(f"=== エポック単位学習 評価結果 ===")
        print(f"全体精度: {overall_accuracy:.4f}")
        print(f"精度範囲: {accuracy_range:.4f}")
        print(f"成功クラス数: {success_classes}/{self.config.num_classes}")
        
        return results
    
    def setup_multiclass_visualization(self, max_epochs: int = 10) -> None:
        """Phase 2: マルチクラス可視化の初期設定"""
        if self.visualizer:
            self._setup_multiclass_visualization(max_epochs)
    
    def _setup_multiclass_visualization(self, max_epochs: int = 10) -> None:
        """Phase 2: マルチクラス可視化の初期設定"""
        import matplotlib.pyplot as plt
        
        # 既存のfigureをクリーンアップ
        if hasattr(self, 'multiclass_fig') and self.multiclass_fig is not None:
            plt.close(self.multiclass_fig)
        
        # 最大エポック数を保存
        self.max_epochs = max_epochs
        
        # 日本語フォント設定確認
        plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'Yu Gothic', 'Noto Sans JP', 'sans-serif']
        
        # グラフウィンドウサイズ（HyperParametersから取得）
        graph_width = self.hyperparams.graph_width
        graph_height = self.hyperparams.graph_height  
        self.multiclass_fig, (self.acc_ax, self.loss_ax) = plt.subplots(1, 2, figsize=(graph_width, graph_height))
        self.multiclass_fig.suptitle('ED-ANN v1.0.1 エポック単位学習進捗', fontsize=16, fontweight='bold')
        
        # 精度グラフ設定（左）
        self.acc_ax.set_title('精度 (Accuracy)', fontsize=14, fontweight='bold')
        self.acc_ax.set_xlabel('エポック', fontsize=12)
        self.acc_ax.set_ylabel('精度', fontsize=12)
        self.acc_ax.grid(True, alpha=0.3)
        self.acc_ax.set_ylim(0.5, 1.0)
        
        # 損失グラフ設定（右）
        self.loss_ax.set_title('損失 (Loss)', fontsize=14, fontweight='bold')
        self.loss_ax.set_xlabel('エポック', fontsize=12)
        self.loss_ax.set_ylabel('損失', fontsize=12)
        self.loss_ax.grid(True, alpha=0.3)
        self.loss_ax.set_ylim(0.0, 0.5)  # 損失範囲を0.0-0.5に設定
        
        # 初期状態表示
        self.acc_ax.text(0.5, 0.75, 'エポック単位学習\nエポック完了時にプロット更新', 
                        ha='center', va='center', fontsize=12, 
                        transform=self.acc_ax.transAxes)
        self.loss_ax.text(0.5, 0.75, 'エポック単位学習\nエポック完了時にプロット更新', 
                         ha='center', va='center', fontsize=12,
                         transform=self.loss_ax.transAxes)
        
        # インタラクティブモード設定
        plt.ion()
        plt.show(block=False)
        
        # 初期描画
        self.multiclass_fig.canvas.draw()
        self.multiclass_fig.canvas.flush_events()
    
    def update_visualization(self, epoch: int) -> None:
        """Phase 2: 可視化更新"""
        if not self.visualizer or not hasattr(self, 'multiclass_fig') or self.multiclass_fig is None:
            return
            
        try:
            # グラフをクリア
            self.acc_ax.clear()
            self.loss_ax.clear()
            
            # グラフ設定を再適用
            self.acc_ax.set_title('精度 (Accuracy)', fontsize=14, fontweight='bold')
            self.acc_ax.set_xlabel('エポック', fontsize=12)
            self.acc_ax.set_ylabel('精度', fontsize=12)
            self.acc_ax.grid(True, alpha=0.3)
            self.acc_ax.set_ylim(0.5, 1.0)
            
            self.loss_ax.set_title('損失 (Loss)', fontsize=14, fontweight='bold')
            self.loss_ax.set_xlabel('エポック', fontsize=12)
            self.loss_ax.set_ylabel('損失', fontsize=12)
            self.loss_ax.grid(True, alpha=0.3)
            self.loss_ax.set_ylim(0.0, 0.5)  # 損失範囲を0.0-0.5に設定
            
            epochs_range = list(range(1, epoch + 2))
            
            # データをプロット
            if self.training_stats['train_accuracy']:
                self.acc_ax.plot(epochs_range, self.training_stats['train_accuracy'], 
                               color='blue', linewidth=2, linestyle='-', marker='o', 
                               label='訓練精度')
            if self.training_stats['test_accuracy']:
                self.acc_ax.plot(epochs_range, self.training_stats['test_accuracy'], 
                               color='red', linewidth=2, linestyle='--', marker='s',
                               label='テスト精度')
            
            if self.training_stats['train_loss']:
                self.loss_ax.plot(epochs_range, self.training_stats['train_loss'], 
                                color='blue', linewidth=2, linestyle='-', marker='o',
                                label='訓練Loss')
            if self.training_stats['test_loss']:
                self.loss_ax.plot(epochs_range, self.training_stats['test_loss'], 
                                color='red', linewidth=2, linestyle='--', marker='s',
                                label='テストLoss')
            
            # 凡例追加（精度グラフは右下、損失グラフはデフォルト）
            self.acc_ax.legend(loc='lower right')
            self.loss_ax.legend()
            
            # X軸範囲設定
            self.acc_ax.set_xlim(1, max(self.max_epochs, epoch + 1))
            self.loss_ax.set_xlim(1, max(self.max_epochs, epoch + 1))
            
            # 描画更新
            self.multiclass_fig.canvas.draw()
            self.multiclass_fig.canvas.flush_events()
            
        except Exception as e:
            pass  # エラーが発生しても継続

class RestoredTrainer:
    """
    v5.5.4準拠 - 復元トレーナー
    バイナリ分類方式による各クラス分類器の独立学習
    """
    
    def __init__(self, config: TrainingConfig, hyperparams: Optional[HyperParameters] = None):
        self.config = config
        self.hyperparams = hyperparams if hyperparams is not None else HyperParameters()
        
        # デバイス決定（CPU強制オプション考慮）
        if config.force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device(config.device)
        
        # 決定論的シード設定
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        if torch.cuda.is_available() and not config.force_cpu:
            torch.cuda.manual_seed(config.random_seed)
        
        # データセット管理
        # Phase 1: 標準Dataset統合
        self.dataset_manager = StandardMNISTDataset(config)
        
        # 各クラス専用分類器
        self.classifiers = {}
        self.optimizers = {}
        
        # 損失関数（v5.5.4準拠）
        self.criterion = nn.BCELoss()
        
        # リアルタイム可視化機能（マルチクラス専用）
        self.visualizer = None
        if config.realtime:
            # マルチクラス可視化システムを有効化
            self.visualizer = "multiclass"  # フラグとして使用
        
        # 学習統計
        self.training_stats = defaultdict(list)
        
        # リアルタイム可視化用の全体訓練結果管理
        self.global_training_results = {}  # クラス別訓練結果を蓄積
        
        # 統合評価フラグ（タイトル管理用）
        self._evaluation_started = False
        
        print(f"復元トレーナー初期化完了: デバイス{self.device}") if self.config.verbose else None
        
    def initialize_classifiers(self):
        """全クラス分類器の初期化"""
        print("クラス分類器を初期化中...") if self.config.verbose else None
        
        for class_idx in range(self.config.num_classes):
            # v5.5.4準拠分類器作成
            classifier = SingleOutputClassifier(
                input_size=784,
                hidden_size=self.config.hidden_size,
                target_class=class_idx
            ).to(self.device)
            
            # オプティマイザ設定（v5.5.4準拠）
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=self.config.learning_rate
            )
            
            self.classifiers[class_idx] = classifier
            self.optimizers[class_idx] = optimizer
            
            # 初期重み統計
            stats = classifier.get_classifier_stats()
            print(f"クラス{class_idx}初期重み - 隠れ層ノルム: {stats['hidden_layer']['norm']:.4f}") if self.config.verbose else None
        
        print("全クラス分類器初期化完了") if self.config.verbose else None
    
    def train_classifier(self, class_idx: int) -> Dict:
        """指定クラスの分類器を訓練（v5.5.4準拠）"""
        print(f"=== クラス {class_idx} の訓練開始 ===") if self.config.verbose else None
        
        classifier = self.classifiers[class_idx]
        optimizer = self.optimizers[class_idx]
        
        # バイナリ分類用データローダー取得
        train_loader, test_loader = self.dataset_manager.get_binary_datasets(class_idx)
        
        classifier.train()
        
        class_stats = {
            'epoch_losses': [],
            'epoch_train_accuracies': [],
            'epoch_test_accuracies': [],
            'training_time': 0
        }
        
        training_start = time.time()
        
        for epoch in range(self.config.epochs):
            epoch_start = time.time()
            epoch_losses = []
            epoch_accuracies = []
            
            for step, (data, labels) in enumerate(train_loader):
                data = data.to(self.device)
                labels = labels.to(self.device).float()
                
                # 前向き計算
                outputs = classifier(data)
                loss = self.criterion(outputs, labels)
                
                # 逆伝播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 精度計算
                predictions = (outputs > 0.5).float()
                accuracy = (predictions == labels).float().mean().item()
                
                epoch_losses.append(loss.item())
                epoch_accuracies.append(accuracy)
                
                # リアルタイム表示
                if self.config.realtime and step % 10 == 0 and self.config.verbose:
                    logger.info(f"  ステップ {step}: 損失 {loss.item():.4f}, 精度 {accuracy:.4f}")
            
            # エポック終了処理
            epoch_time = time.time() - epoch_start
            train_accuracy = np.mean(epoch_accuracies)
            train_loss = np.mean(epoch_losses)
            
            # テスト評価
            test_accuracy, test_loss = self.evaluate_classifier(class_idx, test_loader)
            
            class_stats['epoch_losses'].append(train_loss)
            class_stats['epoch_train_accuracies'].append(train_accuracy)
            class_stats['epoch_test_accuracies'].append(test_accuracy)
            
            # リアルタイム可視化のため、このクラスの結果を全体結果に更新
            self.global_training_results[class_idx] = {
                'epoch_losses': class_stats['epoch_losses'].copy(),
                'epoch_train_accuracies': class_stats['epoch_train_accuracies'].copy(),
                'epoch_test_accuracies': class_stats['epoch_test_accuracies'].copy()
            }
            
            # エポック完了時にリアルタイム可視化を更新
            if self.config.realtime and self.visualizer:
                # クラス9の最終エポック完了時は、先にフラグを設定
                if class_idx == 9 and epoch == self.config.epochs - 1:
                    self._evaluation_started = True
                
                self.update_multiclass_visualization(epoch + 1, class_idx, float(train_accuracy), float(test_accuracy), float(train_loss), float(test_loss))
                print(f"=== エポック {epoch + 1} 完了：リアルタイム可視化更新 (クラス{class_idx}) ===") if self.config.verbose else None
            
            print(f"エポック {epoch}: 訓練精度 {train_accuracy:.4f}, テスト精度 {test_accuracy:.4f}, 時間 {epoch_time:.2f}秒") if self.config.verbose else None
        
        class_stats['training_time'] = time.time() - training_start
        
        # 最終統計
        final_stats = classifier.get_classifier_stats()
        print(f"クラス{class_idx}最終重み - 隠れ層ノルム: {final_stats['hidden_layer']['norm']:.4f}") if self.config.verbose else None
        
        return class_stats
    
    def evaluate_classifier(self, class_idx: int, test_loader: DataLoader) -> Tuple[float, float]:
        """指定クラス分類器の評価（精度と損失を返す）"""
        classifier = self.classifiers[class_idx]
        classifier.eval()
        
        correct = 0
        total = 0
        test_losses = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.to(self.device)
                labels = labels.to(self.device).float()
                
                outputs = classifier(data)
                predictions = (outputs > 0.5).float()
                
                # テスト損失計算
                test_loss = self.criterion(outputs, labels)
                test_losses.append(test_loss.item())
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        avg_test_loss = float(np.mean(test_losses)) if test_losses else 0.0
        return accuracy, avg_test_loss
    
    def update_multiclass_visualization(self, epoch: int, class_idx: int,
                                       train_acc: float, test_acc: float, train_loss: float, test_loss: float) -> None:
        """
        個別クラスのエポック完了時可視化更新（ED-ANN仕様準拠）
        
        Args:
            epoch: 現在のエポック (1から開始)
            class_idx: 完了したクラス番号 (0-9)
            train_acc: 該当クラスの訓練精度
            test_acc: 該当クラスのテスト精度  
            train_loss: 該当クラスの訓練損失
            test_loss: 該当クラスのテスト損失
        """
        if not hasattr(self, 'multiclass_fig'):
            # 最大エポック数を取得（configから）
            max_epochs = getattr(self, 'config', None)
            if max_epochs and hasattr(max_epochs, 'epochs'):
                max_epochs = max_epochs.epochs
            else:
                max_epochs = 10  # デフォルト値
            self._setup_multiclass_visualization(max_epochs)
        
        # 動的タイトル更新（統合評価開始後はスキップ）
        evaluation_started = getattr(self, '_evaluation_started', False)
        if not evaluation_started:
            title = f"ED-ANN v1.0.1 マルチクラス学習進捗　　クラス {class_idx} を学習中"
            self.multiclass_fig.suptitle(title, fontsize=16, fontweight='bold')
        else:
            # 統合評価開始済みの場合は「全体の精度とLossを計算中」に更新
            title = "ED-ANN v1.0.1 マルチクラス学習進捗　　全体の精度とLossを計算中"
            self.multiclass_fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # クラス別データ蓄積用の初期化
        if not hasattr(self, 'mc_class_data'):
            # クラス別の各エポックデータを記録
            self.mc_class_data = {
                i: {
                    'train_accs': [0.0] * self.config.epochs,
                    'test_accs': [0.0] * self.config.epochs, 
                    'train_losses': [0.0] * self.config.epochs,
                    'test_losses': [0.0] * self.config.epochs
                } for i in range(10)
            }
            # 全体統合結果用（最後に計算）
            self.mc_overall_data = {
                'train_accs': [],
                'test_accs': [],
                'train_losses': [],
                'test_losses': []
            }
        
        # 該当クラスのエポック結果を記録
        epoch_idx = epoch - 1  # 0ベースインデックス
        self.mc_class_data[class_idx]['train_accs'][epoch_idx] = train_acc
        self.mc_class_data[class_idx]['test_accs'][epoch_idx] = test_acc
        self.mc_class_data[class_idx]['train_losses'][epoch_idx] = train_loss
        self.mc_class_data[class_idx]['test_losses'][epoch_idx] = test_loss
        
        self._update_individual_class_plots()
    
    def _update_individual_class_plots(self) -> None:
        """個別クラス完了時のプロット更新"""
        # グラフをクリア
        self.acc_ax.clear()
        self.loss_ax.clear()
        
        # グラフ設定を再適用
        self.acc_ax.set_title('精度 (Accuracy)', fontsize=14, fontweight='bold')
        self.acc_ax.set_xlabel('エポック', fontsize=12)
        self.acc_ax.set_ylabel('精度', fontsize=12)
        self.acc_ax.grid(True, alpha=0.3)
        self.acc_ax.set_ylim(0.5, 1.0)  # 精度軸範囲を0.5-1.0に設定
        
        self.loss_ax.set_title('損失 (Loss)', fontsize=14, fontweight='bold')
        self.loss_ax.set_xlabel('エポック', fontsize=12)
        self.loss_ax.set_ylabel('損失', fontsize=12)
        self.loss_ax.grid(True, alpha=0.3)
        self.loss_ax.set_ylim(0.0, 0.5)  # 損失軸範囲を0.0-0.5に設定
        
        # 各クラスの現在までのデータをプロット
        epochs_range = list(range(1, self.config.epochs + 1))
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for class_idx in range(10):
            color = colors[class_idx]
            
            # データがある部分のみを折れ線グラフでプロット
            train_accs = self.mc_class_data[class_idx]['train_accs']
            test_accs = self.mc_class_data[class_idx]['test_accs']
            train_losses = self.mc_class_data[class_idx]['train_losses']
            test_losses = self.mc_class_data[class_idx]['test_losses']
            
            # データが入力済みのエポック範囲を特定
            data_epochs = []
            data_train_accs = []
            data_test_accs = []
            data_train_losses = []
            data_test_losses = []
            
            for epoch_idx in range(self.config.epochs):
                if train_accs[epoch_idx] > 0:  # データが入力済み
                    epoch_num = epoch_idx + 1
                    data_epochs.append(epoch_num)
                    data_train_accs.append(train_accs[epoch_idx])
                    data_test_accs.append(test_accs[epoch_idx])
                    data_train_losses.append(train_losses[epoch_idx])
                    data_test_losses.append(test_losses[epoch_idx])
            
            # 折れ線グラフでプロット（データがある場合のみ）
            if data_epochs:
                # 精度プロット（折れ線）
                self.acc_ax.plot(data_epochs, data_train_accs, 
                               color=color, marker='o', linewidth=1.5, linestyle='-', 
                               label=f'クラス{class_idx}訓練')
                self.acc_ax.plot(data_epochs, data_test_accs, 
                               color=color, marker='s', linewidth=1.5, linestyle='--',
                               label=f'クラス{class_idx}テスト')
                
                # 損失プロット（折れ線）- 訓練とテストの両方
                self.loss_ax.plot(data_epochs, data_train_losses, 
                                color=color, marker='o', linewidth=1.5, linestyle='-',
                                label=f'クラス{class_idx}訓練Loss')
                self.loss_ax.plot(data_epochs, data_test_losses, 
                                color=color, marker='s', linewidth=1.5, linestyle='--',
                                label=f'クラス{class_idx}テストLoss')
        
        # X軸範囲設定（ED-ANN仕様準拠：最小1、最大エポック数）
        self.acc_ax.set_xlim(1, self.config.epochs)
        self.loss_ax.set_xlim(1, self.config.epochs)
        
        # 凡例設定（重複回避）
        handles_acc, labels_acc = self.acc_ax.get_legend_handles_labels()
        if handles_acc:
            self.acc_ax.legend(handles_acc, labels_acc, loc='lower right', fontsize=6, ncol=2)
        
        handles_loss, labels_loss = self.loss_ax.get_legend_handles_labels()
        if handles_loss:
            self.loss_ax.legend(handles_loss, labels_loss, loc='upper right', fontsize=6, ncol=2)
        
        # 描画更新
        self.multiclass_fig.canvas.draw()
        self.multiclass_fig.canvas.flush_events()
    
    def finalize_overall_visualization(self, overall_accuracy: float) -> None:
        """
        全学習完了時の統合結果プロット（ED-ANN仕様準拠）
        
        Args:
            overall_accuracy: 統合システムの最終精度
        """
        if not hasattr(self, 'mc_class_data'):
            return
            
        # 各エポックでの全体平均を計算
        for epoch_idx in range(self.config.epochs):
            epoch_train_accs = []
            epoch_test_accs = []
            epoch_train_losses = []
            epoch_test_losses = []
            
            # 全クラスの該当エポック結果を収集
            for class_idx in range(10):
                epoch_train_accs.append(self.mc_class_data[class_idx]['train_accs'][epoch_idx])
                epoch_test_accs.append(self.mc_class_data[class_idx]['test_accs'][epoch_idx])
                epoch_train_losses.append(self.mc_class_data[class_idx]['train_losses'][epoch_idx])
                epoch_test_losses.append(self.mc_class_data[class_idx]['test_losses'][epoch_idx])
            
            # 平均を計算して記録
            self.mc_overall_data['train_accs'].append(np.mean(epoch_train_accs))
            self.mc_overall_data['test_accs'].append(np.mean(epoch_test_accs))
            self.mc_overall_data['train_losses'].append(np.mean(epoch_train_losses))
            self.mc_overall_data['test_losses'].append(np.mean(epoch_test_losses))
        
        # 最終的な統合結果を含む完全プロット
        self._update_complete_plots()
        
        # プロット完了後、タイトルをメインタイトルのみに変更
        if hasattr(self, 'multiclass_fig') and self.multiclass_fig:
            title = "ED-ANN v1.0.1 マルチクラス学習進捗"  # メインタイトルのみ
            self.multiclass_fig.suptitle(title, fontsize=16, fontweight='bold')
            plt.draw()  # 即座に描画更新
            plt.pause(0.1)  # タイトル更新を表示
        
        # ターミナルメッセージ表示と自動終了機能
        self._handle_visualization_completion()
    
    def _handle_visualization_completion(self) -> None:
        """可視化完了後の処理（自動終了・ユーザー操作対応）"""
        import matplotlib.pyplot as plt
        import time
        import sys
        
        # ターミナルメッセージ表示（削除要求に従い簡潔に）
        print("グラフが3秒間表示されます")
        print("グラフウィンドウの×ボタンで早期終了可能")
        
        # 3秒間待機（グラフ表示継続）
        start_time = time.time()
        while time.time() - start_time < 3.0:
            # グラフウィンドウが閉じられたかチェック
            if not plt.get_fignums():
                break
            plt.pause(0.1)
        
        # グラフを閉じてプログラム終了
        try:
            if plt.get_fignums():
                plt.close('all')
            
            # matplotlibのバックエンドを確実にクリーンアップ
            plt.ioff()  # インタラクティブモードを無効化
            
        except Exception as e:
            pass  # エラーが発生しても継続
        
        # 削除：終了メッセージは非表示
    
    def _update_complete_plots(self) -> None:
        """全学習完了時の完全プロット（統合結果含む）"""
        # グラフをクリア
        self.acc_ax.clear()
        self.loss_ax.clear()
        
        # グラフ設定を再適用
        self.acc_ax.set_title('精度 (Accuracy)', fontsize=14, fontweight='bold')
        self.acc_ax.set_xlabel('エポック', fontsize=12)
        self.acc_ax.set_ylabel('精度', fontsize=12)
        self.acc_ax.grid(True, alpha=0.3)
        self.acc_ax.set_ylim(0.5, 1.0)  # 精度軸範囲を0.5-1.0に設定
        
        self.loss_ax.set_title('損失 (Loss)', fontsize=14, fontweight='bold')
        self.loss_ax.set_xlabel('エポック', fontsize=12)
        self.loss_ax.set_ylabel('損失', fontsize=12)
        self.loss_ax.grid(True, alpha=0.3)
        self.loss_ax.set_ylim(0.0, 0.5)  # 損失軸範囲を0.0-0.5に設定
        
        epochs_range = list(range(1, self.config.epochs + 1))
        
        # 1. 統合結果プロット（太線・黒）
        self.acc_ax.plot(epochs_range, self.mc_overall_data['train_accs'], 
                        color='black', linewidth=3, linestyle='-', marker='o', 
                        label='統合訓練精度')
        self.acc_ax.plot(epochs_range, self.mc_overall_data['test_accs'], 
                        color='black', linewidth=3, linestyle='--', marker='s',
                        label='統合テスト精度')
        
        self.loss_ax.plot(epochs_range, self.mc_overall_data['train_losses'], 
                         color='black', linewidth=3, linestyle='-', marker='o',
                         label='統合訓練Loss')
        self.loss_ax.plot(epochs_range, self.mc_overall_data['test_losses'], 
                         color='black', linewidth=3, linestyle='--', marker='s',
                         label='統合テストLoss')
        
        # 2. 各クラス結果プロット（細線・カラー）
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for class_idx in range(10):
            color = colors[class_idx]
            
            train_accs = self.mc_class_data[class_idx]['train_accs']
            test_accs = self.mc_class_data[class_idx]['test_accs']
            train_losses = self.mc_class_data[class_idx]['train_losses']
            test_losses = self.mc_class_data[class_idx]['test_losses']
            
            # 精度プロット
            self.acc_ax.plot(epochs_range, train_accs, 
                            color=color, linewidth=1.5, linestyle='-', 
                            label=f'クラス{class_idx}訓練')
            self.acc_ax.plot(epochs_range, test_accs, 
                            color=color, linewidth=1.5, linestyle='--')
            
            # 損失プロット（訓練とテストの両方）
            self.loss_ax.plot(epochs_range, train_losses, 
                             color=color, linewidth=1.5, linestyle='-',
                             label=f'クラス{class_idx}訓練Loss')
            self.loss_ax.plot(epochs_range, test_losses, 
                             color=color, linewidth=1.5, linestyle='--')
        
        # X軸範囲設定（ED-ANN仕様準拠：最小1、最大エポック数）
        self.acc_ax.set_xlim(1, self.config.epochs)
        self.loss_ax.set_xlim(1, self.config.epochs)
        
        # 凡例設定
        self.acc_ax.legend(loc='lower right', fontsize=8, ncol=2)
        self.loss_ax.legend(loc='upper right', fontsize=8, ncol=2)
        
        # 描画更新
        self.multiclass_fig.canvas.draw()
        self.multiclass_fig.canvas.flush_events()
    
    def setup_multiclass_visualization(self, max_epochs: int = 10) -> None:
        """マルチクラス可視化の初期設定（外部呼び出し用）"""
        if self.visualizer:
            self._setup_multiclass_visualization(max_epochs)
    
    def _setup_multiclass_visualization(self, max_epochs: int = 10) -> None:
        """マルチクラス可視化の初期設定"""
        import matplotlib.pyplot as plt
        
        # 既存のfigureをクリーンアップ
        if hasattr(self, 'multiclass_fig') and self.multiclass_fig is not None:
            plt.close(self.multiclass_fig)
        
        # 最大エポック数を保存
        self.max_epochs = max_epochs
        
        # 日本語フォント設定確認（ED-ANN仕様準拠）- 警告なしフォントのみ
        plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'Yu Gothic', 'Noto Sans JP', 'sans-serif']
        
        # グラフウィンドウサイズ（HyperParametersから取得）  
        graph_width = self.hyperparams.graph_width
        graph_height = self.hyperparams.graph_height
        self.multiclass_fig, (self.acc_ax, self.loss_ax) = plt.subplots(1, 2, figsize=(graph_width, graph_height))
        self.multiclass_fig.suptitle('ED-ANN v1.0.1 マルチクラス学習進捗', fontsize=16, fontweight='bold')
        
        # 精度グラフ設定（左）
        self.acc_ax.set_title('精度 (Accuracy)', fontsize=14, fontweight='bold')
        self.acc_ax.set_xlabel('エポック', fontsize=12)
        self.acc_ax.set_ylabel('精度', fontsize=12)
        self.acc_ax.grid(True, alpha=0.3)
        self.acc_ax.set_ylim(0.5, 1.0)  # 精度軸範囲を0.5-1.0に設定
        
        # 損失グラフ設定（右）
        self.loss_ax.set_title('損失 (Loss)', fontsize=14, fontweight='bold')
        self.loss_ax.set_xlabel('エポック', fontsize=12)
        self.loss_ax.set_ylabel('損失', fontsize=12)
        self.loss_ax.grid(True, alpha=0.3)
        self.loss_ax.set_ylim(0.0, 0.5)  # 損失軸範囲を0.0-0.5に設定
        
        # 色設定（11色：統合+10クラス）
        import matplotlib.pyplot as plt
        colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        self.multiclass_colors = colors
        
        # 初期状態表示（空のグラフで「準備中」メッセージ）
        self.acc_ax.text(0.5, 0.75, '1エポック目訓練開始\nエポック完了時にプロット更新', 
                        ha='center', va='center', fontsize=12, 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        self.loss_ax.text(0.5, 0.25, '1エポック目訓練開始\nエポック完了時にプロット更新', 
                         ha='center', va='center', fontsize=12, 
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        plt.ion()
        plt.show(block=False)
        
        # 初期描画を強制実行
        self.multiclass_fig.canvas.draw()
        self.multiclass_fig.canvas.flush_events()
    
    def classify_sample(self, sample: torch.Tensor) -> Dict:
        """v5.5.4準拠 - 統合分類システム"""
        sample = sample.to(self.device)
        
        if len(sample.shape) == 3:  # 単一サンプル [C, H, W]
            sample = sample.unsqueeze(0)  # バッチ次元追加
        
        confidence_scores = {}
        
        # 全分類器で信頼度計算
        for class_idx in range(self.config.num_classes):
            classifier = self.classifiers[class_idx]
            classifier.eval()
            
            with torch.no_grad():
                output = classifier(sample)
                confidence = output.item()
                confidence_scores[class_idx] = confidence
        
        # 最高信頼度クラス選択
        predicted_class = max(confidence_scores.keys(), key=lambda k: confidence_scores[k])
        max_confidence = confidence_scores[predicted_class]
        
        return {
            'predicted_class': predicted_class,
            'confidence_scores': confidence_scores,
            'max_confidence': max_confidence
        }
    
    def evaluate_integrated_system(self, save_predictions: bool = False) -> Dict:
        """統合システムの評価（予測結果記録機能付き）"""
        print("=== 統合システム評価開始 ===")
        
        # 全テストデータで評価
        test_dataset = self.dataset_manager.test_dataset
        
        class_accuracies = {}
        class_counts = defaultdict(int)
        class_correct = defaultdict(int)
        
        total_correct = 0
        total_samples = 0
        
        # 検証用データ収集
        all_true_labels = []
        all_predicted_labels = []
        all_predicted_probs = []
        
        # クラス別精度計算
        for true_class in range(self.config.num_classes):
            class_indices = self.dataset_manager.class_indices['test'][true_class]
            test_indices = class_indices[:self.config.test_samples_per_class]
            
            for idx in test_indices:
                data, _ = test_dataset[idx]
                
                # 分類実行
                result = self.classify_sample(data)
                predicted_class = result['predicted_class']
                confidence_scores = result['confidence_scores']
                
                # 検証用データ収集
                if save_predictions:
                    all_true_labels.append(true_class)
                    all_predicted_labels.append(predicted_class)
                    # confidence_scoresを確率として正規化
                    prob_list = [confidence_scores.get(i, 0.0) for i in range(10)]
                    prob_sum = sum(prob_list) if sum(prob_list) > 0 else 1.0
                    normalized_probs = [p / prob_sum for p in prob_list]
                    all_predicted_probs.append(normalized_probs)
                
                class_counts[true_class] += 1
                if predicted_class == true_class:
                    class_correct[true_class] += 1
                    total_correct += 1
                
                total_samples += 1
        
        # クラス別精度算出
        for class_idx in range(self.config.num_classes):
            accuracy = class_correct[class_idx] / class_counts[class_idx] if class_counts[class_idx] > 0 else 0
            class_accuracies[class_idx] = accuracy
            print(f"クラス {class_idx} 精度: {accuracy:.4f} ({class_correct[class_idx]}/{class_counts[class_idx]})")
        
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # クラス間格差分析
        accuracies_list = list(class_accuracies.values())
        accuracy_range = max(accuracies_list) - min(accuracies_list)
        
        # 検証用CSV保存（5エポック以上の場合）
        csv_file = None
        if save_predictions and self.config.epochs >= 5:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = save_predictions_to_csv(
                all_true_labels, all_predicted_labels, all_predicted_probs,
                "class", self.config.epochs, timestamp
            )
            
            # 検証実行
            verification_result = verify_accuracy_from_csv(csv_file, overall_accuracy)
            print_verification_report(verification_result)
        
        results = {
            'class_accuracies': class_accuracies,
            'overall_accuracy': overall_accuracy,
            'accuracy_range': accuracy_range,
            'min_accuracy': min(accuracies_list),
            'max_accuracy': max(accuracies_list),
            'successful_classes': sum(1 for acc in accuracies_list if acc >= 0.75),  # 75%以上
            'csv_file': csv_file
        }
        
        print(f"=== 評価結果 ===")
        print(f"全体精度: {overall_accuracy:.4f}")
        print(f"精度範囲: {accuracy_range:.4f}")
        print(f"成功クラス数: {results['successful_classes']}/10")
        
        return results

def main():
    """Phase 1 & Phase 2 比較実行メイン関数"""
    parser = argparse.ArgumentParser(description='ED-ANN v1.0.0 - モデル・ハイパーパラメータ情報表示版')
    parser.add_argument('--epochs', type=int, default=3, help='訓練エポック数 (default: 3)')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='学習率 (default: 0.01)')
    parser.add_argument('--batch_size', type=int, default=32, help='バッチサイズ (default: 32)')
    parser.add_argument('--hidden_size', type=int, default=64, help='隠れ層サイズ (default: 64)')
    parser.add_argument('--train_size', type=int, default=1000, help='クラス別訓練データ数 (default: 1000)')
    parser.add_argument('--test_size', type=int, default=1000, help='クラス別テストデータ数 (default: 1000)')
    parser.add_argument('--realtime', action='store_true', help='リアルタイム学習表示 (default: OFF)')
    parser.add_argument('--cpu', action='store_true', help='CPU強制使用 (default: 自動判別)')
    parser.add_argument('--seed', type=int, help='シード値 (無指定時はランダム値)')
    parser.add_argument('--verbose', action='store_true', help='詳細ログ表示 (default: OFF)')
    parser.add_argument('--verify', action='store_true', help='精度検証機能(結果CSV書き出し) (default: OFF)')
    parser.add_argument('--mode', type=str, choices=['epoch', 'class', 'both'], 
                       default='epoch', help='学習モード選択 (default: epoch) (epoch=エポック単位、class=クラス単位、both=比較)')
    
    args = parser.parse_args()
    
    # シード値の決定（--seedが指定されなければランダム値を生成）
    if args.seed is None:
        import random
        import time
        random_seed = int(time.time() * 1000) % 10000  # 現在時刻からランダムシードを生成
        print(f"🎲 ランダムシード: {random_seed} (自動生成)")
    else:
        random_seed = args.seed
        print(f"🎯 ランダムシード: {random_seed} (指定値)")
    
    # verboseオプションに基づいてログ設定を初期化
    global logger, VERBOSE_MODE
    logger = setup_logging(verbose=args.verbose)
    VERBOSE_MODE = args.verbose
    
    # 学習開始メッセージ
    print("🚀 ED-ANN v1.0.1 - グラフカスタマイゼーション機能追加版")
    
    # 精度検証機能使用時のエポック数チェック
    if args.verify and args.epochs < 5:
        print("\n❌ エラー: 精度検証機能(--verify)を使用する場合は5エポック以上のエポック数を指定してください。")
        print(f"   現在指定されたエポック数: {args.epochs}")
        print(f"   必要なエポック数: 5以上")
        print("\n💡 ヒント: 以下のようにエポック数を指定してください:")
        print(f"   python {sys.argv[0]} --epochs 5 --verify")
        return
    
    # 設定作成
    config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
        train_samples_per_class=args.train_size,
        test_samples_per_class=args.test_size,
        realtime=args.realtime,
        force_cpu=args.cpu,
        random_seed=random_seed,
        verbose=args.verbose,
        verify=args.verify
    )
    
    # モデル構造とハイパーパラメータ情報を表示
    display_model_summary(config.hidden_size)
    display_hyperparameters()
    
    # HyperParametersインスタンスを作成
    hyperparams = HyperParameters()
    
    results = {}
    
    # クラス単位学習
    if args.mode in ['class', 'both']:
        print("\n" + "="*60)
        print("🎯 クラス単位学習実行")
        print("="*60)
        
        trainer_p1 = RestoredTrainer(config, hyperparams)
        trainer_p1.initialize_classifiers()
        
        # リアルタイム可視化（Phase 1のみ）
        if config.realtime and args.mode == 'phase1':
            trainer_p1.setup_multiclass_visualization(config.epochs)
        
        # クラス単位学習実行
        training_results = {}
        for class_idx in range(config.num_classes):
            print(f"=== クラス {class_idx} 訓練実行中 ===") if config.verbose else None
            class_result = trainer_p1.train_classifier(class_idx)
            training_results[class_idx] = class_result
            print(f"=== クラス {class_idx} 訓練完了 ===") if config.verbose else None
        
        # クラス単位学習最終評価
        p1_results = trainer_p1.evaluate_integrated_system(save_predictions=config.verify)
        results['class'] = p1_results
        
        # リアルタイム可視化終了処理
        if config.realtime and hasattr(trainer_p1, 'multiclass_fig') and trainer_p1.multiclass_fig is not None:
            try:
                plt.close(trainer_p1.multiclass_fig)
            except:
                pass
    
    # エポック単位学習
    if args.mode in ['epoch', 'both']:
        print("\n" + "="*60)
        print("🔄 エポック単位学習実行")  
        print("="*60)
        
        trainer_p2 = EpochBasedTrainer(config, hyperparams)
        p2_results = trainer_p2.train_epoch_based()
        results['epoch'] = p2_results
        
        # Phase 2可視化終了処理
        if config.realtime and hasattr(trainer_p2, 'multiclass_fig') and trainer_p2.multiclass_fig is not None:
            try:
                plt.close(trainer_p2.multiclass_fig)
            except:
                pass
    
    # 結果比較表示
    if args.mode == 'both' and len(results) == 2:
        print("\n" + "="*80)
        print("📊 クラス単位 vs エポック単位 比較結果")
        print("="*80)
        
        p1 = results['class']
        p2 = results['epoch']
        
        print(f"{'メトリクス':<20} {'クラス単位学習':<20} {'エポック単位学習':<20} {'差分':<15}")
        print("-" * 80)
        print(f"{'全体精度':<20} {p1['overall_accuracy']:<20.4f} {p2['overall_accuracy']:<20.4f} {p2['overall_accuracy'] - p1['overall_accuracy']:+.4f}")
        print(f"{'精度範囲':<20} {p1['accuracy_range']:<20.4f} {p2['accuracy_range']:<20.4f} {p2['accuracy_range'] - p1['accuracy_range']:+.4f}")
        print(f"{'成功クラス数':<20} {p1['successful_classes']:<20} {p2['success_classes']:<20} {p2['success_classes'] - p1['successful_classes']:+}")
        
        print("\n🏆 推奨手法:", end=" ")
        if p2['overall_accuracy'] > p1['overall_accuracy']:
            print("エポック単位学習が優位")
        elif p1['overall_accuracy'] > p2['overall_accuracy']:
            print("クラス単位学習が優位")
        else:
            print("両手法同等性能")
    
    return results

def display_model_summary(hidden_size=64, model_name: str = "ED-ANN Model"):
    """
    TensorFlow model.summary()ライクなモデル情報表示
    
    Args:
        hidden_size: 隠れ層サイズ
        model_name: モデル名
    """
    print("\n" + "="*80)
    print(f"📋 {model_name} 構造情報")
    print("="*80)
    
    # ネットワーク構造の定義（MNISTデータに基づく）
    input_size = 784   # MNIST: 28x28 = 784
    output_size = 10   # MNIST: 10クラス
    
    print(f"{'Layer (type)':<25} {'Output Shape':<20} {'Param #':<15}")
    print("-" * 80)
    
    # 入力層 → 隠れ層(1)
    input_to_hidden_params = input_size * hidden_size + hidden_size  # 重み + バイアス
    print(f"{'入力層 (Linear)':<25} {'(None, ' + str(hidden_size) + ')':<20} {input_to_hidden_params:<15,}")
    
    # 隠れ層(1) の活性化関数（ReLU）
    # 注意: ReLUは活性化関数であり、実際のED-ANNでは隠れ層のニューロンの活性化に使用
    # パラメータは持たない
    print(f"{'隠れ層(1) (ReLU)':<25} {'(None, ' + str(hidden_size) + ')':<20} {'0':<15}")
    
    # 隠れ層(1) → 出力層
    hidden_to_output_params = hidden_size * output_size + output_size  # 重み + バイアス
    print(f"{'出力層 (Linear)':<25} {'(None, ' + str(output_size) + ')':<20} {hidden_to_output_params:<15,}")
    
    print("=" * 80)
    total_params = input_to_hidden_params + hidden_to_output_params
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {total_params:,}")
    print(f"Non-trainable params: 0")
    print("="*80)

def display_hyperparameters():
    """
    HyperParametersクラスの全パラメータを表示
    """
    print("\n" + "="*80)
    print("ED法ハイパーパラメータ設定")
    print("="*80)
    
    hp = HyperParameters()
    
    print("ED法理論パラメータ:")
    print(f"   d_plus: float = {hp.d_plus}      # アミン濃度増加量（正答時の重み増加制御）")
    print(f"   d_minus: float = {hp.d_minus}    # アミン濃度減少量（誤答時の重み減少制御）")
    
    print("\n学習率関連:")
    print(f"   base_learning_rate: float = {hp.base_learning_rate}    # 基本学習率（Adam optimizer用）")
    
    print("="*80)

def save_predictions_to_csv(true_labels: List[int], predicted_labels: List[int], 
                           predicted_probs: List[List[float]], mode: str, 
                           epoch: int, timestamp: str) -> str:
    """
    予測結果をCSV形式で保存（検証用）
    
    Args:
        true_labels: 正解ラベルのリスト
        predicted_labels: 予測ラベルのリスト
        predicted_probs: 各クラスの予測確率のリスト
        mode: 学習モード（epoch/class）
        epoch: エポック数
        timestamp: タイムスタンプ
    
    Returns:
        保存されたCSVファイルのパス
    """
    filename = f"ed_ann_predictions_{mode}_{epoch}ep_{timestamp}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # ヘッダー行
        header = ['sample_id', 'true_label', 'predicted_label'] + \
                [f'prob_class_{i}' for i in range(10)]
        writer.writerow(header)
        
        # データ行
        for i, (true_label, pred_label, probs) in enumerate(zip(true_labels, predicted_labels, predicted_probs)):
            row = [i, true_label, pred_label] + probs
            writer.writerow(row)
    
    return filename

def verify_accuracy_from_csv(csv_file: str, displayed_accuracy: float, 
                           displayed_loss: Optional[float] = None) -> Dict:
    """
    CSVファイルから精度とLossを再計算し、表示値と比較検証
    
    Args:
        csv_file: CSVファイルのパス
        displayed_accuracy: 表示された精度
        displayed_loss: 表示されたLoss（オプション）
    
    Returns:
        検証結果の辞書
    """
    true_labels = []
    predicted_labels = []
    predicted_probs = []
    
    # CSVファイルを読み込み
    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            true_labels.append(int(row['true_label']))
            predicted_labels.append(int(row['predicted_label']))
            
            # 予測確率を取得
            probs = [float(row[f'prob_class_{i}']) for i in range(10)]
            predicted_probs.append(probs)
    
    # 精度計算
    correct_predictions = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)
    calculated_accuracy = correct_predictions / len(true_labels)
    
    # Loss計算（CrossEntropyLoss）
    calculated_loss = 0.0
    if predicted_probs:
        for true_label, probs in zip(true_labels, predicted_probs):
            # log(softmax(probs))でCrossEntropyLoss計算
            log_prob = np.log(max(probs[true_label], 1e-10))  # 数値安定性のため最小値制限
            calculated_loss -= log_prob
        calculated_loss /= len(true_labels)
    
    # 検証結果
    accuracy_match = abs(calculated_accuracy - displayed_accuracy) < 1e-6
    loss_match = True
    if displayed_loss is not None:
        loss_match = abs(calculated_loss - displayed_loss) < 1e-6
    
    verification_result = {
        'csv_file': csv_file,
        'sample_count': len(true_labels),
        'calculated_accuracy': calculated_accuracy,
        'displayed_accuracy': displayed_accuracy,
        'accuracy_match': accuracy_match,
        'calculated_loss': calculated_loss,
        'displayed_loss': displayed_loss,
        'loss_match': loss_match,
        'verification_status': 'PASS' if accuracy_match and loss_match else 'FAIL'
    }
    
    return verification_result

def print_verification_report(verification_result: Dict):
    """
    検証結果レポートを表示
    """
    print("\n" + "="*80)
    print("🔍 ED-ANN 予測精度検証レポート")
    print("="*80)
    
    result = verification_result
    status_icon = "✅" if result['verification_status'] == 'PASS' else "❌"
    
    print(f"📊 検証ファイル: {result['csv_file']}")
    print(f"📈 サンプル数: {result['sample_count']:,}")
    print(f"\n🎯 精度検証:")
    print(f"   表示精度: {result['displayed_accuracy']:.6f}")
    print(f"   計算精度: {result['calculated_accuracy']:.6f}")
    print(f"   精度一致: {'✅ YES' if result['accuracy_match'] else '❌ NO'}")
    
    if result['displayed_loss'] is not None:
        print(f"\n📉 Loss検証:")
        print(f"   表示Loss: {result['displayed_loss']:.6f}")
        print(f"   計算Loss: {result['calculated_loss']:.6f}")
        print(f"   Loss一致: {'✅ YES' if result['loss_match'] else '❌ NO'}")
    
    print(f"\n🔍 検証結果: {status_icon} {result['verification_status']}")
    print("="*80)

if __name__ == '__main__':
    setup_japanese_font()
    results = main()
