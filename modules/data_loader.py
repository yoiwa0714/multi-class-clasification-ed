from typing import Optional
import numpy as np


class MiniBatchDataLoader:
    """
    ED法用ミニバッチデータローダー
    
    注：金子勇氏のED理論にはバッチ処理概念なし
    大規模データ対応のための現代的機能拡張
    """
    
    def __init__(self, inputs: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool = True, original_indices: Optional[np.ndarray] = None):
        """
        ミニバッチデータローダーの初期化
        
        Args:
            inputs: 入力データ配列
            labels: ラベル配列
            batch_size: バッチサイズ
            shuffle: データシャッフル有無
            original_indices: 元データセットのインデックス配列（MNIST使用状況追跡用）
        """
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(inputs)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        
        # 元インデックス追跡機能
        if original_indices is not None:
            self.original_indices = original_indices
            self.track_usage = True
            self.usage_count = {}  # インデックス別使用回数
        else:
            self.original_indices = None
            self.track_usage = False
            self.usage_count = {}
        
        self._reset()
    
    def _reset(self):
        """エポック開始時のリセット処理"""
        if self.shuffle:
            indices = np.random.permutation(self.num_samples)
            self.inputs = self.inputs[indices]
            self.labels = self.labels[indices]
            if self.track_usage and self.original_indices is not None:
                self.original_indices = self.original_indices[indices]
        self.current_batch = 0
    
    def __iter__(self):
        """イテレータ初期化"""
        self._reset()
        return self
    
    def __next__(self):
        """次のバッチを取得"""
        if self.current_batch >= self.num_batches:
            raise StopIteration
        
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        
        batch_inputs = self.inputs[start_idx:end_idx]
        batch_labels = self.labels[start_idx:end_idx]
        
        # 使用状況記録
        if self.track_usage and self.original_indices is not None:
            batch_original_indices = self.original_indices[start_idx:end_idx]
            # 各データの使用回数をカウント
            for orig_idx in batch_original_indices:
                self.usage_count[orig_idx] = self.usage_count.get(orig_idx, 0) + 1
        
        self.current_batch += 1
        return batch_inputs, batch_labels
    
    def get_usage_statistics(self):
        """データ使用統計を取得"""
        if not self.track_usage:
            return None
        
        if len(self.usage_count) == 0:
            return {"total_data": 0, "usage_distribution": {}, "max_usage": 0, "min_usage": 0}
        
        usage_values = list(self.usage_count.values())
        return {
            "total_data": len(self.usage_count),
            "usage_distribution": dict(self.usage_count),
            "max_usage": max(usage_values),
            "min_usage": min(usage_values),
            "avg_usage": np.mean(usage_values),
            "usage_variance": np.var(usage_values)
        }
    
    def __len__(self):
        """バッチ数を返す"""
        return self.num_batches
