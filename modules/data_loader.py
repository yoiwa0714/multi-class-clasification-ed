"""
data_loader.py
純正ED法（Error Diffusion Learning Algorithm）Python実装 v0.2.0
Original C implementation by Isamu Kaneko (1999)
"""

import numpy as np


class MiniBatchDataLoader:
    """
    ED法用ミニバッチデータローダー
    
    注：金子勇氏のED理論にはバッチ処理概念なし
    大規模データ対応のための現代的機能拡張
    """
    
    def __init__(self, inputs: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool = True):
        """
        ミニバッチデータローダーの初期化
        
        Args:
            inputs: 入力データ配列
            labels: ラベル配列
            batch_size: バッチサイズ
            shuffle: データシャッフル有無
        """
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(inputs)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size
        self._reset()
    
    def _reset(self):
        """エポック開始時のリセット処理"""
        if self.shuffle:
            indices = np.random.permutation(self.num_samples)
            self.inputs = self.inputs[indices]
            self.labels = self.labels[indices]
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
        
        self.current_batch += 1
        return batch_inputs, batch_labels
    
    def __len__(self):
        """バッチ数を返す"""
        return self.num_batches
