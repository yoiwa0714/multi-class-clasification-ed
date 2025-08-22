# ED法を用いたANNネットワークのマルチクラス分類法

## 概要

従来のED（Error Diffusion）法は二値分類を前提として設計されています。本実装は、ANNネットワークにマルチクラスED法を適用してマルチクラス分類を可能にしたものです。

## 特徴

- **従来のED法の拡張**: 二値分類限定であったED法をマルチクラス分類に対応
- **重み選択メカニズム**: クラス別専用重み配列による効率的な学習
- **MNIST対応**: 手書き数字認識（10クラス分類）での実装例を提供
- **PyTorchベース**: 現代的な機械学習フレームワークによる実装

## インストール方法

### 必要な環境
- Python 3.8以上
- PyTorch
- torchvision
- matplotlib
- numpy

### インストール手順

```bash
# リポジトリをクローン
git clone https://github.com/yoiwa0714/ed-ann.git
cd ed-ann

# 仮想環境の作成（推奨）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# または
.venv\Scripts\activate  # Windows

# 依存関係のインストール
pip install torch torchvision matplotlib numpy
```

## 使用方法

### 基本的な使用例

```python
import torch
from ed_ann import MulticlassEDANN

# モデルの初期化
model = MulticlassEDANN(input_size=784, hidden_size=256, num_classes=10)

# MNISTデータでの学習
python train_mnist.py
```

### オプション設定例

#### クラス別学習モード
```python
# クラス別学習（各クラス順次学習）
python train_mnist.py --learning_mode class_sequential --epochs 10

# 学習率の調整
python train_mnist.py --learning_rate 0.01 --batch_size 64
```

#### エポック順学習モード
```python
# エポック順学習（全クラス同時学習）
python train_mnist.py --learning_mode epoch_based --epochs 20

# 詳細ログ出力
python train_mnist.py --verbose --save_logs
```

#### 推論の実行
```python
# 学習済みモデルでの推論
python inference.py --model_path saved_models/best_model.pth --input_image test_image.png
```

### カスタマイズオプション

- `--hidden_size`: 隠れ層のサイズ指定
- `--learning_rate`: 学習率の設定
- `--weight_decay`: 正則化パラメータ
- `--save_interval`: モデル保存間隔
- `--device`: 計算デバイス（cpu/cuda）の指定

## ファイル構成

```
ed-ann/
├── README.md
├── train_mnist.py          # MNIST学習スクリプト
├── inference.py            # 推論スクリプト
├── ed_ann/                 # メインモジュール
│   ├── __init__.py
│   ├── model.py           # マルチクラスED-ANNモデル
│   ├── trainer.py         # 学習ロジック
│   └── utils.py           # ユーティリティ関数
├── examples/              # 使用例
├── docs/                  # ドキュメント
│   └── multiclass_ed_comprehensive_explanation.md
└── tests/                 # テストコード
```

## 動作原理

### 重み選択メカニズム
- クラス0〜9それぞれに専用の重み配列W[0:9]を用意
- 正解ラベルに対応する重み配列を動的に選択
- 選択された重みでクラス特化ネットワークを構成

### 学習方式
1. **クラス別学習**: 各クラス専用データで順次学習
2. **エポック順学習**: 全クラスデータを混合して学習

詳細な原理説明は[docs/multiclass_ed_comprehensive_explanation.md](docs/multiclass_ed_comprehensive_explanation.md)をご参照ください。

## 実験結果

MNISTデータセットでの実験により、従来の手法と比較して高い分類精度を達成しています。

## ライセンス

本プロジェクトはMITライセンスの下で公開されています。詳細は[LICENSE](LICENSE)ファイルをご参照ください。

```
MIT License

Copyright (c) 2025 yoiwa0714

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 貢献

プルリクエストやイシューの報告を歓迎します。貢献方法については[CONTRIBUTING.md](CONTRIBUTING.md)をご参照ください。

## 参考文献

- Error Diffusion法に関する基礎研究
- マルチクラス分類に関する機械学習文献

## 連絡先

質問や提案がございましたら、GitHubのIssueまでお知らせください。
