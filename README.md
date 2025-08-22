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

```bash
# MNISTデータでの学習実行（デフォルト設定）
python ed_ann_simple_v100.py

# ヘルプオプションで利用可能なオプションを確認
python ed_ann_simple_v100.py --help

# 基本的なパラメータ調整例
python ed_ann_simple_v100.py --epochs 5 --learning_rate 0.005 --batch_size 64
```

プログラム内では以下のような構成になっています：

```python
# マルチクラスED-ANNモデルの初期化
model = MulticlassEDANN(input_size=784, hidden_size=64, num_classes=10)

# MNIST学習の実行
# - 学習モード（epoch/class/both）をコマンドラインで選択可能
# - 各種パラメータはコマンドラインオプションで設定可能
```

### オプション設定例

#### 基本的なパラメータ調整

```bash
# エポック数とバッチサイズを調整
python ed_ann_simple_v100.py --epochs 10 --batch_size 128

# 学習率と隠れ層サイズを調整
python ed_ann_simple_v100.py --learning_rate 0.005 --hidden_size 128

# CPU強制使用での実行
python ed_ann_simple_v100.py --cpu

# 詳細ログとリアルタイム表示を有効化
python ed_ann_simple_v100.py --verbose --realtime

# シード値を指定して再現可能な実行
python ed_ann_simple_v100.py --seed 42

# シード値を指定しない場合はランダム値が自動生成される
python ed_ann_simple_v100.py  # 毎回異なるランダムシードで実行
```

#### 学習モード選択

```bash
# エポック単位学習（デフォルト）
python ed_ann_simple_v100.py --mode epoch

# クラス単位学習
python ed_ann_simple_v100.py --mode class

# 両方のモードで比較実行
python ed_ann_simple_v100.py --mode both
```

#### 詳細な設定例

```bash
# 全パラメータを指定した例
python ed_ann_simple_v100.py \
  --epochs 5 \
  --learning_rate 0.008 \
  --batch_size 64 \
  --hidden_size 128 \
  --mode both \
  --realtime \
  --verbose \
  --verify \
  --seed 123
```

### カスタマイズオプション

`ed_ann_simple_v100.py`では以下のコマンドラインオプションが利用可能です：

#### 基本パラメータ
- `--epochs`: 訓練エポック数（デフォルト: 3）
- `--learning_rate`: 学習率（デフォルト: 0.01）  
- `--batch_size`: バッチサイズ（デフォルト: 32）
- `--hidden_size`: 隠れ層サイズ（デフォルト: 64）

#### 実行モード・オプション
- `--mode`: 学習モード選択（デフォルト: epoch）
  - `epoch`: エポック単位学習
  - `class`: クラス単位学習  
  - `both`: 比較実行
- `--realtime`: リアルタイム学習表示の有効化
- `--verbose`: 詳細ログ表示の有効化
- `--verify`: 精度検証機能（結果CSV書き出し）の有効化

#### システム設定
- `--cpu`: CPU強制使用（GPU自動判別を無効化）
- `--seed`: シード値（無指定時はランダム値）

#### 使用例：

```bash
# 全オプション指定例
python ed_ann_simple_v100.py \
  --epochs 10 \
  --learning_rate 0.005 \
  --batch_size 64 \
  --hidden_size 128 \
  --mode both \
  --realtime \
  --verbose \
  --verify \
  --cpu \
  --seed 789
```

## ファイル構成

```
ed-ann/
├── README.md                           # このファイル
├── ed_ann_simple_v100.py               # メインの実行ファイル
└── docs/                               # ドキュメント
    ├── multiclass_ed_comprehensive_explanation.md
    ├── 通常のEDネットワーク例.png
    ├── MNIST対応 マルチクラスEDネットワーク.png
    ├── クラス別学習 学習順序.png
    └── エポック順学習 学習順序.png
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
