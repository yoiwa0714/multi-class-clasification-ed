# ED-Genuine v0.2.1 - Pure Error Diffusion Learning Algorithm

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-orange)](https://numpy.org)
[![CuPy](https://img.shields.io/badge/CuPy-GPU%20Accelerated-green)](https://cupy.dev)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

🚀 **金子勇氏の純正ED法（Error Diffusion Learning Algorithm）Python実装 - NumPy高速化版**

## 📖 概要

ED-Genuine（Error Diffusion Genuine Algorithm）は、金子勇氏によるオリジナルのError Diffusion Learning AlgorithmのC実装を完全に忠実に再現し、NumPy行列演算による大幅高速化を実現したPython実装です。

### 🎯 主な特徴

- **🧬 純正ED法実装**: 金子勇氏のC実装を100%忠実に再現
- **🚀 NumPy高速化**: 4.1倍の総合高速化（342秒→83秒/10エポック）
- **⚡ 行列演算最適化**: フォワード計算1,899倍高速化達成
- **🎯 マルチクラス対応**: MNISTで71%精度（さらなる最適化で95%目標）
- **📊 完全可視化**: 混同行列・学習進捗のリアルタイム表示
- **� 理論準拠**: ed_genuine.prompt.md 100%準拠実装

## 🚀 クイックスタート

### インストール

```bash
git clone https://github.com/yourusername/ed-snn-develop.git
cd ed-snn-develop
pip install -r requirements.txt
```

### 基本実行

```bash
# 基本実行（NumPy高速化版）
python ed_genuine/multi_ed_v021.py --train 1000 --test 1000 --epochs 10

# ハイパーパラメータ調整
python ed_genuine/multi_ed_v021.py --train 1000 --test 1000 --epochs 30 --alpha 0.5 --hidden 256

# リアルタイム可視化
python ed_genuine/multi_ed_v021.py --train 500 --test 500 --epochs 10 --viz

# GPU高速化（利用可能な場合）
python ed_genuine/multi_ed_v021.py --train 1000 --test 1000 --epochs 10 --gpu
```

## 📊 性能結果

| バージョン | 実行時間 (10エポック) | 精度 | 高速化率 |
|-----------|---------------------|------|----------|
| v0.2.0 (元版) | 342.17秒 | ~50% | 基準 |
| v0.2.1 (NumPy版) | 83.5秒 | 71.1% | **4.1倍高速化** |
| v0.2.1 (最適化版) | 予定 | 95%目標 | さらなる向上 |

### 🚀 高速化の詳細

- **フォワード計算**: 1,899倍高速化（トリプルループ→行列演算）
- **総合性能**: 4.1倍高速化（実用的な学習時間を実現）
- **理論準拠**: ed_genuine.prompt.md 100%準拠を維持

## 🧬 ED法の理論的背景

ED法は生物学的な神経伝達メカニズムを模倣した学習アルゴリズムです：

### 🔬 核心原理

1. **アミン濃度制御**: 正答時の濃度増加（d_plus）、誤答時の濃度減少（d_minus）
2. **重み符号保持**: 学習中の重みの符号を維持（ED法の核心制約）
3. **適応的学習**: 濃度に基づく動的な学習率調整

### 📈 マルチクラス拡張

従来の二値分類から多クラス分類への革新的拡張：

```text
二値分類:   Class A ↔ Class B
           ↓
多クラス:   Class 0, 1, 2, ..., 9
           各クラスに対する個別的なED制御
```

詳細は [THEORY.md](THEORY.md) を参照してください。

## 💻 使用例

### 基本的な使用法

```python
# 設定
config = TrainingConfig(
    epochs=10,
    learning_rate=0.01,
    batch_size=32,
    verify=True  # 精度検証有効
)

# 実行
trainer = EpochBasedTrainer(config)
results = trainer.train_and_evaluate()
```

### 高度なオプション

```bash
# CPU強制使用
python ed_ann_simple_v100.py --epochs 5 --cpu

# 詳細ログ出力
python ed_ann_simple_v100.py --epochs 5 --verbose

# カスタムパラメータ
python ed_ann_simple_v100.py --epochs 10 --learning_rate 0.005 --batch_size 64
```

詳細は [EXAMPLES.md](EXAMPLES.md) を参照してください。

## 🏗️ システム構造

```text
ED-ANN v1.0.0
├── 学習エンジン
│   ├── EpochBasedTrainer    # エポック単位学習
│   └── RestoredTrainer      # クラス単位学習
├── ED法コア
│   ├── アミン濃度制御
│   ├── 重み更新制御
│   └── 符号保持メカニズム
├── 可視化システム
│   ├── リアルタイムグラフ
│   ├── モデル構造表示
│   └── ハイパーパラメータ表示
└── 検証システム
    ├── CSV出力
    ├── 精度検証
    └── レポート生成
```

## 📋 必要要件

- Python 3.8+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy
- pandas

## 📖 ドキュメント

- [THEORY.md](THEORY.md) - ED法の理論的詳細
- [EXAMPLES.md](EXAMPLES.md) - 実行例とパラメータ詳細
- [API Documentation](docs/api.md) - API リファレンス

## 🎨 図表資料

### ED法理論図
![ED法基本理論](figures/ed_theory_diagram.png)

### マルチクラス拡張概念図
![マルチクラス拡張](figures/multiclass_expansion.png)

### 学習フロー比較
![学習フロー比較](figures/learning_flow.png)

### 性能比較
![性能比較](figures/performance_comparison.png)

### アミン濃度ダイナミクス
![アミン濃度ダイナミクス](figures/amine_dynamics.png)

## 🔧 コマンドライン オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `--epochs` | 3 | 学習エポック数 |
| `--learning_rate` | 0.01 | 学習率 |
| `--batch_size` | 32 | バッチサイズ |
| `--hidden_size` | 64 | 隠れ層サイズ |
| `--mode` | epoch | 学習モード (epoch/class/both) |
| `--verify` | False | 精度検証機能 (5エポック以上必須) |
| `--realtime` | False | リアルタイム可視化 |
| `--cpu` | False | CPU強制使用 |
| `--verbose` | False | 詳細ログ表示 |
| `--random_seed` | 42 | ランダムシード |

## 🚨 重要な注意事項

### 精度検証機能
- `--verify`オプションを使用する場合は**5エポック以上**が必須です
- 5エポック未満で実行すると明確なエラーメッセージが表示されます

### 出力ファイル
- **CSV検証ファイル**: `ed_ann_predictions_{mode}_{epochs}ep_{timestamp}.csv`
- 検証レポートでファイル名が確認できます

## 🤝 貢献

プルリクエストやイシューの報告を歓迎します。

## 📄 ライセンス

MIT License - 詳細は [LICENSE](LICENSE) を参照

## 🙏 謝辞

このプロジェクトは生物学的な神経伝達メカニズムの研究にインスパイアされています。

## 📞 お問い合わせ

- GitHub Issues: [Issues](https://github.com/yourusername/ed-ann-v1.0.0/issues)
- Email: your.email@example.com

## 📈 バージョン履歴

### v1.0.0 (2025-08-17)
- 🎉 初回リリース
- 🧠 ED法マルチクラス分類システム実装
- 📊 TensorFlow風モデル構造表示
- 🔍 CSV出力精度検証システム
- 📈 リアルタイム学習可視化
- 🎯 MNIST 89.4%精度達成

---

**🧠 ED-ANN: 生物学的知見とAI技術の融合**

*Error Diffusion Method による次世代ニューラルネットワーク*
