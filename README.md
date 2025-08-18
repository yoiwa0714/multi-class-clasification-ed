# ED-ANN v1.0.0 - Error Diffusion Artificial Neural Network

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

🧠 **ED法（Error Diffusion Method）によるマルチクラス分類システム**

## 📖 概要

ED-ANN（Error Diffusion Artificial Neural Network）は、生物学的な神経伝達メカニズムにヒントを得た「ED法（誤差拡散法）」を用いて、マルチクラス分類問題を解決するニューラルネットワークシステムです。

### 🎯 主な特徴

- **🧬 生物学的学習原理**: アミン濃度による学習制御メカニズム
- **🎯 マルチクラス対応**: 10クラス分類（MNIST）で90%以上の精度
- **⚙️ 柔軟な学習モード**: エポック単位・クラス単位・比較モード
- **📊 詳細な可視化**: リアルタイム学習状況・モデル構造表示
- **🔍 精度検証システム**: CSV出力による厳密な精度検証

## 🚀 クイックスタート

### インストール

```bash
git clone https://github.com/yourusername/ed-ann-v1.0.0.git
cd ed-ann-v1.0.0
pip install -r requirements.txt
```

### 基本実行

```bash
# 基本実行（エポック単位学習）
python ed_ann_simple_v100.py --epochs 5

# 精度検証付き実行
python ed_ann_simple_v100.py --epochs 5 --verify

# クラス単位学習
python ed_ann_simple_v100.py --epochs 5 --mode class

# リアルタイム可視化
python ed_ann_simple_v100.py --epochs 10 --realtime
```

## 📊 性能結果

| 学習モード | 精度 | 成功クラス数 | 特徴 |
|-----------|------|-------------|------|
| エポック単位 | 89.4% | 10/10 | 安定した全クラス学習 |
| クラス単位 | 88.3% | 10/10 | クラス特化型学習 |
| 比較モード | - | - | 両手法の詳細比較 |

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
