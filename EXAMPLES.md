# ED-ANN 実行例とパラメータ詳細

## 🚀 基本的な実行例

### 1. シンプル実行

最もシンプルな実行例：

```bash
# デフォルト設定での実行
python ed_ann_simple_v100.py

# 出力例:
🚀 ED-ANN v1.0.0 - モデル・ハイパーパラメータ情報表示版
================================================================================
📋 ED-ANN Model 構造情報
================================================================================
Layer (type)              Output Shape         Param #        
--------------------------------------------------------------------------------
入力層 (Linear)              (None, 64)           50,240         
隠れ層(1) (ReLU)             (None, 64)           0              
出力層 (Linear)              (None, 10)           650            
================================================================================
Total params: 50,890
```

### 2. エポック数指定

```bash
# 10エポックでの学習
python ed_ann_simple_v100.py --epochs 10

# より長期の学習
python ed_ann_simple_v100.py --epochs 20
```

### 3. 精度検証付き実行

```bash
# 精度検証機能を有効化（5エポック以上必須）
python ed_ann_simple_v100.py --epochs 5 --verify

# 出力例:
================================================================================
🔍 ED-ANN 予測精度検証レポート
================================================================================
📊 検証ファイル: ed_ann_predictions_epoch_5ep_20250817_120000.csv
📈 サンプル数: 9,786
🎯 精度検証:
   表示精度: 0.904251
   計算精度: 0.904251
   精度一致: ✅ YES
🔍 検証結果: ✅ PASS
```

## 🎯 学習モード別実行例

### エポック単位学習

全クラスを同時に学習する標準的なアプローチ：

```bash
# エポック単位学習（デフォルト）
python ed_ann_simple_v100.py --epochs 10 --mode epoch

# 特徴:
# - 高速な収束
# - 安定した全クラス学習
# - 平均精度: 89.4%
```

### クラス単位学習

各クラスを個別に学習するアプローチ：

```bash
# クラス単位学習
python ed_ann_simple_v100.py --epochs 10 --mode class

# 特徴:
# - クラス特化型学習
# - バランスの取れた精度
# - 平均精度: 88.3%
```

### 比較モード

両手法を同時実行して比較：

```bash
# 比較実行
python ed_ann_simple_v100.py --epochs 10 --mode both

# 出力例:
============================================================
🔄 エポック単位学習実行
============================================================
=== エポック単位学習結果 ===
全体精度: 0.8940

============================================================
🎯 クラス単位学習実行
============================================================
=== 統合システム評価開始 ===
全体精度: 0.8831
```

## 🎨 可視化オプション

### リアルタイム学習可視化

```bash
# リアルタイムグラフ表示
python ed_ann_simple_v100.py --epochs 15 --realtime

# 機能:
# - 学習曲線のリアルタイム更新
# - 精度・損失の動的表示
# - 学習進捗の視覚的確認
```

### 詳細ログ出力

```bash
# 詳細な学習ログを表示
python ed_ann_simple_v100.py --epochs 10 --verbose

# 出力内容:
# - 各エポックの詳細統計
# - 重み更新の詳細情報
# - ED法パラメータの変化
```

## ⚙️ ハードウェア制御

### CPU強制使用

```bash
# GPU環境でもCPUを強制使用
python ed_ann_simple_v100.py --epochs 10 --cpu

# 用途:
# - デバッグ時の再現性確保
# - GPU環境でのCPU性能テスト
# - メモリ制約がある場合
```

### GPU自動選択（デフォルト）

```bash
# GPU自動検出・使用
python ed_ann_simple_v100.py --epochs 10

# 環境:
# - CUDA利用可能 → GPU使用
# - CUDA利用不可 → CPU使用
```

## 🔧 パラメータカスタマイズ

### 学習率調整

```bash
# 学習率を0.005に設定
python ed_ann_simple_v100.py --epochs 10 --learning_rate 0.005

# 効果:
# - 低学習率 → 安定だが収束が遅い
# - 高学習率 → 高速だが不安定になる可能性
```

### バッチサイズ調整

```bash
# バッチサイズを64に設定
python ed_ann_simple_v100.py --epochs 10 --batch_size 64

# 効果:
# - 大きなバッチ → 安定した勾配、多メモリ使用
# - 小さなバッチ → ノイジーな勾配、少メモリ使用
```

### 隠れ層サイズ調整

```bash
# 隠れ層を128ニューロンに設定
python ed_ann_simple_v100.py --epochs 10 --hidden_size 128

# 効果:
# - 大きなサイズ → 高表現力、多パラメータ
# - 小さなサイズ → 軽量、高速学習
```

### ランダムシード固定

```bash
# 再現性のためのシード固定
python ed_ann_simple_v100.py --epochs 10 --random_seed 12345

# 効果:
# - 結果の完全再現
# - デバッグ時の一貫性
```

## 📊 高度な実行例

### 高精度設定

最高精度を目指す設定：

```bash
python ed_ann_simple_v100.py \
  --epochs 20 \
  --learning_rate 0.005 \
  --batch_size 32 \
  --hidden_size 128 \
  --mode epoch \
  --realtime \
  --verify
```

### 高速実験設定

素早い実験用設定：

```bash
python ed_ann_simple_v100.py \
  --epochs 5 \
  --batch_size 64 \
  --hidden_size 32 \
  --mode epoch
```

### 詳細分析設定

詳細な分析用設定：

```bash
python ed_ann_simple_v100.py \
  --epochs 15 \
  --mode both \
  --verbose \
  --realtime \
  --verify \
  --random_seed 42
```

## 📈 出力ファイル

### CSV検証ファイル

`--verify`オプション使用時に生成：

```
ファイル名形式: ed_ann_predictions_{mode}_{epochs}ep_{timestamp}.csv
内容: 実際の予測値と正解ラベルの詳細記録
用途: 精度検証、エラー分析、詳細統計
```

### ファイル構造例

```csv
true_label,predicted_label,prediction_prob_0,prediction_prob_1,...,prediction_prob_9
7,7,0.001,0.002,0.001,0.001,0.002,0.001,0.001,0.985,0.003,0.003
2,2,0.001,0.001,0.994,0.001,0.001,0.001,0.001,0.001,0.001,0.001
...
```

## 🚨 エラーハンドリング

### よくあるエラーと対処法

#### 1. エポック数不足エラー

```bash
python ed_ann_simple_v100.py --epochs 3 --verify
# エラー: 精度検証機能を使用する場合は5エポック以上のエポック数を指定してください。
# 解決: --epochs 5 以上を指定
```

#### 2. メモリ不足エラー

```bash
# 対処法: バッチサイズを小さくする
python ed_ann_simple_v100.py --epochs 10 --batch_size 16 --cpu
```

#### 3. CUDA関連エラー

```bash
# 対処法: CPU強制使用
python ed_ann_simple_v100.py --epochs 10 --cpu
```

## 🎯 使用場面別推奨設定

### 研究・開発用

```bash
# 詳細な分析が必要な場合
python ed_ann_simple_v100.py --epochs 15 --mode both --verbose --verify --realtime
```

### 教育・デモ用

```bash
# 素早い結果表示が必要な場合
python ed_ann_simple_v100.py --epochs 5 --realtime
```

### 本番・評価用

```bash
# 厳密な性能評価が必要な場合
python ed_ann_simple_v100.py --epochs 20 --verify --random_seed 42
```

## 📋 パラメータ一覧表

| パラメータ | デフォルト | 範囲 | 効果 |
|-----------|-----------|------|------|
| `--epochs` | 3 | 1-∞ | 学習反復回数 |
| `--learning_rate` | 0.01 | 0.0001-0.1 | 学習速度制御 |
| `--batch_size` | 32 | 1-1000 | 一度に処理するサンプル数 |
| `--hidden_size` | 64 | 16-512 | 隠れ層ニューロン数 |
| `--random_seed` | 42 | 0-∞ | 再現性制御 |
| `--mode` | epoch | epoch/class/both | 学習モード |

---

**🚀 ED-ANN: 柔軟な設定で様々な用途に対応**
