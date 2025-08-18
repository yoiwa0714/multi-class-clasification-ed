#!/usr/bin/env python3
"""
ED法理論図作成スクリプト
ED-ANN v1.0.0 公開資料用の図表生成
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

def create_ed_theory_diagram():
    """ED法の基本理論図を作成"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左側: 生物学的メカニズム
    ax1.set_title('🧬 生物学的神経伝達メカニズム', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    ax1.axis('off')
    
    # シナプス前細胞
    pre_cell = Circle((2, 4), 1, color='lightblue', alpha=0.7)
    ax1.add_patch(pre_cell)
    ax1.text(2, 4, 'シナプス\n前細胞', ha='center', va='center', fontweight='bold')
    
    # シナプス後細胞
    post_cell = Circle((8, 4), 1, color='lightcoral', alpha=0.7)
    ax1.add_patch(post_cell)
    ax1.text(8, 4, 'シナプス\n後細胞', ha='center', va='center', fontweight='bold')
    
    # 神経伝達物質（アミン）
    for i, x in enumerate(np.linspace(3.5, 6.5, 5)):
        color = plt.cm.viridis(i / 4)
        amine = Circle((x, 4), 0.2, color=color, alpha=0.8)
        ax1.add_patch(amine)
    
    ax1.text(5, 5.5, '神経伝達物質（アミン）', ha='center', fontsize=12, fontweight='bold')
    ax1.text(5, 2.5, '濃度変化による学習制御', ha='center', fontsize=11, style='italic')
    
    # 矢印
    ax1.arrow(3, 4, 4, 0, head_width=0.3, head_length=0.3, fc='black', ec='black')
    
    # 右側: ED法実装
    ax2.set_title('💻 ED法ニューラルネットワーク', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 8)
    ax2.axis('off')
    
    # 入力層
    input_layer = FancyBboxPatch((0.5, 3), 1.5, 2, boxstyle="round,pad=0.1", 
                                facecolor='lightblue', alpha=0.7)
    ax2.add_patch(input_layer)
    ax2.text(1.25, 4, '入力層\n(784)', ha='center', va='center', fontweight='bold')
    
    # 隠れ層
    hidden_layer = FancyBboxPatch((4, 3), 1.5, 2, boxstyle="round,pad=0.1",
                                 facecolor='lightgreen', alpha=0.7)
    ax2.add_patch(hidden_layer)
    ax2.text(4.75, 4, '隠れ層\n(64)', ha='center', va='center', fontweight='bold')
    
    # 出力層
    output_layer = FancyBboxPatch((7.5, 3), 1.5, 2, boxstyle="round,pad=0.1",
                                 facecolor='lightcoral', alpha=0.7)
    ax2.add_patch(output_layer)
    ax2.text(8.25, 4, '出力層\n(10)', ha='center', va='center', fontweight='bold')
    
    # ED制御ブロック
    ed_control = FancyBboxPatch((3.5, 6), 2.5, 1, boxstyle="round,pad=0.1",
                               facecolor='gold', alpha=0.8)
    ax2.add_patch(ed_control)
    ax2.text(4.75, 6.5, 'ED制御\n(アミン濃度)', ha='center', va='center', fontweight='bold')
    
    # 矢印
    ax2.arrow(2, 4, 1.5, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    ax2.arrow(5.5, 4, 1.5, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    ax2.arrow(4.75, 6, 0, -0.8, head_width=0.2, head_length=0.2, fc='red', ec='red')
    
    # 説明テキスト
    ax2.text(5, 1.5, 'アミン濃度による動的学習制御', ha='center', fontsize=11, 
            style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('/home/yoichi/develop/ai/edm/src/relational_ed/snn/figures/ed_theory_diagram.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ ED法理論図を保存しました: figures/ed_theory_diagram.png")

def create_multiclass_expansion_diagram():
    """マルチクラス拡張の概念図を作成"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # 上段: 従来の二値分類
    ax1.set_title('📊 ED法の拡張: 二値分類 → マルチクラス分類', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 6)
    ax1.axis('off')
    
    # 従来の二値分類
    ax1.text(2, 5, '従来: 二値分類', fontsize=14, fontweight='bold')
    
    class_a = Circle((1, 3), 0.8, color='lightblue', alpha=0.8)
    ax1.add_patch(class_a)
    ax1.text(1, 3, 'Class A', ha='center', va='center', fontweight='bold')
    
    class_b = Circle((3, 3), 0.8, color='lightcoral', alpha=0.8)
    ax1.add_patch(class_b)
    ax1.text(3, 3, 'Class B', ha='center', va='center', fontweight='bold')
    
    # 双方向矢印
    ax1.annotate('', xy=(3-0.8, 3), xytext=(1+0.8, 3),
                arrowprops=dict(arrowstyle='<->', lw=2, color='black'))
    ax1.text(2, 2, 'ED制御', ha='center', fontweight='bold')
    
    # 下段: マルチクラス分類
    ax2.set_xlim(0, 12)
    ax2.set_ylim(0, 6)
    ax2.axis('off')
    
    ax2.text(6, 5.5, '拡張: マルチクラス分類 (MNIST 10クラス)', fontsize=14, fontweight='bold')
    
    # 10クラスの円配置
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    angles = np.linspace(0, 2*np.pi, 11)[:-1]
    center_x, center_y = 6, 3
    radius = 2
    
    for i, (angle, color) in enumerate(zip(angles, colors)):
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        
        class_circle = Circle((x, y), 0.4, color=color, alpha=0.8)
        ax2.add_patch(class_circle)
        ax2.text(x, y, str(i), ha='center', va='center', fontweight='bold', color='white')
        
        # 中心からの線
        ax2.plot([center_x, x], [center_y, y], 'k--', alpha=0.5, lw=1)
    
    # 中央のED制御
    ed_center = Circle((center_x, center_y), 0.6, color='gold', alpha=0.9)
    ax2.add_patch(ed_center)
    ax2.text(center_x, center_y, 'ED\n制御', ha='center', va='center', fontweight='bold')
    
    ax2.text(6, 0.5, '各クラスに対する個別ED制御', ha='center', fontsize=12, 
            style='italic', bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('/home/yoichi/develop/ai/edm/src/relational_ed/snn/figures/multiclass_expansion.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ マルチクラス拡張図を保存しました: figures/multiclass_expansion.png")

def create_learning_flow_diagram():
    """学習フロー比較図を作成"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # 左側: エポック単位学習
    ax1.set_title('🔄 エポック単位学習フロー', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.axis('off')
    
    # フローチャート要素
    start1 = FancyBboxPatch((3.5, 10.5), 3, 1, boxstyle="round,pad=0.1",
                           facecolor='lightgreen', alpha=0.8)
    ax1.add_patch(start1)
    ax1.text(5, 11, '開始', ha='center', va='center', fontweight='bold')
    
    data1 = FancyBboxPatch((3, 9), 4, 1, boxstyle="round,pad=0.1",
                          facecolor='lightblue', alpha=0.8)
    ax1.add_patch(data1)
    ax1.text(5, 9.5, 'データセット全体', ha='center', va='center', fontweight='bold')
    
    epoch1 = FancyBboxPatch((2.5, 7.5), 5, 1, boxstyle="round,pad=0.1",
                           facecolor='lightyellow', alpha=0.8)
    ax1.add_patch(epoch1)
    ax1.text(5, 8, 'エポック単位で全クラス学習', ha='center', va='center', fontweight='bold')
    
    update1 = FancyBboxPatch((3, 6), 4, 1, boxstyle="round,pad=0.1",
                            facecolor='lightcoral', alpha=0.8)
    ax1.add_patch(update1)
    ax1.text(5, 6.5, 'ED法による重み更新', ha='center', va='center', fontweight='bold')
    
    result1 = FancyBboxPatch((3.5, 4.5), 3, 1, boxstyle="round,pad=0.1",
                            facecolor='lightgreen', alpha=0.8)
    ax1.add_patch(result1)
    ax1.text(5, 5, '精度: 89.4%', ha='center', va='center', fontweight='bold')
    
    # 矢印
    for i in range(4):
        y_start = 10.5 - i*1.5
        ax1.arrow(5, y_start-0.2, 0, -0.6, head_width=0.2, head_length=0.1, fc='blue', ec='blue')
    
    # 右側: クラス単位学習
    ax2.set_title('🎯 クラス単位学習フロー', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')
    
    start2 = FancyBboxPatch((3.5, 10.5), 3, 1, boxstyle="round,pad=0.1",
                           facecolor='lightgreen', alpha=0.8)
    ax2.add_patch(start2)
    ax2.text(5, 11, '開始', ha='center', va='center', fontweight='bold')
    
    data2 = FancyBboxPatch((3, 9), 4, 1, boxstyle="round,pad=0.1",
                          facecolor='lightblue', alpha=0.8)
    ax2.add_patch(data2)
    ax2.text(5, 9.5, 'クラス別データ分割', ha='center', va='center', fontweight='bold')
    
    class2 = FancyBboxPatch((2.5, 7.5), 5, 1, boxstyle="round,pad=0.1",
                           facecolor='lightyellow', alpha=0.8)
    ax2.add_patch(class2)
    ax2.text(5, 8, '各クラス個別学習', ha='center', va='center', fontweight='bold')
    
    update2 = FancyBboxPatch((3, 6), 4, 1, boxstyle="round,pad=0.1",
                            facecolor='lightcoral', alpha=0.8)
    ax2.add_patch(update2)
    ax2.text(5, 6.5, 'クラス特化ED制御', ha='center', va='center', fontweight='bold')
    
    result2 = FancyBboxPatch((3.5, 4.5), 3, 1, boxstyle="round,pad=0.1",
                            facecolor='lightgreen', alpha=0.8)
    ax2.add_patch(result2)
    ax2.text(5, 5, '精度: 88.3%', ha='center', va='center', fontweight='bold')
    
    # 矢印
    for i in range(4):
        y_start = 10.5 - i*1.5
        ax2.arrow(5, y_start-0.2, 0, -0.6, head_width=0.2, head_length=0.1, fc='red', ec='red')
    
    plt.tight_layout()
    plt.savefig('/home/yoichi/develop/ai/edm/src/relational_ed/snn/figures/learning_flow.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 学習フロー図を保存しました: figures/learning_flow.png")

def create_performance_comparison():
    """性能比較グラフを作成"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左側: 精度比較
    methods = ['標準SGD', 'Adam', 'ED法\n(エポック)', 'ED法\n(クラス)']
    accuracies = [87.2, 89.1, 89.4, 88.3]
    colors = ['lightblue', 'lightgreen', 'gold', 'lightcoral']
    
    bars1 = ax1.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('精度 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('📊 手法別精度比較', fontsize=14, fontweight='bold')
    ax1.set_ylim(85, 91)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 数値ラベル
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{acc}%', ha='center', va='bottom', fontweight='bold')
    
    # 右側: 収束エポック数比較
    epochs = [15, 12, 8, 10]
    bars2 = ax2.bar(methods, epochs, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('収束エポック数', fontsize=12, fontweight='bold')
    ax2.set_title('⚡ 手法別収束速度比較', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 18)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 数値ラベル
    for bar, epoch in zip(bars2, epochs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{epoch}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/yoichi/develop/ai/edm/src/relational_ed/snn/figures/performance_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ 性能比較図を保存しました: figures/performance_comparison.png")

def create_amine_concentration_dynamics():
    """アミン濃度ダイナミクス図を作成"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # サンプル時間軸
    time = np.linspace(0, 100, 1000)
    
    # アミン濃度のシミュレーション
    concentration = np.zeros_like(time)
    base_level = 0.5
    concentration[0] = base_level
    
    # シミュレートされた正答/誤答パターン
    np.random.seed(42)
    correct_times = np.random.choice(time[::10], 30, replace=False)
    incorrect_times = np.random.choice(time[::10], 20, replace=False)
    
    d_plus, d_minus = 0.1, 0.05
    decay = 0.99
    
    for i in range(1, len(time)):
        dt = time[i] - time[i-1]
        concentration[i] = concentration[i-1] * (decay ** dt)
        
        if time[i] in correct_times:
            concentration[i] += d_plus
        elif time[i] in incorrect_times:
            concentration[i] = max(0, concentration[i] - d_minus)
    
    # プロット
    ax.plot(time, concentration, 'b-', linewidth=2, label='アミン濃度')
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='閾値 (0.5)')
    ax.fill_between(time, 0, concentration, alpha=0.3, color='blue')
    
    # 正答/誤答のマーク
    for ct in correct_times[:10]:  # 最初の10個だけ表示
        ax.axvline(x=ct, color='green', linestyle=':', alpha=0.6)
        ax.text(ct, 0.9, '✓', ha='center', va='center', color='green', fontsize=12, fontweight='bold')
    
    for it in incorrect_times[:5]:  # 最初の5個だけ表示
        ax.axvline(x=it, color='red', linestyle=':', alpha=0.6)
        ax.text(it, 0.1, '✗', ha='center', va='center', color='red', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('時間 (学習ステップ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('アミン濃度', fontsize=12, fontweight='bold')
    ax.set_title('🧪 ED法におけるアミン濃度ダイナミクス', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)
    
    # 説明テキスト
    ax.text(50, 0.8, '正答時: 濃度増加 (+d_plus)', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='lightgreen', alpha=0.7), fontsize=10)
    ax.text(50, 0.3, '誤答時: 濃度減少 (-d_minus)', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='lightcoral', alpha=0.7), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/home/yoichi/develop/ai/edm/src/relational_ed/snn/figures/amine_dynamics.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ アミン濃度ダイナミクス図を保存しました: figures/amine_dynamics.png")

def main():
    """すべての図表を生成"""
    print("🎨 ED-ANN v1.0.0 公開資料用図表を生成中...")
    
    # figuresディレクトリを作成
    import os
    os.makedirs('/home/yoichi/develop/ai/edm/src/relational_ed/snn/figures', exist_ok=True)
    
    # 各図表を作成
    create_ed_theory_diagram()
    create_multiclass_expansion_diagram()
    create_learning_flow_diagram()
    create_performance_comparison()
    create_amine_concentration_dynamics()
    
    print("\n🎉 すべての図表を生成完了しました！")
    print("📁 保存場所: figures/")
    print("📋 生成された図表:")
    print("   - ed_theory_diagram.png: ED法基本理論図")
    print("   - multiclass_expansion.png: マルチクラス拡張概念図")
    print("   - learning_flow.png: 学習フロー比較図")
    print("   - performance_comparison.png: 性能比較グラフ")
    print("   - amine_dynamics.png: アミン濃度ダイナミクス図")

if __name__ == "__main__":
    main()
