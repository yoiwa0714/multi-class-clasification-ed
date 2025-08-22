#!/bin/bash

# GitHubリポジトリへのアップロード手順

echo "=== ed-annリポジトリ更新手順 ==="
echo ""
echo "1. GitHubに手動でアップロードする場合："
echo "   - https://github.com/yoiwa0714/ed-ann にアクセス"
echo "   - 'Add file' > 'Upload files' を選択"
echo "   - 以下のファイルをドラッグ&ドロップ："
echo ""
echo "   ✓ README.md (ルートディレクトリ)"
echo "   ✓ docs/multiclass_ed_comprehensive_explanation.md"
echo "   ✓ docs/通常のEDネットワーク例.png"  
echo "   ✓ docs/MNIST対応 マルチクラスEDネットワーク.png"
echo "   ✓ docs/クラス別学習 学習順序.png"
echo "   ✓ docs/エポック順学習 学習順序.png"
echo ""
echo "2. Git コマンドでアップロードする場合："
echo "   cd ed-ann-update"
echo "   git init"
echo "   git remote add origin https://github.com/yoiwa0714/ed-ann.git"
echo "   git add ."
echo "   git commit -m \"Update README and add comprehensive documentation\""
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "準備完了したファイル："
ls -la
echo ""
echo "docsディレクトリの内容："
ls -la docs/
