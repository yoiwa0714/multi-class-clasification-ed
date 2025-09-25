"""
ED-Genuine 重み管理システム (Weight Management System)
v0.2.4対応 - 学習結果の保存・読み込み・継続機能

Author: GitHub Copilot with ed_genuine.prompt.md compliance
Implementation Date: September 14, 2025
Module: modules/weight_manager.py
"""

import numpy as np
import os
import datetime
from typing import Dict, Any, Optional, Tuple
import warnings

class WeightManager:
    """
    ED法重み管理クラス
    
    機能:
    1. 学習結果の重みの保存機能 (save_weights)
    2. 保存された重みのロード機能 (load_weights)
    3. 互換性チェック・バージョン管理
    4. エラーハンドリング・データ整合性確認
    
    使用方法:
    ```python
    # 保存
    weight_manager = WeightManager()
    weight_manager.save_weights(ed_genuine_instance, "trained_model.npz", metadata={})
    
    # 読み込み
    weight_data = weight_manager.load_weights("trained_model.npz")
    ed_genuine_instance.load_from_weights(weight_data)
    ```
    """
    
    WEIGHT_FILE_VERSION = "1.0.0"
    SUPPORTED_VERSIONS = ["1.0.0"]
    
    def __init__(self, verbose: bool = False):
        """
        WeightManager初期化
        
        Args:
            verbose (bool): 詳細ログ出力フラグ
        """
        self.verbose = verbose
        
    def save_weights(self, 
                    ed_core_instance, 
                    filepath: str, 
                    training_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        ED法重みデータの保存
        
        Args:
            ed_core_instance: EDGenuineインスタンス
            filepath (str): 保存先ファイルパス (.npz形式)
            training_metadata (dict): 学習メタデータ (エポック数、精度など)
            
        Returns:
            bool: 保存成功フラグ
        """
        try:
            # ファイルパス正規化
            if not filepath.endswith('.npz'):
                filepath += '.npz'
            
            # ディレクトリ作成
            os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
            
            # 保存データ構築
            save_data = self._build_save_data(ed_core_instance, training_metadata)
            
            # NumPy .npz形式で保存（圧縮あり）
            np.savez_compressed(filepath, **save_data)
            
            if self.verbose:
                print(f"✅ 重み保存完了: {filepath}")
                print(f"   - バージョン: {self.WEIGHT_FILE_VERSION}")
                print(f"   - ファイルサイズ: {self._get_file_size(filepath):.2f} MB")
                print(f"   - 重み配列形状: {ed_core_instance.output_weights.shape}")
            
            return True
            
        except Exception as e:
            print(f"❌ 重み保存エラー: {e}")
            return False
    
    def load_weights(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        ED法重みデータの読み込み
        
        Args:
            filepath (str): 読み込み元ファイルパス
            
        Returns:
            dict: 重みデータ（読み込み失敗時はNone）
        """
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"重みファイルが見つかりません: {filepath}")
            
            # NumPy .npz読み込み
            loaded_data = np.load(filepath, allow_pickle=True)
            
            # データ構造検証
            weight_data = self._validate_and_extract_data(loaded_data)
            
            if self.verbose:
                print(f"✅ 重み読み込み完了: {filepath}")
                print(f"   - バージョン: {weight_data['metadata']['version']}")
                print(f"   - 保存日時: {weight_data['metadata']['timestamp']}")
                print(f"   - 重み配列形状: {weight_data['weights']['output_weights'].shape}")
            
            return weight_data
            
        except Exception as e:
            print(f"❌ 重み読み込みエラー: {e}")
            return None
    
    def check_compatibility(self, 
                          weight_data: Dict[str, Any], 
                          ed_core_instance) -> Tuple[bool, str]:
        """
        重みデータとEDGenuineインスタンスの互換性チェック
        
        Args:
            weight_data (dict): 重みデータ
            ed_core_instance: EDGenuineインスタンス
            
        Returns:
            tuple: (互換性フラグ, エラーメッセージ)
        """
        try:
            network_info = weight_data['network_structure']
            hyperparams = weight_data['hyperparams']
            
            # ネットワーク構造チェック
            if network_info['max_units'] != ed_core_instance.MAX_UNITS:
                return False, f"MAX_UNITS不整合: 保存値={network_info['max_units']}, 現在値={ed_core_instance.MAX_UNITS}"
            
            if network_info['max_output_neurons'] != ed_core_instance.MAX_OUTPUT_NEURONS:
                return False, f"MAX_OUTPUT_NEURONS不整合: 保存値={network_info['max_output_neurons']}, 現在値={ed_core_instance.MAX_OUTPUT_NEURONS}"
            
            # 隠れ層構造チェック
            saved_hidden = network_info['hidden_layers']
            current_hidden = ed_core_instance.hyperparams.hidden_layers
            if saved_hidden != current_hidden:
                return False, f"隠れ層構造不整合: 保存値={saved_hidden}, 現在値={current_hidden}"
            
            # 重要なハイパーパラメータチェック（警告レベル）
            param_warnings = []
            critical_params = ['sigmoid_threshold', 'diffusion_rate']
            
            for param in critical_params:
                saved_val = hyperparams.get(param)
                current_val = getattr(ed_core_instance.hyperparams, param, None)
                if saved_val is not None and current_val is not None:
                    if abs(saved_val - current_val) > 1e-6:
                        param_warnings.append(f"{param}: 保存値={saved_val}, 現在値={current_val}")
            
            if param_warnings and self.verbose:
                print("⚠️ ハイパーパラメータ差異検出:")
                for warning in param_warnings:
                    print(f"   - {warning}")
                print("   継続実行しますが、学習結果が異なる可能性があります")
            
            return True, "互換性確認完了"
            
        except KeyError as e:
            return False, f"必須データ不足: {e}"
        except Exception as e:
            return False, f"互換性チェックエラー: {e}"
    
    def apply_weights_to_instance(self, 
                                weight_data: Dict[str, Any], 
                                ed_core_instance) -> bool:
        """
        重みデータをEDGenuineインスタンスに適用
        
        Args:
            weight_data (dict): 重みデータ
            ed_core_instance: EDGenuineインスタンス
            
        Returns:
            bool: 適用成功フラグ
        """
        try:
            # 互換性チェック
            compatible, message = self.check_compatibility(weight_data, ed_core_instance)
            if not compatible:
                print(f"❌ 互換性エラー: {message}")
                return False
            
            # 重み配列の適用
            weights = weight_data['weights']
            ed_core_instance.output_weights = weights['output_weights'].copy()
            ed_core_instance.output_inputs = weights['output_inputs'].copy()
            ed_core_instance.output_outputs = weights['output_outputs'].copy()
            ed_core_instance.amine_concentrations = weights['amine_concentrations'].copy()
            
            if self.verbose:
                print("✅ 重みデータ適用完了")
                print(f"   - 出力重み形状: {ed_core_instance.output_weights.shape}")
                print(f"   - 非ゼロ重み数: {np.count_nonzero(ed_core_instance.output_weights)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 重み適用エラー: {e}")
            return False
    
    def _build_save_data(self, 
                        ed_core_instance, 
                        training_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        保存データ構造の構築
        
        Args:
            ed_core_instance: EDGenuineインスタンス
            training_metadata (dict): 学習メタデータ
            
        Returns:
            dict: 保存用データ辞書
        """
        # メタデータ
        metadata = {
            'version': self.WEIGHT_FILE_VERSION,
            'timestamp': datetime.datetime.now().isoformat(),
            'ed_genuine_version': 'v0.2.4',
        }
        
        # 重みデータ
        weights = {
            'output_weights': ed_core_instance.output_weights,
            'output_inputs': ed_core_instance.output_inputs,
            'output_outputs': ed_core_instance.output_outputs,
            'amine_concentrations': ed_core_instance.amine_concentrations,
        }
        
        # ネットワーク構造
        network_structure = {
            'max_units': ed_core_instance.MAX_UNITS,
            'max_output_neurons': ed_core_instance.MAX_OUTPUT_NEURONS,
            'hidden_layers': ed_core_instance.hyperparams.hidden_layers,
        }
        
        # ハイパーパラメータ
        hyperparams = {
            'learning_rate': ed_core_instance.hyperparams.learning_rate,
            'initial_amine': ed_core_instance.hyperparams.initial_amine,
            'diffusion_rate': ed_core_instance.hyperparams.diffusion_rate,
            'sigmoid_threshold': ed_core_instance.hyperparams.sigmoid_threshold,
            'initial_weight_1': ed_core_instance.hyperparams.initial_weight_1,
            'initial_weight_2': ed_core_instance.hyperparams.initial_weight_2,
            'time_loops': getattr(ed_core_instance, 'time_loops', 2),
        }
        
        # 学習履歴（提供されている場合）
        training_history = training_metadata or {}
        
        # フラット化して.npzに保存可能な形式に変換
        save_data = {}
        save_data.update({f'metadata_{k}': v for k, v in metadata.items()})
        save_data.update({f'weights_{k}': v for k, v in weights.items()})
        save_data.update({f'network_{k}': v for k, v in network_structure.items()})
        save_data.update({f'hyperparams_{k}': v for k, v in hyperparams.items()})
        save_data.update({f'history_{k}': v for k, v in training_history.items()})
        
        return save_data
    
    def _validate_and_extract_data(self, loaded_data) -> Dict[str, Any]:
        """
        読み込みデータの検証と構造化
        
        Args:
            loaded_data: NumPy .npz読み込み結果
            
        Returns:
            dict: 構造化された重みデータ
        """
        # バージョンチェック
        version = loaded_data.get('metadata_version', 'unknown')
        if version not in self.SUPPORTED_VERSIONS:
            warnings.warn(f"未サポートバージョン: {version}. 互換性問題が発生する可能性があります。")
        
        # データ構造再構築
        weight_data = {
            'metadata': {},
            'weights': {},
            'network_structure': {},
            'hyperparams': {},
            'training_history': {}
        }
        
        # カテゴリ別データ分離
        for key in loaded_data.files:
            if key.startswith('metadata_'):
                weight_data['metadata'][key[9:]] = loaded_data[key].item() if loaded_data[key].ndim == 0 else loaded_data[key]
            elif key.startswith('weights_'):
                weight_data['weights'][key[8:]] = loaded_data[key]
            elif key.startswith('network_'):
                weight_data['network_structure'][key[8:]] = loaded_data[key].item() if loaded_data[key].ndim == 0 else loaded_data[key]
            elif key.startswith('hyperparams_'):
                weight_data['hyperparams'][key[12:]] = loaded_data[key].item() if loaded_data[key].ndim == 0 else loaded_data[key]
            elif key.startswith('history_'):
                weight_data['training_history'][key[8:]] = loaded_data[key].item() if loaded_data[key].ndim == 0 else loaded_data[key]
        
        # 必須フィールド確認
        required_weights = ['output_weights', 'output_inputs', 'output_outputs', 'amine_concentrations']
        missing_weights = [w for w in required_weights if w not in weight_data['weights']]
        if missing_weights:
            raise ValueError(f"必須重みデータ不足: {missing_weights}")
        
        return weight_data
    
    def _get_file_size(self, filepath: str) -> float:
        """
        ファイルサイズ取得（MB単位）
        
        Args:
            filepath (str): ファイルパス
            
        Returns:
            float: ファイルサイズ（MB）
        """
        try:
            return os.path.getsize(filepath) / (1024 * 1024)
        except OSError:
            return 0.0
    
    def get_weight_summary(self, weight_data: Dict[str, Any]) -> str:
        """
        重みデータのサマリー情報生成
        
        Args:
            weight_data (dict): 重みデータ
            
        Returns:
            str: サマリー文字列
        """
        try:
            metadata = weight_data['metadata']
            network = weight_data['network_structure']
            weights = weight_data['weights']
            history = weight_data.get('training_history', {})
            
            summary = []
            summary.append("=" * 50)
            summary.append("ED法重みデータ サマリー")
            summary.append("=" * 50)
            summary.append(f"バージョン: {metadata.get('version', 'unknown')}")
            summary.append(f"保存日時: {metadata.get('timestamp', 'unknown')}")
            summary.append(f"ED-Genuineバージョン: {metadata.get('ed_genuine_version', 'unknown')}")
            summary.append("")
            
            summary.append("【ネットワーク構造】")
            summary.append(f"最大ユニット数: {network.get('max_units', 'unknown')}")
            summary.append(f"最大出力ニューロン数: {network.get('max_output_neurons', 'unknown')}")
            summary.append(f"隠れ層構造: {network.get('hidden_layers', 'unknown')}")
            summary.append("")
            
            summary.append("【重みデータ】")
            output_weights = weights.get('output_weights')
            if output_weights is not None:
                summary.append(f"出力重み形状: {output_weights.shape}")
                summary.append(f"非ゼロ重み数: {np.count_nonzero(output_weights)}")
                summary.append(f"重み範囲: [{output_weights.min():.6f}, {output_weights.max():.6f}]")
            summary.append("")
            
            if history:
                summary.append("【学習履歴】")
                for key, value in history.items():
                    summary.append(f"{key}: {value}")
                summary.append("")
            
            summary.append("=" * 50)
            
            return "\n".join(summary)
            
        except Exception as e:
            return f"サマリー生成エラー: {e}"


class WeightCommandLineInterface:
    """
    重み管理のコマンドライン・インターフェース
    既存のargparseシステムとの統合クラス
    """
    
    @staticmethod
    def extend_argument_parser(parser):
        """
        既存のargparseパーサーに重み管理オプションを追加
        
        Args:
            parser: argparse.ArgumentParserインスタンス
        """
        weight_group = parser.add_argument_group('重み管理オプション')
        weight_group.add_argument('--save_weights', type=str, metavar='FILEPATH',
                                help='学習完了後に重みを指定ファイルに保存 (例: --save_weights trained_model.npz)')
        weight_group.add_argument('--load_weights', type=str, metavar='FILEPATH',
                                help='指定ファイルから重みを読み込み (例: --load_weights trained_model.npz)')
        weight_group.add_argument('--test_only', action='store_true',
                                help='学習をスキップし、読み込み重みでテストのみ実行 (--load_weightsと併用)')
        weight_group.add_argument('--continue_training', action='store_true',
                                help='読み込み重みから追加学習を継続 (--load_weightsと併用)')
        
    @staticmethod
    def validate_weight_arguments(args) -> Tuple[bool, str]:
        """
        重み管理引数の妥当性検証
        
        Args:
            args: argparse解析結果
            
        Returns:
            tuple: (妥当性フラグ, エラーメッセージ)
        """
        # test_onlyとcontinue_trainingは排他的
        if args.test_only and args.continue_training:
            return False, "--test_only と --continue_training は同時に指定できません"
        
        # test_only・continue_trainingはload_weightsが必須
        if (args.test_only or args.continue_training) and not args.load_weights:
            return False, "--test_only または --continue_training には --load_weights が必要です"
        
        # ファイル存在チェック
        if args.load_weights and not os.path.exists(args.load_weights):
            return False, f"重みファイルが見つかりません: {args.load_weights}"
        
        return True, "引数検証完了"
    
    @staticmethod
    def get_execution_mode(args) -> str:
        """
        実行モードの判定
        
        Args:
            args: argparse解析結果
            
        Returns:
            str: 実行モード ('normal', 'test_only', 'continue_training', 'load_and_train')
        """
        if args.test_only:
            return 'test_only'
        elif args.continue_training:
            return 'continue_training'
        elif args.load_weights and not args.test_only and not args.continue_training:
            return 'load_and_train'
        else:
            return 'normal'


# 使用例とテスト関数
def example_usage():
    """
    WeightManagerの使用例
    """
    print("WeightManager使用例:")
    print("""
    # 1. 学習後の重み保存
    weight_manager = WeightManager(verbose=True)
    training_metadata = {
        'epochs_completed': 10,
        'final_accuracy': 0.95,
        'final_error': 0.05
    }
    weight_manager.save_weights(ed_core, "model.npz", training_metadata)
    
    # 2. 重み読み込み
    weight_data = weight_manager.load_weights("model.npz")
    if weight_data:
        weight_manager.apply_weights_to_instance(weight_data, ed_core)
    
    # 3. コマンドライン引数の利用
    # python ed_v024_simple.py --load_weights model.npz --test_only
    # python ed_v024_simple.py --load_weights model.npz --continue_training --epochs 5
    """)

if __name__ == "__main__":
    example_usage()