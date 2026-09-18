#!/usr/bin/env python3
"""
環境チェックスクリプト
このプロジェクトで Python コードを実行する前に、必ずこのスクリプトを実行してください。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / "environment.yml"
CODEX_DIR = ROOT_DIR / ".codex"
CODEX_CONFIG = CODEX_DIR / "config.toml"

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11+ では不要
    tomllib = None


def parse_args() -> argparse.Namespace:
    """CLI 引数を解析する。"""
    parser = argparse.ArgumentParser(
        description="環境と主要設定ファイルを検証します。",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI モード。conda の有効化を必須にせず、requirements.txt ベースの確認を許可する。",
    )
    return parser.parse_args()


def parse_environment_yml(env_file: Path) -> list[str]:
    """environment.yml から pip パッケージ名の一覧を取得する。"""
    packages = []

    try:
        with open(env_file, "r", encoding="utf-8") as f:
            content = f.read()

        # pip セクションからパッケージを抽出
        pip_section = re.search(r"pip:\s*\n((?:\s+-\s+.*\n?)*)", content)
        if pip_section:
            pip_packages = pip_section.group(1)
            for line in pip_packages.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    package = line[2:]
                    package_name = re.split(r"[\[<>=!~]", package)[0].strip()
                    if package_name:
                        packages.append(package_name)
    except FileNotFoundError:
        print(f"❌ エラー: {env_file.name} が見つかりません")
        return []
    except Exception as exc:
        print(f"❌ エラー: {env_file.name} の解析に失敗: {exc}")
        return []

    return packages


def check_python_env(ci_mode: bool) -> tuple[bool, str]:
    """Python 実行環境を確認する。"""
    print("🔍 Python 実行環境をチェック中...")

    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        print(f"✅ conda 環境がアクティブです（環境名: {conda_env}）")
        if conda_env == "base":
            print("⚠️ 警告: 'base' 環境での開発は推奨されません。専用の仮想環境を使用してください。")
        return True, conda_env

    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        env_name = Path(venv_path).name or venv_path
        print(f"✅ virtualenv / venv がアクティブです（環境名: {env_name}）")
        print("⚠️ ローカル標準は conda ですが、互換環境として続行します。")
        return True, env_name

    if ci_mode:
        print("✅ CI モードのため conda 有効化チェックをスキップします")
        return True, "CI"

    print("❌ エラー: conda 環境または virtualenv / venv がアクティブではありません")
    print("実行例: conda activate <your-env-name>")
    print("⚠️ 注意: README と environment.yml を正本として環境を用意してください")
    return False, "不明"


def check_dependencies(ci_mode: bool) -> bool:
    """依存関係を確認する。"""
    print("🔍 依存関係をチェック中...")

    required_packages = parse_environment_yml(ENV_FILE)
    if not required_packages:
        print(f"❌ エラー: {ENV_FILE.name} からパッケージリストを取得できませんでした")
        return False

    print(f"📦 チェック対象パッケージ数: {len(required_packages)}")
    missing_packages = []

    for package in required_packages:
        try:
            import_name = package.replace("-", "_")

            if package == "python-dateutil":
                import dateutil
            elif package == "python-dotenv":
                import dotenv
            elif package == "typing-extensions":
                import typing_extensions
            elif package == "pyyaml":
                import yaml
            else:
                __import__(import_name)

        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ 不足している依存関係: {', '.join(missing_packages)}")
        if ci_mode:
            print("実行してください: python -m pip install -r requirements.txt")
        else:
            print("実行してください: conda env update -f environment.yml")
        return False

    print("✅ 主要な依存関係がインストールされています")
    return True


def check_environment_files() -> bool:
    """environment.yml の存在を確認する。"""
    print("🔍 environment.yml ファイルをチェック中...")

    if not ENV_FILE.exists():
        print("⚠️ 警告: environment.yml が見つかりません")
        return False

    print("✅ environment.yml ファイルが存在します")
    return True


def load_toml_config(config_path: Path) -> dict:
    """TOML設定ファイルを読み込む。"""
    if tomllib is not None:
        with open(config_path, "rb") as f:
            return tomllib.load(f)

    try:
        import tomli
    except ModuleNotFoundError as exc:  # pragma: no cover - 保険
        raise RuntimeError("TOMLの読み込みに tomllib / tomli が必要です") from exc

    with open(config_path, "rb") as f:
        return tomli.load(f)


def main() -> None:
    """メイン関数。"""
    args = parse_args()
    ci_mode = args.ci or os.environ.get("CI", "").lower() == "true"

    print("🚀 環境チェックを開始します...")
    print("=" * 50)

    env_ok, active_env = check_python_env(ci_mode)
    print()

    env_files_ok = check_environment_files()
    print()

    deps_ok = check_dependencies(ci_mode)
    print()

    print("🔍 Codex 設定をチェック中...")
    if CODEX_DIR.exists() and CODEX_CONFIG.exists():
        try:
            config = load_toml_config(CODEX_CONFIG)
            policy = config.get("approval_policy")
            sandbox = config.get("sandbox_mode")
            model = config.get("model")
            web_search = config.get("web_search")
            network_access = (
                config.get("sandbox_workspace_write", {}) or {}
            ).get("network_access")
            print("✅ .codex/config.toml を検出")
            if model:
                print(f"   • model: {model}")
            if policy:
                print(f"   • approval_policy: {policy}")
            if sandbox:
                print(f"   • sandbox_mode: {sandbox}")
            if web_search:
                print(f"   • web_search: {web_search}")
            if network_access is not None:
                print(f"   • shell network_access: {network_access}")
            if policy == "on-failure":
                print("   ⚠️ approval_policy=on-failure は非推奨です。on-request を検討してください。")
        except Exception as exc:
            print(f"⚠️ .codex/config.toml の読み取りに失敗: {exc}")
    else:
        print("ℹ️ .codex/config.toml が見つかりません。Codex CLI/IDE を使う場合は作成してください。")
    print()

    print("=" * 50)

    if env_ok and deps_ok and env_files_ok:
        print("🎉 環境チェック完了！実行準備OK")
        print(f"✅ 仮想環境: {active_env}")
        print("✅ 依存関係: インストール済み")
        print("✅ 設定ファイル: 存在確認済み")
        sys.exit(0)

    print("⚠️ 環境に問題があります。上記の指示に従って修正してください")
    print()
    print("📋 修正手順:")
    if ci_mode:
        print("1. python -m pip install -r requirements.txt")
        print("2. python utility/check_env.py --ci  # 再チェック")
    else:
        print("1. conda activate <your-env-name>")
        print("2. conda env update -f environment.yml")
        print("3. python utility/check_env.py  # 再チェック")
    print("\n⚠️ 注意: README と environment.yml を正本として運用してください")
    sys.exit(1)

if __name__ == "__main__":
    main()
