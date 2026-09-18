#!/usr/bin/env bash

# プロジェクトセットアップスクリプト
# environment.yml の name を読み取り、ワークスペース内の
# .conda/envs/<name> に conda 環境を作成または更新します。

set -euo pipefail

print_info() {
    echo -e "\033[1;34mℹ️  $1\033[0m"
}

print_success() {
    echo -e "\033[1;32m✅ $1\033[0m"
}

print_warning() {
    echo -e "\033[1;33m⚠️  $1\033[0m"
}

print_error() {
    echo -e "\033[1;31m❌ $1\033[0m"
}

print_step() {
    echo -e "\033[1;36m📋 $1\033[0m"
}

parse_env_name() {
    awk -F': *' '/^name:/{print $2; exit}' "$1"
}

main() {
    local root_dir env_file env_name env_root env_path

    root_dir="$(cd "$(dirname "$0")/.." && pwd)"
    env_file="${root_dir}/environment.yml"

    echo "🚀 プロジェクト起動スクリプトを開始します..."
    echo "============================================================"

    print_step "ステップ 1: conda の確認"
    if ! command -v conda >/dev/null 2>&1; then
        print_error "conda がインストールされていません"
        echo "   Anaconda または Miniconda をインストールしてください"
        echo "   https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    print_success "conda がインストールされています"
    echo

    print_step "ステップ 2: environment.yml の確認"
    if [[ ! -f "${env_file}" ]]; then
        print_error "environment.yml が見つかりません"
        exit 1
    fi

    env_name="$(parse_env_name "${env_file}")"
    if [[ -z "${env_name}" ]]; then
        print_error "environment.yml から環境名を取得できませんでした"
        exit 1
    fi

    env_root="${root_dir}/.conda/envs"
    env_path="${env_root}/${env_name}"
    mkdir -p "${env_root}"

    print_success "environment.yml が存在します"
    print_success "環境名: ${env_name}"
    print_success "作成先: ${env_path}"
    echo

    print_step "ステップ 3: conda 環境の確認・作成・更新"
    if [[ ! -d "${env_path}" ]]; then
        print_info "環境 '${env_name}' が存在しません。ワークスペース内に作成します..."
        if conda env create -p "${env_path}" -f "${env_file}"; then
            print_success "環境 '${env_name}' を作成しました"
        else
            print_error "環境の作成に失敗しました"
            exit 1
        fi
    else
        print_success "環境 '${env_name}' が存在します"
        print_info "environment.yml で環境を更新します..."
        if conda env update -p "${env_path}" -f "${env_file}"; then
            print_success "環境を更新しました"
        else
            print_error "環境の更新に失敗しました"
            exit 1
        fi
    fi
    echo

    print_step "ステップ 4: 環境のアクティベート"
    print_info "次のコマンドを実行してください:"
    echo "   conda activate \"${env_path}\""
    echo

    print_step "ステップ 5: 環境チェック"
    if [[ -f "${root_dir}/utility/check_env.py" ]]; then
        if conda run -p "${env_path}" python "${root_dir}/utility/check_env.py"; then
            print_success "環境チェックが完了しました"
        else
            print_warning "環境チェックで問題が検出されました"
            echo "   手動で確認してください: conda run -p \"${env_path}\" python utility/check_env.py"
        fi
    else
        print_warning "utility/check_env.py が見つかりません。スキップします。"
    fi
    echo

    echo "============================================================"
    print_success "セットアップ完了！"
    echo
    echo "📋 次のステップ:"
    echo "1. conda activate \"${env_path}\""
    echo "2. python utility/check_env.py  # 環境チェック"
    echo "3. プロジェクト固有のコマンドを実行"
    echo
    echo "💡 ヒント:"
    echo "   - 環境を非アクティブにする: conda deactivate"
    echo "   - 環境を削除する: conda env remove -p \"${env_path}\""
    echo "   - 環境を更新する: conda env update -p \"${env_path}\" -f environment.yml"
    echo "   - 利用可能な環境を確認: conda env list"
}

main "$@"
