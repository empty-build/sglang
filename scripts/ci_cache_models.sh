#!/bin/bash
set -euxo pipefail

models=$(python3 -c "from sglang.test.test_utils import _get_default_models; print(_get_default_models())" | jq -r '.[]')

if [ -z "$models" ]; then
    echo "Failed to get default models."
    exit 1
fi

# 获取 DEFAULT_MODEL_CACHE_DIR 环境变量的值
cache_dir="${DEFAULT_MODEL_CACHE_DIR:-}"

if [ -z "$cache_dir" ]; then
    echo "DEFAULT_MODEL_CACHE_DIR environment variable is not set."
    exit 1
fi

for model in $models; do
    echo "Caching model: $model"
    # 使用 cache_dir 作为下载路径
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$model', local_dir='$cache_dir', local_dir_use_symlinks=False)"
    if [ $? -ne 0 ]; then
        echo "Failed to cache model: $model"
    else
        echo "Successfully cached model: $model"
    fi
done
