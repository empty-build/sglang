
export CCACHE_DIR=/mnt/nvmen01/jack/sglang-w8a8-cutlass_cache
export CCACHE_BACKEND=""
export CCACHE_KEEP_LOCAL_STORAGE="TRUE"
unset CCACHE_READONLY
#python -m uv build --wheel -Cbuild-dir=build --color=always .
#make build
pip install -e . --no-build-isolation -vvv
