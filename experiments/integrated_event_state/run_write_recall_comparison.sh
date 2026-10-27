#!/usr/bin/env bash
# 所属模块：独立写入期召回实验启动器。
# 输入：输出目录、只读来源图、可选配置；输出：运行日志及明确退出状态。
# 不删除目录或历史图。启动后由调用方通知用户路径，不持续轮询后台任务。
set -u
cd "$(dirname "$0")/../.." || exit 2
output_dir="$1"
source_graph="$2"
config_file="${3:-config/Memory_EventState_sample0_rebuild.yaml}"
python_bin="${LICOMEMORY_PYTHON:-/home/cwf/miniconda3/envs/licomemory/bin/python}"
mkdir -p "$output_dir"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
date -u +%FT%TZ > "$output_dir/started_at.txt"
"$python_bin" -u experiments/integrated_event_state/compare_write_recall.py \
  --source "$source_graph" --output "$output_dir" --config "$config_file" \
  --candidate-top-k 5 --audit-concurrency 4
status=$?
printf '%s\n' "$status" > "$output_dir/exit_code.txt"
date -u +%FT%TZ > "$output_dir/finished_at.txt"
exit "$status"
