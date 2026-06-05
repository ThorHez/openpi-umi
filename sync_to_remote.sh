#!/bin/bash
# 后台批量 scp 传输脚本
# 用法: bash sync_to_remote.sh
# 传输完成后日志写入 /tmp/scp_sync_YYYYMMDD.log

PASSWORD="test@1811"   # <-- 改这里
REMOTE="root@115.190.212.169"
REMOTE_DIR="/data2/hzl_workspace_for_pi/openpi-umi/data"
SRC_DIR="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd"  # 源文件夹所在目录
LOG="/tmp/scp_sync_$(date +%Y%m%d_%H%M%S).log"

FOLDERS=(
    "hitl_replay_buffer_0510"
    "hitl_replay_buffer_0511"
    "hitl_replay_buffer_0511_afternoon"
)

echo "=== SCP 批量传输启动 $(date) ===" | tee "$LOG"
echo "日志文件: $LOG"

PIDS=()
for folder in "${FOLDERS[@]}"; do
    src="$SRC_DIR/$folder"
    if [ ! -d "$src" ]; then
        echo "[SKIP] $src 不存在，跳过" | tee -a "$LOG"
        continue
    fi
    folder_log="/tmp/scp_${folder}.log"
    echo "[START] $folder -> $REMOTE:$REMOTE_DIR/ (子日志: $folder_log)" | tee -a "$LOG"
    nohup sshpass -p "$PASSWORD" scp -r -o StrictHostKeyChecking=no \
        "$src" "$REMOTE:$REMOTE_DIR/" \
        > "$folder_log" 2>&1 &
    pid=$!
    PIDS+=($pid)
    echo "  PID: $pid" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "所有传输已在后台启动，PID: ${PIDS[*]}" | tee -a "$LOG"
echo "监控进度: tail -f $LOG"
echo "查看传输进程: ps aux | grep scp"
echo ""

# 等待所有后台任务完成
for pid in "${PIDS[@]}"; do
    wait "$pid"
    code=$?
    if [ $code -eq 0 ]; then
        echo "[DONE] PID $pid 传输成功" | tee -a "$LOG"
    else
        echo "[FAIL] PID $pid 传输失败，退出码=$code" | tee -a "$LOG"
    fi
done

echo "" | tee -a "$LOG"
echo "=== 全部传输完成 $(date) ===" | tee -a "$LOG"
