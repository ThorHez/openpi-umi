#!/bin/bash
# 完整数据集分析脚本 - 一键运行所有分析

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 打印函数
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <dataset_path> [output_dir]"
    echo ""
    echo "示例:"
    echo "  $0 /data/umi_lerobot_dataset_v3"
    echo "  $0 /data/umi_lerobot_dataset_v3 ./my_analysis_output"
    exit 1
fi

DATASET_PATH=$1
OUTPUT_DIR=${2:-"${DATASET_PATH}/full_analysis_$(date +%Y%m%d_%H%M%S)"}

# 检查数据集是否存在
if [ ! -d "$DATASET_PATH" ]; then
    print_error "数据集路径不存在: $DATASET_PATH"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
print_success "创建输出目录: $OUTPUT_DIR"

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 开始分析
print_header "开始完整数据集分析"
echo "数据集: $DATASET_PATH"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 1. 快速统计
print_header "1/3 快速统计"
python "$SCRIPT_DIR/quick_stats.py" "$DATASET_PATH" --detailed | tee "$OUTPUT_DIR/quick_stats.txt"
print_success "快速统计完成"

# 2. 综合分析
print_header "2/3 综合分析和可视化"
python "$SCRIPT_DIR/analyze_dataset.py" "$DATASET_PATH" --output "$OUTPUT_DIR/analysis_report"
print_success "综合分析完成"

# 3. 数据质量检查
print_header "3/3 数据质量检查"
python "$SCRIPT_DIR/check_data_quality.py" "$DATASET_PATH" --sample-size 100 | tee "$OUTPUT_DIR/quality_check.txt"

# 移动质量报告
if [ -f "${DATASET_PATH}/data_quality_report.json" ]; then
    mv "${DATASET_PATH}/data_quality_report.json" "$OUTPUT_DIR/"
    print_success "数据质量检查完成"
fi

# 生成汇总报告
print_header "生成汇总报告"

SUMMARY_FILE="$OUTPUT_DIR/SUMMARY.txt"

{
    echo "========================================"
    echo "数据集分析汇总报告"
    echo "========================================"
    echo ""
    echo "分析时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "数据集: $DATASET_PATH"
    echo "输出目录: $OUTPUT_DIR"
    echo ""
    echo "========================================"
    echo "生成的文件"
    echo "========================================"
    echo ""
    echo "1. 文本报告:"
    echo "   - quick_stats.txt         快速统计结果"
    echo "   - quality_check.txt       质量检查结果"
    echo "   - data_quality_report.json  质量问题详情(JSON)"
    echo ""
    echo "2. 可视化图表 (analysis_report/):"
    echo "   - episode_lengths.png     Episode长度分布"
    echo "   - state_distribution.png  状态分布"
    echo "   - action_distribution.png 动作分布"
    echo "   - trajectory_samples.png  轨迹样本"
    echo ""
    echo "3. 数据文件 (analysis_report/):"
    echo "   - episode_stats.csv       Episode统计"
    echo "   - state_stats.csv         状态统计"
    echo "   - action_stats.csv        动作统计"
    echo ""
    echo "========================================"
    echo "快速信息摘要"
    echo "========================================"
    echo ""
    
    # 从quick_stats.txt提取关键信息
    if [ -f "$OUTPUT_DIR/quick_stats.txt" ]; then
        grep -E "(总Episodes|总帧数|平均长度|总时长|Parquet文件总大小)" "$OUTPUT_DIR/quick_stats.txt" || true
    fi
    
    echo ""
    echo "========================================"
    echo "质量检查摘要"
    echo "========================================"
    echo ""
    
    # 从quality_check.txt提取摘要
    if [ -f "$OUTPUT_DIR/quality_check.txt" ]; then
        grep -A 10 "数据质量检查总结" "$OUTPUT_DIR/quality_check.txt" || true
    fi
    
    echo ""
    echo "========================================"
    echo "下一步建议"
    echo "========================================"
    echo ""
    echo "1. 查看可视化图表:"
    echo "   cd $OUTPUT_DIR/analysis_report"
    echo "   # 使用图片查看器打开 *.png 文件"
    echo ""
    echo "2. 查看详细统计数据:"
    echo "   cd $OUTPUT_DIR/analysis_report"
    echo "   # 使用Excel或pandas查看 *.csv 文件"
    echo ""
    echo "3. 如果发现问题，查看详细报告:"
    echo "   cat $OUTPUT_DIR/data_quality_report.json"
    echo ""
    
} > "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

# 完成
print_header "分析完成！"
print_success "所有结果保存在: $OUTPUT_DIR"
print_success "查看汇总报告: $OUTPUT_DIR/SUMMARY.txt"

echo ""
echo "主要输出文件:"
echo "  📄 $OUTPUT_DIR/SUMMARY.txt"
echo "  📄 $OUTPUT_DIR/quick_stats.txt"
echo "  📄 $OUTPUT_DIR/quality_check.txt"
echo "  📊 $OUTPUT_DIR/analysis_report/*.png"
echo "  📊 $OUTPUT_DIR/analysis_report/*.csv"
echo ""

