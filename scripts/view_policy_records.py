#!/usr/bin/env python3
"""查看 policy_records 文件夹中的内容。

用法:
    python scripts/view_policy_records.py                    # 查看所有记录的摘要
    python scripts/view_policy_records.py --step 0           # 查看特定 step 的详细内容
    python scripts/view_policy_records.py --step 0 --keys    # 只显示 keys
    python scripts/view_policy_records.py --latest           # 查看最新的记录
    python scripts/view_policy_records.py --plot             # 绘制 action 轨迹
    python scripts/view_policy_records.py --camera           # 显示摄像头画面（所有 step）
    python scripts/view_policy_records.py --camera --step 0  # 显示特定 step 的摄像头画面
    python scripts/view_policy_records.py --camera --video   # 播放摄像头视频
"""

import argparse
from pathlib import Path
import numpy as np


def get_record_files(records_dir: Path) -> list[tuple[int, Path]]:
    """获取所有记录文件，按 step 排序。"""
    files = []
    for f in records_dir.glob("step_*.npy"):
        try:
            step = int(f.stem.split("_")[1])
            files.append((step, f))
        except (ValueError, IndexError):
            continue
    return sorted(files, key=lambda x: x[0])


def print_summary(records_dir: Path):
    """打印记录摘要。"""
    files = get_record_files(records_dir)
    
    if not files:
        print(f"No records found in {records_dir}")
        return
    
    print("=" * 60)
    print("Policy Records Summary")
    print(f"Directory: {records_dir}")
    print("=" * 60)
    
    total_size = sum(f.stat().st_size for _, f in files)
    print(f"Total files: {len(files)}")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
    print(f"Step range: {files[0][0]} - {files[-1][0]}")
    print("-" * 60)
    
    # 加载第一个文件看看结构
    first_data = np.load(files[0][1], allow_pickle=True).item()
    print(f"\nData structure (from step {files[0][0]}):")
    print_dict_structure(first_data, indent=2)


def print_dict_structure(data, indent=0, max_depth=3, current_depth=0):
    """递归打印字典结构。"""
    if current_depth >= max_depth:
        print(" " * indent + "...")
        return
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}: (dict)")
                print_dict_structure(value, indent + 2, max_depth, current_depth + 1)
            elif isinstance(value, np.ndarray):
                print(" " * indent + f"{key}: ndarray, shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(" " * indent + f"{key}: {type(value).__name__}, len={len(value)}")
            else:
                print(" " * indent + f"{key}: {type(value).__name__} = {repr(value)[:50]}")
    elif isinstance(data, np.ndarray):
        print(" " * indent + f"ndarray, shape={data.shape}, dtype={data.dtype}")
    else:
        print(" " * indent + f"{type(data).__name__}")


def view_step(records_dir: Path, step: int, show_keys_only: bool = False):
    """查看特定 step 的详细内容。"""
    file_path = records_dir / f"step_{step}.npy"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    data = np.load(file_path, allow_pickle=True).item()
    
    print("=" * 60)
    print(f"Step {step} Details")
    print(f"File: {file_path}")
    print(f"Size: {file_path.stat().st_size / 1024:.2f} KB")
    print("=" * 60)
    
    if show_keys_only:
        print("\nKeys:")
        print_keys(data)
    else:
        print("\nContent:")
        print_dict_content(data)


def print_keys(data, prefix=""):
    """打印所有 keys。"""
    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(value, dict):
                print_keys(value, full_key)
            else:
                print(f"  {full_key}")


def print_dict_content(data, indent=0, max_array_elements=5):
    """打印字典内容。"""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                print_dict_content(value, indent + 2, max_array_elements)
            elif isinstance(value, np.ndarray):
                print(" " * indent + f"{key}: shape={value.shape}, dtype={value.dtype}")
                if value.size <= max_array_elements * 2:
                    print(" " * indent + f"  values: {value}")
                else:
                    flat = value.flatten()
                    print(" " * indent + f"  first {max_array_elements}: {flat[:max_array_elements]}")
                    print(" " * indent + f"  last {max_array_elements}: {flat[-max_array_elements:]}")
                    print(" " * indent + f"  min: {value.min():.6f}, max: {value.max():.6f}, mean: {value.mean():.6f}")
            else:
                print(" " * indent + f"{key}: {repr(value)[:100]}")


def view_latest(records_dir: Path):
    """查看最新的记录。"""
    files = get_record_files(records_dir)
    if not files:
        print(f"No records found in {records_dir}")
        return
    
    latest_step, _ = files[-1]
    view_step(records_dir, latest_step)


def plot_actions(records_dir: Path, start_step: int = 0, end_step: int = -1):
    """绘制 action 轨迹。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return
    
    files = get_record_files(records_dir)
    if not files:
        print(f"No records found in {records_dir}")
        return
    
    if end_step < 0:
        end_step = files[-1][0]
    
    # 收集 actions
    steps = []
    actions_list = []
    
    for step, file_path in files:
        if start_step <= step <= end_step:
            data = np.load(file_path, allow_pickle=True).item()
            if "actions" in data:
                steps.append(step)
                actions = data["actions"]
                if isinstance(actions, np.ndarray):
                    actions_list.append(actions.flatten()[:10])  # 只取前 10 维
    
    if not actions_list:
        print("No actions found in records")
        return
    
    actions_array = np.array(actions_list)
    
    # 绘图
    _, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 上图：所有维度
    ax1 = axes[0]
    for i in range(min(actions_array.shape[1], 10)):
        ax1.plot(steps, actions_array[:, i], label=f"dim_{i}")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Action Value")
    ax1.set_title("Action Trajectory (all dimensions)")
    ax1.legend(loc="upper right", ncol=5, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 下图：位置 (前3维) 和姿态 (后4维)
    ax2 = axes[1]
    if actions_array.shape[1] >= 7:
        # 位置
        for i in range(3):
            ax2.plot(steps, actions_array[:, i], label=f"pos_{i}", linestyle="-")
        # 姿态
        for i in range(3, 7):
            ax2.plot(steps, actions_array[:, i], label=f"quat_{i-3}", linestyle="--")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Value")
    ax2.set_title("Position (solid) & Quaternion (dashed)")
    ax2.legend(loc="upper right", ncol=4, fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = records_dir / "action_trajectory.png"
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    plt.show()


def get_camera_keys(data: dict) -> list[str]:
    """从数据中获取所有摄像头 key。"""
    camera_keys = []
    inputs = data.get("inputs", data)
    for key in inputs:
        if "rgb" in key.lower() or "camera" in key.lower():
            camera_keys.append(key)
    return sorted(camera_keys)


def convert_image_for_display(img: np.ndarray) -> np.ndarray:
    """将图像转换为可显示格式 (H, W, C) uint8。"""
    # 处理 batch 维度，取第一个
    if img.ndim == 4:
        img = img[0]
    
    # 处理 CHW -> HWC
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = np.transpose(img, (1, 2, 0))
    
    # 处理灰度图
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 1:
        img = np.concatenate([img] * 3, axis=-1)
    
    # 归一化到 0-255
    if img.dtype == np.float32 or img.dtype == np.float64:
        if img.max() <= 1.0:
            img = img * 255
        elif img.min() < 0:
            # 可能是 [-1, 1] 范围
            img = (img + 1) * 127.5
    
    return np.clip(img, 0, 255).astype(np.uint8)


def show_camera_single_step(records_dir: Path, step: int, save: bool = True):
    """显示单个 step 的所有摄像头画面。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return
    
    file_path = records_dir / f"step_{step}.npy"
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    data = np.load(file_path, allow_pickle=True).item()
    inputs = data.get("inputs", data)
    
    camera_keys = get_camera_keys(data)
    if not camera_keys:
        print("No camera data found")
        return
    
    print(f"Found {len(camera_keys)} cameras: {camera_keys}")
    
    # 创建图像网格
    n_cameras = len(camera_keys)
    cols = min(3, n_cameras)
    rows = (n_cameras + cols - 1) // cols
    
    _, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_cameras == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    for i, key in enumerate(camera_keys):
        img = inputs[key]
        img_display = convert_image_for_display(img)
        
        axes[i].imshow(img_display)
        axes[i].set_title(f"{key}\nshape: {img.shape}")
        axes[i].axis("off")
    
    # 隐藏多余的子图
    for i in range(n_cameras, len(axes)):
        axes[i].axis("off")
    
    plt.suptitle(f"Step {step} - Camera Views", fontsize=14)
    plt.tight_layout()
    
    if save:
        output_path = records_dir / f"camera_step_{step}.png"
        plt.savefig(output_path, dpi=150)
        print(f"Saved to: {output_path}")
    
    plt.show()


def show_camera_grid(records_dir: Path, start_step: int = 0, end_step: int = -1, 
                     sample_interval: int = 10, save: bool = True):
    """显示多个 step 的摄像头画面网格。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return
    
    files = get_record_files(records_dir)
    if not files:
        print(f"No records found in {records_dir}")
        return
    
    if end_step < 0:
        end_step = files[-1][0]
    
    # 采样 steps
    sampled_files = [(s, f) for s, f in files 
                     if start_step <= s <= end_step and s % sample_interval == 0]
    
    if not sampled_files:
        print("No files to display")
        return
    
    # 获取第一个文件的摄像头 keys
    first_data = np.load(sampled_files[0][1], allow_pickle=True).item()
    camera_keys = get_camera_keys(first_data)
    
    if not camera_keys:
        print("No camera data found")
        return
    
    # 只显示前 3 个摄像头
    camera_keys = camera_keys[:3]
    
    # 限制最多显示 20 个 step
    max_steps = 20
    if len(sampled_files) > max_steps:
        step_interval = len(sampled_files) // max_steps
        sampled_files = sampled_files[::step_interval][:max_steps]
    
    print(f"Displaying {len(sampled_files)} steps, {len(camera_keys)} cameras")
    
    # 创建图像网格：行是摄像头，列是时间步
    n_rows = len(camera_keys)
    n_cols = len(sampled_files)
    
    _, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2.5 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for col, (step, file_path) in enumerate(sampled_files):
        data = np.load(file_path, allow_pickle=True).item()
        inputs = data.get("inputs", data)
        
        for row, key in enumerate(camera_keys):
            if key in inputs:
                img = inputs[key]
                img_display = convert_image_for_display(img)
                axes[row, col].imshow(img_display)
            
            axes[row, col].axis("off")
            
            if row == 0:
                axes[row, col].set_title(f"Step {step}", fontsize=8)
            if col == 0:
                axes[row, col].set_ylabel(key, fontsize=8)
    
    plt.suptitle(f"Camera Views (Steps {start_step}-{end_step})", fontsize=12)
    plt.tight_layout()
    
    if save:
        output_path = records_dir / "camera_grid.png"
        plt.savefig(output_path, dpi=150)
        print(f"Saved to: {output_path}")
    
    plt.show()


def play_camera_video(records_dir: Path, start_step: int = 0, end_step: int = -1,
                      fps: int = 10, save_video: bool = True):
    """播放摄像头视频。"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return
    
    files = get_record_files(records_dir)
    if not files:
        print(f"No records found in {records_dir}")
        return
    
    if end_step < 0:
        end_step = files[-1][0]
    
    # 过滤文件
    selected_files = [(s, f) for s, f in files if start_step <= s <= end_step]
    
    if not selected_files:
        print("No files to display")
        return
    
    # 获取摄像头 keys
    first_data = np.load(selected_files[0][1], allow_pickle=True).item()
    camera_keys = get_camera_keys(first_data)[:3]  # 最多 3 个摄像头
    
    if not camera_keys:
        print("No camera data found")
        return
    
    print(f"Playing {len(selected_files)} frames, {len(camera_keys)} cameras at {fps} FPS")
    print("Press 'q' to quit")
    
    # 创建图形
    n_cameras = len(camera_keys)
    fig, axes = plt.subplots(1, n_cameras, figsize=(5 * n_cameras, 5))
    if n_cameras == 1:
        axes = [axes]
    
    # 初始化图像
    images = []
    for i, key in enumerate(camera_keys):
        img = first_data.get("inputs", first_data)[key]
        img_display = convert_image_for_display(img)
        im = axes[i].imshow(img_display)
        axes[i].set_title(key)
        axes[i].axis("off")
        images.append(im)
    
    title = fig.suptitle(f"Step {selected_files[0][0]}")
    
    def update(frame_idx):
        step, file_path = selected_files[frame_idx]
        data = np.load(file_path, allow_pickle=True).item()
        inputs = data.get("inputs", data)
        
        for i, key in enumerate(camera_keys):
            if key in inputs:
                img = inputs[key]
                img_display = convert_image_for_display(img)
                images[i].set_data(img_display)
        
        title.set_text(f"Step {step}")
        return images + [title]
    
    anim = FuncAnimation(fig, update, frames=len(selected_files), 
                         interval=1000 // fps, blit=False)
    
    if save_video:
        try:
            output_path = records_dir / "camera_video.mp4"
            anim.save(str(output_path), writer='ffmpeg', fps=fps)
            print(f"Video saved to: {output_path}")
        except Exception as e:
            print(f"Could not save video: {e}")
            print("Try: sudo apt install ffmpeg")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="查看 policy_records 文件夹中的内容")
    parser.add_argument("--dir", type=str, default="policy_records",
                        help="记录文件夹路径 (默认: policy_records)")
    parser.add_argument("--step", type=int, default=None,
                        help="查看特定 step 的详细内容")
    parser.add_argument("--keys", action="store_true",
                        help="只显示 keys")
    parser.add_argument("--latest", action="store_true",
                        help="查看最新的记录")
    parser.add_argument("--plot", action="store_true",
                        help="绘制 action 轨迹")
    parser.add_argument("--start", type=int, default=0,
                        help="起始 step")
    parser.add_argument("--end", type=int, default=-1,
                        help="结束 step")
    
    # 摄像头相关参数
    parser.add_argument("--camera", action="store_true",
                        help="显示摄像头画面")
    parser.add_argument("--video", action="store_true",
                        help="播放摄像头视频")
    parser.add_argument("--fps", type=int, default=10,
                        help="视频帧率 (默认: 10)")
    parser.add_argument("--interval", type=int, default=10,
                        help="网格显示时的采样间隔 (默认: 10)")
    parser.add_argument("--no-save", action="store_true",
                        help="不保存图片/视频")
    
    args = parser.parse_args()
    
    # 确定记录目录
    records_dir = Path(args.dir)
    if not records_dir.is_absolute():
        # 尝试在当前目录和项目根目录查找
        if not records_dir.exists():
            project_root = Path(__file__).parent.parent
            records_dir = project_root / args.dir
    
    if not records_dir.exists():
        print(f"Directory not found: {records_dir}")
        return
    
    save = not args.no_save
    
    if args.camera:
        if args.video:
            # 播放视频
            play_camera_video(records_dir, args.start, args.end, args.fps, save)
        elif args.step is not None:
            # 显示单个 step
            show_camera_single_step(records_dir, args.step, save)
        else:
            # 显示网格
            show_camera_grid(records_dir, args.start, args.end, args.interval, save)
    elif args.step is not None:
        view_step(records_dir, args.step, args.keys)
    elif args.latest:
        view_latest(records_dir)
    elif args.plot:
        plot_actions(records_dir, args.start, args.end)
    else:
        print_summary(records_dir)


if __name__ == "__main__":
    main()
