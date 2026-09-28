import os
from pathlib import Path
import pandas as pd


def load_skip_indices(skip_file: str) -> set[int]:
    """读取 skip.txt 中的索引（支持空格/逗号分隔），返回 set[int]。"""
    skip_indices: set[int] = set()
    try:
        with open(skip_file, "r", encoding="utf-8") as sf:
            for line in sf:
                for item in line.replace(",", " ").split():
                    s = item.strip()
                    if s.lstrip("+").isdigit():
                        skip_indices.add(int(s))
    except FileNotFoundError:
        print(f"⚠️ 未找到 {skip_file}，将不跳过任何行。")
    return skip_indices


def calculate_fn_fp(file1: str, file2: str, skip_indices: set[int]) -> tuple[float, float, float]:
    """
    比较两个TXT文件每一行的数字（0/1），计算 FN、FP 和 ACC。
    FN (False Negative): 将不可达点(0)预测为可达点(1)的比例
    FP (False Positive): 将可达点(1)预测为不可达点(0)的比例
    ACC (Accuracy): 预测值与真实值相同的比例    
    """
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    total_lines = min(len(lines1), len(lines2))
    if total_lines == 0:
        raise ValueError("文件为空或行数为0。")

    acc_count = 0
    fn_count = 0
    fp_count = 0
    valid_count = 0

    valid_skip = skip_indices & set(range(total_lines))  # 只保留在范围内的 skip
    for i in range(total_lines):
        if i in valid_skip:
            continue

        val1 = lines1[i].strip()
        val2 = lines2[i].strip()
        
        # 确保值是有效的 0 或 1
        if val1 not in ["0", "1"] or val2 not in ["0", "1"]:
            continue

        valid_count += 1
        
        # FN: 预测值为 1（可达），真实值为 0（不可达）
        if val1 == "1" and val2 == "0":
            fn_count += 1
        # FP: 预测值为 1（不可达），真实值为 0（可达）
        elif val1 == "0" and val2 == "1":
            fp_count += 1
        elif val1 == val2:
            acc_count += 1

    if valid_count == 0:
        return 0.0, 0.0, 0.0

    fn_ratio = fn_count / valid_count
    fp_ratio = fp_count / valid_count
    acc_ratio = acc_count / valid_count
    return fn_ratio, fp_ratio, acc_ratio


def read_timing(timing_file: str) -> float:
    """
    读取 timing.txt 文件的最后一行数字作为时间值。
    """
    try:
        with open(timing_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if not lines:
                return 0.0
            # 读取最后一行并转换为浮点数
            last_line = lines[-1].strip()
            if last_line.replace(".", "").isdigit():
                return float(last_line)
            return 0.0
    except FileNotFoundError:
        print(f"⚠️ 未找到 {timing_file}，时间设为 0.0。")
        return 0.0


def batch_calculate_and_save_xlsx(
    rendering_root: str = str(Path(__file__).resolve().parent / "outputs" / "rendering"),
    geometric_root: str = str(Path(__file__).resolve().parent / "outputs" / "geometric"),
    output_xlsx: str = str(Path(__file__).resolve().parent / "outputs" / "metrics.xlsx"),
    skip_name: str = "skip.txt",
):
    rendering_root_p = Path(rendering_root)
    geometric_root_p = Path(geometric_root)

    if not rendering_root_p.exists():
        raise FileNotFoundError(f"rendering_root 不存在: {rendering_root}")
    if not geometric_root_p.exists():
        raise FileNotFoundError(f"geometric_root 不存在: {geometric_root}")

    folders = sorted([p for p in rendering_root_p.iterdir() if p.is_dir()], key=lambda x: x.name)

    accs: list[float | None] = []

    times: list[float] = []
    fns: list[float | None] = []
    fps: list[float | None] = []
    valid_times: list[float] = []  # 用于统计平均时间（只统计成功的）
    valid_fns: list[float] = []    # 用于统计平均 FN（只统计成功的）
    valid_fps: list[float] = []    # 用于统计平均 FP（只统计成功的）

    for folder in folders:
        name = folder.name

        # 读取时间
        timing_file = folder / "timing.txt"
        time_val = read_timing(str(timing_file))
        times.append(time_val)
        valid_times.append(time_val)  # 时间总是有效的

        # 计算 FN 和 FP
        file1 = folder / "safety_status.txt"
        geom_sub = geometric_root_p / name
        file2 = geom_sub / "new_safety_status.txt"
        skip_file = geom_sub / skip_name

        if not file1.exists():
            print(f"⚠️ {name}: 缺少 file1: {file1}")
            fns.append(None)
            fps.append(None)
            continue

        if not file2.exists():
            print(f"⚠️ {name}: 缺少 file2: {file2}")
            fns.append(None)
            fps.append(None)
            continue

        skip_indices = load_skip_indices(str(skip_file))

        try:
            fn_ratio, fp_ratio, acc_ratio = calculate_fn_fp(str(file1), str(file2), skip_indices)
            fns.append(fn_ratio)
            fps.append(fp_ratio)
            accs.append(acc_ratio)
            valid_fns.append(fn_ratio)
            valid_fps.append(fp_ratio)

            if acc_ratio < 0.90:
                print(f"model name {folder} acc_ratio is {acc_ratio}")
        except Exception as e:
            print(f"⚠️ {name}: 处理失败 -> {e}")
            fns.append(None)
            fps.append(None)
            accs.append(None)

    # 写入 Excel：三列，不写表头
    df = pd.DataFrame({
        'time': times,
        'fn': fns,
        'fp': fps,
        'acc': accs
    })
    df.to_excel(output_xlsx, index=False, header=False)

    # 输出平均值
    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        print(f"✅ 平均时间（{len(valid_times)}/{len(folders)}）：{avg_time:.6f}")
    else:
        avg_time = None
        print("⚠️ 没有任何有效时间，无法计算平均值。")

    if valid_fns:
        avg_fn = sum(valid_fns) / len(valid_fns)
        print(f"✅ 平均 FN（成功 {len(valid_fns)}/{len(folders)}）：{avg_fn:.6f}")
    else:
        avg_fn = None
        print("⚠️ 没有任何有效 FN，无法计算平均值。")

    if valid_fps:
        avg_fp = sum(valid_fps) / len(valid_fps)
        print(f"✅ 平均 FP（成功 {len(valid_fps)}/{len(folders)}）：{avg_fp:.6f}")
    else:
        avg_fp = None
        print("⚠️ 没有任何有效 FP，无法计算平均值。")

    if accs:
        avg_acc = sum(accs) / len(accs)
        print(f"✅ 平均 ACC（成功 {len(accs)}/{len(folders)}）：{avg_acc:.6f}")
    else:
        avg_acc = None
        print("⚠️ 没有任何有效 ACC，无法计算平均值。")



    print(f"✅ 已写入: {output_xlsx}")
    print(f"文件夹数量: {len(folders)}，写入行数: {len(times)}")


if __name__ == "__main__":
    batch_calculate_and_save_xlsx()
