import os
import sys


def rename_files_and_folders(root_dir, start_pos, end_pos):
    """
    批量重命名文件和文件夹，将固定位置替换为全1
    """
    count = 0

    # 处理文件夹
    for item in os.listdir(root_dir):
        old_path = os.path.join(root_dir, item)

        if os.path.isdir(old_path) and len(item) >= end_pos:
            new_name = item[:start_pos] + '1' * (end_pos - start_pos) + item[end_pos:]
            new_path = os.path.join(root_dir, new_name)

            if new_name != item:
                os.rename(old_path, new_path)
                # print(f"重命名文件夹: {item} -> {new_name}")
                count += 1

    # 处理文件
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(('.obj', '.txt')) and len(file) >= end_pos:
                old_path = os.path.join(root, file)
                new_name = file[:start_pos] + '1' * (end_pos - start_pos) + file[end_pos:]
                new_path = os.path.join(root, new_name)

                if new_name != file:
                    os.rename(old_path, new_path)
                    # print(f"重命名文件: {file} -> {new_name}")
                    count += 1

    return count


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("用法: python rename_files.py <根目录> <开始位置> <结束位置>")
        print("示例: python rename_files.py ./data 9 32")
        sys.exit(1)

    root_dir = sys.argv[1]
    start_pos = int(sys.argv[2])
    end_pos = int(sys.argv[3])

    if not os.path.exists(root_dir):
        print(f"错误: 目录 '{root_dir}' 不存在")
        sys.exit(1)

    print(f"开始处理目录: {root_dir}")
    print(f"替换位置: [{start_pos}:{end_pos}]")

    total = rename_files_and_folders(root_dir, start_pos, end_pos)
    print(f"处理完成! 共重命名 {total} 个项目")