import os
import shutil
from pathlib import Path


def organize_files():
    """
    主功能函数：按规则分类复制特定txt文件
    文件命名规则：
    - 包含'collision_detection'的复制到models文件夹
    - 包含'cutter'的复制到models_cutter文件夹
    """
    # 配置路径（使用Path对象确保跨平台兼容性）
    BASE_DIR = Path('data/final_data')
    SOURCE_DIR = BASE_DIR  # 源文件夹
    TARGET_DIR = BASE_DIR / 'raw_data'  # 目标根目录

    # 目标子目录配置
    TARGET_MAP = {
        'collision_detection': TARGET_DIR / 'models',
        'cutter': TARGET_DIR / 'models_cutter'
    }

    try:
        # 创建目标目录结构
        for folder in TARGET_MAP.values():
            folder.mkdir(parents=True, exist_ok=True)

        # 遍历源目录
        for root, _, files in os.walk(SOURCE_DIR):
            # 跳过目标目录自身（防止循环复制）
            if Path(root) == TARGET_DIR:
                continue

            for filename in files:
                src_file = Path(root) / filename
                # 仅处理txt文件
                if src_file.suffix != '.txt':
                    continue

                # 分类复制文件
                for key, target_dir in TARGET_MAP.items():
                    if key in filename:
                        dest_file = target_dir / filename
                        shutil.copy2(src_file, dest_file)
                        print(f'[SUCCESS] 已复制: {src_file} -> {dest_file}')
                        break

        print(f'操作完成！文件已分类至 {TARGET_DIR}')

    except Exception as e:
        print(f'[ERROR] 操作失败: {str(e)}')


if __name__ == '__main__':
    organize_files()