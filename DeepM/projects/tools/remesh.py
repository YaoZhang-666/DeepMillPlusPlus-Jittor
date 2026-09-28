import os
import shutil
from pathlib import Path
from tqdm import tqdm
import pymeshlab
import random


def remesh(ms, output_file_path, targetlen_num):
    """
    对网格进行重采样处理
    """
    # 应用网格细分过滤器
    ms.apply_filter('meshing_isotropic_explicit_remeshing',
                    iterations=20,
                    featuredeg=15,
                    targetlen=pymeshlab.Percentage(targetlen_num))

    # 获取面的数量
    face_count = ms.current_mesh().face_number()
    if face_count > 250000:
        return False

    # 保存细分后的模型
    ms.save_current_mesh(output_file_path)
    return True


def process_mesh_file(input_file_path, destination_directory, target_faces):
    """
    处理单个网格文件
    """
    # 创建目标目录
    os.makedirs(destination_directory, exist_ok=True)

    # 创建一个新的MeshSet对象
    ms = pymeshlab.MeshSet()

    # 加载网格文件
    ms.load_new_mesh(input_file_path)

    # 获取初始面数
    initial_face_count = ms.current_mesh().face_number()
    print(f"初始面数: {initial_face_count}")

    # # 如果面数已经大于100000，直接保存
    # if initial_face_count > 150000:
    #     output_path = os.path.join(destination_directory, os.path.basename(input_file_path))
    #     ms.save_current_mesh(output_path)
    #     return True

    # 逐步增加细化程度直到达到目标面数
    targetlen_num = 1.0
    max_iterations = random.randint(-2, 6)  # 防止无限循环
    #max_iterations = 1
    iteration = 0
    print('最大迭代次数为{}'.format(max_iterations))
    while ms.current_mesh().face_number() < 250000 and iteration < max_iterations:
        # 应用重采样过滤器
        ms.apply_filter('meshing_isotropic_explicit_remeshing',
                        iterations=15,
                        featuredeg=15,
                        targetlen=pymeshlab.Percentage(targetlen_num))

        current_faces = ms.current_mesh().face_number()
        print(f"迭代 {iteration + 1}: 面数 {current_faces}")

        # 检查是否超过目标面数
        if current_faces > 250000:
            break

        targetlen_num -= 0.1
        iteration += 1

    # 保存处理后的模型
    output_filename = os.path.basename(input_file_path)
    output_path = os.path.join(destination_directory, output_filename[:8]+'.obj')
    print(output_path)
    ms.save_current_mesh(output_path)

    final_face_count = ms.current_mesh().face_number()
    print(f"最终面数: {final_face_count}")

    return True


def find_obj_files(root_directory):
    """
    在目录树中查找所有OBJ文件
    """
    obj_files = []
    root_path = Path(root_directory)

    for file_path in root_path.rglob("*.obj"):
        obj_files.append(str(file_path))

    return obj_files


def main():
    """
    主函数：处理指定目录下的所有OBJ文件
    """
    # 源目录路径
    source_directory = r"E:\ZhangYao\绘图\done_1"

    # 目标目录路径
    destination_directory = r"mesh_100k"

    # 查找所有OBJ文件
    print("正在搜索OBJ文件...")
    obj_files = find_obj_files(source_directory)
    print(f"找到 {len(obj_files)} 个OBJ文件")

    # 处理每个OBJ文件
    for i, input_file_path in enumerate(tqdm(obj_files, desc="处理网格文件")):
        print(f"\n处理文件 ({i + 1}/{len(obj_files)}): {input_file_path}")
        process_mesh_file(input_file_path, destination_directory, 20000)

    print("所有文件处理完成!")


if __name__ == "__main__":
    main()
