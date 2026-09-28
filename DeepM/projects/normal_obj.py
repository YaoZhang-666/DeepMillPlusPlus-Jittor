import os
import numpy as np


def normalize_obj_points(input_folder, output_folder):
    """
    对OBJ文件中的顶点进行归一化处理，保持面信息不变
    Parameters:
        input_folder (str): 输入文件夹路径
        output_folder (str): 输出文件夹路径
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".obj"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取OBJ文件
            with open(input_path, 'r') as f:
                lines = f.readlines()

            # 分别存储顶点和面数据
            vertices = []
            faces = []
            other_lines = []

            for line in lines:
                if line.startswith('v '):
                    # 提取顶点坐标
                    vertex_data = line.strip().split()[1:]
                    vertices.append([float(x) for x in vertex_data])
                elif line.startswith('f '):
                    # 保存面数据
                    faces.append(line)
                else:
                    # 保存其他行（如vt, vn等）
                    other_lines.append(line)

            # 转换顶点为numpy数组
            vertices = np.array(vertices)

            # 归一化处理
            # 1. 计算中心点并居中
            center = np.mean(vertices, axis=0)
            vertices_centered = vertices - center

            # 2. 计算缩放因子
            max_extent = np.max(np.abs(vertices_centered), axis=0)
            max_scale = np.max(max_extent)

            # 3. 归一化到[-0.8, 0.8]范围
            vertices_normalized = vertices_centered / max_scale * 0.8

            # 写入新的OBJ文件
            with open(output_path, 'w') as f:
                # 写入归一化后的顶点
                for vertex in vertices_normalized:
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

                # 写入面数据和其他行
                for face in faces:
                    f.write(face)
                for other_line in other_lines:
                    f.write(other_line)

            print(f"Processed and saved: {filename}")


if __name__ == "__main__":
    # 使用示例
    input_folder = "data/models_obj"
    output_folder = "visual/obj"
    normalize_obj_points(input_folder, output_folder)
    print("所有OBJ文件处理完成!")