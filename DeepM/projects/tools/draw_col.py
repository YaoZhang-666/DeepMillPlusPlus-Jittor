import os


def read_safety_status(status_file):
    """读取安全状态文件，返回每个点的安全状态列表"""
    safety_status = []
    with open(status_file, 'r') as f:
        for line in f:
            safety_status.append(int(line.strip()))
    return safety_status


def read_unreachable_vertices(unreachable_file):
    """读取不可达顶点文件，返回不可达顶点索引集合"""
    unreachable_indices = set()
    if os.path.exists(unreachable_file):
        with open(unreachable_file, 'r') as f:
            for line in f:
                if line.strip():
                    # 注意：文件中的索引是从0开始的
                    unreachable_indices.add(int(line.strip()))
    return unreachable_indices


def process_obj_file(obj_path, safety_status, unreachable_indices, output_0_path, output_1_path, output_2_path):
    """处理OBJ文件，根据安全状态和不可达点分离面"""
    # 读取OBJ文件中的所有行
    with open(obj_path, 'r') as f:
        lines = f.readlines()

    # 分离顶点、法向量和面
    vertices = []  # 存储所有顶点行
    normals = []  # 存储所有法向量行
    faces = []  # 存储所有面行

    for line in lines:
        if line.startswith('v '):
            vertices.append(line)
        elif line.startswith('vn '):
            normals.append(line)
        elif line.startswith('f '):
            faces.append(line)

    # 确保安全状态数量与顶点数量匹配
    if len(safety_status) != len(vertices):
        raise ValueError(f"安全状态数量({len(safety_status)})与顶点数量({len(vertices)})不匹配")

    # 分离面到三个文件
    faces_0 = []  # 存储不安全的面(0)
    faces_1 = []  # 存储安全的面(1)
    faces_2 = []  # 存储保守不可达的面(2)

    for face_line in faces:
        # 解析面索引 (OBJ面索引从1开始)
        face_indices = []
        face_parts = face_line.strip().split()[1:]  # 去掉'f'前缀

        for part in face_parts:
            # 处理可能包含纹理/法向量的索引格式 (如 1/2/3 或 1//3)
            vertex_index = int(part.split('/')[0])
            face_indices.append(vertex_index)

        # 检查面中顶点是否在不可达点中（注意索引转换）
        unreachable_count = 0
        for idx in face_indices:
            # OBJ索引从1开始，不可达文件索引从0开始，所以需要减1
            if (idx - 1) in unreachable_indices:
                unreachable_count += 1

        # 优先处理保守不可达面
        if unreachable_count >= 2:
            faces_2.append(face_line)
        else:
            # 检查面中顶点的安全状态
            unsafe_count = 0
            for idx in face_indices:
                # 注意：OBJ索引从1开始，列表索引从0开始
                if safety_status[idx - 1] == 0:  # 0表示不安全(碰撞)
                    unsafe_count += 1

            # 根据不安全顶点数量决定面的归属
            if unsafe_count >= 2:
                faces_0.append(face_line)
            else:
                faces_1.append(face_line)

    # 写入保守不可达文件 (_2.obj)
    with open(output_2_path, 'w') as f:
        f.writelines(vertices)
        if normals:
            f.writelines(normals)
        f.writelines(faces_2)

    # 写入不安全文件 (_0.obj)
    with open(output_0_path, 'w') as f:
        f.writelines(vertices)
        if normals:
            f.writelines(normals)
        f.writelines(faces_0)

    # 写入安全文件 (_1.obj)
    with open(output_1_path, 'w') as f:
        f.writelines(vertices)
        if normals:
            f.writelines(normals)
        f.writelines(faces_1)

    # 验证面数量
    total_faces = len(faces_0) + len(faces_1) + len(faces_2)
    if total_faces != len(faces):
        raise ValueError(f"面数量不匹配: 分离后{total_faces} != 原始{len(faces)}")

    return len(faces_0), len(faces_1), len(faces_2)


def main():
    # 定义路径
    gallery_folder = r'E:\ZhangYao\Results\results_gallery_uniform\our'
    geometric_folder = r'E:\ZhangYao\Results\results_gallery_uniform\geometric'
    uniform_folder = r'E:\ZhangYao\Results\uniform'
    output_base_folder = r'E:\ZhangYao\绘图\draw_result\uniform'

    # 确保输出基础文件夹存在
    os.makedirs(output_base_folder, exist_ok=True)

    # 遍历gallery文件夹中的所有子文件夹
    for subfolder_name in os.listdir(gallery_folder):
        subfolder_path = os.path.join(gallery_folder, subfolder_name)

        # 确保是文件夹
        if os.path.isdir(subfolder_path):
            # 构建相关路径
            safety_status_path = os.path.join(subfolder_path, 'safety_status.txt')
            unreachable_path = os.path.join(geometric_folder, subfolder_name, 'unreachable_vertices.txt')
            obj_source_path = os.path.join(uniform_folder, subfolder_name, f'{subfolder_name}.obj')
            output_subfolder_path = os.path.join(output_base_folder, subfolder_name)
            output_0_path = os.path.join(output_subfolder_path, f'{subfolder_name}_0.obj')
            output_1_path = os.path.join(output_subfolder_path, f'{subfolder_name}_1.obj')
            output_2_path = os.path.join(output_subfolder_path, f'{subfolder_name}_2.obj')

            # 检查必要文件是否存在
            if not os.path.exists(safety_status_path):
                print(f"警告: {subfolder_name} 中未找到 safety_status.txt 文件，跳过处理")
                continue

            if not os.path.exists(obj_source_path):
                print(f"警告: {subfolder_name} 中未找到对应的 OBJ 文件，跳过处理")
                continue

            # 创建输出子文件夹
            os.makedirs(output_subfolder_path, exist_ok=True)

            try:
                # 读取安全状态
                safety_status = read_safety_status(safety_status_path)

                # 读取不可达顶点
                unreachable_indices = read_unreachable_vertices(unreachable_path)

                # 处理OBJ文件
                count_0, count_1, count_2 = process_obj_file(
                    obj_source_path, safety_status, unreachable_indices,
                    output_0_path, output_1_path, output_2_path
                )

                print(f"成功处理: {subfolder_name} (不可达:{count_0}, 可达:{count_1}, 几何不可达:{count_2})")
            except Exception as e:
                print(f"处理 {subfolder_name} 时出错: {str(e)}")

    print("所有文件处理完成!")


if __name__ == "__main__":
    main()
