from jittor_compat import torch
import math


class CylinderVisualizer:
    def __init__(self, radius=0.01, segments=8):
        self.radius = radius
        self.segments = segments

    def create_cylinder_vertices(self, start_point, end_point):
        """创建圆柱体顶点"""
        # 计算圆柱体方向向量
        direction = end_point - start_point
        length = torch.norm(direction)
        if length < 1e-6:
            return []

        direction = direction / length

        # 创建垂直于方向向量的两个向量
        if abs(direction[2]) < 0.9:
            up = torch.tensor([0.0, 0.0, 1.0])
        else:
            up = torch.tensor([1.0, 0.0, 0.0])

        right = torch.cross(direction, up)
        right = right / torch.norm(right)
        up = torch.cross(right, direction)
        up = up / torch.norm(up)

        vertices = []

        # 创建底面和顶面的顶点
        for i in range(self.segments):
            angle = 2 * math.pi * i / self.segments
            offset = (right * math.cos(angle) + up * math.sin(angle)) * self.radius

            # 底面顶点
            vertices.append(start_point + offset)
            # 顶面顶点
            vertices.append(start_point + offset + direction * length)

        # 添加中心点（用于封顶）
        vertices.append(start_point)  # 底面中心
        vertices.append(start_point + direction * length)  # 顶面中心

        return vertices

    def create_cylinder_faces(self, start_index):
        """创建圆柱体面"""
        faces = []

        # 侧面
        for i in range(self.segments):
            next_i = (i + 1) % self.segments
            # 侧面四边形分解为两个三角形
            faces.append([start_index + i * 2, start_index + next_i * 2, start_index + i * 2 + 1])
            faces.append([start_index + next_i * 2, start_index + next_i * 2 + 1, start_index + i * 2 + 1])

        # 底面
        center_bottom = start_index + self.segments * 2
        for i in range(self.segments):
            next_i = (i + 1) % self.segments
            faces.append([start_index + i * 2, start_index + next_i * 2, center_bottom])

        # 顶面
        center_top = start_index + self.segments * 2 + 1
        for i in range(self.segments):
            next_i = (i + 1) % self.segments
            faces.append([start_index + i * 2 + 1, center_top, start_index + next_i * 2 + 1])

        return faces

    def visualization_new(self, points, pred_points, columns, obj_folder, pred_folder, hash_ord):
        # 获取原始tensor的形状和元素总数
        original_shape = pred_points.shape
        total_elements = pred_points.numel()

        # 计算合适的行数
        if total_elements % columns != 0:
            raise ValueError(f"无法将包含{total_elements}个元素的tensor重塑为{columns}列，请检查输入tensor的维度")

        rows = total_elements // columns

        # 重塑tensor
        reshaped_tensor = pred_points.reshape(rows, columns)

        # 计算每行最大值的索引
        max_indices = torch.argmax(reshaped_tensor, dim=1)

        # 收集所有圆柱体的顶点和面
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        # 为每个方向绘制圆柱体
        for i in range(rows):
            batch_points = points[i]
            sample_point = 0.05 * hash_ord[max_indices[i]]

            # 计算起点和终点
            start_point = torch.tensor([batch_points.points[0], batch_points.points[1], batch_points.points[2]])
            end_point = start_point + sample_point

            # 创建圆柱体
            vertices = self.create_cylinder_vertices(start_point, end_point)
            if vertices:
                faces = self.create_cylinder_faces(vertex_offset)

                all_vertices.extend(vertices)
                all_faces.extend(faces)
                vertex_offset += len(vertices)

        # 写入OBJ文件
        with open(pred_folder, 'w') as obj_file:
            # 写入顶点
            for vertex in all_vertices:
                obj_file.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

            # 写入面
            for face in all_faces:
                obj_file.write(f"f {' '.join(str(idx + 1) for idx in face)}\n")

        #print(f"已生成包含{len(all_vertices)}个顶点和{len(all_faces)}个面的圆柱体模型")