import os
import numpy as np


def calculate_normal(v1, v2, v3):
    edge1 = v2 - v1
    edge2 = v3 - v1
    normal = np.cross(edge1, edge2)
    return normal / np.linalg.norm(normal)


def process_obj(obj_path, txt_path, cutter_path):
    vertices = []
    normals = []
    with open(obj_path, 'r') as obj_file:
        lines = obj_file.readlines()

    for line in lines:
        if line.startswith('v '):
            vertices.append(np.array(list(map(float, line.split()[1:]))))
        # elif line.startswith('vn '):
        #     #print(line)
        #     normals.append(np.array(list(map(float, line.split()[1:]))))
        elif line.startswith('f '):
            if '//' in line:
                indexs = line.split()[1:]
                indexs = [index.split('//')[0] for index in indexs]
                face = list(map(int, indexs))
            else:
                face = list(map(int, line.split()[1:]))
            v1, v2, v3 = vertices[face[0] - 1], vertices[face[1] - 1], vertices[face[2] - 1]
            normal = calculate_normal(v1, v2, v3)
            normals.append(normal)

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    os.makedirs(os.path.dirname(cutter_path), exist_ok=True)

    with open(txt_path, 'w') as txt_file:
        for line in lines:
            if line.startswith('v '):
                txt_file.write(line[2:].strip() + ' ' + ' '.join(map(str, normals.pop(0))) + ' 0' * 102 + '\n')
            # else:
            #     txt_file.write(line)
    with open(cutter_path, 'w') as f:
        f.write('1.5 20 15 30')


def process_files():
    models_obj_dir = 'data/models'
    raw_data_models_dir = 'data/raw_data/models'
    cutter_dir = 'data/raw_data/models_cutter'

    for obj_file in os.listdir(models_obj_dir):
        if obj_file.endswith('.obj'):
            obj_path = os.path.join(models_obj_dir, obj_file)
            txt_file = os.path.splitext(obj_file)[0] + '.txt'
            txt_path = os.path.join(raw_data_models_dir, txt_file)
            cutter_path = os.path.join(cutter_dir, os.path.splitext(obj_file)[0] + '_cutter.txt')
            process_obj(obj_path, txt_path, cutter_path)


if __name__ == "__main__":
    process_files()