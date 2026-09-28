import os
import random


def modify_ply_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.ply'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r+', encoding='utf-8', errors='ignore') as file:
                lines = file.readlines()
                file.seek(0)
                file.truncate()
                print(len(lines))
                for line in lines:
                    if line.strip():
                        random_bits = ''.join(str(random.randint(0, 2)) for _ in range(98))
                        modified_line = line.strip() + ' ' + random_bits + '\n'
                        file.write(modified_line)

            print(f"已处理: {filepath}")


if __name__ == '__main__':
    target_folder = 'data/points/test_model'  # 修改为你的目标文件夹路径
    modify_ply_files(target_folder)
    print("处理完成！")