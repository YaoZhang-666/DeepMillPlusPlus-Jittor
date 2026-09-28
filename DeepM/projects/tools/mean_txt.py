import os

def calculate_average(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data = [float(line[16:]) for line in lines]
    data = data[1:]
    return float(sum(data)) / float(len(data)) if data else 0

if __name__ == "__main__":
    file_path = "visual/time.txt"
    print(calculate_average(file_path))