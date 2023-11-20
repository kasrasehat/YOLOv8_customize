import os
import glob


def process_line(line):
    parts = line.split()
    class_id = parts[0]
    coordinates = list(map(float, parts[1:]))
    xs = coordinates[0::2]
    ys = coordinates[1::2]

    # Calculate the center, width, and height
    x_center = sum(xs) / 4
    y_center = sum(ys) / 4
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    return f"{class_id} {x_center} {y_center} {width} {height}\n"


def process_file(file_path, output_folder):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    base_name = os.path.basename(file_path)
    new_file_path = os.path.join(output_folder, base_name)

    with open(new_file_path, 'w') as new_file:
        for line in lines:
            new_line = process_line(line)
            new_file.write(new_line)


def main(main_folder):
    output_folder = os.path.join(main_folder, 'labels')
    os.makedirs(output_folder, exist_ok=True)

    for txt_file in glob.glob(os.path.join(main_folder + '/raw_labels', '*.txt')):
        process_file(txt_file, output_folder)


if __name__ == '__main__':
    # Replace 'path_to_main_folder' with the path to your main folder
    main_folder = '/home/kasra/PycharmProjects/YOLOv8_customize/data/validation'
    main(main_folder)
