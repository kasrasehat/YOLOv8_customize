from ultralytics import YOLO
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
import pickle
import tqdm
import cv2
import numpy as np
import os
import pandas as pd
from ast import literal_eval


def save_image(image, path):
    """
    Save an image to a specified path.

    Parameters:
        image (ndarray): The image to be saved.
        path (str): The path where the image will be saved.
    """
    # Create the directory if it doesn't exist
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Save the image
    cv2.imwrite(path, image)


def convert_to_persian_numbers(arabic_str):
    """
    Convert a string containing Arabic numerals to Persian numerals.

    Parameters:
        arabic_str (str): The string containing Arabic numerals.

    Returns:
        str: The string with Arabic numerals replaced by Persian numerals.
    """
    arabic_to_persian = {
        '0': '۰',
        '1': '۱',
        '2': '۲',
        '3': '۳',
        '4': '۴',
        '5': '۵',
        '6': '۶',
        '7': '۷',
        '8': '۸',
        '9': '۹'
    }

    return ''.join(arabic_to_persian.get(char, char) for char in arabic_str)


def search_by_national_id(national_id, cls, dataframe):
    """
    Search for a given national ID in the dataframe and return other columns' values and index if found.

    Parameters:
        national_id (int or str): The national ID to search for.
        dataframe (pd.DataFrame): The DataFrame to search in.

    Returns:
        dict: A dictionary containing the other columns' values and index if the national ID is found.
        None: If the national ID is not found.
    """
    # Search for the national ID in the 'NATIONAL_ID' column
    matching_row = None
    matching_row_list = dataframe[
        (dataframe['NATIONAL_ID'] == int(national_id)) & (dataframe['CLASS'] == int(cls))].index.tolist()
    if len(matching_row_list) == 0 and int(cls) in [2, 3]:
        if int(cls) == 2:
            cls = 3
            matching_row_list = dataframe[
                (dataframe['NATIONAL_ID'] == int(national_id)) & (dataframe['CLASS'] == int(cls))].index.tolist()
        elif int(cls) == 3:
            cls = 2
            matching_row_list = dataframe[
                (dataframe['NATIONAL_ID'] == int(national_id)) & (dataframe['CLASS'] == int(cls))].index.tolist()

    if len(matching_row_list) != 0:
        matching_row = matching_row_list[0]

    # If a match is found
    if matching_row is not None and not pd.isna(dataframe.loc[matching_row, 'PERSON_COORD']):
        # Extract the first match (assuming national IDs are unique)
        row = dataframe.iloc[matching_row]

        # Prepare the result dictionary
        a = str(int(row['NATIONAL_ID']))
        while len(list(a)) != 10:
            a = '0' + a

        if row['SCALE'] == 0:
            scale = 1
        else:
            scale = row['SCALE']

        result = {
            'National_ID': a,
            'First Name': row['FIRST_NAME'],
            'Last Name': row['LAST_NAME'],
            'Father Name': row['FATHER_NAME'],
            'Birth Date': row['BIRTH_DATE'],
            'Persian Birth Date': convert_to_persian_numbers(row['PERSIAN_BIRTH_DATE']),
            'National ID Serial': row['NATIONAL_ID_SERIAL'].upper(),
            'Class': int(row['CLASS']),
            'Person_coordinate': np.round(np.array(literal_eval(row['PERSON_COORD'])).reshape(8) / 720, 2),
            'Rotation': (row['ROTATION'] + 1) / 2,
            'Scale': scale,
            'Transport': np.round(((np.array(literal_eval(row['TRANSPORT'])).reshape(2) / 360) +
                                   np.ones_like(np.array(literal_eval(row['TRANSPORT'])).reshape(2))) / 2, 2),
            'Index': row.name  # row.name contains the index

        }

        return result
    else:
        return None


def replace_tokens(passage):
    passage = passage.replace('a', 'روی کارت ملی')
    passage = passage.replace('b', 'پشت کارت ملی')
    passage = passage.replace('c', 'شناسنامه جدید')
    passage = passage.replace('d', 'شناسنامه قدیم')
    return passage


def create_csv_if_not_exists(csv_path):
    if not os.path.exists(csv_path):
        # Create an empty DataFrame with the same columns as BB
        df = pd.DataFrame(columns=['location','csv_file'])

        # Save the empty DataFrame to a new CSV file named "AAA.csv"
        df.to_csv(csv_path, encoding='UTF-8-SIG', index=False)
        print(f"csv file '{csv_path}' created.")
    else:
        print(f"csv file '{csv_path}' already exists.")


def create_folder_if_not_exists(folder_path):
    """
    Create a new folder if it does not already exist.

    Parameters:
    - folder_path (str): The path of the folder to create.

    Returns:
    - None
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")


def display_image(batch, index):
    # Extract the ith image
    img = batch[index].numpy()

    # Convert from CHW to HWC
    img = np.transpose(img, (1, 2, 0))
    img = (img + np.ones_like(img))/2

    # Convert pixel values from [0, 1] to [0, 255]
    img = (img * 255).astype(np.uint8)

    # Display the image using cv2
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_and_save_detections_cv2(yolo_output, image, save_folder, resize_to=(128, 128)):
    """
    Processes YOLO output, crops, resizes, and saves detected objects using OpenCV.

    yolo_output: Output from YOLO model.
    image: Image read by OpenCV (cv2) on which detections were made.
    save_folder: Folder to save cropped images.
    resize_to: Tuple (width, height) to resize cropped images.
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, detection in enumerate(yolo_output):
        # Extract bounding box coordinates
        x1, y1, x2, y2, confidence, label = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Crop and resize the image
        cropped_image = image[y1:y2, x1:x2]
        resized_image = cv2.resize(cropped_image, resize_to)

        # Save the image
        save_path = os.path.join(save_folder, f'detection_{i}_label_{label}.jpg')
        cv2.imwrite(save_path, resized_image)


lbs = {0: 'back_national_card',
       1: 'birth_date',
       2: 'expiration_date',
       3: 'family_name',
       4: 'father_name',
       5: 'front_national_card',
       6: 'image',
       7: 'name',
       8: 'national_number',
       9: 'new_birth_certificate',
       10: 'old_birth_certificate',
       11: 'serial_number'}

height, width = 640, 640
batch_size = 16
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((height, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

model = YOLO("/home/kasra/PycharmProjects/YOLOv8_customize/runs/detect/train/weights/best.pt")  # load a pretrained model (recommended for training)

cut_image_folder_path = '/home/kasra/kasra_files/data-shenasname/cut_image_folder'
create_folder_if_not_exists(cut_image_folder_path)
csv_path = '/home/kasra/kasra_files/data-shenasname/cut_data_loc_ID_card'
files = glob.glob('/home/kasra/kasra_files/data-shenasname/*.CSV') + glob.glob('/home/kasra/kasra_files/data-shenasname/*.csv')
file_list = []
create_csv_if_not_exists(csv_path)
new_csv = pd.read_csv(csv_path, encoding='UTF-8-SIG')
for file in files:
    if file.split('.')[0][-1] != 'A':
        file_list.append([file, file.split('.')[0].replace('metadata', 'files')])
p = 0
for index, file in tqdm.tqdm(enumerate(file_list)):

    labels = pd.read_csv(file[0], encoding='UTF-8-SIG', converters={'feature': literal_eval})
    for i, image_file in enumerate(os.listdir(file[1])):
        national_id, cls = image_file.split('.')[0].split('_')
        if cls in [0, 1]:
            if search_by_national_id(int(national_id), cls, labels) is not None:

                image = cv2.imread(image_file)
                new_width = 640
                new_height = 640
                # Resize the image
                image = cv2.resize(image, (new_width, new_height))
                # Inference
                img = '/home/kasra/PycharmProjects/YOLOv8_customize/extra_files/image1.jpg'

                save_image(image, img)
                results = model.predict(img, save=True, imgsz=640, conf=0.3, save_txt=True)[0].boxes.data
                # print(results[0].boxes.data)  # returns xyxy of bounding box + confidence and class number
                new_csv.loc[p, 'location'] = file[1] + f'/{image_file}'
                new_csv.loc[p, 'csv_file'] = file[0]
                p += 1

        if i % 100 == 0:
            print(index/len(file_list), i/len(os.listdir(file[1])))

new_csv.to_csv(csv_path, encoding='UTF-8-SIG', index=False)