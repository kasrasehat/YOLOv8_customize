from ultralytics import YOLO
import torch
from torchsummary import summary
import cv2
import os
from PIL import Image

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


def process_tensor(tensor, image, file_name):
    for i in range(tensor.size(0)):
        # Check if the last element (label) is 11
        if tensor[i, -1] == 11:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, tensor[i, :4])

            # Crop and rotate the bounding box
            cropped_image = image[y1:y2, x1:x2]
            cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)

            # Save the image
            cv2.imwrite(f'/home/kasra/kasra_files/data-shenasname/seria_nembers/{file_name}_{i}', cropped_image)
        elif tensor[i, -1] in [0, 5, 6]:
            continue
        else:
            # Extract coordinates
            x1, y1, x2, y2 = map(int, tensor[i, :4])

            # Crop and rotate the bounding box
            cropped_image = image[y1:y2, x1:x2]
            image = cv2.resize(image, (256, 128))

            # Save the image
            cv2.imwrite(f'/home/kasra/kasra_files/data-shenasname/cropped_data/{YOLO_LABEL}_{MAIN_LABEL}', cropped_image)


# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = YOLO("/home/kasra/PycharmProjects/YOLOv8_customize/runs/detect/train/weights/best.pt")  # load a pretrained model (recommended for training)
for idx, file_name in enumerate(os.listdir('/home/kasra/kasra_files/data-shenasname/poshte_card')):
    img = '/home/kasra/kasra_files/data-shenasname/poshte_card' + '/' + file_name  # or file, Path, PIL, OpenCV, numpy, list
    image = cv2.imread(img)
    new_width = 640
    new_height = 640
    # Resize the image
    image = cv2.resize(image, (new_width, new_height))
    # Inference
    img = '/home/kasra/PycharmProjects/YOLOv8_customize/extra_files/image1.jpg'

    save_image(image, img)
    results = model.predict(img, save=True, imgsz=640, conf=0.3, save_txt=False)
    process_tensor(results[0].boxes.data, image, file_name) # returns xyxy of bounding box + confidence and class number