import requests
import os
from PIL import Image, ImageDraw
#!pip install roboflow
from roboflow import Roboflow
import math

# Function to draw red bounding boxes for all trees regardless of class and no counting
# def process_images_with_roboflow(images_folder):
#     # API endpoint for object detection
#     rf = Roboflow(api_key="7PXFRJH4bJbTNKYyUfww")
#     project = rf.workspace("henry-kjyef").project("treedetection1")
#     model = project.version(7).model

#     # Iterate through images in the folder
#     for image_file in os.listdir(images_folder):
#         if image_file.endswith('.jpg') or image_file.endswith('.png'):
#             # Prepare image data
#             image_path = os.path.join(images_folder, image_file)
#             image_data = open(image_path, 'rb').read()

#             # Make request to Roboflow API
#             response_data = model.predict(str(image_path), confidence=40, overlap=30).json()
#             #response_data = response.json()
#             print(response_data)

#             # Retrieve bounding box data
#             # bounding_boxes = response_data['predictions']

#             bounding_boxes = response_data['predictions']
#             original_image = Image.open(image_path)
#             draw = ImageDraw.Draw(original_image)
#             for box in bounding_boxes:
#                 x, y = box['x'], box['y']
#                 width, height = box['width'], box['height']
#                 x1 = x - (width / 2)
#                 y1 = y - (height / 2)
#                 x2 = x + (width / 2)
#                 y2 = y + (height / 2)
#                 draw.rectangle([x1, y1, x2, y2], outline='red', width=20)

#             # Save annotated image
#             annotated_image_path = os.path.join(images_folder, 'annotated_' + image_file)
#             original_image.save(annotated_image_path)
#             print(f"Annotated image saved: {annotated_image_path}")

def process_images_with_roboflow(images_folder):
    # Mapping classes to colors
    class_colors = {
        "yellow": "turquoise",
        "bare": "pink",
        "trees": "yellow",
        "brown": "red"
    }
    class_counts = {class_name: 0 for class_name in class_colors.keys()}
    total_bbox_count = 0

    # API endpoint for object detection
    rf = Roboflow(api_key="7PXFRJH4bJbTNKYyUfww")
    project = rf.workspace("henry-kjyef").project("treedetection1")
    model = project.version(7).model

    # Iterate through images in the folder
    for image_file in os.listdir(images_folder):
        if image_file.endswith('.jpg') or image_file.endswith('.png'):
            # Prepare image data
            image_path = os.path.join(images_folder, image_file)
            image_data = open(image_path, 'rb').read()

            # Make request to Roboflow API
            response_data = model.predict(str(image_path), confidence=40, overlap=30).json()
            #response_data = response.json()
            print(response_data)

            # Retrieve bounding box data
            bounding_boxes = response_data['predictions']

            original_image = Image.open(image_path)
            draw = ImageDraw.Draw(original_image)
            for box in bounding_boxes:
                x, y = box['x'], box['y']
                width, height = box['width'], box['height']
                class_name = box['class']
                class_counts[class_name] += 1
                total_bbox_count += 1

                color = class_colors.get(class_name, "white")  # Default to white if class color not found
                draw.rectangle([x - (width / 2), y - (height / 2), x + (width / 2), y + (height / 2)],
                               outline=color, width=20)

            # Save annotated image
            annotated_image_path = os.path.join(images_folder, 'annotated_' + image_file)
            original_image.save(annotated_image_path)
            print(f"Annotated image saved: {annotated_image_path}")

    print("Bounding box counts by class:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    print(f"Total bounding boxes: {total_bbox_count}")


def tile_and_resize_annotated_images(images_folder, target_size=(2560, 2560)):
    annotated_images = []
    for image_file in sorted(os.listdir(images_folder)):
        if image_file.startswith('annotated_') and image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_folder, image_file)
            annotated_images.append(Image.open(image_path))
    num_images = len(annotated_images)
    if num_images == 0:
        print("No annotated images found in the folder.")
        return

    # Calculate the number of rows and columns needed for a square grid
    num_cols = math.ceil(math.sqrt(num_images))
    num_rows = math.ceil(num_images / num_cols)

    # Get the maximum width and height of annotated images
    max_width = max(img.size[0] for img in annotated_images)
    max_height = max(img.size[1] for img in annotated_images)

    # Create a new blank image to hold the tiled images
    tiled_width = max_width * num_cols
    tiled_height = max_height * num_rows
    tiled_image = Image.new('RGB', (tiled_width, tiled_height))

    # Paste annotated images into the tiled image
    x_offset = 0
    y_offset = 0
    for img in annotated_images:
        tiled_image.paste(img, (x_offset, y_offset))
        x_offset += max_width
        if x_offset >= tiled_width:
            x_offset = 0
            y_offset += max_height

    # Resize the tiled image to fit within the target size
    tiled_image_resized = tiled_image.resize(target_size, Image.ANTIALIAS)

    # Save the resized tiled image
    tiled_image_resized.save('resized_tiled_annotated_output.jpg')
    print("Resized tiled annotated image saved: resized_tiled_annotated_output.jpg")

# Main function
def main():
    images_folder = "/content/images3"
    process_images_with_roboflow(images_folder)
    tile_and_resize_annotated_images(images_folder)

main()
