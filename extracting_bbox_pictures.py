import os
import cv2
from tqdm import tqdm  # Импортируем tqdm для отображения прогресса

# Пути к изображениям и меткам
image_dir = 'data/images/train'
label_dir = 'data/labels/train'
output_dir = 'data/cropped_images'

# Создаем директорию для сохранения вырезанных изображений
os.makedirs(output_dir, exist_ok=True)

# Функция для преобразования YOLO формата в координаты пикселей
def yolo_to_pixels(bbox, img_width, img_height):
    x_center, y_center, width, height = bbox
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    return x_min, y_min, x_max, y_max

# Получаем список всех изображений
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

# Используем tqdm для отображения прогресса
for image_name in tqdm(image_files, desc="Обработка изображений"):
    # Загружаем изображение
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    img_height, img_width, _ = image.shape

    # Загружаем соответствующий файл с метками
    label_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
    label_path = os.path.join(label_dir, label_name)

    if os.path.exists(label_path):
        with open(label_path, 'r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                # Парсим ограничивающую рамку
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                bbox = (x_center, y_center, width, height)

                # Преобразуем YOLO формат в координаты пикселей
                x_min, y_min, x_max, y_max = yolo_to_pixels(bbox, img_width, img_height)

                # Вырезаем область изображения
                cropped_image = image[y_min:y_max, x_min:x_max]

                # Сохраняем вырезанное изображение
                output_image_name = f"{image_name.split('.')[0]}_bbox_{i}.jpg"
                output_image_path = os.path.join(output_dir, output_image_name)
                cv2.imwrite(output_image_path, cropped_image)

print(len(os.listdir(output_dir)), "изображений успешно извлечены из рамок и сохранены в директорию", output_dir)