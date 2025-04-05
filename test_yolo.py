import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os


cl = ["cat", "chicken", "cow", "dog", "fox", "goat", "horse", "person", "racoon", "skunk"]

cl_dict = {i: c for i, c in enumerate(cl)}
print(cl_dict)

model = YOLO("best.pt")

target_class = "racoon"

image_folder = "test_img"

image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".jpeg", ".png"))]

# Цикл поиска по изображениям в каталоге
for image_file in image_files:
  image_path = os.path.join(image_folder, image_file)
  image = Image.open(image_path)
  image = np.asarray(image)

  # Предсказание на изображении
  results = model.predict(image)

  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  # Проверка наличия целевого класса на изображении
  found_target = False
  for box in results[0].boxes.data:
    x1, y1, x2, y2, score, label = box
    if cl_dict[int(label)] == target_class:
      found_target = True

      # Рисуем bounding box
      cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

      # Добавляем текст с именем класса и уверенностью
      text = f"{cl_dict[int(label)]} {score:.2f}"
      cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 2)

  # Вывод изображения, если найден целевой класс
  if found_target:
    plt.imshow(image)
    plt.show()