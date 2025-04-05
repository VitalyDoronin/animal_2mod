from flask import Flask, render_template, request, jsonify, url_for
from PIL import Image
import os

app = Flask(__name__)

# Путь к исходному изображению
IMAGE_PATH = os.path.join('static', 'images', 'image.jpg')
# Путь к измененному изображению
RESIZED_IMAGE_PATH = os.path.join('static', 'images', 'resized_image.jpg')

def resize_image(width=300):
    """Изменяет размер изображения до указанной ширины с пропорциональной высотой."""
    with Image.open(IMAGE_PATH) as img:
        # Вычисляем пропорциональную высоту
        w_percent = width / float(img.size[0])
        height = int(float(img.size[1]) * float(w_percent))
        # Изменяем размер
        resized_img = img.resize((width, height), Image.Resampling.LANCZOS)
        # Сохраняем измененное изображение
        resized_img.save(RESIZED_IMAGE_PATH)

@app.route('/')
def index():
    # Изменяем размер изображения перед отображением
    resize_image(300)
    return render_template('index2.html', image_url=url_for('static', filename='images/resized_image.jpg'))

@app.route('/save_coords', methods=['POST'])
def save_coords():
    data = request.json
    with open('coordinates.txt', 'w') as f:
        for box in data:
            f.write(f"Class: {box['class']}, Coordinates: {box['topLeft']}, {box['topRight']}, {box['bottomLeft']}, {box['bottomRight']}\n")
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)