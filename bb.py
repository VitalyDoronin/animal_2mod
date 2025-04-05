from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Конфигурация
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Категории
CATEGORIES = ["cat", "chicken", "cow", "dog", "fox", "goat", "horse", "person", "racoon", "skunk"]


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index3.html', categories=CATEGORIES)


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return jsonify({'filename': filename})

    return jsonify({'error': 'Invalid file type'})


@app.route('/save_annotations', methods=['POST'])
def save_annotations():
    data = request.json
    filename = data['filename']
    annotations = data['annotations']

    # Создаем имя файла txt на основе имени изображения
    txt_filename = os.path.splitext(filename)[0] + '.txt'
    txt_path = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)

    # Записываем аннотации в формате YOLO
    with open(txt_path, 'w') as f:
        for ann in annotations:
            class_id = CATEGORIES.index(ann['category'])
            # YOLO формат: class_id center_x center_y width height
            line = f"{class_id} {ann['x']} {ann['y']} {ann['w']} {ann['h']}\n"
            f.write(line)

    return jsonify({'success': True})


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)