from minio import Minio
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from flask import Flask, render_template, jsonify, request, redirect, session, url_for, send_file, make_response, send_from_directory, flash
import base64
import zipfile
import io
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo
from werkzeug.security import generate_password_hash, check_password_hash
import psycopg2
from minio.error import S3Error
import json
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import ssd300_vgg16

app = Flask(__name__)
app.config['SECRET_KEY'] = 'J5e-t4s-0t4-PuyE-TrWQ'

# Подключение к базе данных PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(
        host="postgres",
        port=5432,
        database="app-db",
        user="mlflow",
        password="password"
    )
    return conn

# MinIO конфигурация
MINIO_ACCESS_KEY = 'mlflow'
MINIO_SECRET_KEY = 'password'
MINIO_ENDPOINT = 's3:9000'
MINIO_BUCKET_NAME = 'myappbucket'

# Проверка переменных окружения
if not all([MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_ENDPOINT, MINIO_BUCKET_NAME]):
    raise ValueError("Необходимо задать переменные окружения MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_ENDPOINT и MINIO_BUCKET_NAME")

# Подключение к MinIO
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False  # True для HTTPS
)

# Проверка существования бакета и его создание, если необходимо
try:
    if not minio_client.bucket_exists(MINIO_BUCKET_NAME):
        minio_client.make_bucket(MINIO_BUCKET_NAME)
except Exception as e:
    print(f"Ошибка при работе с бакетом: {e}")

# Функция для установки политики бакета
def set_bucket_policy(minio_client, bucket_name, policy):
    try:
        minio_client.set_bucket_policy(bucket_name, json.dumps(policy))
        print(f"Политика для бакета {bucket_name} успешно установлена")
    except S3Error as e:
        print(f"Ошибка при установке политики для бакета {bucket_name}: {e}")

# Политика для бакета dataset (разрешить запись и чтение всем пользователям)
dataset_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": "*"},
            "Action": ["s3:PutObject"],
            "Resource": [f"arn:aws:s3:::dataset/*"]
        },
        {
            "Effect": "Allow",
            "Principal": {"AWS": "*"},
            "Action": ["s3:GetObject"],
            "Resource": [f"arn:aws:s3:::dataset/*"]
        }
    ]
}

app.config['UPLOAD_FOLDER'] = 'uploads'  # Папка для временного хранения загруженных файлов
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

cl = ["cat", "chicken", "cow", "dog", "fox", "goat", "horse", "person", "racoon", "skunk"]
cl_dict = {i: c for i, c in enumerate(cl)}
print(cl_dict)

# Папки с моделями
YOLO_MODELS_DIR = "yola_models"
SSD_MODELS_DIR = "ssd_models"

# Функция для получения списка моделей YOLO
def get_yolo_models():
    return [f for f in os.listdir(YOLO_MODELS_DIR) if f.endswith('.pt')]

# Функция для получения списка моделей SSD
def get_ssd_models():
    return [f for f in os.listdir(SSD_MODELS_DIR) if f.endswith('.pth')]

# Загрузка моделей (глобальные переменные)
current_model = None
current_model_type = None

# Функция для загрузки выбранной модели
def load_model(model_name, model_type):
    global current_model, current_model_type
    if model_type == "yolo":
        model_path = os.path.join(YOLO_MODELS_DIR, model_name)
        current_model = YOLO(model_path)
        current_model_type = "yolo"
    elif model_type == "ssd":
        model_path = os.path.join(SSD_MODELS_DIR, model_name)
        # Используем weights_backbone=None вместо deprecated pretrained_backbone
        model = ssd300_vgg16(weights=None, weights_backbone=None)
        num_classes = 11  # 10 классов + фон
        in_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = [4, 6, 6, 6, 4, 4]
        model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        # Загружаем только наши веса
        checkpoint = torch.load(model_path, map_location='cpu')  # Загружаем на CPU, чтобы избежать проблем с GPU
        model.load_state_dict(checkpoint)
        model.eval()
        current_model = model
        current_model_type = "ssd"
    else:
        raise ValueError("Неизвестный тип модели")

# Кэш для результатов (словарь: (класс, порог, тип модели, имя модели) -> список найденных объектов)
yolo_results_cache = {}

# Форма регистрации
class RegistrationForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired(), Length(min=2, max=20)])
    password = PasswordField('Пароль', validators=[DataRequired()])
    confirm_password = PasswordField('Подтвердите пароль', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Зарегистрироваться')

# Форма входа
class LoginForm(FlaskForm):
    username = StringField('Имя пользователя', validators=[DataRequired()])
    password = PasswordField('Пароль', validators=[DataRequired()])
    submit = SubmitField('Войти')

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        hashed_password = generate_password_hash(password)  # Хэширование пароля

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (%s, %s) RETURNING id", (username, hashed_password))
        user_id = cur.fetchone()[0]  # Получаем ID нового пользователя
        conn.commit()
        cur.close()
        conn.close()

        # Создание бакета для пользователя
        user_bucket_name = f"{MINIO_BUCKET_NAME}-{user_id}"  # Имя бакета, основанное на ID пользователя
        try:
            if not minio_client.bucket_exists(user_bucket_name):
                minio_client.make_bucket(user_bucket_name)
        except Exception as e:
            print(f"Ошибка при создании бакета {user_bucket_name}: {e}")

        flash('Регистрация прошла успешно! Теперь вы можете войти в систему.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    message = None  # Переменная для хранения сообщения
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        user = cur.fetchone()
        cur.close()
        conn.close()

        if user and check_password_hash(user[2], password):  # user[2] - это хэш пароля
            session['username'] = username  # Сохранение имени пользователя в сессии
            message = 'Вы вошли в систему!'  # Устанавливаем сообщение
            return redirect(url_for('index2', username=username))  # Переход на главную страницу с именем пользователя
        else:
            message = 'Неправильное имя пользователя или пароль'  # Устанавливаем сообщение

    return render_template('login.html', form=form, message=message)

@app.route('/index/<username>', methods=['GET', 'POST'])
def index2(username):
    global current_model, current_model_type
    confidence_threshold = None  # Установите значение по умолчанию

    # Обновляем списки моделей при каждом запросе
    yolo_models = get_yolo_models()
    ssd_models = get_ssd_models()

    if request.method == "POST":
        selected_class = request.form.get("class")
        confidence_threshold = request.form.get("confidence")  # Получаем значение из формы
        model_name = request.form.get("model_name")
        model_type = request.form.get("model_type")

        if selected_class and confidence_threshold and model_name and model_type:
            try:
                confidence_threshold = float(confidence_threshold)  # Приводим к float
                # Загружаем выбранную модель
                load_model(model_name, model_type)
                return redirect(url_for("show_results", selected_class=selected_class, confidence_threshold=confidence_threshold))
            except ValueError:
                flash('Пожалуйста, введите допустимое значение для порога уверенности.', 'danger')
                return redirect(url_for('index2', username=username))

    return render_template("index.html", classes=cl, username=username, confidence_threshold=confidence_threshold,
                           yolo_models=yolo_models, ssd_models=ssd_models)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'username' not in session:
        return jsonify({'error': 'Необходимо войти в систему'}), 403

    username = session['username']

    # Получаем ID пользователя из базы данных
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    user_id = cur.fetchone()[0]
    cur.close()
    conn.close()

    user_bucket_name = f"{MINIO_BUCKET_NAME}-{user_id}"  # Имя бакета пользователя
    files = request.files.getlist('files[]')

    if not files:
        return jsonify({'error': 'Нет файлов'}), 400

    # Создаем директорию uploads, если она не существует
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                minio_client.fput_object(user_bucket_name, filename, filepath)
                os.remove(filepath)
                filenames.append(filename)
            except Exception as e:
                return jsonify({'error': f'Ошибка при загрузке файла {filename} в MinIO: {e}'}), 500
        else:
            return jsonify({'error': f'Недопустимый тип файла: {file.filename}'}), 400

    return jsonify({'message': 'Файлы успешно загружены', 'filenames': filenames}), 200

def non_max_suppression(boxes, scores, threshold):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        order = order[np.where(iou <= threshold)[0] + 1]

    return keep

# Трансформация для предсказаний SSD
ssd_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_image(image_np, selected_class, confidence_threshold):
    global current_model, current_model_type
    height, width, _ = image_np.shape
    new_width = 480
    new_height = int(height * (new_width / width))
    resized_image = cv2.resize(image_np, (new_width, new_height))

    if current_model_type == "yolo":
        results = current_model.predict(resized_image)
        image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

        boxes = []
        scores = []
        labels = []

        for box in results[0].boxes.data:
            x1, y1, x2, y2, score, label = box
            if cl_dict[int(label)] == selected_class and score >= confidence_threshold:
                boxes.append([x1, y1, x2, y2])
                scores.append(score)
                labels.append(int(label))

    elif current_model_type == "ssd":
        try:
            # Преобразуем изображение для SSD
            image_pil = Image.fromarray(image_np)
            image_tensor = ssd_transform(image_pil).unsqueeze(0)  # Добавляем batch dimension

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            current_model.to(device)
            image_tensor = image_tensor.to(device)

            # Предсказание
            with torch.no_grad():
                predictions = current_model(image_tensor)

            # Извлекаем предсказания
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()

            # Фильтрация по классу и порогу уверенности
            filtered_boxes = []
            filtered_scores = []
            filtered_labels = []

            for i in range(len(boxes)):
                label = labels[i]
                score = scores[i]
                if label - 1 >= 0 and label - 1 < len(cl_dict):  # Проверяем, что label в допустимом диапазоне
                    if cl_dict[label - 1] == selected_class and score >= confidence_threshold:  # label - 1, т.к. SSD включает фон как класс 0
                        box = boxes[i]
                        # Масштабируем координаты обратно к размеру resized_image
                        x1 = box[0] * new_width / 300
                        y1 = box[1] * new_height / 300
                        x2 = box[2] * new_width / 300
                        y2 = box[3] * new_height / 300
                        filtered_boxes.append([x1, y1, x2, y2])
                        filtered_scores.append(score)
                        filtered_labels.append(label - 1)  # Учитываем смещение классов

            boxes = filtered_boxes
            scores = filtered_scores
            labels = filtered_labels
            image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"Ошибка при обработке SSD модели: {e}")
            boxes = []
            scores = []
            labels = []
            image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)

    else:
        raise ValueError("Модель не загружена или тип модели неизвестен")

    # Применяем Non-Maximum Suppression
    boxes = np.array(boxes)
    scores = np.array(scores)
    indices = non_max_suppression(boxes, scores, threshold=0.4)  # Порог IoU (перекрытие рамок)

    for i in indices:
        box = boxes[i]
        x1, y1, x2, y2 = box
        score = scores[i]
        text = f"{cl_dict[labels[i]]} {score:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        if y1 - text_height - 10 >= 0:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(image, text, (int(x1 + 5), int(y1 + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    _, image_buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(image_buffer).decode('utf-8')
    return image_base64

def find_images(selected_class, confidence_threshold):
    global current_model, current_model_type
    if 'username' not in session:
        return []

    username = session['username']

    # Получаем ID пользователя из базы данных
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    user_id = cur.fetchone()[0]
    cur.close()
    conn.close()

    user_bucket_name = f"{MINIO_BUCKET_NAME}-{user_id}"
    print(user_bucket_name)
    found_images = []

    objects = minio_client.list_objects(user_bucket_name, prefix="")
    for obj in objects:
        if obj.object_name.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                response = minio_client.get_object(user_bucket_name, obj.object_name)
                image_bytes = response.read()
                image_np = np.asarray(Image.open(BytesIO(image_bytes)))

                if current_model_type == "yolo":
                    results = current_model.predict(image_np)
                    for box in results[0].boxes.data:
                        x1, y1, x2, y2, score, label = box
                        if cl_dict[int(label)] == selected_class and score >= confidence_threshold:
                            found_images.append(obj.object_name)
                            break

                elif current_model_type == "ssd":
                    try:
                        # Преобразуем изображение для SSD
                        image_pil = Image.fromarray(image_np)
                        image_tensor = ssd_transform(image_pil).unsqueeze(0)  # Добавляем batch dimension

                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        current_model.to(device)
                        image_tensor = image_tensor.to(device)

                        # Предсказание
                        with torch.no_grad():
                            predictions = current_model(image_tensor)

                        # Извлекаем предсказания
                        boxes = predictions[0]['boxes'].cpu().numpy()
                        scores = predictions[0]['scores'].cpu().numpy()
                        labels = predictions[0]['labels'].cpu().numpy()

                        for i in range(len(boxes)):
                            label = labels[i]
                            score = scores[i]
                            if label - 1 >= 0 and label - 1 < len(cl_dict):  # Проверяем, что label в допустимом диапазоне
                                if cl_dict[label - 1] == selected_class and score >= confidence_threshold:  # label - 1, т.к. SSD включает фон как класс 0
                                    found_images.append(obj.object_name)
                                    break
                    except Exception as e:
                        print(f"Ошибка при обработке SSD модели: {e}")
                        continue

            except Exception as e:
                print(f"Ошибка обработки изображения {obj.object_name}: {e}")
                continue
    return found_images

@app.route("/results/<selected_class>/<float:confidence_threshold>")
def show_results(selected_class, confidence_threshold):
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']
    images = []
    cache_key = (selected_class, confidence_threshold, current_model_type, current_model.__class__.__name__)

    # Получаем ID пользователя из базы данных
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    user_id = cur.fetchone()[0]
    cur.close()
    conn.close()

    user_bucket_name = f"{MINIO_BUCKET_NAME}-{user_id}"

    if cache_key not in yolo_results_cache:
        found_images = find_images(selected_class, confidence_threshold)  # Получаем список файлов
        yolo_results_cache[cache_key] = found_images
    else:
        found_images = yolo_results_cache[cache_key]

    for filename in found_images:
        try:
            response = minio_client.get_object(user_bucket_name, filename)
            image_bytes = response.read()
            image_np = np.asarray(Image.open(BytesIO(image_bytes)))
            image_base64 = process_image(image_np, selected_class, confidence_threshold)
            images.append({'filename': filename, 'image_base64': image_base64})
        except Exception as e:
            print(f"Ошибка обработки изображения {filename}: {e}")
            continue

    return render_template("results.html", images=images, selected_class=selected_class, confidence_threshold=confidence_threshold,
                           download_url=url_for('download_results', selected_class=selected_class, confidence_threshold=confidence_threshold),
                           username=username)

@app.route("/download/<selected_class>/<float:confidence_threshold>")
def download_results(selected_class, confidence_threshold):
    cache_key = (selected_class, confidence_threshold, current_model_type, current_model.__class__.__name__)
    found_images = yolo_results_cache.get(cache_key, [])  # Получаем из кэша

    if not found_images:
        return "Нет изображений, соответствующих критериям поиска.", 404

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for filename in found_images:
            try:
                response = minio_client.get_object(MINIO_BUCKET_NAME, filename)
                zf.writestr(filename, response.read())
            except Exception as e:
                print(f"Ошибка загрузки изображения {filename} в архив: {e}")
                continue

    memory_file.seek(0)
    response = make_response(memory_file.read())
    response.headers["Content-Disposition"] = "attachment; filename=found_images.zip"
    response.headers["Content-Type"] = "application/zip"
    return response

@app.route('/logout')
def logout():
    session.pop('username', None)  # Удаляем имя пользователя из сессии
    return redirect(url_for('login'))  # Перенаправляем на страницу входа

@app.route('/edit_image', methods=['GET', 'POST'])
def edit_image():
    if 'username' not in session:
        return redirect(url_for('login'))

    username = session['username']

    # Извлекаем параметры из request.args (для GET и POST)
    filename = request.args.get('filename')
    selected_class = request.args.get('selected_class')
    confidence_threshold = request.args.get('confidence_threshold')

    if not filename:
        print("Ошибка: параметр filename отсутствует в запросе")
        return jsonify({'error': 'Изображение не найдено: параметр filename отсутствует'}), 400

    try:
        confidence_threshold = float(confidence_threshold) if confidence_threshold else 0.5
    except (ValueError, TypeError):
        confidence_threshold = 0.5  # Значение по умолчанию

    # Получаем ID пользователя из базы данных
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = %s", (username,))
    user_id = cur.fetchone()[0]
    cur.close()
    conn.close()

    if request.method == 'POST':
        print("Получен POST-запрос")
        print(f"Заголовки запроса: {request.headers}")
        print(f"Тело запроса: {request.get_data(as_text=True)}")

        if not request.is_json:
            print("Ошибка: данные не в формате JSON")
            return jsonify({'error': 'Отсутствуют данные в формате JSON'}), 400

        data = request.json
        print(f"Полученные данные: {data}")

        annotations = data.get('annotations', [])
        if not annotations:
            print("Ошибка: аннотации отсутствуют в запросе")
            return jsonify({'error': 'Аннотации отсутствуют в запросе'}), 400

        user_bucket_name = f"{MINIO_BUCKET_NAME}-{user_id}"

        # Извлекаем изображение из MinIO
        try:
            response = minio_client.get_object(user_bucket_name, filename)
            image_bytes = response.read()
            if not image_bytes:
                print(f"Ошибка: изображение {filename} пустое или повреждено")
                return jsonify({'error': 'Изображение пустое или повреждено'}), 400
        except S3Error as e:
            print(f"Ошибка при извлечении изображения {filename} из MinIO: {e}")
            return jsonify({'error': f"Ошибка при загрузке изображения из MinIO: {e}"}), 500

        # Декодируем изображение в байты
        image_io = BytesIO(image_bytes)
        image_io.seek(0)

        # Формируем уникальное имя файла с user_id и временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"annotated_{user_id}_{timestamp}_{filename}"
        dataset_bucket_name = "dataset"

        # Сохраняем изображение в бакет dataset
        try:
            if not minio_client.bucket_exists(dataset_bucket_name):
                print(f"Создаем бакет {dataset_bucket_name}")
                minio_client.make_bucket(dataset_bucket_name)
                set_bucket_policy(minio_client, dataset_bucket_name, dataset_policy)

            print(f"Сохраняем изображение {new_filename} в бакет {dataset_bucket_name}")
            minio_client.put_object(
                dataset_bucket_name,
                new_filename,
                image_io,
                length=len(image_bytes),
                content_type='image/jpeg'
            )
            print(f"Изображение {new_filename} успешно сохранено в бакет {dataset_bucket_name}")
        except S3Error as e:
            print(f"Ошибка при сохранении изображения в MinIO: {e}")
            return jsonify({'error': f'Ошибка при сохранении изображения в MinIO: {e}'}), 500

        # Сохраняем аннотации в текстовый файл в формате YOLO
        txt_filename = f"annotated_{user_id}_{timestamp}_{os.path.splitext(filename)[0]}.txt"
        txt_io = BytesIO()
        for ann in annotations:
            try:
                class_id = cl.index(ann['category'])
                line = f"{class_id} {ann['x']} {ann['y']} {ann['w']} {ann['h']}\n"
                txt_io.write(line.encode('utf-8'))
            except KeyError as e:
                print(f"Ошибка в данных аннотации: отсутствует ключ {e}")
                return jsonify({'error': f'Ошибка в данных аннотации: отсутствует ключ {e}'}), 400
        txt_io.seek(0)
        if txt_io.getbuffer().nbytes == 0:
            print("Ошибка: аннотации пустые")
            return jsonify({'error': 'Аннотации пустые'}), 400

        try:
            print(f"Сохраняем аннотации {txt_filename} в бакет {dataset_bucket_name}")
            minio_client.put_object(
                dataset_bucket_name,
                txt_filename,
                txt_io,
                length=txt_io.getbuffer().nbytes,
                content_type='text/plain'
            )
            print(f"Аннотации {txt_filename} успешно сохранены в бакет {dataset_bucket_name}")
        except S3Error as e:
            print(f"Ошибка при сохранении аннотаций в MinIO: {e}")
            return jsonify({'error': f'Ошибка при сохранении аннотаций в MinIO: {e}'}), 500

        print("Аннотации успешно сохранены")
        return jsonify({'success': True, 'message': 'Аннотации успешно сохранены'})

    # Если GET-запрос, отображаем страницу редактирования
    user_bucket_name = f"{MINIO_BUCKET_NAME}-{user_id}"

    # Извлекаем изображение из MinIO
    try:
        response = minio_client.get_object(user_bucket_name, filename)
        image_bytes = response.read()
        if not image_bytes:
            print(f"Ошибка: изображение {filename} пустое или повреждено")
            return jsonify({'error': 'Изображение пустое или повреждено'}), 400
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    except S3Error as e:
        print(f"Ошибка при извлечении изображения {filename} из MinIO: {e}")
        return jsonify({'error': f"Ошибка при загрузке изображения из MinIO: {e}"}), 500

    return render_template('edit_image.html', image_base64=image_base64, categories=cl, username=username, filename=filename, selected_class=selected_class, confidence_threshold=confidence_threshold)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
