import mlflow
from ultralytics import YOLO, settings
import os

mlflow.set_tracking_uri("http://localhost:5050")
def main():
    # Включаем ведение журнала MLflow
    # settings.update({"mlflow": True})

    os.environ["MLFLOW_EXPERIMENT_NAME"] = "YOLO_animal_proj"
    os.environ["MLFLOW_RUN"] = "YOLO_run5"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "mlflow"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password"

    # Загружаем модель
    model = YOLO('yolo11n.pt')

    # Указываем путь к данным
    custom_data_path = r"/Users/vi/Animals_Project/Animal/train_models/animals.v2-release.yolov11/data.yaml"

    # Указываем параметры обучения модели
    results = model.train(
        data=custom_data_path,
        imgsz = 320,
        epochs=2,
        batch=32,
        optimizer='Adam',
        name='temp'
    )

if __name__ == "__main__":
    main()
