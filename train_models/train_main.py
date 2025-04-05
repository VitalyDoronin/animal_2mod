# from ultralytics import YOLO, settings
# import mlflow
# import os
#
# # Update a setting
# settings.update({"mlflow": True})
#
# mlflow.set_tracking_uri("http://localhost:5050")
#
# model = YOLO("yolo11n.pt")
# results = model.train(data='/Users/vi/Animals_Project/Animal/train_models/animals.v2-release.yolov11/data.yaml', epochs=1, imgsz=640, batch=16)
#
# # Export to ONNX
# model.export(format='onnx', path='model.onnx')
#
# # Assuming you've set up MLflow as shown in your code
# mlflow.log_artifact('model.onnx', artifact_path="models")
#
# # Register the model in MLflow Model Registry
# mlflow.register_model("models/model.onnx", "my_registered_model")

import mlflow
from ultralytics import YOLO, settings
import os

mlflow.set_tracking_uri("http://localhost:5050")
def main():
    # Enable MLflow logging
    # settings.update({"mlflow": True})

    # Set project and run names
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "YOLO_animal_proj"
    os.environ["MLFLOW_RUN"] = "YOLO_run5"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "mlflow"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "password"

    # Load the model
    model = YOLO('yolo11n.pt')

    # Define custom data path
    custom_data_path = r"/Users/vi/Animals_Project/Animal/train_models/animals.v2-release.yolov11/data.yaml"

    # Train the model
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