# Используем базовый образ PyTorch
FROM pytorch/pytorch:latest

# Устанавливаем необходимые пакеты, включая git
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /workspace

# Устанавливаем переменные окружения для Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Открываем порт для приложения
EXPOSE 5000

CMD ["flask", "run"]
