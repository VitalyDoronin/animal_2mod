let startX, startY; // Начальные координаты рамки
let isDrawing = false; // Флаг, указывающий, что мы в процессе рисования
let currentBox = null; // Текущая рамка
let boxes = []; // Массив для хранения всех рамок

const canvas = document.getElementById('canvas');
const classSelector = document.getElementById('classSelector');

canvas.addEventListener('click', function(e) {
    if (!isDrawing) {
        // Первый клик: начало рисования рамки
        isDrawing = true;
        startX = e.offsetX;
        startY = e.offsetY;

        // Создаём элемент рамки
        currentBox = document.createElement('div');
        currentBox.className = 'box';
        currentBox.style.position = 'absolute'; // Позиционируем рамку абсолютно
        currentBox.style.border = '2px solid red'; // Стиль рамки
        canvas.appendChild(currentBox);
    } else {
        // Второй клик: завершение рисования рамки
        isDrawing = false;

        // Сохраняем рамку в массив
        const selectedClass = classSelector.value;
        boxes.push({ element: currentBox, class: selectedClass });
        currentBox = null; // Сбрасываем текущую рамку
    }
});

canvas.addEventListener('mousemove', function(e) {
    if (isDrawing && currentBox) {
        // Текущие координаты курсора
        const endX = e.offsetX;
        const endY = e.offsetY;

        // Вычисляем ширину и высоту рамки
        const width = Math.abs(endX - startX);
        const height = Math.abs(endY - startY);

        // Устанавливаем позицию и размер рамки
        currentBox.style.left = Math.min(startX, endX) + 'px';
        currentBox.style.top = Math.min(startY, endY) + 'px';
        currentBox.style.width = width + 'px';
        currentBox.style.height = height + 'px';
    }
});

function clearBoxes() {
    // Удаляем все рамки
    boxes.forEach(box => box.element.remove());
    boxes = [];
}

function saveCoords() {
    // Сохраняем координаты рамок
    const coords = boxes.map(box => {
        const rect = box.element.getBoundingClientRect();
        const canvasRect = canvas.getBoundingClientRect();
        const img = document.querySelector('#canvas img');
        const scaleX = img.naturalWidth / img.clientWidth;
        const scaleY = img.naturalHeight / img.clientHeight;
        return {
            class: box.class,
            topLeft: { x: (rect.left - canvasRect.left) * scaleX, y: (rect.top - canvasRect.top) * scaleY },
            topRight: { x: (rect.right - canvasRect.left) * scaleX, y: (rect.top - canvasRect.top) * scaleY },
            bottomLeft: { x: (rect.left - canvasRect.left) * scaleX, y: (rect.bottom - canvasRect.top) * scaleY },
            bottomRight: { x: (rect.right - canvasRect.left) * scaleX, y: (rect.bottom - canvasRect.top) * scaleY }
        };
    });

    // Отправляем координаты на сервер
    fetch('/save_coords', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(coords)
    })
    .then(response => response.json())
    .then(data => console.log(data))
    .catch((error) => console.error('Error:', error));
}
