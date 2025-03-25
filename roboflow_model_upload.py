from roboflow import Roboflow

# Передать API-ключ при инициализации Roboflow
rf = Roboflow(api_key="v3cFAranxYzTNutcgON4")

# Получаем проект
project = rf.workspace("rye-jo3uz").project("my-first-project-7i8pk")

# Указываем версию проекта (замените VERSION_NUMBER на номер версии, например, 1)
version = project.version(1)  # Убедитесь, что версия существует в вашем проекте

# Загружаем веса модели
version.deploy(
    model_type="yolov8-obb",  # Тип модели (например, yolov8-obb)
    model_path="last.pt"      # Путь к файлу с весами модели
)