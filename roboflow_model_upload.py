from roboflow import Roboflow

# Передать API-ключ при инициализации Roboflow
rf = Roboflow(api_key="v3cFAranxYzTNutcgON4")

# Получаем проект
workspace = rf.workspace("rye-jo3uz")

# Загружаем веса модели
workspace.deploy_model(
    model_type="yolov11-obb",  # Тип модели (например, yolov8-obb)
    model_path="best.pt",      # Путь к файлу с весами модели
    project_ids=['my-first-project-7i8pk'],
    model_name="last"
)
'https://detect.roboflow.com/my-first-project-7i8pk/last?api_key=v3cFAranxYzTNutcgON4'