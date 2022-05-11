# 9_evaluation_selection
Задание 7
![image](https://user-images.githubusercontent.com/75991746/167729482-b9f12c4e-fce5-462e-82fe-6ec563715bcd.png)

Задание 8
![image](https://user-images.githubusercontent.com/75991746/167732018-e43b3455-51ab-440f-a443-3f83da8a93bc.png)

Задание 9
![image](https://user-images.githubusercontent.com/75991746/167841002-97cc793b-4456-4a06-8de6-d76dcc0ad6a1.png)

Задание 10
Инструкция:
1. Клонируйте репозиторий https://github.com/devbrest/9_evaluation_selection.git 
2. Создайте папку data в корне репозитория
3. Скопируйте файлы csv с данными forest dataset  из домашнего задания в папку data
4. Убедитесь что Вы используете нужную версию Python должна быть 3.9
5. Установите poetry 
6. Установите зависимости проекта poetry install --no-dev
7. Для запуска тренировки модели используйте poetry run train
8. Для запуска mlflow используйте poetry run mlflow ui
9. Для запуска 9-го задания в терминале запустите poetry run train_nested без параметров, в этом случае будет запущен поиск 
оптимальных параметров
10. poetry run train_nested --use-nested=False --max_depth=... --n_estimators=... --max_features=... --criterion='...' используется для запуска с оптимальными параметрами
11. Для следующих заданий нужно запустить poetry install
12. Для запуска теста выполнить poetry run pytest
13. Для установки nox  в терминале ввести pip install --user --upgrade nox
