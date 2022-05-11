# 9_evaluation_selection
Задание 2
Домашняя работа овормлена как package с использованием src layout. Выбор обусловлен удобством в дальнейшем использовать Poetry для запуска кода в таком случае.

Задание 5

![image](https://user-images.githubusercontent.com/75991746/167882335-1c900bb8-e72b-4149-b1e3-c53c924c636c.png)

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
14. Для проверки заданий 12, 13, 14 запустите nox,  или nox  с указанием полного пути к папке установки скрипта

Задание 12
![image](https://user-images.githubusercontent.com/75991746/167876179-ded43384-3fc1-4d76-ab73-82aaa4ae0d4c.png)

![image](https://user-images.githubusercontent.com/75991746/167881073-c93a8081-69a3-4f32-89e8-ae21cf6e8fc4.png)

Задание 13
![image](https://user-images.githubusercontent.com/75991746/167876463-bd8c5c5d-4039-4b08-8ea0-4ba31d3c8413.png)

Задание 14
![image](https://user-images.githubusercontent.com/75991746/167877389-6c38c51d-c7e2-4e74-a98c-10e3e2bef791.png)
