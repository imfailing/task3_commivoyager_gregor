 Задача 3. Рекомендательная система для пользователей московских библиотек
 
Модуль подготовки модели

Формирует обученную модель на основе предложенных датасетов
Пример запуска:

model_creation.py --book <путь до датасета с книгами (books.jsn)> --users <путь до датасета с заказами пользователей(dataset_knigi_1.xlsx)> --model_dataset <путь для сохранения предобработанного датасета для модели> --model <путь для сохранения модели, по умолчанию models/final_model.sav>

Модуль взаимодействия

REST сервис, использующий фреймворк Flask и сохраненную модель

Пример запуска:
app.py 

Все зависимости лежат в requirements.txt