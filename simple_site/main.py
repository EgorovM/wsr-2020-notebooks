import datetime

import joblib
import numpy as np

import bottle
from bottle import route, request


# Загрузим модель
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')


@route('/')
def index():
    # Переводим в нужную страниу
    return bottle.redirect('/make_response/')


@route('/make_response/')
def make_response():
    return bottle.template(
        """
    <h2> Определите время вашей поездки </h2>
    <form action="/predict/" method="POST">
        <input name="lat_o" type="text" placeholder="Latitude отбытия" required><br>
        <input name="log_o" type="text" placeholder="Longitude отбытия" required><br>
        <input name="lat_p" type="text" placeholder="Latitude прибытия" required><br>
        <input name="log_p" type="text" placeholder="Longitude прибытия" required><br>
        <input name="send" type="submit" value="Отправить"><br>
    </form>
        """
    )


@route('/predict/', method='GET')
@route('/predict/', method='POST')
def predict():
    # Собираем введенные данные
    latitude_depart = request.forms.get('lat_o')
    longitude_depart = request.forms.get('log_o')
    latitude_arrive = request.forms.get('lat_p')
    longitude_arrive = request.forms.get('log_p')

    # Если какой из нибудь None, то переводим в /make_request/

    if not (latitude_depart and longitude_depart and latitude_arrive and longitude_arrive):
        return bottle.redirect('/make_response/')

    latitude_depart = float(latitude_depart)
    longitude_depart = float(longitude_depart)
    latitude_arrive = float(latitude_arrive)
    longitude_arrive = float(longitude_arrive)

    date = datetime.datetime.now()
    d, w = date.day, date.month

    # Допустим фичи: адрес прибытия, адрес отбытия, 
    # день и месяц сегодняшено числа
    # Нужно по хорошему их привести к такому же виду,
    # К которому приводится train/test

    # Нормализуем
    scaled_coords = scaler.transform([
        [latitude_depart, longitude_depart, latitude_arrive, longitude_arrive]
    ])

    encoded_f = encoder.transform([
        [d, w]
    ]).toarray()

    features = np.hstack((scaled_coords, encoded_f))

    time_duration = model.predict(features)[0]

    return bottle.template(
        """
        <p> Ваша поездка составит: {{time}} минут </p>
        <a href="/make_response/"> Повторить </a>
        """, time=time_duration)

    
bottle.run(host='localhost', port='8080')