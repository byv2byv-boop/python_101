import requests
from flask import Flask, render_template, jsonify
from configuer import WEATHER_API_KEY
CITY = "Москва"

app = Flask(__name__)
app.json.ensure_ascii = False


#ПОлучаем данные о погоде
def weather():
    params = {
    "key": WEATHER_API_KEY,
    "q": CITY,
    "lang": "ru"
}


    response = requests.get(
        "http://api.weatherapi.com/v1/current.json",
        params=params,
        timeout=10
    )
    data=response.json()
#во тпо такой ссылке мы это найдем https://api.weatherapi.com/v1/current.json?q=%22%D0%9C%D0%BE%D1%81%D0%BA%D0%B2%D0%B0%22&key=f8fcdc3d2d1d4972872185213262109&lang=%22ru%22

    our_data = { 
            "city": data["location"] ["name"],
            "timezone": data["location"] ["tz_id"],
            "temp": round (data["current"] ["temp_c"]),
            "feels": round(data["current"] ["feelslike_c"]),
            "text": data["current"] ["condition"] ["text"]
            }
    return our_data

@app.route("/")
def index():
    return render_template("index.html",w=weather())

@app.route("/api/weather")
def api():
    return jsonify(weather())

app.run(debug=True,port=5004)
print("ЕЕЕ ЙиииХА!")