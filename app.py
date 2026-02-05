# app.py

from flask import Flask, render_template, request
from vecm_model import build_vecm_and_forecast

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    historical_html = None
    forecast_html = None

    if request.method == "POST":
        hist_df, forecast_df = build_vecm_and_forecast(steps=15)

        historical_html = hist_df.to_html(classes='table table-bordered')
        forecast_html = forecast_df.to_html(classes='table table-bordered')

    return render_template(
        "index.html",
        historical_data=historical_html,
        forecast_data=forecast_html
    )


if __name__ == "__main__":
    app.run(debug=True)
