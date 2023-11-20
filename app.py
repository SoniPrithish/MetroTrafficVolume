from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


application=Flask(__name__)

app=application


def parse_bool(value):
    return str(value).strip().lower() in {'true', '1', 'yes', 'y', 'on'}


def describe_prediction(volume):
    if volume < 1500:
        return (
            'Low flow',
            'Light traffic conditions with headroom for normal operations.',
            'low',
        )
    if volume < 3500:
        return (
            'Steady flow',
            'Traffic is active but should remain manageable through most of the hour.',
            'steady',
        )
    if volume < 5500:
        return (
            'High flow',
            'Expect sustained demand with a higher chance of slower segments and queue buildup.',
            'high',
        )

    return (
        'Peak congestion',
        'Demand is approaching peak corridor pressure and may require proactive intervention.',
        'peak',
    )


@app.route('/')
def home_page():
    return render_template('index.html', title='Metro Traffic Volume Prediction')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html', title='Run Traffic Forecast')
    
    else:
        temp = float(request.form.get('Temp'))
        rain = float(request.form.get('Rain'))
        snow = float(request.form.get('Snow'))
        clouds = float(request.form.get('Clouds'))
        holiday = parse_bool(request.form.get('Holiday'))
        weather = request.form.get('Weather')
        time = int(request.form.get('Time'))
        month = int(request.form.get('Month'))
        day = request.form.get('Day')

        data=CustomData(
            Temp=temp,
            Rain=rain,
            Snow=snow,
            Clouds=clouds,
            Holiday=holiday,
            Weather=weather,
            Time=time,
            Month=month,
            Day=day
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        result_value = round(pred[0], 2)
        traffic_band, traffic_note, band_slug = describe_prediction(result_value)

        submitted_data = {
            'Temperature': f'{temp:.1f} K ({temp - 273.15:.1f} C)',
            'Rainfall': f'{rain:.1f} mm',
            'Snowfall': f'{snow:.1f} mm',
            'Cloud Cover': f'{clouds:.0f}%',
            'Holiday': 'Yes' if holiday else 'No',
            'Weather': weather,
            'Hour': f'{time:02d}:00',
            'Month': month,
            'Day': day,
        }

        return render_template(
            'results.html',
            title='Forecast Result',
            final_result=result_value,
            traffic_band=traffic_band,
            traffic_note=traffic_note,
            band_slug=band_slug,
            submitted_data=submitted_data,
        )






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=False)
