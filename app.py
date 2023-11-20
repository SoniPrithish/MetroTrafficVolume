from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application


@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
            Temp=float(request.form.get('Temp')),
            Rain = float(request.form.get('Rain')),
            Snow = float(request.form.get('Snow')),
            Clouds = float(request.form.get('Clouds')),
            Holiday = bool(request.form.get('Holiday')),
            Weather=  (request.form.get('Weather')),
            Time = int(request.form.get('Time')),
            Month= int(request.form.get('Month')),
            Day = request.form.get('Day')
        )
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)
        print(results)

        return render_template('results.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)