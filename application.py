import cv2
from flask import Flask, render_template, request, Response, jsonify
from src.pipelines.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

def generate_frames():
    cap = cv2.VideoCapture(0) # 0 for webcam
    fps = 32 # 16 -- will be less and 48 may be greater and later it can be treated as batch size.
    cap.set(cv2.CAP_PROP_FPS, fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break






@app.route("/")
def home_page():
    return render_template('index.html')


@app.route('/data_profile_report')
def data_analysis():
    return render_template('credit_card_data_profile_report.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        ## Take all data from form
        data = CustomData(
            V1= float(request.form.get('V1')), V2= float(request.form.get('V2')),
            V3= float(request.form.get('V3')), V4= float(request.form.get('V4')),
            V5= float(request.form.get('V5')), V6= float(request.form.get('V6')),
            V7= float(request.form.get('V7')),
            V8= float(request.form.get('V8')),
            V9= float(request.form.get('V9')), V10= float(request.form.get('V10')),
            V11= float(request.form.get('V11')), V12= float(request.form.get('V12')),
            V13= float(request.form.get('V13')), V14= float(request.form.get('V14')),
            V15= float(request.form.get('V15')), V16= float(request.form.get('V16')),
            V17= float(request.form.get('V17')),
            V18= float(request.form.get('V18')), V20= float(request.form.get('V20')),
            V21= float(request.form.get('V21')), V22= float(request.form.get('V22')),
            V23= float(request.form.get('V23')),
            V24= float(request.form.get('V24')),
            V25= float(request.form.get('V25')), V26= float(request.form.get('V26')),
            V27= float(request.form.get('V27')), V28= float(request.form.get('V28')),
            Amount= float(request.form.get('Amount'))
        )
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        results = round(pred[0],2)

        return render_template('results.html', final_result = results)
    

#
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)