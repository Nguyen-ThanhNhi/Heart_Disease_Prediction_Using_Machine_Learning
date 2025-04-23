import os
import numpy as np
import joblib
import json
from flask import Flask, render_template, request

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load scaler, mô hình, và ngưỡng
loaded_scaler = joblib.load(os.path.join(BASE_DIR, 'model/scaler.pkl'))
loaded_model_rf = joblib.load(os.path.join(BASE_DIR, 'model/random_forest_model.pkl'))
with open(os.path.join(BASE_DIR, 'model/threshold.json'), 'r') as f:
    THRESHOLD = json.load(f)['threshold']

# Load danh sách cột đặc trưng
X_columns = np.load(os.path.join(BASE_DIR, 'model/X_columns.npy')).tolist()
numeric_columns = [
    'Age', 'Sleep Hours', 'BMI', 'Homocysteine Level', 'Cholesterol Level',
    'Blood Pressure', 'Triglyceride Level', 'CRP Level', 'Fasting Blood Sugar'
]
numeric_idx = [X_columns.index(col) for col in numeric_columns]

def predict_heart_disease(SL, AGE, SH, BMI, HL, CL, BP, TL, CRP, FBS, model, threshold=THRESHOLD):
    # Tạo vector đặc trưng theo thứ tự trong X_columns
    x = np.array([SL, AGE, SH, BMI, HL, CL, BP, TL, CRP, FBS])
    
    # Debug: In vector đặc trưng trước chuẩn hóa
    print("Vector đặc trưng trước chuẩn hóa:", x)
    
    # Chuẩn hóa các cột số
    x[numeric_idx] = loaded_scaler.transform([x[numeric_idx]])[0]
    
    # Debug: In vector đặc trưng sau chuẩn hóa
    print("Vector đặc trưng sau chuẩn hóa:", x)
    
    # Dự đoán với ngưỡng
    prob = model.predict_proba([x])[0][1]  # Xác suất lớp 1 (bệnh)
    prediction = 1 if prob >= threshold else 0  # Áp dụng ngưỡng
    prob_percent = prob * 100
    
    # Debug: In dự đoán, xác suất, và ngưỡng
    print(f"Dự đoán: {prediction}, Xác suất lớp 1 (%): {prob_percent:.2f}, Ngưỡng: {threshold}")
    
    return prediction, prob_percent

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/index', methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            us_name = request.form["name"]
            # Đọc input từ form
            age = float(request.form["Age"])
            sh = float(request.form["Sleep Hours"])
            bmi = float(request.form["BMI"])
            hl = float(request.form["Homocysteine Level"])
            chol = float(request.form["Cholesterol Level"])
            bp = float(request.form["Blood Pressure"])
            tl = float(request.form["Triglyceride Level"])
            crp = float(request.form["CRP Level"])
            fbs = float(request.form["Fasting Blood Sugar"])

            # Stress Level => số
            sl_text = request.form["Stress Level"].lower()
            if sl_text == 'cao':
                sl = 1.0
            elif sl_text in ['thấp', 'thap']:
                sl = 0.0
            else:
                sl = 0.5

            # Debug: In dữ liệu đầu vào
            print("Dữ liệu đầu vào:", {
                'name': us_name, 'Age': age, 'Sleep Hours': sh, 'BMI': bmi,
                'Homocysteine Level': hl, 'Cholesterol Level': chol, 'Blood Pressure': bp,
                'Triglyceride Level': tl, 'CRP Level': crp, 'Fasting Blood Sugar': fbs,
                'Stress Level': sl
            })

            # Gọi hàm predict với mô hình Random Forest
            prediction, prob = predict_heart_disease(sl, age, sh,bmi,hl, chol,bp,tl,crp,fbs, loaded_model_rf)
            if prediction == 1:
                return render_template("result1.html", name=us_name, prob=prob)
            else:
                return render_template("result2.html", name=us_name, prob=prob)
        except Exception as e:
            print("Lỗi:", str(e))
            return render_template("index.html", error=f"Lỗi: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)