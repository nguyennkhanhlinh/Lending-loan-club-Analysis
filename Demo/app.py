from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model đã huấn luyện
with open(r'C:\Users\Laptop\All Lending Club loan data\Demo\model_1.sav', 'rb') as f:
    model = pickle.load(f)
print(f"Model type: {type(model)}")
# Từ điển encode
encode = {
    "emp_length": {
        '< 1 year': 0.0, '1 year': 1.0, '2 years': 2.0, '3 years': 3.0,
        '4 years': 4.0, '5 years': 5.0, '6 years': 6.0, '7 years': 7.0,
        '8 years': 8.0, '9 years': 9.0, '10+ years': 10.0
    },
    "home_ownership": {
        'ANY': 0.0, 'NONE': 1.0, 'OTHER': 2.0, 'OWN': 3.0, 'MORTGAGE': 4.0, 'RENT': 5.0
    },
    "sub_grade": {
        grade: i + 1.0 for i, grade in enumerate([
            "A1", "A2", "A3", "A4", "A5",
            "B1", "B2", "B3", "B4", "B5",
            "C1", "C2", "C3", "C4", "C5",
            "D1", "D2", "D3", "D4", "D5",
            "E1", "E2", "E3", "E4", "E5",
            "F1", "F2", "F3", "F4", "F5",
            "G1", "G2", "G3", "G4", "G5"
        ])
    },
    "verification_status": {
        'Not Verified': 0.0,
        'Source Verified': 1.0,
        'Verified': 2.0
    }
}

# Danh sách cột đúng thứ tự mô hình
input_features = [
    'loan_amnt', 'int_rate', 'annual_inc', 'dti', 'revol_util',
    'chargeoff_within_12_mths', 'collections_12_mths_ex_med', 'emp_length',
    'collection_recovery_fee', 'home_ownership', 'sub_grade',
    'verification_status', 'acc_now_delinq', 'delinq_2yrs',
    'acc_open_past_24mths', 'mort_acc', 'recoveries', 'last_fico_range_avg'
]

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()

    missing_fields = [feature for feature in input_features if not data.get(feature)]
    if missing_fields:
        return render_template('index.html', prediction="Bạn cần nhập đầy đủ thông tin!")
    # Chuyển dữ liệu vào list theo thứ tự đúng
    processed_input = []
    for feature in input_features:
        value = data.get(feature)

        # Áp dụng encode nếu là feature phân loại
        if feature in encode:
            value = encode[feature].get(value, 0.0)  # fallback nếu giá trị không hợp lệ
        else:
            value = float(value)  # chuyển về float nếu là số

        processed_input.append(value)

    # Chuyển thành định dạng mảng numpy
    X = np.array([processed_input])

    # Dự đoán
    prediction = model.predict(X)[0]
    if prediction == 1:
        message = "Người này không đủ điều kiện để vay tiền (The person is not eligible to pay for the loan)."
    else:
        message = "Người này đủ điều kiện để vay tiền (The person is eligible to pay for the loan)."
    return render_template('index.html', prediction=message)

if __name__ == '__main__':
    app.run(debug=True)