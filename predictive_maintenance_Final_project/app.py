from flask import Flask, render_template, request, send_file
import joblib
import pandas as pd
import io

app = Flask(__name__)

# 1. تحميل الموديل والـ Label Encoder (تأكد من صحة الأسماء لديك)
try:
    model_pipeline = joblib.load('predictive_maintenance_xgboost_model.pkl')
    le = joblib.load('label_encoder.pkl')
except:
    print("خطأ: تأكد من وجود ملفات الموديل بنفس الأسماء في مجلد المشروع")

# 2. قاموس ترجمة الأعطال للعربية
failure_translations = {
    "No Failure": "لا يوجد عطل (الماكينة سليمة)",
    "Heat Dissipation Failure": "عطل بسبب تبديد الحرارة",
    "Power Failure": "عطل في نظام الطاقة/العزم",
    "Overstrain Failure": "عطل ناتج عن الإجهاد الميكانيكي",
    "Tool Wear Failure": "عطل بسبب تآكل العدة",
    "Random Failures": "أعطال عشوائية غير محددة"
}


@app.route('/')
def index():
    return render_template('index.html', inputs={}, fail_type_text="", prediction_text="")


@app.route('/bulk')
def bulk():
    return render_template('bulk.html')


# --- الجزء الخاص بالتنبؤ اليدوي ---
@app.route('/predict', methods=['POST'])
def predict():
    inputs = {
        'type': request.form.get('type', '0'),
        'air_temp': request.form.get('air_temp', ''),
        'proc_temp': request.form.get('proc_temp', ''),
        'speed': request.form.get('speed', ''),
        'torque': request.form.get('torque', ''),
        'tool_wear': request.form.get('tool_wear', '')
    }

    try:
        if not all([inputs['air_temp'], inputs['proc_temp'], inputs['speed'], inputs['torque'], inputs['tool_wear']]):
            return render_template('index.html', inputs=inputs, prediction_text="", fail_type_text="", risk_level="")

        type_mapping = {
            '0': 'L',
            '1': 'M',
            '2': 'H'
        }
        data = pd.DataFrame([{
            'Type': type_mapping.get(inputs['type'], 'L'),
            'Air temperature [K]': float(inputs['air_temp']),
            'Process temperature [K]': float(inputs['proc_temp']),
            'Rotational speed [rpm]': float(inputs['speed']),
            'Torque [Nm]': float(inputs['torque']),
            'Tool wear [min]': float(inputs['tool_wear'])
        }])

        # تطبيق الحماية (Clipping)
        data['Air temperature [K]'] = data['Air temperature [K]'].clip(280, 320)
        data['Process temperature [K]'] = data['Process temperature [K]'].clip(280, 330)
        data['Rotational speed [rpm]'] = data['Rotational speed [rpm]'].clip(0, 5000)
        data['Torque [Nm]'] = data['Torque [Nm]'].clip(0, 150)
        data['Tool wear [min]'] = data['Tool wear [min]'].clip(0, 350)

        pred_numeric = model_pipeline.predict(data)[0]
        fail_type_en = le.inverse_transform([pred_numeric])[0]
        fail_type_ar = failure_translations.get(fail_type_en, fail_type_en)

        if fail_type_en == "No Failure":
            risk, p_text, f_msg = "Low", "الحالة: الماكينة مستقرة ✅", ""
        else:
            risk, p_text, f_msg = "High", "انتباه: تم رصد خلل محتمل! ⚠️", f"التشخيص: {fail_type_ar}"

        return render_template('index.html', inputs=inputs, prediction_text=p_text, fail_type_text=f_msg,
                               risk_level=risk)

    except ValueError:
        return render_template('index.html', inputs=inputs, prediction_text="يرجى إدخال أرقام صحيحة", risk_level="High")


# --- الجزء الخاص بالتنبؤ الجماعي (الملفات) ---
@app.route('/predict_file', methods=['POST'])
def predict_file():
    if 'file' not in request.files:
        return "لم يتم رفع أي ملف"

    file = request.files['file']
    if file.filename == '':
        return "اسم الملف فارغ"

    try:
        df = pd.read_csv(file)

        # تنظيف وحماية بيانات الملف قبل التنبؤ
        df['Air temperature [K]'] = df['Air temperature [K]'].clip(280, 320)
        df['Process temperature [K]'] = df['Process temperature [K]'].clip(280, 330)
        df['Rotational speed [rpm]'] = df['Rotational speed [rpm]'].clip(0, 5000)
        df['Torque [Nm]'] = df['Torque [Nm]'].clip(0, 150)
        df['Tool wear [min]'] = df['Tool wear [min]'].clip(0, 350)

        # التنبؤ الجماعي
        preds_numeric = model_pipeline.predict(df)
        preds_en = le.inverse_transform(preds_numeric)

        # إضافة عمود النتيجة بالعربي
        df['النتيجة (التشخيص)'] = [failure_translations.get(p, p) for p in preds_en]

        # تحويل البيانات إلى JSON لغرض التحميل لاحقاً
        raw_json = df.to_json(orient='records')

        # عرض الجدول في الصفحة
        table_html = df.to_html(classes='table table-striped table-hover mt-3', index=False)

        return render_template('bulk.html', tables=[table_html], raw_data_json=raw_json)
    except Exception as e:
        return f"خطأ في معالجة الملف: تأكد من تطابق أسماء الأعمدة. الخطأ: {str(e)}"


@app.route('/download_results', methods=['POST'])
def download_results():
    data_json = request.form.get('table_data')
    df = pd.read_json(io.StringIO(data_json))
    proxy = io.StringIO()
    df.to_csv(proxy, index=False)
    proxy.seek(0)
    return send_file(io.BytesIO(proxy.getvalue().encode()), mimetype='text/csv', as_attachment=True,
                     download_name='maintenance_results.csv')


if __name__ == '__main__':
    app.run(debug=True)