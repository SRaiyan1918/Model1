from flask import Flask,render_template,request,redirect
import joblib
import pandas as pd
import numpy as np
import sklearn

app = Flask(__name__)

df = pd.read_csv('static/datasets/data.csv')
model = joblib.load('static/model/(SR) Laptop price prediction model(RandomForest).sralgo')

@app.route('/')
def index():
    companies = sorted(df['Company'].unique())
    typenames = sorted(df['TypeName'].unique())
    rams = sorted(df['Ram'].unique())
    opsyss = sorted(df['OpSys'].dropna().astype(str).unique())
    displays = sorted(df['Display'].unique())
    hdds = sorted(df['HDD'].unique())
    ssds = sorted(df['SSD'].unique())
    hybrids = sorted(df['Hybrid'].unique())
    flash_storages = sorted(df['Flash_Storage'].unique())
    gpu_brandss = sorted(df['Gpu_brand'].unique())
    cpu_names = sorted(df['Cpu_name'].unique())
    return render_template('index.html',
                            companies=companies,
                            typenames=typenames,
                            rams=rams,
                            opsyss=opsyss,
                            displays=displays,
                            hdds=hdds,
                            ssds=ssds,
                            hybrids=hybrids,
                            flash_storages=flash_storages,
                            gpu_brandss=gpu_brandss,
                            cpu_names=cpu_names)

@app.route('/predict',methods=['POST'])
def predict():
    company = request.form.get('company')
    typename = request.form.get('typename')
    ram = request.form.get('ram')
    opsys = request.form.get('opsys')
    weight = request.form.get('weight')
    ips = int(request.form.get('ips'))
    touchscreen = int(request.form.get('touchscreen'))
    display = request.form.get('display')
    ppi = request.form.get('ppi')
    hdd = request.form.get('hdd')
    ssd = request.form.get('ssd')
    hybrid = request.form.get('hybrid')
    flash_storage = request.form.get('flash_storage')
    gpu_brand = request.form.get('gpu_brands')
    cpu_name = request.form.get('cpu_name')
    
    prediction=model.predict(pd.DataFrame(columns=['Company', 'TypeName', 'Ram', 'OpSys', 'Weight', 'IPS', 'Touchscreen', 'Display', 'PPI', 'HDD', 'SSD', 'Hybrid', 'Flash_Storage', 'Gpu_brand', 'Cpu_name'],
                                  data=np.array([company,typename,ram,opsys,weight,ips,touchscreen,display,ppi,hdd,ssd,hybrid,flash_storage,gpu_brand,cpu_name]).reshape(1, 15)))
    print(prediction)
    
    return str(np.round(np.exp(prediction[0]),2))


if __name__ == '__main__':
    app.run(debug=True)
