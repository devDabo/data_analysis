from flask import Flask, render_template, jsonify
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Use Agg backend for Matplotlib
plt.switch_backend('Agg')

app = Flask(__name__)

# Database connection details
db_type = 'postgresql'
db_driver = 'psycopg2'
db_user = 'myuser'
db_pass = 'password' 
db_host = 'localhost'
db_port = '5432'
db_name = 'mydatabase'

# Create the database connection
engine = create_engine(f'{db_type}+{db_driver}://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}')

@app.route('/')
def index():
    query = """
    SELECT customerid, name, arpu, mrc, churnstatus, monthlyusagegb, contracttype, devicetype, lastinteractiondate, productholding, churnpredictionscore
    FROM customerbehavior
    """
    df = pd.read_sql(query, engine)
    
    required_columns = ['arpu', 'mrc', 'monthlyusagegb', 'churnpredictionscore', 'lastinteractiondate']
    for col in required_columns:
        if col not in df.columns:
            df[col] = None
    
    if 'lastinteractiondate' in df.columns:
        df['lastinteractiondate'] = pd.to_datetime(df['lastinteractiondate'])
    else:
        df['lastinteractiondate'] = pd.NaT 
    
    df.fillna({
        'arpu': df['arpu'].mean() if 'arpu' in df.columns else 0,
        'mrc': df['mrc'].mean() if 'mrc' in df.columns else 0,
        'monthlyusagegb': df['monthlyusagegb'].mean() if 'monthlyusagegb' in df.columns else 0,
        'churnpredictionscore': df['churnpredictionscore'].mean() if 'churnpredictionscore' in df.columns else 0
    }, inplace=True)
    
    df_encoded = pd.get_dummies(df, columns=['contracttype', 'devicetype', 'productholding'])
    
    X = df_encoded.drop(['customerid', 'name', 'churnstatus', 'lastinteractiondate'], axis=1)
    y = df_encoded['churnstatus'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    img_cm = io.BytesIO()
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(img_cm, format='png')
    img_cm.seek(0)
    cm_plot_url = base64.b64encode(img_cm.getvalue()).decode('utf8')
    
    cr_df = pd.DataFrame(cr).transpose()
    cr_html = cr_df.to_html(classes="table table-striped table-bordered")
    
    img_arpu = io.BytesIO()
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='churnstatus', y='arpu', data=df)
    plt.title('ARPU Distribution by Churn Status')
    plt.savefig(img_arpu, format='png')
    img_arpu.seek(0)
    arpu_plot_url = base64.b64encode(img_arpu.getvalue()).decode('utf8')
    
    img_usage = io.BytesIO()
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='contracttype', y='monthlyusagegb', data=df)
    plt.title('Monthly Usage by Contract Type')
    plt.savefig(img_usage, format='png')
    img_usage.seek(0)
    usage_plot_url = base64.b64encode(img_usage.getvalue()).decode('utf8')

    return render_template('index.html', 
                           arpu_plot_url=arpu_plot_url, 
                           usage_plot_url=usage_plot_url,
                           cm_plot_url=cm_plot_url,
                           cr_html=cr_html)

@app.route('/dataset')
def dataset():
    query = """
    SELECT * FROM customerbehavior
    """
    df = pd.read_sql(query, engine)
    data_html = df.to_html(classes="table table-striped table-bordered", table_id="dataTable")
    return render_template('dataset.html', data_html=data_html)

if __name__ == '__main__':
    app.run(debug=True)