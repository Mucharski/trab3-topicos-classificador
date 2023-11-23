from flask import Flask, render_template, request, redirect, url_for
from sklearn.linear_model import LogisticRegression
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import pandas as pd
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


# Função para verificar se a extensão do arquivo é permitida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

def preprocess_data(df):
    df.fillna(0, inplace=True)
    
    df['date'] = pd.to_datetime(df['date'])
    
    df = df.drop(columns=['date'])
    
    df['state'] = df['state'].astype('category').cat.codes
    
    numeric_columns = df.select_dtypes(include=['float64']).columns
    df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
    
    return df


# Função para treinar e avaliar o modelo
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Salvar a matriz de confusão como uma imagem
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Salvar a imagem em BytesIO
    img_stream = BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)
    img_str = base64.b64encode(img_stream.read()).decode('utf-8')

    return accuracy, f1_macro, img_str


# Rota principal
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        classifier = request.form['classifier']
        param1 = float(request.form['param1'])
        param2 = float(request.form['param2'])
        param3 = float(request.form['param3'])

        if 'csv_file' not in request.files:
            return redirect(request.url)

        file = request.files['csv_file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            df = pd.read_csv(file_path)

            df = preprocess_data(df)

            # Separar variáveis independentes (X) e variável dependente (y)
            X = df.drop(columns=['deaths_covid19'])  # Exclua a variável alvo dos recursos
            y = df['deaths_covid19']

            # Dividir os dados em conjuntos de treinamento e teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if classifier == 'RL':
                model = LogisticRegression()
            elif classifier == 'KNN': 
                model = KNeighborsClassifier(n_neighbors=int(param1), leaf_size=int(param2), p=int(param3))
                
            accuracy, f1_macro, conf_matrix_img = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)

            return render_template('result.html', accuracy=accuracy, f1_macro=f1_macro, conf_matrix_img=conf_matrix_img)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
