import os
import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
import uuid

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            data_processor = DataPreprocessor(file_path)
            data = data_processor.preprocess()

            X = data['area'].values  
            y = data['price'].values  

            model_trainer = ModelTrainer(X.reshape(-1, 1), y)
            model, r2 = model_trainer.train_model()

            y_pred = model.predict(X.reshape(-1, 1))

       
            unique_filename = str(uuid.uuid4()) + '.png'
            plot_file_path = os.path.join('static', 'plots', unique_filename)
            generate_plot(y, y_pred, plot_file_path)

            return render_template('index.html', mse=mean_squared_error(y, y_pred),
                                   r2=r2, plot_filename=unique_filename)

    return render_template('index.html')

def generate_plot(y, y_pred, file_path):

    model = LinearRegression()
    model.fit(y_pred.reshape(-1, 1), y)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y, color='blue', label='Actual vs. Predicted')

    line_x = np.linspace(min(y_pred), max(y_pred), 100)
    line_y = model.predict(line_x.reshape(-1, 1))
    plt.plot(line_x, line_y, color='red', linestyle='--', label='Regression Line')

    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Scatter Plot of Actual vs. Predicted with Regression Line')
    plt.legend()
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

if __name__ == '__main__':
    app.run(debug=True)
