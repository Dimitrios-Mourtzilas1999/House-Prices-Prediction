import os
import uuid
import numpy as np
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from generate_plot import PlotGenerator 

app = Flask(__name__)

plot_generator = PlotGenerator()

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

            plot_generator.generate_plot(y, y_pred, plot_file_path)

            return render_template('index.html', mse=mean_squared_error(y, y_pred),
                                   r2=r2, plot_filename=unique_filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
