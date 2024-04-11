import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class PlotGenerator:
    def __init__(self):
        self.model = LinearRegression()

    def generate_plot(self, y, y_pred, file_path):
        self.model.fit(y_pred.reshape(-1, 1), y)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, y, color='blue', label='Actual vs. Predicted')

        line_x = np.linspace(min(y_pred), max(y_pred), 100)
        line_y = self.model.predict(line_x.reshape(-1, 1))
        plt.plot(line_x, line_y, color='red', linestyle='--', label='Regression Line')

        plt.xlabel('Predicted Values')
        plt.ylabel('Actual Values')
        plt.title('Scatter Plot of Actual vs. Predicted with Regression Line')
        plt.legend()
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()


