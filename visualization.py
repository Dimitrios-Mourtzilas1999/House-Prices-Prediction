import matplotlib.pyplot as plt

class ResultVisualizer:
    def __init__(self, y_test, y_pred):
        self.y_test = y_test
        self.y_pred = y_pred

    def plot_results(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test, self.y_pred, color='blue', alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], color='red', linestyle='--', linewidth=2)
        plt.title('Linear Regression Prediction vs. Actual')
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.grid(True)
        plt.show()
