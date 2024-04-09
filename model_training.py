from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.scaler = StandardScaler()

    def train_model(self):
        x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        model = LinearRegression()
        model.fit(x_train_scaled, y_train)
        y_pred = model.predict(x_test_scaled)
        r2 = r2_score(y_test, y_pred)
        return model, r2

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
