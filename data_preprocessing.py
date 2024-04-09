import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)

    def preprocess(self):
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                self.data[col] = LabelEncoder().fit_transform(self.data[col])
        return self.data
