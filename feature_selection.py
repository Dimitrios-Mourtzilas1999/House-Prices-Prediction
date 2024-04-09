class FeatureSelector:
    def __init__(self, data, target_variable, threshold=0.5):
        self.data = data
        self.target_variable = target_variable
        self.threshold = threshold

    def select_features(self):
        correlations = self.data.corr()[self.target_variable].abs()
        low_corr_columns = correlations[correlations < self.threshold].index.tolist()
        filtered_data = self.data.drop(columns=low_corr_columns)
        return filtered_data
