from data_preprocessing import DataPreprocessor
from feature_selection import FeatureSelector
from model_training import ModelTrainer
from visualization import ResultVisualizer

def main():
    # Data preprocessing
    data_processor = DataPreprocessor('Housing.csv')
    data = data_processor.preprocess()

    # Feature selection
    target_variable = 'price'
    correlation_threshold = 0.3
    feature_selector = FeatureSelector(data, target_variable, correlation_threshold)
    filtered_data = feature_selector.select_features()

    # Model training
    X = filtered_data.drop(columns='price', axis=1)
    y = filtered_data['price']
    model_trainer = ModelTrainer(X, y)
    model, r2 = model_trainer.train_model()

    # Predict using the trained model
    y_pred = model.predict(model_trainer.scaler.transform(X))

    # Display results
    result_visualizer = ResultVisualizer(y, y_pred)
    result_visualizer.plot_results()

    print("R-squared Score:", r2)
    slope = model.coef_[0]
    intercept = model.intercept_
    print("Linear Equation:")
    print(f"y = {slope:.2f} * X + {intercept:.2f}")

if __name__ == "__main__":
    main()
