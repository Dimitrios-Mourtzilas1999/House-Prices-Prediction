import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def drop_low_corr_columns(df, target_variable, threshold=0.5):

    correlations = df.corr()[target_variable].abs()
    
    low_corr_columns = correlations[correlations < threshold].index.tolist()
    
    df_dropped = df.drop(columns=low_corr_columns)
    
    return df_dropped

def main():
    dataset = pd.read_csv('Housing.csv')
    
    for col in dataset.columns:
        if dataset[col].dtype == 'object':
            dataset[col] = LabelEncoder().fit_transform(dataset[col])
    
    target_variable = 'price'
    correlation_threshold = 0.3
    
    df_filtered = drop_low_corr_columns(dataset, target_variable, correlation_threshold)
    
    X = df_filtered.drop(columns='price', axis=1)
    y = df_filtered['price']
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    linear = LinearRegression()
    linear.fit(x_train_scaled, y_train)
    
    y_pred = linear.predict(x_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    print("R-squared Score:", r2)
    
    slope = linear.coef_[0]
    intercept = linear.intercept_
    
    print("Linear Equation:")
    print(f"y = {slope:.2f} * X + {intercept:.2f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
    plt.title('Linear Regression Prediction vs. Actual')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
