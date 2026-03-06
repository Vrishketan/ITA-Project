import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests
import io

def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'\n{name}')
    print(f'RMSE: {round(rmse, 4)}')
    print(f'MAE : {round(mae, 4)}')
    print(f'R2  : {round(r2, 4)}')

def run_analysis(url):
    print(f"Fetching data from GitHub: {url}")
    response = requests.get(url)
    if response.status_code != 200:
        print("Error: Could not retrieve file from GitHub.")
        return
    
    df = pd.read_csv(io.StringIO(response.text))
    print("Dataset Loaded Successfully")

    df['Peak_Hour_Flag'] = df['Hour'].apply(lambda x: 1 if x in [6, 7, 8, 18, 19, 20] else 0)
    df['Daily_Avg_kWh'] = df.groupby('Household_ID')['Energy_Consumption_kWh'].transform('mean')
    df['Daily_Variance'] = df.groupby('Household_ID')['Energy_Consumption_kWh'].transform('var').fillna(0)

    household_features = df.groupby('Household_ID').agg({'Daily_Avg_kWh': 'mean', 'Daily_Variance': 'mean'}).reset_index()
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    household_features['Cluster'] = kmeans.fit_predict(household_features[['Daily_Avg_kWh', 'Daily_Variance']])
    df = df.merge(household_features[['Household_ID', 'Cluster']], on='Household_ID', how='left')

    features = ['Hour', 'Peak_Hour_Flag', 'Daily_Avg_kWh', 'Daily_Variance']
    target = 'Energy_Consumption_kWh'
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(random_state=42).fit(X_train, y_train)
    gb = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
    evaluate(y_test, rf.predict(X_test), "Global Random Forest")
    evaluate(y_test, gb.predict(X_test), "Global Gradient Boosting")

    for cluster in sorted(df['Cluster'].unique()):
        print(f'\n--- Cluster {cluster} ---')
        c_df = df[df['Cluster'] == cluster]
        Xc, yc = c_df[features], c_df[target]
        Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.2, random_state=42)
        evaluate(yc_te, RandomForestRegressor(random_state=42).fit(Xc_tr, yc_tr).predict(Xc_te), "Cluster RF")

if __name__ == '__main__':
    url = 'https://raw.githubusercontent.com/Vrishketan/ITA-Project/main/sample_data/energy_consumption_1000_households.csv'
    run_analysis(url)