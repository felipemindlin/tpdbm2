import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from geopy.distance import geodesic

# Load the dataset
file_path = "/home/pipemind/tpdbm2/Motor_Vehicle_Collisions_-_Crashes.csv"
data = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')

# Verify if all rows are read
csv_row_count = sum(1 for row in open(file_path, 'r', encoding='utf-8')) - 1  # Subtract 1 for header
data_row_count = len(data)
if data_row_count == csv_row_count:
    print(f"All rows read successfully: {data_row_count} rows.")
else:
    print(f"Mismatch in row count. CSV reports {csv_row_count} rows, but DataFrame has {data_row_count} rows.")
    print(f"Missing {csv_row_count - data_row_count} rows during reading. Check file for errors.")

# -------------------- DATA CLEANING --------------------
def clean_data(df):
    print("Starting data cleaning...")
    
    # Drop duplicates
    df = df.drop_duplicates()

    # Fill missing numerical values with median
    num_cols = [
        'NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED',
        'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
        'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
        'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED'
    ]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Drop rows with invalid or missing location data
    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    df = df[(df['LATITUDE'].between(40, 41)) & (df['LONGITUDE'].between(-74, -73))]

    # Combine CRASH DATE and CRASH TIME into a single datetime column
    if 'CRASH DATE' in df.columns and 'CRASH TIME' in df.columns:
        df['CRASH DATETIME'] = pd.to_datetime(
            df['CRASH DATE'] + ' ' + df['CRASH TIME'], errors='coerce'
        )
        df.drop(columns=['CRASH DATE', 'CRASH TIME'], inplace=True, errors='ignore')

    # Extract features from datetime
    if 'CRASH DATETIME' in df.columns:
        df['HOUR'] = df['CRASH DATETIME'].dt.hour
        df['DAY_OF_WEEK'] = df['CRASH DATETIME'].dt.day_name()
        df['PEAK_HOURS'] = df['HOUR'].apply(lambda x: 'Peak' if 7 <= x <= 9 or 16 <= x <= 19 else 'Off-Peak')
        df['WEEKEND'] = df['DAY_OF_WEEK'].isin(['Saturday', 'Sunday'])
        df['SEASON'] = df['CRASH DATETIME'].dt.month % 12 // 3 + 1
        df['SEASON'] = df['SEASON'].map({1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'})

    # Handle contributing factors
    factor_cols = ['CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2', 
                   'CONTRIBUTING FACTOR VEHICLE 3', 'CONTRIBUTING FACTOR VEHICLE 4', 
                   'CONTRIBUTING FACTOR VEHICLE 5']
    for col in factor_cols:
        if col not in df.columns:
            df[col] = 'Unknown'
        else:
            df[col] = df[col].fillna('Unknown')

    print(f"Data cleaning completed. Number of entries remaining: {len(df)}")
    return df

cleaned_data = clean_data(data)

# -------------------- CLASSIFICATION --------------------
def classify_crashes(df):
    print("Starting classification...")
    
    # Convert contributing factors to numerical values (one-hot encoding)
    df_encoded = pd.get_dummies(df, columns=[col for col in df.columns if col.startswith('CONTRIBUTING FACTOR')])
    
    # Define features AFTER encoding
    features = ['LATITUDE', 'LONGITUDE'] + [col for col in df_encoded.columns if 'CONTRIBUTING FACTOR' in col]

    # Ensure all features exist in the encoded dataframe
    X = df_encoded[features]
    y = (df['NUMBER OF PERSONS INJURED'] > 0).astype(int)  # Binary target variable

    # Align columns in case of mismatch
    X = X.loc[:, ~X.columns.duplicated()]

    # Split data
    print("Splitting data for classification...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a RandomForestClassifier
    print("Training RandomForestClassifier...")
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    print(f"Classification completed. Accuracy: {clf.score(X_test, y_test):.2f}")

classify_crashes(cleaned_data)

# -------------------- CLUSTERING --------------------
def cluster_severity_hotspots(df, n_clusters=8):
    print("Starting clustering for crash severity hotspots...")

    # Calculate severity score
    df['SEVERITY_SCORE'] = (df['NUMBER OF PERSONS INJURED'] + df['NUMBER OF PEDESTRIANS INJURED'] +
                            df['NUMBER OF CYCLIST INJURED'] + df['NUMBER OF MOTORIST INJURED'])

    # Filter for meaningful clusters (exclude rows with zero severity)
    severity_data = df[df['SEVERITY_SCORE'] > 0][['LATITUDE', 'LONGITUDE', 'SEVERITY_SCORE']]

    if severity_data.empty:
        print("No crashes with injuries to cluster.")
        return df, []

    # Normalize severity score for clustering
    severity_data['SEVERITY_SCORE'] = (severity_data['SEVERITY_SCORE'] - severity_data['SEVERITY_SCORE'].mean()) / severity_data['SEVERITY_SCORE'].std()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    print("Fitting KMeans for severity hotspots...")
    severity_data['Cluster'] = kmeans.fit_predict(severity_data[['LATITUDE', 'LONGITUDE', 'SEVERITY_SCORE']])

    print("Clustering completed. Cluster centers:")
    cluster_info = []
    for i, center in enumerate(kmeans.cluster_centers_):
        print(f"Cluster {i}: Location ({center[0]:.4f}, {center[1]:.4f}), Severity Score: {center[2]:.2f}")
        cluster_info.append({
            'Cluster': i,
            'Latitude': center[0],
            'Longitude': center[1],
            'Severity_Score': center[2]
        })

    # Calculate cluster radius
    print("Calculating cluster radii...")
    for i in range(n_clusters):
        cluster_points = severity_data[severity_data['Cluster'] == i]
        distances = cluster_points.apply(lambda row: geodesic((row['LATITUDE'], row['LONGITUDE']),
                                                              (kmeans.cluster_centers_[i][0], kmeans.cluster_centers_[i][1])).meters, axis=1)
        cluster_radius = distances.max() if not distances.empty else 0
        cluster_info[i]['Radius_meters'] = cluster_radius
        print(f"Cluster {i} radius: {cluster_radius:.2f} meters")

    # Merge cluster labels back to the main dataframe
    df = df.merge(severity_data[['LATITUDE', 'LONGITUDE', 'Cluster']], on=['LATITUDE', 'LONGITUDE'], how='left')
    print("Cluster assignment completed.")

    return df, cluster_info

clustered_data, cluster_details = cluster_severity_hotspots(cleaned_data)

print("Clustered Data Sample:")
print(clustered_data[['LATITUDE', 'LONGITUDE', 'SEVERITY_SCORE', 'Cluster']].dropna().sample(5))

print("Cluster Details:")
for cluster in cluster_details:
    print(cluster)

# Save cleaned and clustered data to a new CSV file
output_file_path = "/home/pipemind/tpdbm2/cleaned_and_clustered_crashes.csv"
clustered_data.to_csv(output_file_path, index=False)
print(f"Cleaned and clustered data saved to {output_file_path}")
