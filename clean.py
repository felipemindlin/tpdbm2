import pandas as pd
import numpy as np

# Load the dataset
file_path = "/home/pipemind/tpdbm2/Motor_Vehicle_Collisions_-_Crashes.csv"

# Handle potential row corruption by skipping bad lines
data = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip')

# Verify if all rows are read
csv_row_count = sum(1 for row in open(file_path, 'r', encoding='utf-8')) - 1  # Subtract 1 for header
data_row_count = len(data)
if data_row_count == csv_row_count:
    print(f"All rows read successfully: {data_row_count} rows.")
else:
    print(f"Mismatch in row count. CSV reports {csv_row_count} rows, but DataFrame has {data_row_count} rows.")

# Diagnose missing or invalid rows
if data_row_count < csv_row_count:
    print(f"Missing {csv_row_count - data_row_count} rows during reading. Check file for errors.")

# Display the first few rows to understand the structure
data.head()

# Step 1: Handle missing data
# Identify missing values summary
missing_summary = data.isnull().sum()
print("Missing Values Summary:\n", missing_summary)

# Drop columns with excessive missing values (threshold > 50%)
missing_threshold = len(data) * 0.5
cleaned_data = data.loc[:, data.isnull().sum() <= missing_threshold].copy()  # Work on a copy to avoid SettingWithCopyWarning

# Impute missing values where feasible
# For numeric columns, fill with median
numeric_cols = cleaned_data.select_dtypes(include=[np.number]).columns
cleaned_data.loc[:, numeric_cols] = cleaned_data[numeric_cols].fillna(cleaned_data[numeric_cols].median())

# For categorical columns, fill with a placeholder 'Unknown'
categorical_cols = cleaned_data.select_dtypes(include=['object']).columns
cleaned_data.loc[:, categorical_cols] = cleaned_data[categorical_cols].fillna("Unknown")

# Step 2: Format Date and Time
# Combine CRASH DATE and CRASH TIME into a single datetime column
if 'CRASH DATE' in cleaned_data.columns and 'CRASH TIME' in cleaned_data.columns:
    cleaned_data.loc[:, 'CRASH DATETIME'] = pd.to_datetime(
        cleaned_data['CRASH DATE'] + ' ' + cleaned_data['CRASH TIME'], errors='coerce'
    )
    # Drop the original CRASH DATE and CRASH TIME columns
    cleaned_data.drop(columns=['CRASH DATE', 'CRASH TIME'], inplace=True, errors='ignore')

# Extract features from datetime (e.g., hour, day of the week)
if 'CRASH DATETIME' in cleaned_data.columns:
    cleaned_data.loc[:, 'HOUR'] = cleaned_data['CRASH DATETIME'].dt.hour
    cleaned_data.loc[:, 'DAY_OF_WEEK'] = cleaned_data['CRASH DATETIME'].dt.day_name()

# Step 3: Remove duplicates
cleaned_data = cleaned_data.drop_duplicates()

# Step 4: Drop columns with excessive missing data (e.g., LOCATION, ON STREET NAME, CROSS STREET NAME)
columns_to_drop = ['LOCATION', 'ON STREET NAME', 'CROSS STREET NAME', 'OFF STREET NAME']
# Check for existence before dropping
existing_columns_to_drop = [col for col in columns_to_drop if col in cleaned_data.columns]
cleaned_data = cleaned_data.drop(columns=existing_columns_to_drop, axis=1)

# Display the cleaned dataset summary
cleaned_data.info()

# Save cleaned data to a new CSV file
output_file_path = "/home/pipemind/tpdbm2/cleaned_crashes2.csv"
cleaned_data.to_csv(output_file_path, index=False)
print(f"Cleaned data saved to {output_file_path}")
