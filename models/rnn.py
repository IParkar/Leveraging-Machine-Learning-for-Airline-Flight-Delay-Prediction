import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Loading data
year_2007 = pd.read_csv('2007.csv', encoding='latin-1')

# Creating binary arrival delay column
year_2007['ArrDelay_Binary'] = (year_2007['ArrDelay'] > 0).astype(int)

# Creating TimePeriod column
year_2007['ArrTime_Trim'] = year_2007['ArrTime'].astype(str).str.strip('.0')

def categorize_time_period(hour):
    try:
        hour = int(hour)
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 18:
            return 'Afternoon'
        elif 18 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    except ValueError:
        return np.nan

year_2007['TimePeriod'] = year_2007['ArrTime_Trim'].apply(categorize_time_period)
year_2007['TimePeriod'] = year_2007['TimePeriod'].fillna('Unknown')

# Filtering data
filtered_data = year_2007[(year_2007['ArrDelay'] > 0) & (year_2007['ArrDelay'] < 180)]
year_2007['Filtered_Distance'] = np.nan  # Initialize with NaN
if 'Distance' in year_2007.columns:
    year_2007.loc[filtered_data.index, 'Filtered_Distance'] = filtered_data['Distance'].values

# Checking for missing values in 'Filtered_Distance'
year_2007['Filtered_Distance'].fillna(year_2007['Filtered_Distance'].mean(), inplace=True)

# Validating columns before proceeding
print("Columns in the DataFrame:", year_2007.columns)

# Defining features
features = ['DayOfWeek', 'Month', 'Filtered_Distance', 'TimePeriod']
# Ensuring all features exist
for feature in features:
    if feature not in year_2007.columns:
        raise KeyError(f"{feature} not found in DataFrame columns.")

X = pd.get_dummies(year_2007[features], drop_first=True)
y = year_2007['ArrDelay_Binary']

# Handling missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Reshaping for LSTM input: [samples, time steps, features]
X_imputed = X_imputed.reshape((X_imputed.shape[0], 1, X_imputed.shape[1]))

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Checking class distribution
print("Class distribution before SMOTE:")
print(y_train.value_counts())

# Applying SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
X_train_resampled = X_train_resampled.reshape((X_train_resampled.shape[0], 1, -1))  # Reshape back for LSTM

# Checking new class distribution
print("Class distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

# Building the LSTM model
model = Sequential()
model.add(LSTM(128, input_shape=(X_train_resampled.shape[1], X_train_resampled.shape[2]), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Training the model
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
model.fit(X_train_resampled, y_train_resampled, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[reduce_lr])

# Making predictions on the test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred.flatten() > 0.5).astype(int)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred_binary)
confusion_mat = confusion_matrix(y_test, y_pred_binary)
classification_rep = classification_report(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_mat)
print("Classification Report:\n", classification_rep)
print("F1 Score:", f1)
