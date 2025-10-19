# Project: Ho Chi Minh Temperture Forecasting _ Group 6
1. Introduction
This project focuses on forecasting daily (hourly) temperature in Ho Chi Minh using 10 years of historical weather data from Visual Crossing. By analyzing meteorological features such as humidity, solar radiation, precipitation, and moon phase, we aim to build an accurate machine learning model for short-term temperature prediction. The workflow includes data processing, feature engineering, model training, hyperparameter tuning, and deployment via a user-friendly web interface.
3. Process (process is designed to be more suitable with the pipeline)
- Basic cleaning:
  + Dropping unecessary columns such as description, venue, longtitude,...
  + Normalizing data dtypes : datetime, sunrise, sunset (to datetime type)
- Split data:
  + Train set (60%): x_train, y_train 
  + Validation set (20%): x_val, y_val
  + Test set (20%): x_test, y_test
- Feature engineer: We perform feature engineering before preprocessing and after splitting the data to avoid data leakage. This is because when creating lag or rolling features, we need access to past target values (y). If this process were done inside the pipeline, the model might indirectly “see” future information during training, leading to data leakage and overly optimistic results. By applying feature engineering after the split, each subset (train, validation, test) only uses information available up to its own time range. Additionally, generating lag and rolling features naturally produces NaN values at the beginning of each subset (due to missing past observations). These initial rows are carefully removed or aligned later in the pipeline to ensure that X and y remain properly synchronized for model training.
  + Extract datetime into features such as month, week,...
  + Creating new feature: day_light_hour from sunset, sunrise 
  + Creating lag features and rolling mean features for X includes columns ['temp', 'tempmax', 'tempmin', 'feelslike', 'dew', 'humidity', 'windspeed', 'windgust', 'sealevelpressure', 'precip', 'solarradiation', 'solarenergy', 'cloudcover'] => Continuous variable bring meaning trend can be utilized.
  + Extract x_train, y_train
=> Disadvantage: must do munually for validation and test set => Deployment carefully
- Pipeline:
  + Preprocessing: Includes handle missing value by intepolate for numerical, ffill and bfill for categorical, encoding categorical, drop nan from lag/rolling features
  + SafeAlign: Because after preprocessing, X and y do not have the same size so that design a custom that accept to return both X and y
  + Model and hyperparameter tuning
- Deployment into web/app
4. Practical discussion
5. Simulated result 
