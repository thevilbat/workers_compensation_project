import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os

def analysis_and_model_page():
    st.title("Прогнозирование стоимости страховых выплат")

    # ===================== ЗАГРУЗКА ДАННЫХ =====================
    if st.button("Загрузить данные"):
        with st.spinner("Загрузка данных..."):
            data_path = "data/workers_compensation.csv"
            arff_path = "data/workers_compensation.arff"
            df = None

            # Попытка 1: CSV
            if os.path.exists(data_path):
                try:
                    df = pd.read_csv(data_path)
                    st.success(f"Данные загружены из CSV! ({df.shape[0]:,} записей)")
                except Exception as e:
                    st.warning(f"Ошибка чтения CSV: {e}")

            # Попытка 2: ARFF
            if df is None and os.path.exists(arff_path):
                try:
                    import arff
                    with open(arff_path, 'r') as f:
                        arff_data = arff.load(f)
                    df = pd.DataFrame(arff_data['data'], columns=[attr[0] for attr in arff_data['attributes']])
                    st.success(f"Данные загружены из ARFF! ({df.shape[0]:,} записей)")
                except Exception as e:
                    st.error(f"Ошибка чтения ARFF: {e}")

            # Попытка 3: fetch_openml
            if df is None:
                try:
                    data = fetch_openml(data_id=42876, as_frame=True, parser='auto')
                    df = data.frame
                    os.makedirs("data", exist_ok=True)
                    df.to_csv(data_path, index=False)
                    st.success("Данные загружены через OpenML и сохранены в data/")
                except Exception as e:
                    st.error("Не удалось загрузить данные автоматически.")
                    st.info("Положите файл workers_compensation.arff или .csv в папку data/")
                    st.stop()

            st.session_state['df'] = df
            st.dataframe(df.head())

    # ===================== ОСНОВНАЯ РАБОТА =====================
    if 'df' in st.session_state:
        df = st.session_state['df']

        st.subheader("Просмотр данных")
        st.write(df.head())

        st.subheader("Статистика")
        st.write(df.describe())

        # ===================== ПРЕДОБРАБОТКА =====================
        if st.button("Выполнить предобработку"):
            with st.spinner("Предобработка данных..."):
                data = df.copy()

                # Удаляем идентификаторы заявок
                id_cols = [col for col in data.columns if col.lower() in ['claimnumber', 'claimid', 'id', 'policy']]
                data = data.drop(columns=id_cols, errors='ignore')

                # Обработка дат
                if 'DateTimeOfAccident' in data.columns:
                    data['DateTimeOfAccident'] = pd.to_datetime(data['DateTimeOfAccident'])
                if 'DateReported' in data.columns:
                    data['DateReported'] = pd.to_datetime(data['DateReported'])

                if 'DateTimeOfAccident' in data.columns and 'DateReported' in data.columns:
                    data['AccidentMonth'] = data['DateTimeOfAccident'].dt.month
                    data['AccidentDayOfWeek'] = data['DateTimeOfAccident'].dt.dayofweek
                    data['ReportingDelay'] = (data['DateReported'] - data['DateTimeOfAccident']).dt.days
                    data = data.drop(columns=['DateTimeOfAccident', 'DateReported'], errors='ignore')

                # Кодирование категориальных признаков
                categorical_columns = ['Gender', 'MaritalStatus', 'PartTimeFullTime', 'ClaimDescription']
                label_encoders = {}

                for col in categorical_columns:
                    if col in data.columns:
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col].astype(str))
                        label_encoders[col] = le
                        # Сохраняем классы для формы предсказания
                        st.session_state[f'{col}_classes'] = list(le.classes_)

                st.session_state['label_encoders'] = label_encoders

                # Масштабирование числовых признаков
                numerical_features = ['Age', 'DependentChildren', 'DependentsOther', 'WeeklyPay',
                                      'HoursWorkedPerWeek', 'DaysWorkedPerWeek', 'InitialCaseEstimate',
                                      'AccidentMonth', 'AccidentDayOfWeek', 'ReportingDelay']

                numerical_features = [col for col in numerical_features if col in data.columns]

                scaler = StandardScaler()
                data[numerical_features] = scaler.fit_transform(data[numerical_features])

                st.session_state['scaler'] = scaler
                st.session_state['numerical_features'] = numerical_features
                st.session_state['data_preprocessed'] = data

                st.success("Предобработка успешно завершена!")
                st.write(f"Форма данных: {data.shape}")
                st.write("Столбцы:", list(data.columns))

        # ===================== ОБУЧЕНИЕ МОДЕЛЕЙ =====================
        if 'data_preprocessed' in st.session_state:
            if st.button("Обучить модели"):
                with st.spinner("Обучение моделей..."):
                    data = st.session_state['data_preprocessed']
                    target = 'UltimateIncurredClaimCost'

                    X = data.drop(columns=[target], errors='ignore')
                    y = data[target]

                    # Проверка на нечисловые столбцы
                    non_numeric = X.select_dtypes(include=['object']).columns.tolist()
                    if non_numeric:
                        st.error(f"Остались нечисловые столбцы: {non_numeric}")
                        st.stop()

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    models = {
                        "Linear Regression": LinearRegression(),
                        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
                        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
                    }

                    results = {}
                    trained_models = {}

                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        trained_models[name] = model

                        y_pred = model.predict(X_test)
                        mae = mean_absolute_error(y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        r2 = r2_score(y_test, y_pred)

                        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}

                    st.session_state['trained_models'] = trained_models
                    st.session_state['model_results'] = results
                    st.session_state['X_test'] = X_test
                    st.session_state['y_test'] = y_test

                    st.success("Модели успешно обучены!")

                    # Вывод результатов
                    st.subheader("Сравнение моделей")
                    results_df = pd.DataFrame(results).T
                    st.dataframe(results_df.style.format({"MAE": "${:.2f}", "RMSE": "${:.2f}", "R2": "{:.4f}"}))

                    best_model_name = max(results, key=lambda x: results[x]["R2"])
                    st.info(f"**Лучшая модель: {best_model_name}** (R² = {results[best_model_name]['R2']:.4f})")

            # ===================== ПРЕДСКАЗАНИЕ =====================
            if 'trained_models' in st.session_state:
                st.header("Предсказание стоимости возмещения")

                le_dict = st.session_state.get('label_encoders', {})

                with st.form("prediction_form"):
                    st.write("Введите параметры случая:")

                    col1, col2 = st.columns(2)
                    with col1:
                        age = st.number_input("Возраст", min_value=13, max_value=76, value=35)
                        gender = st.selectbox("Пол", ["M", "F"])
                        marital_options = st.session_state.get('MaritalStatus_classes', ["Single", "Married", "Divorced", "Widowed"])
                        marital_status = st.selectbox("Семейное положение", marital_options)
                        weekly_pay = st.number_input("Еженедельная зарплата ($)", min_value=0, value=800)

                    with col2:
                        initial_estimate = st.number_input("Начальная оценка ($)", min_value=0, value=5000)
                        hours_per_week = st.number_input("Часов в неделю", min_value=0, max_value=168, value=40)
                        days_per_week = st.number_input("Дней в неделю", min_value=0, max_value=7, value=5)
                        part_options = st.session_state.get('PartTimeFullTime_classes', ["Full Time", "Part Time"])
                        part_time = st.selectbox("Тип занятости", part_options)

                    claim_options = st.session_state.get('ClaimDescription_classes', ["Unknown"])
                    claim_description = st.selectbox("Описание заявки", claim_options)

                    submit_button = st.form_submit_button("Предсказать")

                    if submit_button:
                        input_data = pd.DataFrame({
                            'Age': [age],
                            'Gender': [gender],
                            'MaritalStatus': [marital_status],
                            'DependentChildren': [0],
                            'DependentsOther': [0],
                            'WeeklyPay': [weekly_pay],
                            'PartTimeFullTime': [part_time],
                            'HoursWorkedPerWeek': [hours_per_week],
                            'DaysWorkedPerWeek': [days_per_week],
                            'ClaimDescription': [claim_description],
                            'InitialCaseEstimate': [initial_estimate],
                            'AccidentMonth': [6],
                            'AccidentDayOfWeek': [2],
                            'ReportingDelay': [5]
                        })

                        # Кодирование категориальных признаков
                        for col in ['Gender', 'MaritalStatus', 'PartTimeFullTime', 'ClaimDescription']:
                            if col in le_dict:
                                try:
                                    input_data[col] = le_dict[col].transform(input_data[col])
                                except ValueError:
                                    st.error(f"Значение '{input_data[col].iloc[0]}' для {col} не встречалось при обучении.")
                                    st.stop()

                        # Масштабирование
                        scaler = st.session_state['scaler']
                        num_features = st.session_state['numerical_features']
                        input_data[num_features] = scaler.transform(input_data[num_features])

                        # Предсказание
                        model_results = st.session_state['model_results']
                        best_model_name = max(model_results, key=lambda x: model_results[x]["R2"])
                        best_model = st.session_state['trained_models'][best_model_name]

                        prediction = best_model.predict(input_data)[0]

                        st.success(f"**Предсказанная итоговая стоимость: ${prediction:,.2f}**")
                        st.info(f"Модель: **{best_model_name}** (R² = {model_results[best_model_name]['R2']:.4f})")

    else:
        st.info("Нажмите кнопку «Загрузить данные» для начала работы.")

analysis_and_model_page()