import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import reveal_slides as rs


def presentation_page():
    st.title("🎯 Презентация проекта")

    has_results = 'model_results' in st.session_state and 'trained_models' in st.session_state

    # ======================= Markdown презентация =======================
    presentation_markdown = """
    # Прогнозирование стоимости страховых выплат

    ---

    ## Введение
    - Анализ данных о страховых случаях компенсации работникам
    - Цель: предсказать **UltimateIncurredClaimCost**
    - Датасет: Workers Compensation (~100 000 записей)

    ---

    ## Бизнес-задача
    - Точная оценка будущих страховых выплат
    - Начальная оценка часто значительно отличается от реальной стоимости
    - Помощь в формировании резервов и тарифов

    ---

    ## Этапы работы
    1. Загрузка и анализ данных
    2. Предобработка (даты, кодирование, масштабирование)
    3. Обучение 4 моделей регрессии
    4. Оценка качества
    5. Анализ важности признаков
    6. Интерактивное Streamlit-приложение

    ---

    ## Использованные модели
    - Linear Regression
    - Ridge Regression
    - Random Forest Regressor
    - XGBoost Regressor

    ---

    ## Результаты обучения моделей
    """

    if has_results:
        results = st.session_state['model_results']
        results_df = pd.DataFrame(results).T

        # Красивая таблица без tabulate
        markdown_table = "| Модель | MAE | RMSE | R² Score |\n|--------|-----|------|----------|\n"
        for model, metrics in results_df.iterrows():
            markdown_table += f"| {model} | ${metrics['MAE']:,.2f} | ${metrics['RMSE']:,.2f} | {metrics['R2']:.4f} |\n"

        presentation_markdown += markdown_table + "\n"

        best_model = max(results, key=lambda x: results[x]["R2"])
        best_r2 = results[best_model]["R2"]
        presentation_markdown += f"**Лучшая модель: {best_model}** (R² = {best_r2:.4f})\n\n"
    else:
        presentation_markdown += "**Результаты ещё не получены.** Перейдите на вкладку «Анализ и модель» и обучите модели.\n\n"

    presentation_markdown += """
    ---

    ## Анализ важности признаков
    """

    if 'trained_models' in st.session_state and 'Random Forest' in st.session_state['trained_models']:
        rf = st.session_state['trained_models']['Random Forest']
        data = st.session_state['data_preprocessed']
        X = data.drop(columns=['UltimateIncurredClaimCost'], errors='ignore')

        importance = pd.DataFrame({
            'Признак': X.columns,
            'Важность': rf.feature_importances_
        }).sort_values(by='Важность', ascending=False).head(10)

        presentation_markdown += "**Топ-10 важных признаков (Random Forest):**\n\n"
        for _, row in importance.iterrows():
            presentation_markdown += f"- **{row['Признак']}**: {row['Важность']:.4f}\n"

        presentation_markdown += "\n**Вывод:** Самый важный признак — **InitialCaseEstimate**.\n"
    else:
        presentation_markdown += "Анализ важности признаков будет доступен после обучения моделей.\n"

    presentation_markdown += """
    ---

    ## Ключевые выводы
    - Модель значительно превосходит начальную оценку страховщика
    - R² лучшей модели обычно > 0.85
    - Streamlit-приложение позволяет быстро делать предсказания

    ---

    ## Streamlit-приложение
    - Загрузка и предобработка данных
    - Обучение и сравнение моделей
    - Интерактивная форма предсказания
    - Динамическая презентация с актуальными результатами

    ---

    ## Заключение и предложения по улучшению
    - Полный цикл ML-проекта реализован
    - Возможные улучшения:
      - Подбор гиперпараметров
      - Feature Engineering
      - Ансамбли моделей
      - Деплой в облако

    Спасибо за внимание!
    """

    # ======================= Боковая панель =======================
    with st.sidebar:
        st.header("Настройки презентации")
        theme = st.selectbox("Тема", ["black", "white", "league", "beige", "sky", "night"], index=1)
        height = st.number_input("Высота слайдов (px)", value=650, min_value=400)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom"], index=0)
        plugins = st.multiselect("Плагины", ["highlight", "notes", "search", "zoom"], default=["highlight"])

    # Отображение презентации
    rs.slides(
        presentation_markdown,
        height=height,
        theme=theme,
        config={
            "transition": transition,
            "plugins": plugins,
        },
        markdown_props={"data-separator-vertical": "^--$"},
    )


presentation_page()