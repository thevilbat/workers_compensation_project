import streamlit as st

# Правильный формат: список объектов st.Page
pages = [
    st.Page("analysis_and_model.py", title="Анализ и модель"),
    st.Page("presentation.py", title="Презентация"),
]

# Навигация
current_page = st.navigation(
    pages,
    position="sidebar",
    expanded=True
)

current_page.run()