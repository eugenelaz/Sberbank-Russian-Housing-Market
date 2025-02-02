import streamlit as st
import pandas as pd
import lightgbm as lgb

# Загрузка модели
model = lgb.Booster(model_file='best_model_lgbm.pkl')

# Список признаков
selected_features = ['full_sq', 'life_sq', 'floor', 'max_floor', 'kremlin_km']

st.title('💰 Предсказание цены квартиры в Москве')

# Поля для ввода данных
inputs = {}
col1, col2 = st.columns(2)
with col1:
    inputs['full_sq'] = st.number_input('Общая площадь (м²)', min_value=10, max_value=500, value=50)
    inputs['life_sq'] = st.number_input('Жилая площадь (м²)', min_value=10, max_value=300, value=30)
with col2:
    inputs['floor'] = st.number_input('Этаж', min_value=1, max_value=50, value=5)
    inputs['max_floor'] = st.number_input('Всего этажей', min_value=1, max_value=100, value=10)
inputs['kremlin_km'] = st.number_input('Расстояние до Кремля (км)', min_value=0.0, max_value=50.0, value=5.0)

# Предсказание
if st.button('Рассчитать цену'):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)
    st.success(f"**Предсказанная цена:** {prediction[0]:,.0f} руб.")