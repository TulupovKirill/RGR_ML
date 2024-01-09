import streamlit as st
import pandas as pd

st.title("Ближайшие объекты к Земле")

st.header("О датасете")

st.write("Этот файл содержит различные параметры/характеристики, на основании которых конкретный астероид, который уже классифицируется как ближайший земной объект, может быть или не быть опасным.")

df = pd.read_csv("neo_task.csv")
st.table(df.head())

st.write("Всего в датасете 90836 объектов и 8 признаков")

st.header("Описание признаков")

st.write("id - индентификатор, float64")
st.write("name - название объекта, object")
st.write("est_diameter_min - оценочный минимальный диаметр тела, float64")
st.write("est_diameter_max - оценочный максимальный диаметр тела, float64")
st.write("relative_velocity - относительная скорость, float64")
st.write("miss_distance - расстояние от Земли до тела, float64")
st.write("absolute_magnitude - абсолютная величина, float64")
st.write("hazardous - опасность для Земли, bool")

st.header("Предобработка данных")

st.write("В датасете присутствуют пропущенные значения")

st.table(df.isna().sum())

st.subheader("Заполним пропущенные значения:")
st.write("столбцы name и id модой этого же признака")
st.write("absolute_magnitude медианой est_diameter_min")
st.write("relative_velocity средним miss_distance")
st.write("est_diameter_max средней разницей между est_diameter_max и est_diameter_min")

code = '''x = lambda x: x.mean() if x.notna().any() else 0
group = df.groupby('miss_distance')['relative_velocity'].transform('mean').iat[0]
df['relative_velocity'].fillna(group, inplace=True)

x = lambda x: x.median() if x.notna().any() else 0
group = df.groupby('est_diameter_min')['absolute_magnitude'].transform(x)
df['absolute_magnitude'].fillna(group, inplace=True)

table = df['est_diameter_max'] - df['est_diameter_min']
df['est_diameter_max'].fillna(df['est_diameter_min'] + table.mean(), inplace=True)

df['name'].fillna(df['name'].mode()[0], inplace=True)

df['id'].fillna(df['id'].mode()[1], inplace=True)'''
st.code(code, language='python')

st.write("Теперь в датасете нет пропущенных значений.")

df = pd.read_csv('new_neo_task.csv')

st.table(df.isna().sum())

st.subheader("Проверим наличие дубликатов")

st.write(df.duplicated().sum())

st.subheader("Удаление аномалий")

st.write("В стоблце absolute_magnitude присутвуют значения равные 0, поэтому удалим их")

code = '''df = df.drop(index=df.loc[df['absolute_magnitude'] == 0].index)'''
st.code(code, language='python')

st.write("Все типы признаков нас устраивают, пропущенных значений нет, как и дубликатов, а также удалили аномалии. На этом предобратока данных закончена.")