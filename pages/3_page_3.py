import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np

df = pd.read_csv("neo_task.csv")

st.title("Разведывательный анализ данных")

st.subheader("Выведем основные статистические данные")

st.table(df.describe())

variable_1 = st.sidebar.radio('Выберите первый признак для точечного графика', ('relative_velocity', 'miss_distance', 'absolute_magnitude'))
variable_2 = st.sidebar.radio('Выберите второй признак для точечного графика', ('relative_velocity', 'miss_distance', 'absolute_magnitude'), 
index=2)

st.subheader("Точечный график")

fig2, ax2 = plt.subplots()
ax2.scatter(df[variable_1], df[variable_2])
plt.xlabel(variable_1)
plt.ylabel(variable_2)
st.pyplot(fig2)

st.markdown("**Вывод:**")
st.markdown("Исходя из данных на графике можно сделать вывод о том, между признаками *relative_velocity* и *miss_distance* видна линейность. С увеличением дистанции от Земли скорость объектов растёт. Помимо этого с увеличением *absolute_magnitude* падает значение *relative_velocity*.")

st.subheader("Гистограммы")

variable = st.sidebar.radio('Выберите признак для гистограммы', ('relative_velocity', 'miss_distance', 'absolute_magnitude'))

new_df = df[variable]
fig1, ax1 = plt.subplots()
plt.title(f"{variable}")
ax1.hist(new_df)
st.pyplot(fig1)

st.markdown("**Вывод:**")
st.markdown("На гистограмме признака *relative_velocity* значения смещены влево, значит значение среднего меньше медианы")
st.markdown("На гистограмме признака *absolute_magnitude* значения смещены немного вправо, значит значение среднего больше медианы")

new_df = np.unique(df['hazardous'], return_counts=True)
fig3, ax3 = plt.subplots()
st.subheader("Круговая диаграмма")
plt.title("hazardous")
ax3.pie(new_df[1],labels=["False", "True"], colors=['red', 'blue'], autopct='%1.1f%%')
st.pyplot(fig3)

st.markdown("**Вывод:**")
st.markdown("На круговой диаграмме виден дисбаланс классов по признаку *hazardous* в пользу класса False")

st.subheader("Тепловая карта датасета")

df_ohne_name = df.drop('name', axis=1)
fig4, ax4 = plt.subplots()
correlation_matrix = df_ohne_name.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
st.pyplot(fig4)

st.markdown("**Вывод:**")
st.markdown("На данной матрице корреляции видно, что признаки *est_diameter_min* и *est_diameter_max* корреллируют абсолютно, следовательно один из этих признаков стоило бы убрать, оставив всего один")
