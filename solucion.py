import pandas as pd
<<<<<<< HEAD

# Suponiendo que tienes una lista de diccionarios con tus datos
data = [
    {'nombre': 'Juan', 'edad': 25, 'ciudad': 'Madrid'},
    {'nombre': 'Ana', 'edad': 30, 'ciudad': 'Barcelona'},
    {'nombre': 'Pedro', 'edad': 35, 'ciudad': 'Madrid'},
    # Añade más datos según sea necesario
]

# Convierte los datos en un DataFrame
df = pd.DataFrame(data)

# Imprime el DataFrame para verificar los datos
print(df)

# Categoriza los datos en grupos, por ejemplo, por ciudad
grouped = df.groupby('ciudad')

# Imprime los grupos para verificar
for name, group in grouped:
    print(name)
    print(group)

# Exporta cada grupo a un archivo CSV separado si es necesario
for name, group in grouped:
    group.to_csv(f'{name}.csv', index=False)
=======
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
# Asegúrate de reemplazar 'data.csv' con la ruta real a tu archivo de datos
df = pd.read_csv('data.csv')

# Ver las primeras filas del DataFrame para entender su estructura
print(df.head())

# Dividir los datos en conjuntos de entrenamiento y prueba
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar un modelo de Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Imprimir el reporte de clasificación y la matriz de confusión
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Visualizar la importancia de las características
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(inplace=True, ascending=False)

sns.barplot(x=feature_importances.values, y=feature_importances.index)
plt.xlabel('Importancia de las características')
plt.ylabel('Características')
plt.show()
>>>>>>> 43acd621e96457524e5530dee399ede07fcfb3ab
