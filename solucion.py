import pandas as pd

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