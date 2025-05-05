# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 08:57:14 2025

@author: Lenovo-PC
"""

# Importación
from pydataxm.pydatasimem import CatalogSIMEM

# Crear una instancia de catalogo con el tipo
catalogo_conjuntos = CatalogSIMEM('Datasets')

# Extraer información a utilizar
print("Nombre: ", catalogo_conjuntos.get_name())
print("Metadata: ", catalogo_conjuntos.get_metadata())
print("Columnas: ", catalogo_conjuntos.get_columns())

#  Dataframe con información de los conjuntos de datos
data = catalogo_conjuntos.get_data()
print(data)

path = r'D:\DATOS\MTE\Datasets.csv'
data.to_csv(path, index=False)


#%%

# Crear una instancia de catalogo con el tipo
catalogo_vbles = CatalogSIMEM('variables')

# Extraer información a utilizar
print("Nombre: ", catalogo_vbles.get_name())
print("Metadata: ", catalogo_vbles.get_metadata())
print("Columnas: ", catalogo_vbles.get_columns())

# Dataframe con información de las variables
data = catalogo_vbles.get_data()
print(data)

path = r'D:\DATOS\MTE\variables.csv'
data.to_csv(path, index=False)

#%%
from pydataxm import *                           #Se realiza la importación de las librerias necesarias para ejecutar
import datetime as dt  

objetoAPI = pydataxm.ReadDB()    
df = objetoAPI.get_collections() #El método get_collection sin argumentos retorna todas las variables que se tienen disponible en la API y que se pueden consultar  
df.head()   
#%%
path = r'D:\DATOS\MTE\Datos_XM.csv'
df.to_csv(path, index=False)

#%%
df_variable = objetoAPI.request_data(
                        "DemaReal",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Agente",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2022, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2022, 1, 15)) #Corresponde a la fecha final de la consulta

df_variable.head(5)  

#%%

path = r'D:\DATOS\MTE\Data_Demanda_Agente.csv'
df_variable.to_csv(path, index=False)

#%%
df_variable = objetoAPI.request_data(
                        "ListadoAGPE",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Agente",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2025, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 1, 2)) #Corresponde a la fecha final de la consulta

df_variable.head(5)  

#%%

path = r'D:\DATOS\MTE\ListadoAGPE.csv'
df_variable.to_csv(path, index=False)

#%%
df_variable = objetoAPI.request_data(
                        "ListadoMetricas",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2025, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 1, 2)) #Corresponde a la fecha final de la consulta

df_variable.head(5)  

path = r'D:\DATOS\MTE\ListadoMetricas.csv'
df_variable.to_csv(path, index=False)

#%%
df_variable = objetoAPI.request_data(
                        "ListadoEmbalses",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2025, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 1, 2)) #Corresponde a la fecha final de la consulta

df_variable.head(5)  

path = r'D:\DATOS\MTE\XM_API\ListadoEmbalses.csv'
df_variable.to_csv(path, index=False)

#%%
df_variable = objetoAPI.request_data(
                        "ListadoRios",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2025, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 1, 2)) #Corresponde a la fecha final de la consulta

df_variable.head(5)  

path = r'D:\DATOS\MTE\XM_API\ListadoRios.csv'
df_variable.to_csv(path, index=False)

#%%
df_variable = objetoAPI.request_data(
                        "ListadoAgentes",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2025, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 1, 2)) #Corresponde a la fecha final de la consulta

df_variable.head(5)  

path = r'D:\DATOS\MTE\XM_API\ListadoAgentes.csv'
df_variable.to_csv(path, index=False, encoding='utf-8')

#%%

from pydataxm import *                           #Se realiza la importación de las librerias necesarias para ejecutar
import datetime as dt  

objetoAPI = pydataxm.ReadDB()    
df = objetoAPI.get_collections() #El método get_collection sin argumentos retorna todas las variables que se tienen disponible en la API y que se pueden consultar  
df.head()  

#%% LISTADO DE RECURSOS
df_variable = objetoAPI.request_data(
                        "ListadoRecursos",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2025, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 1, 2)) #Corresponde a la fecha final de la consulta

df_variable.head(5)  

path = r'D:\DATOS\MTE\XM_API\ListadoRecursos_Sistema.csv'
df_variable.to_csv(path, index=False, encoding='utf-8')

#%%

import matplotlib.pyplot as plt

# Diccionario para nombres más legibles
titulos_legibles = {
    'Values_Type': 'Tipo de Tecnología',
    'Values_Disp': 'Tipo de Despacho',
    'Values_RecType': 'Tipo de Unidad',
    'Values_EnerSource': 'Fuente de Energía',
    'Values_State': 'Estado Operativo'
}

# Lista de columnas a graficar
columnas = list(titulos_legibles.keys())

for col in columnas:
    conteo = df_variable[col].value_counts()
    total = conteo.sum()

    # Crear etiquetas con nombre + porcentaje
    etiquetas = [f'{cat} ({(val/total)*100:.1f}%)' for cat, val in zip(conteo.index, conteo)]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Crear el gráfico sin etiquetas en la torta
    wedges, _ = ax.pie(conteo, labels=None, startangle=140)

    # Agregar leyenda con nombre y porcentaje
    ax.legend(
        wedges,
        etiquetas,
        title=titulos_legibles[col],
        loc='lower right',
        bbox_to_anchor=(1, 0),
        fontsize=9
    )

    # Título descriptivo
    ax.set_title(f'Distribución por {titulos_legibles[col]}', fontsize=14)

    ax.axis('equal')  # Forma de círculo
    plt.tight_layout()
    plt.show()

#%% VOLUMEN UTIL DIARI ENERGIA
df_variable = objetoAPI.request_data(
                        "VoluUtilDiarEner",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2022, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 15)) #Corresponde a la fecha final de la consulta

df_variable.head(5)  

# path = r'D:\DATOS\MTE\XM_API\VoluUtilDiarEner.csv'
# df_variable.to_csv(path, index=False, encoding='utf-8')

#%%
import matplotlib.pyplot as plt
import pandas as pd

# Asegúrate de que 'Date' esté en formato datetime
df_variable['Date'] = pd.to_datetime(df_variable['Date'], format='%d/%m/%Y')

# Establecer la columna 'Date' como índice
df_variable.set_index('Date', inplace=True)

# Graficar la evolución de los valores
plt.figure(figsize=(10, 5))
plt.plot(df_variable.index, df_variable['Value'], marker='o', linestyle='-', color='royalblue')
plt.title('Volumen Util Diario')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% APORTES DIARIOS
df_variable = objetoAPI.request_data(
                        "PorcApor",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2022, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 15)) #Corresponde a la fecha final de la consulta

df_variable.head(5)  

# path = r'D:\DATOS\MTE\XM_API\VoluUtilDiarEner.csv'
# df_variable.to_csv(path, index=False, encoding='utf-8')

#%%
import matplotlib.pyplot as plt
import pandas as pd

# Asegúrate de que 'Date' esté en formato datetime
df_variable['Date'] = pd.to_datetime(df_variable['Date'], format='%d/%m/%Y')

# Establecer la columna 'Date' como índice
df_variable.set_index('Date', inplace=True)

# Graficar la evolución de los valores
plt.figure(figsize=(10, 5))
plt.plot(df_variable.index, df_variable['Value'], marker='o', linestyle='-', color='royalblue')
plt.title('Aportes Diarios')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Media Histórica de Aportes del SIN
df_variable = objetoAPI.request_data(
                        "CapaUtilDiarEner",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2022, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 15)) #Corresponde a la fecha final de la consulta

df_variable.head(5)  

# path = r'D:\DATOS\MTE\XM_API\VoluUtilDiarEner.csv'
# df_variable.to_csv(path, index=False, encoding='utf-8')

#%%
import matplotlib.pyplot as plt
import pandas as pd

# Asegúrate de que 'Date' esté en formato datetime
df_variable['Date'] = pd.to_datetime(df_variable['Date'], format='%d/%m/%Y')

# Establecer la columna 'Date' como índice
df_variable.set_index('Date', inplace=True)

# Graficar la evolución de los valores
plt.figure(figsize=(10, 5))
plt.plot(df_variable.index, df_variable['Value'], marker='o', linestyle='-', color='royalblue')
plt.title('Media Histórica de Aportes del SIN')
plt.xlabel('Fecha')
plt.ylabel('Valor')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Demanda Comercial por Sistema
df_variable = objetoAPI.request_data(
                        "DemaCome",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2025, 4, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 21)) #Corresponde a la fecha final de la consulta

df_variable.head(5) 

# path = r'D:\DATOS\MTE\XM_API\DemaCome.csv'
# df_variable.to_csv(path, index=False, encoding='utf-8')

#%%
import pandas as pd
import matplotlib.pyplot as plt

# Asegúrate de que la fecha esté bien formateada
df_variable['Date'] = pd.to_datetime(df_variable['Date'], format='%d/%m/%Y')

# Tomamos solo las columnas que contienen las horas
cols_horas = [col for col in df_variable.columns if 'Values_Hour' in col]

# Lista para guardar los valores
fechas_horas = []
valores = []

# Iterar sobre cada fila (día)
for idx, fila in df_variable.iterrows():
    fecha_base = fila['Date']
    for i, col in enumerate(cols_horas):
        dt = fecha_base + pd.Timedelta(hours=i)
        fechas_horas.append(dt)
        valores.append(fila[col])

# Crear un DataFrame final
df_final = pd.DataFrame({'Datetime': fechas_horas, 'Valor': valores})
df_final.set_index('Datetime', inplace=True)

# Graficar
plt.figure(figsize=(14,6))
plt.plot(df_final.index, df_final['Valor'], color='steelblue')
plt.title('Demanda Comercial por Sistema')
plt.xlabel('Fecha y hora')
plt.ylabel('kWh')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Demanda Máxima Potencia
df_variable = objetoAPI.request_data(
                        "DemaMaxPot",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2024, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 15)) #Corresponde a la fecha final de la consulta

df_variable.head(5) 

# path = r'D:\DATOS\MTE\XM_API\DemaCome.csv'
# df_variable.to_csv(path, index=False, encoding='utf-8')

#%%
import matplotlib.pyplot as plt
import pandas as pd

# Asegúrate de que 'Date' esté en formato datetime
df_variable['Date'] = pd.to_datetime(df_variable['Date'], format='%d/%m/%Y')

# Establecer la columna 'Date' como índice
df_variable.set_index('Date', inplace=True)

# Graficar la evolución de los valores
plt.figure(figsize=(10, 5))
plt.plot(df_variable.index, df_variable['Value'], marker='o', linestyle='-', color='royalblue')
plt.title('Demanda Máxima Potencia')
plt.xlabel('Fecha')
plt.ylabel('kW')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Demanda Energía Escenario UPME (Alto/Medio/Bajo):
df_variable_A = objetoAPI.request_data(
                        "EscDemUPMEAlto",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2022, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 15)) #Corresponde a la fecha final de la consulta

df_variable_M = objetoAPI.request_data(
                        "EscDemUPMEMedio",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2022, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 15)) #Corresponde a la fecha final de la consulta

df_variable_B = objetoAPI.request_data(
                        "EscDemUPMEBajo",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2022, 1, 1),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 15)) #Corresponde a la fecha final de la consulta

# df_variable.head(5) 

# path = r'D:\DATOS\MTE\XM_API\DemaCome.csv'
# df_variable.to_csv(path, index=False, encoding='utf-8')

#%%
import matplotlib.pyplot as plt
import pandas as pd

# Asegúrate de que 'Date' esté en formato datetime
df_variable_A['Date'] = pd.to_datetime(df_variable_A['Date'], format='%d/%m/%Y')
# Establecer la columna 'Date' como índice
df_variable_A.set_index('Date', inplace=True)

# Asegúrate de que 'Date' esté en formato datetime
df_variable_M['Date'] = pd.to_datetime(df_variable_M['Date'], format='%d/%m/%Y')
# Establecer la columna 'Date' como índice
df_variable_M.set_index('Date', inplace=True)

# Asegúrate de que 'Date' esté en formato datetime
df_variable_B['Date'] = pd.to_datetime(df_variable_B['Date'], format='%d/%m/%Y')
# Establecer la columna 'Date' como índice
df_variable_B.set_index('Date', inplace=True)

# Graficar la evolución de los valores
plt.figure(figsize=(10, 5))
plt.plot(df_variable_A.index, df_variable_A['Value'], marker='o', linestyle='-', color='red')
plt.plot(df_variable_M.index, df_variable_M['Value'], marker='o', linestyle='-', color='yellow')
plt.plot(df_variable_B.index, df_variable_B['Value'], marker='o', linestyle='-', color='green')
plt.title('Demanda Energía Escenario UPME (Alto/Medio/Bajo)')
plt.xlabel('Fecha')
plt.ylabel('kW')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% Generación Real Total y Generación Real por Recurso:
    
objetoAPI = pydataxm.ReadDB()    
# df = objetoAPI.get_collections() #El método get_collection sin argumentos retorna todas las variables que se tienen disponible en la API y que se pueden consultar  
    
df_variable = objetoAPI.request_data(
                        "Gene",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2024, 4, 15),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 15)) #Corresponde a la fecha final de la consulta

df_variable.head(5) 

# path = r'D:\DATOS\MTE\XM_API\Gene.csv'
# df_variable.to_csv(path, index=False, encoding='utf-8')

#%%
import pandas as pd
import matplotlib.pyplot as plt

# Asegúrate de que la fecha esté bien formateada
df_variable['Date'] = pd.to_datetime(df_variable['Date'], format='%d/%m/%Y')

# Tomamos solo las columnas que contienen las horas
cols_horas = [col for col in df_variable.columns if 'Values_Hour' in col]

# Lista para guardar los valores
fechas_horas = []
valores = []

# Iterar sobre cada fila (día)
for idx, fila in df_variable.iterrows():
    fecha_base = fila['Date']
    for i, col in enumerate(cols_horas):
        dt = fecha_base + pd.Timedelta(hours=i)
        fechas_horas.append(dt)
        valores.append(fila[col])

# Crear un DataFrame final
df_final = pd.DataFrame({'Datetime': fechas_horas, 'Valor': valores})
df_final.set_index('Datetime', inplace=True)

# Graficar
plt.figure(figsize=(14,6))
plt.plot(df_final.index, df_final['Valor'], color='steelblue')
plt.title('Generación Real Total')
plt.xlabel('Fecha y hora')
plt.ylabel('kWh')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Consumo de Combustible por Recurso::
    
# objetoAPI = pydataxm.ReadDB()    
# df = objetoAPI.get_collections() #El método get_collection sin argumentos retorna todas las variables que se tienen disponible en la API y que se pueden consultar  
    
df_variable = objetoAPI.request_data(
                        "ConsCombustibleMBTU",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Combustible",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2025, 1, 15),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 15)) #Corresponde a la fecha final de la consulta

# df_variable.head(5) 

# path = r'D:\DATOS\MTE\XM_API\ConsCombustibleMBTU.csv'
# df_variable.to_csv(path, index=False, encoding='utf-8')

#%%
import pandas as pd
import matplotlib.pyplot as plt

# Asegúrate de que la fecha esté bien formateada
df_variable['Date'] = pd.to_datetime(df_variable['Date'], format='%d/%m/%Y')

# Tomamos solo las columnas que contienen las horas
cols_horas = [col for col in df_variable.columns if 'Values_Hour' in col]

# Lista para guardar los valores
fechas_horas = []
valores = []

# Iterar sobre cada fila (día)
for idx, fila in df_variable.iterrows():
    fecha_base = fila['Date']
    for i, col in enumerate(cols_horas):
        dt = fecha_base + pd.Timedelta(hours=i)
        fechas_horas.append(dt)
        valores.append(fila[col])

# Crear un DataFrame final
df_final = pd.DataFrame({'Datetime': fechas_horas, 'Valor': valores})
df_final.set_index('Datetime', inplace=True)

# Graficar
plt.figure(figsize=(14,6))
plt.plot(df_final.index, df_final['Valor'], color='steelblue')
plt.title('Consumo de Combustible')
plt.xlabel('Fecha y hora')
plt.ylabel('MBTU')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Precio Bolsa Nacional Ponderado por Sistema
    
# objetoAPI = pydataxm.ReadDB()    
# df = objetoAPI.get_collections() #El método get_collection sin argumentos retorna todas las variables que se tienen disponible en la API y que se pueden consultar  
# df.head()  
    
df_variable = objetoAPI.request_data(
                        "PrecBolsNaci",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2022, 1, 15),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 15)) #Corresponde a la fecha final de la consulta

# df_variable.head(5) 

path = r'D:\DATOS\MTE\XM_API\PrecBolsNaci.csv'
df_variable.to_csv(path, index=False, encoding='utf-8')

#%%
import pandas as pd
import matplotlib.pyplot as plt

# Asegúrate de que la fecha esté bien formateada
df_variable['Date'] = pd.to_datetime(df_variable['Date'], format='%d/%m/%Y')

# Tomamos solo las columnas que contienen las horas
cols_horas = [col for col in df_variable.columns if 'Values_Hour' in col]

# Lista para guardar los valores
fechas_horas = []
valores = []

# Iterar sobre cada fila (día)
for idx, fila in df_variable.iterrows():
    fecha_base = fila['Date']
    for i, col in enumerate(cols_horas):
        dt = fecha_base + pd.Timedelta(hours=i)
        fechas_horas.append(dt)
        valores.append(fila[col])

# Crear un DataFrame final
df_final = pd.DataFrame({'Datetime': fechas_horas, 'Valor': valores})
df_final.set_index('Datetime', inplace=True)

# Graficar
plt.figure(figsize=(14,6))
plt.plot(df_final.index, df_final['Valor'], color='steelblue')
plt.title('Precio Bolsa Nacional')
plt.xlabel('Fecha y hora')
plt.ylabel('COP/kWh')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Precio de Oferta de Despacho por Recurso
    
# objetoAPI = pydataxm.ReadDB()    
# df = objetoAPI.get_collections() #El método get_collection sin argumentos retorna todas las variables que se tienen disponible en la API y que se pueden consultar  
# df.head()  
    
df_variable = objetoAPI.request_data(
                        "PrecOferDesp",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Recurso",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2025, 1, 15),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 15)) #Corresponde a la fecha final de la consulta

# df_variable.head(5) 

path = r'D:\DATOS\MTE\XM_API\PrecOferDesp.csv'
df_variable.to_csv(path, index=False, encoding='utf-8')

#%% Exportaciones EnergÃ­a por Sistema
    
from pydataxm import *                           #Se realiza la importación de las librerias necesarias para ejecutar
import datetime as dt  

objetoAPI = pydataxm.ReadDB()    
df = objetoAPI.get_collections() #El método get_collection sin argumentos retorna todas las variables que se tienen disponible en la API y que se pueden consultar  
df.head()  
    
#%%
df_variable = objetoAPI.request_data(
                        "ExpoEner",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2025, 1, 15),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 15)) #Corresponde a la fecha final de la consulta

# df_variable.head(5) 

path = r'D:\DATOS\MTE\XM_API\ExpoEner.csv'
df_variable.to_csv(path, index=False, encoding='utf-8')

#%%
import pandas as pd
import matplotlib.pyplot as plt

# Asegúrate de que la fecha esté bien formateada
df_variable['Date'] = pd.to_datetime(df_variable['Date'], format='%d/%m/%Y')

# Tomamos solo las columnas que contienen las horas
cols_horas = [col for col in df_variable.columns if 'Values_Hour' in col]

# Lista para guardar los valores
fechas_horas = []
valores = []

# Iterar sobre cada fila (día)
for idx, fila in df_variable.iterrows():
    fecha_base = fila['Date']
    for i, col in enumerate(cols_horas):
        dt = fecha_base + pd.Timedelta(hours=i)
        fechas_horas.append(dt)
        valores.append(fila[col])

# Crear un DataFrame final
df_final = pd.DataFrame({'Datetime': fechas_horas, 'Valor': valores})
df_final.set_index('Datetime', inplace=True)

# Graficar
plt.figure(figsize=(14,6))
plt.plot(df_final.index, df_final['Valor'], color='steelblue')
plt.title('Exportaciones EnergÃ­a')
plt.xlabel('Fecha y hora')
plt.ylabel('kWh')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% Importaciones EnergÃ­a por Sistema
    
from pydataxm import *                           #Se realiza la importación de las librerias necesarias para ejecutar
import datetime as dt  

objetoAPI = pydataxm.ReadDB()    
df = objetoAPI.get_collections() #El método get_collection sin argumentos retorna todas las variables que se tienen disponible en la API y que se pueden consultar  
df.head()  
    
#%%
df_variable = objetoAPI.request_data(
                        "ImpoEner",           #Se indica el nombre de la métrica tal como se llama en el campo metricId
                        "Sistema",             #Se indica el nombre de la entidad tal como se llama en el campo Entity
                        dt.date(2025, 1, 15),  #Corresponde a la fecha inicial de la consulta
                        dt.date(2025, 4, 15)) #Corresponde a la fecha final de la consulta

# df_variable.head(5) 

path = r'D:\DATOS\MTE\XM_API\ImpoEner.csv'
df_variable.to_csv(path, index=False, encoding='utf-8')

#%%
import pandas as pd
import matplotlib.pyplot as plt

# Asegúrate de que la fecha esté bien formateada
df_variable['Date'] = pd.to_datetime(df_variable['Date'], format='%d/%m/%Y')

# Tomamos solo las columnas que contienen las horas
cols_horas = [col for col in df_variable.columns if 'Values_Hour' in col]

# Lista para guardar los valores
fechas_horas = []
valores = []

# Iterar sobre cada fila (día)
for idx, fila in df_variable.iterrows():
    fecha_base = fila['Date']
    for i, col in enumerate(cols_horas):
        dt = fecha_base + pd.Timedelta(hours=i)
        fechas_horas.append(dt)
        valores.append(fila[col])

# Crear un DataFrame final
df_final = pd.DataFrame({'Datetime': fechas_horas, 'Valor': valores})
df_final.set_index('Datetime', inplace=True)

# Graficar
plt.figure(figsize=(14,6))
plt.plot(df_final.index, df_final['Valor'], color='steelblue')
plt.title('Importaciones EnergÃ­a')
plt.xlabel('Fecha y hora')
plt.ylabel('kWh')
plt.grid(True)
plt.tight_layout()
plt.show()

#%% CONSULTA API - Exportaciones e Importaciones de Energía

from pydataxm import *      
import datetime as dt  
import pandas as pd
import matplotlib.pyplot as plt

objetoAPI = pydataxm.ReadDB()    

# Consulta de exportaciones
df_expo = objetoAPI.request_data(
    "ExpoEner", "Sistema",
    dt.date(2025, 1, 15),
    dt.date(2025, 4, 15)
)

# Guardar CSV
path_expo = r'D:\DATOS\MTE\XM_API\ExpoEner.csv'
df_expo.to_csv(path_expo, index=False, encoding='utf-8')

# Consulta de importaciones
df_impo = objetoAPI.request_data(
    "ImpoEner", "Sistema",
    dt.date(2025, 1, 15),
    dt.date(2025, 4, 15)
)

# Guardar CSV
path_impo = r'D:\DATOS\MTE\XM_API\ImpoEner.csv'
df_impo.to_csv(path_impo, index=False, encoding='utf-8')

#%% PROCESAMIENTO - Conversión a series temporales horarias

def convertir_a_serie_horaria(df):
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    cols_horas = [col for col in df.columns if 'Values_Hour' in col]
    
    fechas_horas = []
    valores = []

    for idx, fila in df.iterrows():
        fecha_base = fila['Date']
        for i, col in enumerate(cols_horas):
            dt_ = fecha_base + pd.Timedelta(hours=i)
            fechas_horas.append(dt_)
            valores.append(fila[col])
    
    df_final = pd.DataFrame({'Datetime': fechas_horas, 'Valor': valores})
    df_final.set_index('Datetime', inplace=True)
    return df_final

df_expo_horario = convertir_a_serie_horaria(df_expo)
df_impo_horario = convertir_a_serie_horaria(df_impo)

#%% GRAFICAR AMBAS SERIES EN LA MISMA GRÁFICA

plt.figure(figsize=(15,6))
plt.plot(df_expo_horario.index, df_expo_horario['Valor'], label='Exportaciones')
plt.plot(df_impo_horario.index, df_impo_horario['Valor'], label='Importaciones')
plt.title('Exportaciones e Importaciones de Energía por Sistema')
plt.xlabel('Fecha y hora')
plt.ylabel('kWh')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%% Emisiones de CO2 Eq/kWh por Sistema

from pydataxm import *      
import datetime as dt  
import pandas as pd
import matplotlib.pyplot as plt

# Inicializar objeto de conexión
objetoAPI = pydataxm.ReadDB()

# Consulta de datos desde API
df_emisiones = objetoAPI.request_data(
                                        "factorEmisionCO2e", "Sistema",
                                        dt.date(2025, 1, 15),
                                        dt.date(2025, 4, 20)
                                    )

# Guardar CSV
path_emisiones = r'D:\DATOS\MTE\XM_API\factorEmisionCO2e.csv'
df_emisiones.to_csv(path_emisiones, index=False, encoding='utf-8')

#%% Procesamiento - Conversión a serie horaria

# Formato de fecha
df_emisiones['Date'] = pd.to_datetime(df_emisiones['Date'], format='%d/%m/%Y')

# Extraer columnas de valores por hora
cols_horas = [col for col in df_emisiones.columns if 'Values_Hour' in col]

# Convertir a serie horaria
fechas_horas = []
valores = []

for idx, fila in df_emisiones.iterrows():
    fecha_base = fila['Date']
    for i, col in enumerate(cols_horas):
        dt_ = fecha_base + pd.Timedelta(hours=i)
        fechas_horas.append(dt_)
        valores.append(fila[col])

# DataFrame final
df_emisiones_horaria = pd.DataFrame({'Datetime': fechas_horas, 'Valor': valores})
df_emisiones_horaria.set_index('Datetime', inplace=True)

#%% Gráfica

plt.figure(figsize=(15,6))
plt.plot(df_emisiones_horaria.index, df_emisiones_horaria['Valor'], color='darkorange')
plt.title('Emisiones de CO₂ equivalente por kWh – Sistema')
plt.xlabel('Fecha y hora')
plt.ylabel('gCO₂e / kWh')
plt.grid(True)
plt.tight_layout()
plt.show()



