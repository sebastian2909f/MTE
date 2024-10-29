# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:49:21 2024

@author: sebas
"""

import networkx as nx
import matplotlib.pyplot as plt  # Opcional para visualizar el grafo
import pandas as pd             # manipulacion de dataframes
from geopy.distance import great_circle
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch_geometric_temporal.nn.recurrent import GConvGRU
from torch_geometric_temporal.nn.recurrent import GConvLSTM
from torch_geometric_temporal.nn.recurrent import MPNNLSTM
from torch_geometric_temporal.nn.attention import STConv
from torch_geometric_temporal.nn.recurrent import DCRNN
from torch_geometric_temporal.nn.recurrent import GCLSTM
from torch_geometric_temporal.nn.recurrent import EvolveGCNH, EvolveGCNO

# from labml_nn.graphs.gat import GraphAttentionLayer

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import scipy.sparse as sp

import time
import csv

#%% FUNCION DE SECUENCIAS

def split_sequences(features, targets, n_steps_in, n_steps_out, n_sliding_steps, window_type):
    """
    Args:
    * features: Secuencias de entrada que pueden ser univariadas o multivariadas
    * targets: Secuencias de salida que pueden ser univariadas o multivariadas
    * n_steps_in: Longitud de la secuencia de entrada para la ventana deslizante
    * n_steps_out: Longitud de la secuencia de salida.
    * n_sliding_steps: Tamaño del paso de la ventana.
    * window_type: Tipo de ventana 'sliding' or 'expanding'  ('Deslizante o expansiva')
    """
    X, y = list(), list()
    
    # Iterar a través de las secuencias con un paso deslizante
    for i in range(0, len(features), n_sliding_steps):
        # Calcula el final de la secuencia en curso
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # Comprueba si el ciclo está fuera del límite de secuencias
        if out_end_ix > len(features):
            break

        # Recopila las secuencias de entrada y salida del ciclo
        if window_type == 'sliding':  # Ventana deslizante
            seq_x, seq_y = features[i:end_ix], targets[end_ix:out_end_ix]
        else:  # Ventana expansiva
            seq_x, seq_y = features[0:end_ix], targets[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)

#%%
def get_adjacent_nodes(adj_matrix, node):
    adjacent_nodes = []
    for i in range(len(adj_matrix)):
        if adj_matrix[node][i] == 1:
            adjacent_nodes.append(i)
    return adjacent_nodes

#%%
def calcular_distancia(coord1, coord2):
    return great_circle(coord1, coord2).kilometers

#%%
df = pd.read_csv(r'D:\DATOS\MTE\GRAFOS\UBICACIONES_GENERACION.csv')


#%%
# Crear el diccionario
datos_modelo = {
    "NAME MODEL": [],
    "NODE": [],
    "K": [],
    "CELL": [],
    "SLI": [],
    "MSE": [],
    "MAE": [],
    "R2": [],
    "TIME": []
}

#%%
ima = 1
G = nx.Graph()
for i, ubi in df.iterrows():
  # print(i)
#  print(ubi)
  #print(ubi[0])
  #print(ubi[1])
  G.add_node("Nodo"+str(i), latitud = ubi[1], longitud = ubi[2])

# Establecer el color de los nodos especiales
# for i in [92, 93, 95, 96]:
#   G.nodes["Nodo"+str(i)]["color"] = "red"
  
for node in G.nodes:
    if "color" not in G.nodes[node]:
        G.nodes[node]["color"] = "skyblue"  # Color por defecto


#%%
rad_prox = 3.5

for nodo1 in G.nodes:
    # print(nodo1)
    for nodo2 in G.nodes:
        if nodo1 != nodo2:
            distancia = calcular_distancia((G.nodes[nodo1]["latitud"], G.nodes[nodo1]["longitud"]),
                                         (G.nodes[nodo2]["latitud"], G.nodes[nodo2]["longitud"]))
            if distancia <= rad_prox:
                G.add_edge(nodo1, nodo2, distancia=distancia)

# for nodo1 in ['Nodo92', 'Nodo93', 'Nodo95', 'Nodo96']:
#     for nodo2 in ['Nodo92', 'Nodo93', 'Nodo95', 'Nodo96']:
#         if nodo1 != nodo2:
#             distancia = calcular_distancia((G.nodes[nodo1]["latitud"], G.nodes[nodo1]["longitud"]),
#                                          (G.nodes[nodo2]["latitud"], G.nodes[nodo2]["longitud"]))            
#             G.add_edge(nodo1, nodo2, distancia=distancia)            

#%% SE AGREGA MANUALMENTE CONEXIONES ENTRE NODOS

# G.add_edge('Nodo19', 'Nodo16')
# G.add_edge('Nodo19', 'Nodo13')
# G.add_edge('Nodo19', 'Nodo10')
# G.add_edge('Nodo13', 'Nodo16')
# G.add_edge('Nodo16', 'Nodo10')
# G.add_edge('Nodo16', 'Nodo11')
# G.add_edge('Nodo16', 'Nodo12')
# G.add_edge('Nodo12', 'Nodo6')
# G.add_edge('Nodo12', 'Nodo8')
# G.add_edge('Nodo12', 'Nodo11')
# G.add_edge('Nodo12', 'Nodo13')
# G.add_edge('Nodo9', 'Nodo0')
# G.add_edge('Nodo9', 'Nodo2')
# G.add_edge('Nodo9', 'Nodo18')
# G.add_edge('Nodo0', 'Nodo2')
# G.add_edge('Nodo0', 'Nodo7')
# G.add_edge('Nodo0', 'Nodo1')
                
#%%
if ima == 1:
    # Definir un tamaño personalizado para la figura
    fig, ax = plt.subplots(figsize=(6, 6))  # Ajusta las dimensiones según tus necesidades
    pos = {node: (G.nodes[node]["latitud"], G.nodes[node]["longitud"]) for node in G.nodes}
    node_colors = [G.nodes[node]["color"] for node in G.nodes]
    nx.draw(G, pos, with_labels=True, node_size=100, font_size=8, node_color=node_colors)
    plt.show()                

#%%
# Obtiene la matriz de adyacencia en forma de numpy array
adjacency_matrix = nx.to_numpy_array(G)

df2 = pd.DataFrame(adjacency_matrix)
# Configura las opciones de visualización para mostrar todas las filas y columnas del array NumPy
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# La matriz de adyacencia es una matriz de tipo numpy
#print(df2)
#%% MATRIZ DISPERSA
if ima == 1:
    # Graficar la matriz dispersa
    plt.figure(figsize=(3, 3))
    # plt.spy(adjacency_matrix, markersize=10)
    plt.spy(adjacency_matrix, markersize=10, aspect='auto')
    # plt.title('Matriz Dispersa')
    plt.grid(True)
    # Configurar las marcas en los ejes X e Y cada 1 unidad
    plt.xticks(np.arange(0, len(adjacency_matrix), 1))
    plt.yticks(np.arange(0, len(adjacency_matrix), 1))
    # Eliminar los números de los ejes
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    # Activar la cuadrícula con líneas cada 1 unidad
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
#%%
# Cuenta el número de aristas en el grafo (elementos no nulos)
num_edges = np.sum(adjacency_matrix) // 2  # Divide por 2 si el grafo es no dirigido

# Imprime el número de aristas
print(f"El grafo tiene {num_edges} aristas.")

#%%
node_val = []
# Obtenemos los nodos conectados al nodo 93
for i in range(0,16):
    print('------------------------------------')
    adjacent_nodes = get_adjacent_nodes(df2, i)
    print('NODO ', i)
    print(adjacent_nodes)
    node_val = np.append(node_val,adjacent_nodes)
    
#%% ELIMINA VALORES REPETIDOS
node_val = np.unique(node_val)

# print(len(G.nodes))

for i in range(len(G.nodes)):
    if i in  node_val:
        pass
    else:
        G.remove_node('Nodo'+str(i))

node_val = node_val.astype(np.int64)
print(node_val)
#%%GRAFICA GRAFO
if ima == 1:
    # Definir un tamaño personalizado para la figura
    fig, ax = plt.subplots(figsize=(6, 6))  # Ajusta las dimensiones según tus necesidades
    pos = {node: (G.nodes[node]["latitud"], G.nodes[node]["longitud"]) for node in G.nodes}
    node_colors = [G.nodes[node]["color"] for node in G.nodes]
    # nx.draw(G, pos, with_labels=True, node_size=100, font_size=8, node_color=node_colors)
    nx.draw(G, pos, with_labels=False, node_size=100, font_size=8, node_color=node_colors)
    plt.show()   

#%%Obtiene la matriz de adyacencia en forma de numpy array
adjacency_matrix = nx.to_numpy_array(G)

df2 = pd.DataFrame(adjacency_matrix)
# Configura las opciones de visualización para mostrar todas las filas y columnas del array NumPy
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# La matriz de adyacencia es una matriz de tipo numpy
# print(df2)

#%%Cuenta el número de aristas en el grafo (elementos no nulos)
num_edges = np.sum(adjacency_matrix)# // 2  # Divide por 2 si el grafo es no dirigido

# Imprime el número de aristas
print(f"El grafo tiene {num_edges} aristas.")

#%% SE CARGAN LAS UBICACIONES DE LOS NODOS PRESENTES EN EL GRAFO
ubicaciones ={
    'LATITUD':[],
    'LONGITUD':[]}

# print(df_ubi.iloc[:, 0])
ind = 0
for i in df.iloc[:, 0]:
    # print(i)
    if i in node_val:       
        a = df.iloc[ind]
        # print(a)
        # print('-------------------------')        
        # ubicaciones.append()
        ubicaciones['LATITUD'].append(a[1])
        ubicaciones['LONGITUD'].append(a[2])
    ind += 1
    
#%% GPU
print(torch.cuda.is_available())
# Asegúrate de que PyTorch pueda usar la GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
print(device)
#%% OBTENER LA MATRIZ DE INDICES DE BORDES EN FORMATO COO
# forma [2,num_edges] tipo torch.long

# print(len(df2))

aux1 = []  # Lista para almacenar las coordenadas i (filas) de los elementos con valor 1
aux2 = []  # Lista para almacenar las coordenadas j (columnas) de los elementos 
            #con valor 1

for i in range(len(df2)):
    for j in range(len(df2)):
        # Verificar si el elemento en la posición (i, j) es igual a 1
        if df2.iloc[i, j] == 1:
            aux1.append(i)  # Almacenar la coordenada i (fila) en la lista aux1
            aux2.append(j)  # Almacenar la coordenada j (columna) en la lista aux2

# Convertir las listas auxiliares a arreglos de numpy
aux1 = np.array(aux1)
aux2 = np.array(aux2)

# Apilar los arreglos verticalmente para formar una matriz de coordenadas (edge_index)
aux3 = np.vstack((aux1, aux2))

# Convertir la matriz de coordenadas a un tensor de PyTorch y especificar el tipo de dato
edge_index = torch.tensor(aux3)
edge_index = edge_index.to(torch.int64)
edge_index = edge_index.to(device)

# print(edge_index)

#%% OBTIENE DISTANCIAS ENTRE NODOS DEL GRAFO FINAL

dist = []

# for ind, i in enumerate(aux3[0]):
for ij in range(len(aux3[0])): 
    distancia = calcular_distancia((ubicaciones['LATITUD'][aux3[0][ij]-1], 
                                    ubicaciones['LONGITUD'][aux3[0][ij]-1]),
                                          (ubicaciones['LATITUD'][aux3[1][ij]-1], 
                                           ubicaciones['LONGITUD'][aux3[1][ij]-1])) 
    dist.append(distancia)    


#%% OBTENER ATRIBUTOS DE BORDE

# print(dist)
# aux1 = np.ones(len(dist)) 
aux1 = np.ones(int(num_edges))
ed_at = torch.tensor(aux1)
# ed_at = torch.tensor(dist)
edge_attr = ed_at.to(torch.float32)
edge_attr = edge_attr.to(device)

print(edge_attr)

#%%SELECCION DE NODOS EN BASE DE DATOS
'''

'''

# ini = 4*24*30*13*2
ini = 0
# inter = 4*24*30*12*6
inter = -1

columnas_seleccionadas = ['fecha_hora']

for i in node_val:
    columnas_seleccionadas.append('NODO '+str(i))

# Lee el archivo CSV y selecciona solo las filas indicadas y las columnas especificadas
n41m_df = pd.read_csv(r'D:\DATOS\MTE\CEDENAR\SUBESTACIONES\MODIFICADOS\datos_comb.csv', 
                      usecols=columnas_seleccionadas).iloc[ini:inter]

n4_pot_act = n41m_df.copy()
# n4_pot_act.index = n41m_df['fecha_hora']

n4_pot_act = n4_pot_act.drop(n4_pot_act.columns[:1], axis=1)

# Suponiendo que df es tu DataFrame
num_filas, num_columnas = n4_pot_act.shape

print("Número de filas:", num_filas)
print("Número de columnas:", num_columnas)


#%% FIGURA

if ima == 1:
    fig1, axs = plt.subplots(5,1,figsize=(12,8), sharex=True)
    # axs[0].plot(n4_pot_act.iloc[:,'NODO92']) 
    axs[0].plot(n4_pot_act.loc[:, 'NODO 6'])
    axs[1].plot(n4_pot_act.loc[:, 'NODO 7'])
    axs[2].plot(n4_pot_act.loc[:, 'NODO 8'])
    axs[3].plot(n4_pot_act.loc[:, 'NODO 9'])
    axs[4].plot(n4_pot_act.loc[:, 'NODO 10'])
    
    axs[0].set_title('41PA02')
    axs[1].set_title('41CA11')
    axs[2].set_title('41JA16')
    axs[3].set_title('41PA05')
    axs[4].set_title('41CA20')
    
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    axs[3].grid(True)
    axs[4].grid(True)
    
    plt.tight_layout()
    
    plt.show()

#%%ANALISIS ESTADISTICO


# Crear una nueva figura
plt.figure()
plt.boxplot(n4_pot_act)
# Agregar un título al gráfico
plt.title('Distribución del conjunto de datos')
plt.ylabel('Potencia [MW]')
# Crear el diagrama de caja y obtener los valores numéricos
boxplot_dict = plt.boxplot(n4_pot_act.values)

# Cerrar la figura
plt.close()

# Acceder a los valores clave
whiskers = [item.get_ydata() for item in boxplot_dict['whiskers']]
medians = [item.get_ydata() for item in boxplot_dict['medians']]
fliers = [item.get_ydata() for item in boxplot_dict['fliers']]

# Imprimir los valores
# print("Valores de los whiskers:")
# print(whiskers)
# print("Valores de las medianas:")
# print(medians)
# print("Valores de los valores atípicos:")
# print(fliers)

#%%#%%NORMALIZACION DE DATOS POR SERIE TEMPORAL

# Crear copias de los valores mínimos y máximos para cada columna
min_vals = n4_pot_act.min()
max_vals = n4_pot_act.max()

# Normalización
n4_pot_act_nor = n4_pot_act.copy()
for col in n4_pot_act.columns:
    n4_pot_act_nor[col] = (n4_pot_act[col] - min_vals[col]) / (max_vals[col] - min_vals[col])

#%% INDICES DE NODOS OBJETIVOS

n1_ind = n4_pot_act_nor.columns.get_loc('NODO 0')
n2_ind = n4_pot_act_nor.columns.get_loc('NODO 1')
n3_ind = n4_pot_act_nor.columns.get_loc('NODO 2')
n4_ind = n4_pot_act_nor.columns.get_loc('NODO 3')
n5_ind = n4_pot_act_nor.columns.get_loc('NODO 4')

col_n1 = n4_pot_act_nor.columns[n1_ind]
col_n2 = n4_pot_act_nor.columns[n2_ind]
col_n3 = n4_pot_act_nor.columns[n3_ind]
col_n4 = n4_pot_act_nor.columns[n4_ind]
col_n5 = n4_pot_act_nor.columns[n5_ind]

print(f"Indice de nodos objetivos: {col_n1}, {col_n2}, {col_n3}, {col_n4}, {col_n5}")
#%% FIGURA

if ima == 1:
    fig1, axs = plt.subplots(5,1,figsize=(12,7), sharex= True)
    # axs[0].plot(n4_pot_act.iloc[:,'NODO92']) 
    axs[0].plot(n4_pot_act_nor.loc[:, col_n1])
    axs[1].plot(n4_pot_act_nor.loc[:, col_n2])
    axs[2].plot(n4_pot_act_nor.loc[:, col_n3])
    axs[3].plot(n4_pot_act_nor.loc[:, col_n4])
    axs[4].plot(n4_pot_act_nor.loc[:, col_n5])
    
    axs[0].set_title(col_n1)
    axs[1].set_title(col_n2)
    axs[2].set_title(col_n3)
    axs[3].set_title(col_n4)
    axs[4].set_title(col_n5)
    
    axs[0].grid(True)
    axs[1].grid(True)
    axs[2].grid(True)
    axs[3].grid(True)
    axs[4].grid(True)
    
    plt.tight_layout()
    
    plt.show()

#%%CARGAR DATOS DE 4 NODOS 1 MES

# Se tiene 2 parametros:
# 1ro es el lag de la serie, esto significa cuantos datos toma la serie para
# hacer la prediccion
# 2do es los pasos de prediccion, cuantos pasos en el futuro se desea predecir,
# parametro importante, puesto que es el valor que se delizara la ventala

n4_pot_act_arr = np.array(n4_pot_act_nor)
input_seq_length = 60*24*15
output_seq_length = 60*24
sliding_steps = 60*12

X_train, Y_train = split_sequences(n4_pot_act_arr,
                                    n4_pot_act_arr,
                                    n_steps_in = input_seq_length,
                                    n_steps_out = output_seq_length,
                                    n_sliding_steps = sliding_steps,
                                    window_type='sliding')


#%% DIMENSIONES ADECUADAS PARA ENTRENAMIENTO Y PRUEBA
X_train_t = np.transpose(X_train, (0, 2, 1))
# print(len(X_train_t))
# print(X_train_t[0])

Y_train_t = np.transpose(Y_train, (0, 2, 1))
# print(len(Y_train_t))
# print(Y_train_t[0])

# print('--------------------------')

# print(len(X_train_t))
# print(X_train_t[1])

# print(len(Y_train_t))
# print(Y_train_t[1])

features = torch.tensor(X_train_t)
features = features.to(torch.float32)

y = torch.tensor(Y_train_t)
y = y.to(torch.float32)
y = torch.squeeze(y)

# print(y[0])

#%%DIVISION DE SETS

# Determina el tamaño de tu conjunto de entrenamiento (75%)
train_size = int(0.70 * len(features))
val_size = int(0.15 * len(features))

# print(len(features),' ',train_size, '  ' , val_size)

# Divide el tensor en conjuntos de entrenamiento y validación
feature_train_set = features[:train_size].to(device)
feature_val_set = features[train_size:train_size+val_size].to(device)
feature_test_set = features[train_size+val_size:].to(device)

# Imprime las formas de los conjuntos
print("Forma del conjunto de entrenamiento:", feature_train_set.shape)
print("Forma del conjunto de validación:", feature_val_set.shape)
print("Forma del conjunto de testeo:", feature_test_set.shape)

# Divide el tensor en conjuntos de entrenamiento y validación
y_train_set = y[:train_size].to(device)
y_val_set = y[train_size:train_size+val_size].to(device)
y_test_set = y[train_size+val_size:].to(device)

#%% DEFINICION RED NN
# Definición de la clase RecurrentGCN que hereda de torch.nn.Module
class RecurrentGCN(torch.nn.Module):

    # #Red neuronal de paso de mensajes con memoria a largo plazo.
    def __init__(self, node_features,cell, num_nodes):
        super(RecurrentGCN, self).__init__()
        self.recurrent = MPNNLSTM(node_features, cell, num_nodes, 1, 0.2)
        self.linear = torch.nn.Linear(2*cell + node_features, output_seq_length)
 
    # def __init__(self, node_features,cell, filters):
    #     super(RecurrentGCN, self).__init__()
    #     # self.recurrent = GCLSTM(node_features, cell, output_seq_length, None)
    #     self.recurrent = GCLSTM(node_features, cell, filters, None)
    #     self.linear = torch.nn.Linear(cell, output_seq_length)    
    
    # Método forward que realiza el pase hacia adelante
    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = self.linear(h)
        return h  
    
    # def forward(self, x, edge_index, edge_weight, h, c):
    #     h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
    #     h = F.relu(h_0)
    #     h = self.linear(h)
    #     return h, h_0, c_0
    
    # def forward(self, x, edge_index, edge_weight):        
    #     print("Entrada x:", x.size())
    #     print("Entrada edge_index:", edge_index.size())
    #     print("Entrada edge_weight:", edge_weight.size())
    #     h = self.recurrent(x, edge_index, edge_weight)
    #     print("Salida de recurrent:", h.size())
    #     h = F.relu(h)
    #     print("Salida de ReLU:", h.size())
    #     h = self.linear(h)
    #     print("Salida de linear:", h.size())
    #     return h

    
#%% --------> ENTRENAMIENTO

mod = 'MPNNLSTM'
# mod = 'GCLSTM'

nodos_N = [1,2,3,4,5]
CELL = [8,16,32,64]

for _, nodos_n in enumerate(nodos_N):
    for _, cell in enumerate(CELL):
    
        # Definir el número máximo de épocas sin mejora permitidas antes de detener el entrenamiento
        max_epochs_without_improvement = 35
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        train_cost = []
        val_cost = []
        num_nodes = len(node_val) 
        
        epocas = 10        
        k = 1
        nodos ='model_'+str(nodos_n)+'N_'
        
        # Crear una instancia del modelo RecurrentGCN con 4 características de nodo
        # model = RecurrentGCN(node_features = input_seq_length)
        model = RecurrentGCN(node_features = input_seq_length, cell=cell, num_nodes = num_nodes)  
        # model = RecurrentGCN(num_node = 5, node_features = 1)
        # model = RecurrentGCN(node_features = input_seq_length, cell=cell)
        # model = RecurrentGCN(node_features = input_seq_length, cell=cell,filters = k)
        # model = RecurrentGCN(node_features = input_seq_length, window = input_seq_length)
        model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        opt = '_Adam'
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005, alpha=0.9)
        # opt = '_RMSprop'
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
        # opt = '_SGD'
        
        out_txt = nodos+str(mod)+'_Out'+str(output_seq_length)+'_In'+str(input_seq_length)+opt+'_Cell'+str(cell)+'_K'+str(k)+'_E'+str(epocas)+'_R'+str(rad_prox)+'_SLI'+str(sliding_steps)
        print(out_txt)
        print('-------------------------------')
        # print(model)
        
        #%% TRAIN 1
        
        # Guarda el tiempo de inicio
        inicio = time.time() 
        # Inicializar un optimizador Adam para optimizar los parámetros del modelo con 
        # una tasa de aprendizaje de 0.01
        
        # Configurar el modelo en modo de entrenamiento
        model.train()
        
        # Bucle de entrenamiento a lo largo de 200 épocas
        for epoch in tqdm(range(epocas)):
            # Inicializar la variable de costo para la época actual
            cost = 0
        
            # Iterar sobre los instantes de tiempo y capturas en el conjunto de datos 
            # de entrenamiento
            # for time in enumerate(feature_train_set):
            h, c = None, None
            for times in range(len(feature_train_set)):
                # Obtener las predicciones del modelo para las características de nodo, 
                # índices de arista y atributos de arista en el instante de tiempo actual
                # print(feature_train_set[time])
                
                # y_hat = model(feature_train_set[time].to(device), edge_index.to(device), edge_attr.to(device))
                y_hat = model(feature_train_set[times], edge_index, edge_attr)
                # y_hat, h, c = model(feature_train_set[time], edge_index, edge_attr, h, c)        
                        
                # Seleccionar las predicciones y las etiquetas de entrenamiento correspondientes
               
                if nodos_n == 4:        
                    y_hat_p = y_hat[[n1_ind, n2_ind, n3_ind, n4_ind]]
                    y_train_set_p = y_train_set[times, [n1_ind, n2_ind, n3_ind, n4_ind]] 
                
                elif nodos_n == 5:        
                    y_hat_p = y_hat[[n1_ind, n2_ind, n3_ind, n4_ind, n5_ind]]
                    y_train_set_p = y_train_set[times, [n1_ind, n2_ind, n3_ind, n4_ind, n5_ind]] 
                
                elif nodos_n == 3:        
                    y_hat_p = y_hat[[n1_ind, n2_ind, n3_ind]]
                    y_train_set_p = y_train_set[times, [n1_ind, n2_ind, n3_ind]] 
                
                elif nodos_n == 2:
                    y_hat_p = y_hat[[n2_ind, n3_ind]]  # Predicciones para nodos específicos
                    y_train_set_p = y_train_set[times, [n2_ind, n3_ind]]  # Etiquetas reales
                            
                else: 
                    y_hat_p = y_hat[[n3_ind]]  # Predicciones para nodos específicos
                    y_train_set_p = y_train_set[times, [n3_ind]]  # Etiquetas reales
            
                loss = F.mse_loss(y_hat_p, y_train_set_p)
                cost += loss  
        
            # Calcular el costo promedio para el conjunto de datos de entrenamiento en la época actual
            cost = cost / (times + 1)
            # print(" - MSE: {:.4f}".format(cost))
            train_cost.append(cost.item())
            
            # Validación en un conjunto de validación
            h, c = None, None
            model.eval()
            with torch.no_grad():
                val_loss = 0
                
                for val_time in range(len(feature_val_set)):
                    val_y_hat = model(feature_val_set[val_time], edge_index, edge_attr)
                    # val_y_hat, h, c = model(feature_val_set[val_time], edge_index, edge_attr, h, c)            
                    # val_loss += F.mse_loss(val_y_hat, y_val_set[val_time].unsqueeze(1))            
                    
                    # Seleccionar las predicciones y las etiquetas de validación correspondientes           
                    if nodos_n == 4:        
                        val_y_hat_p = val_y_hat[[n1_ind, n2_ind, n3_ind, n4_ind]]
                        y_val_set_p = y_val_set[val_time, [n1_ind, n2_ind, n3_ind, n4_ind]] 
                        
                    elif nodos_n == 5:        
                        val_y_hat_p = val_y_hat[[n1_ind, n2_ind, n3_ind, n4_ind, n5_ind]]
                        y_val_set_p = y_val_set[val_time, [n1_ind, n2_ind, n3_ind, n4_ind, n5_ind]] 
                    
                    elif nodos_n == 3:        
                        val_y_hat_p = val_y_hat[[n1_ind, n2_ind, n3_ind]]
                        y_val_set_p = y_val_set[val_time, [n1_ind, n2_ind, n3_ind]] 
                    
                    elif nodos_n == 2:
                        val_y_hat_p = val_y_hat[[n2_ind, n3_ind]]  # Predicciones para nodos específicos
                        y_val_set_p = y_val_set[val_time, [n2_ind, n3_ind]]  # Etiquetas reales
                    
                    else: 
                        val_y_hat_p = val_y_hat[[n3_ind]]  # Predicciones para nodos específicos
                        y_val_set_p = y_val_set[val_time, [n3_ind]]  # Etiquetas reales
                    
                    # val_loss += F.mse_loss(val_y_hat, y_val_set[val_time])
                    val_loss += F.mse_loss(val_y_hat_p, y_val_set_p)
                    
                val_loss = val_loss / (len(feature_val_set) + 1)
                print(" - MSE: {:.4f}".format(cost)," - Val MSE: {:.5f}".format(val_loss.item()))
                
                # Verificar si la pérdida en el conjunto de validación ha disminuido
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), r'D:\DATOS\MTE\GRAFOS\MODELOS\CIRCUITOS\\'+out_txt+'.pth')
                    print('-------------> BEST MODEL')
                else:
                    epochs_without_improvement += 1
            val_cost.append(val_loss)
            # Volver al modo de entrenamiento
            model.train()
            
            # Detener el entrenamiento si no hay mejora en un número específico de épocas
            if epochs_without_improvement >= max_epochs_without_improvement:
                print("Deteniendo el entrenamiento debido a falta de mejora en el conjunto de validación.")
                break
            
            # Realizar la retropropagación y la optimización de los parámetros del modelo
            cost.backward()
            optimizer.step()
            
            # Reiniciar los gradientes acumulados en el optimizador
            optimizer.zero_grad()
            
        # Guarda el tiempo de finalización
        fin = time.time()
        
        # Calcula el tiempo transcurrido en segundos
        tiempo_transcurrido = fin - inicio
        print("El bloque de código tomó {} segundos en ejecutarse.".format(tiempo_transcurrido))
        print('--------------------------------------------------')
        #%% GRAFICA DE MEAN SQUARED ERROR
        
        # Concatenar los tensores a lo largo de una nueva dimensión (dim=0)
        # val_cost_aux = torch.cat(val_cost, dim=0)
        # val_cost_array = val_cost_aux.numpy()
        
        plt.figure()
        # plt.plot(val_cost, label='Error cuadratico medio','b*')
        plt.plot(train_cost, 'b-o', label='Train MSE')
        plt.plot(val_cost, 'r-o', label='Val MSE')
        
        
        # Agregar etiquetas y leyenda
        plt.xlabel('Epoca')
        plt.ylabel('MSE')
        # plt.title('Entrenamiento'+' MSE = '+str(val_cost[-1]))
        plt.title('Entrenamiento Val-MSE = {:.5f} Test-MSE = {:.5f}'.format(val_cost[-1], train_cost[-1]))
        
        plt.legend()
        plt.grid()
        
        # Guardar la figura en un archivo (puedes especificar el formato, como PNG, PDF, SVG, etc.)
        plt.savefig(r'D:\DATOS\MTE\GRAFOS\MODELOS\CIRCUITOS\FIGURAS\Train_'+out_txt+'.png')
        
        # Mostrar la gráfica
        plt.show()
        
        #%%CARGAR MODELO
        # Crear una nueva instancia del modelo
        # model = RecurrentGCN(node_features=input_seq_length,cell=cell)
        # model = RecurrentGCN(node_features=input_seq_length,cell=cell,filters=k)
        model = RecurrentGCN(node_features = input_seq_length, cell=cell, num_nodes = num_nodes) 
        
        path = r'D:\DATOS\MTE\GRAFOS\MODELOS\CIRCUITOS\\'+out_txt+'.pth'
        print(path)
        # # Cargar los parámetros guardados en el modelo
        model.load_state_dict(torch.load(path))
        print('-----------> MODELO CARGADO')
        
        #%%
        # Establecer el modelo en modo de evaluación
        model.eval()
        
        # Inicializar la variable de costo
        cost = 0
        y_out =[]
        h, c = None, None
        # print(test_dataset.features[0][0])
        
        # Iterar sobre los instantes de tiempo y snapshots en el conjunto de prueba
        for times in range(len(feature_test_set)):
            # Realizar la predicción utilizando el modelo    
            y_hat = model(feature_test_set[times], edge_index, edge_attr)
            
            y_hat_cut = y_hat[:, 0:sliding_steps]
            
            # Almacenar las predicciones en la lista y_out como un array numpy
            y_out.append(y_hat_cut.cpu().detach().numpy())
            
            # Calcular el costo acumulado sumando el error cuadrático medio (MSE)
            # Se utiliza F.mse_loss para calcular la pérdida entre las predicciones y las etiquetas reales
            loss = F.mse_loss(y_hat, y_test_set[times])
            cost += loss  
            
        
        # Calcular el costo promedio dividiendo por el número de instantes de tiempo
        cost = cost / (times + 1)
        
        # Obtener el valor numérico del costo
        cost = cost.item()
        
        # Imprimir el MSE (Mean Squared Error)
        print("MSE: {:.4f}".format(cost))
        
        #%% CREA LISTAS VACIAS PARA ALMACENAR RESULTADOS VERDADEROS
        
        y_out_test = y_test_set.numpy()
        y_out_test_cut = y_out_test[:, :,0:sliding_steps] 
        
        vacio = []
        
        node_0_test = np.array(vacio)
        node_1_test = np.array(vacio)
        node_2_test = np.array(vacio)
        node_3_test = np.array(vacio)
        node_4_test = np.array(vacio)
        
        for i, data in enumerate(y_out_test_cut):
            # print(data)
            # print(i)
            node_0_test = np.append(node_0_test,data[n1_ind])
            node_1_test = np.append(node_1_test,data[n2_ind])
            node_2_test = np.append(node_2_test,data[n3_ind])
            node_3_test = np.append(node_3_test,data[n4_ind])
            node_4_test = np.append(node_4_test,data[n5_ind])
        
        #%CREA LISTAS VACIAS PARA ALMACENAR RESULTADOS PREDICCION -----------------------------------------------------------------------------
        y_out_pre = np.array(y_out)
        
        node_0 = np.array(vacio)
        node_1 = np.array(vacio)
        node_2 = np.array(vacio)
        node_3 = np.array(vacio)
        node_4 = np.array(vacio)
        
        
        for i, data in enumerate(y_out_pre):
            # print(data)
            node_0 = np.append(node_0,data[n1_ind])
            node_1 = np.append(node_1,data[n2_ind])
            node_2 = np.append(node_2,data[n3_ind])
            node_3 = np.append(node_3,data[n4_ind])
            node_4 = np.append(node_4,data[n5_ind])
        
        #%% METRICAS DE ERROR
        
        # Crear listas para almacenar las métricas
        mse_list = []
        mae_list = []
        r2_list = []
        
        # Calcular métricas para cada par de variables
        for i in range(4):  # Suponiendo que tienes 6 nodos
            y_real = globals()[f'node_{i}']
            y_pred = globals()[f'node_{i}_test']
        
            # Calcular el Error Cuadrático Medio (MSE) 
            mse = mean_squared_error(y_real, y_pred)
            mse_list.append(mse)
        
            # Calcular el Error Absoluto Medio (MAE)
            mae = mean_absolute_error(y_real, y_pred)
            mae_list.append(mae)
        
            # Calcular el Coeficiente de Determinación (R²)
            r2 = r2_score(y_real, y_pred)
            r2_list.append(r2)
        
            # Imprimir las métricas para cada nodo
            print(f'\nMétricas para node_{i}:')
            print(f'MSE: {mse}')
            print(f'MAE: {mae}')
            print(f'R²: {r2}')
        
        # Imprimir métricas promedio
        print('\nMétricas promedio:')
        print(f'MSE promedio: {np.mean(mse_list)}')
        print(f'MAE promedio: {np.mean(mae_list)}')
        print(f'R² promedio: {np.mean(r2_list)}')
        
        #%%
        # Número de datos
        n = len(node_0_test)  # Suponiendo que 'node_0_test' es tu array con n datos
        
        # Fecha final
        fecha_final = pd.to_datetime('2024/08/27 00:49')
        
        # Crear un rango de fechas en el eje X con un muestreo de 1 minuto
        frecuencia = 'T'  # 'T' es para minutos
        
        # Restar el tiempo total (n-1) minutos desde la fecha final
        fecha_inicial = fecha_final - pd.to_timedelta(f'{(n - 1)}T')  # Ajustar para 1 minuto entre datos
        
        # Crear un vector de fechas desde la fecha inicial a la final
        fechas = pd.date_range(start=fecha_inicial, end=fecha_final, periods=n)
        #%%GRAFICA COMPARACION
        fig, axs= plt.subplots(4,1,figsize=(12,8),sharex=True)
        
        # Título global para los subgráficos
        plt.suptitle('RESULTADOS MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(cost,np.mean(mae_list),np.mean(r2_list)))
        
        axs[0].plot(fechas,node_0_test)
        axs[0].plot(fechas,node_0,'--')
        axs[1].plot(fechas,node_1_test)
        axs[1].plot(fechas,node_1,'--')
        axs[2].plot(fechas,node_2_test)
        axs[2].plot(fechas,node_2,'--')
        axs[3].plot(fechas,node_3_test)
        axs[3].plot(fechas,node_3,'--')
        
        axs[0].set_title('NODE 0 MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[0],mae_list[0],r2_list[0]))
        axs[1].set_title('NODE 1 MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[1],mae_list[1],r2_list[1]))
        axs[2].set_title('NODE 2 MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[2],mae_list[2],r2_list[2]))
        axs[3].set_title('NODE 3 MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[3],mae_list[3],r2_list[3]))
        
        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)
        axs[3].grid(True)
        
        plt.tight_layout()
        
        # Guardar la figura en un archivo (puedes especificar el formato, como PNG, PDF, SVG, etc.)
        plt.savefig(r'D:\DATOS\MTE\GRAFOS\MODELOS\CIRCUITOS\FIGURAS\Res_'+out_txt+'.png')
        
        plt.show()
        
        #%% GUARDA METRICAS
                    
        # print(out_txt)
            
        datos_modelo['NAME MODEL'].append(out_txt)
        datos_modelo["NODE"].append(nodos_n)
        datos_modelo["K"].append(k)
        datos_modelo["CELL"].append(cell)
        datos_modelo["SLI"].append(sliding_steps)
        datos_modelo["MSE"].append(np.mean(mse_list))
        datos_modelo["MAE"].append(np.mean(mae_list))
        datos_modelo["R2"].append(np.mean(r2_list))
        datos_modelo["TIME"].append(tiempo_transcurrido)
        
        # Nombre del archivo CSV
        # input_seq_length = 4*24*30
        # output_seq_length = 4*24
        nombre_archivo = r"D:\DATOS\MTE\GRAFOS\MODELOS\CIRCUITOS\METRICAS\DATA_"+mod+"_OUT"+str(output_seq_length)+"_IN"+str(input_seq_length)+".csv"
        # nombre_archivo = r"C:\Users\GIIEE\Desktop\MTE_GRAPHS\MODELOS\UNIVERSIDADES\RAD_2\METRICAS\TEST_DATA_"+mod+"_OUT"+str(output_seq_length)+"_IN"+str(input_seq_length)+".csv"
        
        # Escribir los datos en el archivo CSV
        with open(nombre_archivo, mode='w', newline='') as archivo_csv:
            escritor_csv = csv.DictWriter(archivo_csv, fieldnames=datos_modelo.keys())
        
            # Escribir los encabezados
            escritor_csv.writeheader()
        
            # Escribir los datos
            for i in range(len(datos_modelo["NAME MODEL"])):
                fila = {key: datos_modelo[key][i] for key in datos_modelo.keys()}
                escritor_csv.writerow(fila)
        
        # print("El archivo CSV ha sido creado con éxito.")
