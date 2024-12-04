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
from torch_geometric_temporal.nn.recurrent import GCLSTM, EvolveGCNO

# from labml_nn.graphs.gat import GraphAttentionLayer

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
df = pd.read_csv(r'C:\Users\GIIEE\Desktop\MTE_GRAPHS\DATOS\ubicaciones.csv')

#%%

# Crear el diccionario
datos_modelo = {
    "NAME MODEL": [],
    "RAD": [],
    "K": [],
    "CELL": [],
    "MSE": [],
    "MAE": [],
    "R2": [],
    "TIME": []
}

#%%
ima = 0

# rad_prox = 0.25 #<------------------------------------
# rad_prox = [0, 0.25, 0.5, 0.75]
rad_prox = [0.25, 0.5, 0.75]
rad_prox = [0.5, 0.75]
rad_prox = [0.25, 0.5]
rad_prox = [0.75]

for ind, rad_prox in enumerate(rad_prox):
    print(rad_prox)
    G = nx.Graph()
    for i, ubi in df.iterrows():
      #print(i)
    #  print(ubi)
      #print(ubi[0])
      #print(ubi[1])
      G.add_node("Nodo"+str(i), latitud = -ubi[1], longitud = -ubi[2])
    
    # Establecer el color de los nodos especiales
    # for i in [92, 93, 95, 96]:
    for i in [69, 70, 71, 72]:
      G.nodes["Nodo"+str(i)]["color"] = "red"
      
    for node in G.nodes:
        if "color" not in G.nodes[node]:
            G.nodes[node]["color"] = "skyblue"  # Color por defecto
    
    #%%
    # a = df.iloc[0]
    # b = df.iloc[2]
    
    # distancia = calcular_distancia(a, b)
    # print(f"La distancia entre los puntos es {distancia} kilómetros.")  
    
    
    
    #%%
        
    for nodo1 in G.nodes:
        # print(nodo1)
        for nodo2 in G.nodes:
            if nodo1 != nodo2:
                distancia = calcular_distancia((G.nodes[nodo1]["latitud"], G.nodes[nodo1]["longitud"]),
                                             (G.nodes[nodo2]["latitud"], G.nodes[nodo2]["longitud"]))
                if distancia <= rad_prox:
                    G.add_edge(nodo1, nodo2, distancia=distancia)
    
    for nodo1 in ['Nodo69', 'Nodo70', 'Nodo71', 'Nodo72']:
        for nodo2 in ['Nodo69', 'Nodo70', 'Nodo71', 'Nodo72']:
            if nodo1 != nodo2:
                distancia = calcular_distancia((G.nodes[nodo1]["latitud"], G.nodes[nodo1]["longitud"]),
                                             (G.nodes[nodo2]["latitud"], G.nodes[nodo2]["longitud"]))            
                G.add_edge(nodo1, nodo2, distancia=distancia)
    
                    
    #%%
    # Definir un tamaño personalizado para la figura
    if ima == 1:
        fig, ax = plt.subplots(figsize=(12, 6))  # Ajusta las dimensiones según tus necesidades
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
    
    #%%
    # Cuenta el número de aristas en el grafo (elementos no nulos)
    num_edges = np.sum(adjacency_matrix) // 2  # Divide por 2 si el grafo es no dirigido
    
    # Imprime el número de aristas
    # print(f"El grafo tiene {num_edges} aristas.")
    
    #%%
    node_val = []
    # Obtenemos los nodos conectados al nodo 93
    for i in [69, 70, 71, 72]:
        # print('------------------------------------')
        adjacent_nodes = get_adjacent_nodes(df2, i)
        # print('NODO : ', i)
        # print(adjacent_nodes)
        node_val = np.append(node_val,adjacent_nodes)
        
    #%% ELIMINA NODOS SIN INFORMACION
    if rad_prox == 0.25:
        val_del = [] #R=0.25
    
    if rad_prox == 0.5:
        val_del = [] #R=0.5
    
    if rad_prox == 0.75:
        val_del = [6, 9, 45, 61] #R=0.75
    
    # Verificar si alguno de los elementos de val_del está presente en node_val
    if any(val in node_val for val in val_del):
        # Eliminar los elementos de val_del de node_val si están presentes
        for val in val_del:
            node_val = np.delete(node_val, np.where(node_val == val))
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
    # Definir un tamaño personalizado para la figura
    if ima == 1:
        fig, ax = plt.subplots(figsize=(12, 6))  # Ajusta las dimensiones según tus necesidades
        pos = {node: (G.nodes[node]["latitud"], G.nodes[node]["longitud"]) for node in G.nodes}
        node_colors = [G.nodes[node]["color"] for node in G.nodes]
        nx.draw(G, pos, with_labels=True, node_size=100, font_size=8, node_color=node_colors)
        plt.show()   
    
    #%%Obtiene la matriz de adyacencia en forma de numpy array
    adjacency_matrix = nx.to_numpy_array(G)
    
    df2 = pd.DataFrame(adjacency_matrix)
    # Configura las opciones de visualización para mostrar todas las filas y columnas del array NumPy
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    # La matriz de adyacencia es una matriz de tipo numpy
    # print(df2)
    
    #%% MATRIZ DISPERSA
    # Graficar la matriz dispersa
    plt.figure(figsize=(7, 7))
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
    
    #%%Cuenta el número de aristas en el grafo (elementos no nulos)
    num_edges = np.sum(adjacency_matrix)# // 2  # Divide por 2 si el grafo es no dirigido
    
    # Imprime el número de aristas
    # print(f"El grafo tiene {num_edges} aristas.")
    
    #%%
    ubicaciones ={
        'LATITUD':[],
        'LONGITUD':[]}
    
    df_ubi = pd.read_csv(r'C:\Users\GIIEE\Desktop\MTE_GRAPHS\DATOS\UBICACIONES_MAPA-COM.csv')
    # print(df_ubi.iloc[:, 0])
    ind = 0
    for i in df_ubi.iloc[:, 0]:
        # print(i)
        if i in node_val:       
            a = df_ubi.iloc[ind]
            # print(a)
            # print('-------------------------')
            # print('entro: ', i)
            # ubicaciones.append()
            ubicaciones['LATITUD'].append(a[1])
            ubicaciones['LONGITUD'].append(a[2])
        ind += 1
        
    #%% GPU
    # print(torch.cuda.is_available())
    # Asegúrate de que PyTorch pueda usar la GPU
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # print(ubicaciones)
    
    # print(ubicaciones['LATITUD'][0])
    # print(ubicaciones['LONGITUD'][0])
    
    # print(aux3)
    
    for ij in range(len(aux3[0])):
        # print(ij)
        # print('--------------------------------------')
        # print(aux3[0][ij],aux3[1][ij])
        # print(ubicaciones['LATITUD'][aux3[0][ij]],ubicaciones['LONGITUD'][aux3[0][ij]])    
        # print(ubicaciones['LATITUD'][aux3[1][ij]],ubicaciones['LONGITUD'][aux3[1][ij]])   
        distancia = calcular_distancia((ubicaciones['LATITUD'][aux3[0][ij]], ubicaciones['LONGITUD'][aux3[0][ij]]),
                                             (ubicaciones['LATITUD'][aux3[1][ij]], ubicaciones['LONGITUD'][aux3[1][ij]])) 
        dist.append(distancia)
        # print(distancia)
    
    #%% OBTENER ATRIBUTOS DE BORDE
    
    # print(dist)
    aux1 = np.ones(len(dist)) 
    # aux1 = np.ones(int(num_edges))
    #ed_at = torch.tensor(aux1) #<----- BORDES CON PESO UNITARIO
    ed_at = torch.tensor(aux1/dist) #<------ BORDES CON PESO
    edge_attr = ed_at.to(torch.float32)
    edge_attr = edge_attr.to(device)
    
    # print('ATRIBUTOS DE BORDE')
    # print(edge_attr)
    
    #%%SELECCION DE NODOS EN BASE DE DATOS
    '''
    POLLO LA 17	NODO 69
    POLLO LA COLINA	NODO 70
    POLLO NORTE	NODO 71
    POLLO SUR	NODO 72
    '''
    # ini = 4*24*30*5
    # inter = 4*24*30*1
    # inter = 4*24*30*7
    
    ini = 4*24*30*13*2
    # inter = 4*24*30*12*6
    inter = -1
    
    
    columnas_seleccionadas = ['fecha_hora']
    
    for i in node_val:
        columnas_seleccionadas.append('NODO'+str(i))
    
    # Lee el archivo CSV y selecciona solo las filas indicadas y las columnas especificadas
    n41m_df = pd.read_csv(r'C:\Users\GIIEE\Desktop\MTE_GRAPHS\DATOS\NODOS_MAPA_COM.csv', usecols=columnas_seleccionadas).iloc[ini:inter]
    
    n4_pot_act = n41m_df.copy()
    # n4_pot_act.index = n41m_df['fecha_hora']
    
    n4_pot_act = n4_pot_act.drop(n4_pot_act.columns[:1], axis=1)
    
    # Suponiendo que df es tu DataFrame
    num_filas, num_columnas = n4_pot_act.shape
    
    print("Número de filas:", num_filas)
    print("Número de columnas:", num_columnas)
    
    # for i in range(num_columnas):
    #     print(i)
    #     aux = f'med_{i}'
    #     n4_pot_act[aux] = n4_pot_act.iloc[:,0]
    
    #%% FIGURA
    # fig1, axs = plt.subplots(4,1,figsize=(12,8))
    # # axs[0].plot(n4_pot_act.iloc[:,'NODO92']) 
    # axs[0].plot(n4_pot_act.loc[:, 'NODO92'])
    # axs[1].plot(n4_pot_act.loc[:, 'NODO93'])
    # axs[2].plot(n4_pot_act.loc[:, 'NODO95'])
    # axs[3].plot(n4_pot_act.loc[:, 'NODO96'])
    
    # axs[0].set_title('U COOPERTATIVA')
    # axs[1].set_title('U MARIANA')
    # axs[2].set_title('UDENAR')
    # axs[3].set_title('VIPRI')
    
    # axs[0].grid(True)
    # axs[1].grid(True)
    # axs[2].grid(True)
    # axs[3].grid(True)
    
    # plt.tight_layout()
    
    # plt.show()
    
    #%% FIGURA
    
    if ima == 1:
        fig1, axs = plt.subplots(5,1,figsize=(12,8))
        # axs[0].plot(n4_pot_act.iloc[:,'NODO92']) 
        axs[0].plot(n4_pot_act.loc[:, 'NODO69'])
        axs[1].plot(n4_pot_act.loc[:, 'NODO70'])
        axs[2].plot(n4_pot_act.loc[:, 'NODO71'])
        axs[3].plot(n4_pot_act.loc[:, 'NODO72'])
        axs[4].plot(n4_pot_act.loc[:, 'NODO4'])
        
        axs[0].set_title('MR POLLO LA 17')
        axs[1].set_title('MR POLLO LA COLINA')
        axs[2].set_title('MR POLLO NORTE')
        axs[3].set_title('MR POLLO SUR')
        axs[4].set_title('OTRO')        
        
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
    plt.ylabel('Energía [kWh]')
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
    
    #%%
    
    # print(whiskers[7][1])
    # print(whiskers[-1][1])
    
    # print(whiskers[8][1])
    # print(whiskers[-2][1])
    
    
    
    #%% OBTIENEN VALORES SUPERIORES A ELIMINAR
        
    # UNIV. COOPERATIVA	NODO 92
    # UNIVERSIDAD MARIANA	NODO 93
    # UNV. DE NARIÑO - TOROBAJO	NODO 95
    # UNV. DE NARIÑO - VIPRI	NODO 96    
    
    # if rad_prox == 0:
    #     whiskers[-7][1] = 29#n4_pot_act.iloc[:, 0].max()
    #     whiskers[-5][1] = 29#n4_pot_act.iloc[:, 1].max()
    #     whiskers[-3][1] = 57.7#n4_pot_act.iloc[:, 2].max()
    #     whiskers[-1][1] = 10#n4_pot_act.iloc[:, 3].max()
        
    #     whiskers[-8][1] = 2.5
    #     whiskers[-6][1] = 8
    #     whiskers[-4][1] = 15
    #     whiskers[-2][1] = 1.5
        
    #     n0_mm = [whiskers[-7][1],whiskers[-8][1]]
    #     n1_mm = [whiskers[-5][1],whiskers[-6][1]]
    #     n2_mm = [whiskers[-3][1],whiskers[-4][1]]
    #     n3_mm = [whiskers[-1][1],whiskers[-2][1]]
    
    # else:
    
    #     whiskers[-9][1] = 29#n4_pot_act.iloc[:, 0].max()
    #     whiskers[-7][1] = 29#n4_pot_act.iloc[:, 1].max()
    #     whiskers[-3][1] = 57.7#n4_pot_act.iloc[:, 2].max()
    #     whiskers[-1][1] = 10#n4_pot_act.iloc[:, 3].max()
        
    #     whiskers[-10][1] = 2.5
    #     whiskers[-8][1] = 8
    #     whiskers[-4][1] = 15
    #     whiskers[-2][1] = 1.5
        
    #     n0_mm = [whiskers[-9][1],whiskers[-10][1]]
    #     n1_mm = [whiskers[-7][1],whiskers[-8][1]]
    #     n2_mm = [whiskers[-3][1],whiskers[-4][1]]
    #     n3_mm = [whiskers[-1][1],whiskers[-2][1]]
        
    
    # print('MIN NODO 0 ',whiskers[-10][1])
    # print('MAX NODO 0 ',whiskers[-9][1])
    # print('MIN NODO 1 ',whiskers[-8][1])
    # print('MAX NODO 1 ',whiskers[-7][1])
    # print('MIN NODO 2 ',whiskers[-4][1])
    # print('MAX NODO 2 ',whiskers[-3][1])
    # print('MIN NODO 3 ',whiskers[-2][1])
    # print('MAX NODO 3 ',whiskers[-1][1])
    
    #%% CORTE DE VALORES ATIPICOS
    
    # if rad_prox == 0:
    #     n4_pot_act[n4_pot_act.columns[-4]] = n4_pot_act[n4_pot_act.columns[-4]].clip(lower=n0_mm[1],upper=n0_mm[0])
    #     n4_pot_act[n4_pot_act.columns[-3]] = n4_pot_act[n4_pot_act.columns[-3]].clip(lower=n1_mm[1],upper=n1_mm[0])
    #     n4_pot_act[n4_pot_act.columns[-2]] = n4_pot_act[n4_pot_act.columns[-2]].clip(lower=n2_mm[1],upper=n2_mm[0])
    #     n4_pot_act[n4_pot_act.columns[-1]] = n4_pot_act[n4_pot_act.columns[-1]].clip(lower=n3_mm[1],upper=n3_mm[0])
    
    # else: 
    #     n4_pot_act[n4_pot_act.columns[-5]] = n4_pot_act[n4_pot_act.columns[-5]].clip(lower=n0_mm[1],upper=n0_mm[0])
    #     n4_pot_act[n4_pot_act.columns[-4]] = n4_pot_act[n4_pot_act.columns[-4]].clip(lower=n1_mm[1],upper=n1_mm[0])
    #     n4_pot_act[n4_pot_act.columns[-2]] = n4_pot_act[n4_pot_act.columns[-2]].clip(lower=n2_mm[1],upper=n2_mm[0])
    #     n4_pot_act[n4_pot_act.columns[-1]] = n4_pot_act[n4_pot_act.columns[-1]].clip(lower=n3_mm[1],upper=n3_mm[0])
    
    #R=0.25 [ 4  7 17 20 28 32 34 38 52 53 54 56 57 64 70 81 83 89 90 91 97]
    if rad_prox == 0.25:        
        n4_pot_act['NODO4'] = n4_pot_act['NODO4'].clip(lower=0, upper=2.3)       
        n4_pot_act['NODO11'] = n4_pot_act['NODO11'].clip(lower=0, upper=4.8)    #
        n4_pot_act['NODO17'] = n4_pot_act['NODO17'].clip(lower=0, upper=4)        
        n4_pot_act['NODO32'] = n4_pot_act['NODO32'].clip(lower=0, upper=3.4)       
        n4_pot_act['NODO60'] = n4_pot_act['NODO60'].clip(lower=0, upper=0.4)    #       
        n4_pot_act['NODO69'] = n4_pot_act['NODO69'].clip(lower=1, upper=6)
        n4_pot_act['NODO70'] = n4_pot_act['NODO70'].clip(lower=1, upper=6)
        n4_pot_act['NODO71'] = n4_pot_act['NODO71'].clip(lower=4, upper=16)    #
        n4_pot_act['NODO72'] = n4_pot_act['NODO72'].clip(lower=1, upper=7)    #27       
            
    if rad_prox == 0.5:        
        n4_pot_act['NODO4'] = n4_pot_act['NODO4'].clip(lower=0, upper=2.3)
        n4_pot_act['NODO11'] = n4_pot_act['NODO11'].clip(lower=0, upper=4.8)    #
        n4_pot_act['NODO12'] = n4_pot_act['NODO12'].clip(lower=0, upper=6)    #
        n4_pot_act['NODO16'] = n4_pot_act['NODO16'].clip(lower=0, upper=1.4)    #
        n4_pot_act['NODO17'] = n4_pot_act['NODO17'].clip(lower=0, upper=4)
        # n4_pot_act['NODO21'] = n4_pot_act['NODO21'].clip(lower=, upper=)    #
        n4_pot_act['NODO32'] = n4_pot_act['NODO32'].clip(lower=0, upper=3.4)
        # n4_pot_act['NODO40'] = n4_pot_act['NODO40'].clip(lower=, upper=)    #16  
        n4_pot_act['NODO49'] = n4_pot_act['NODO49'].clip(lower=0, upper=3.1)    #19  
        n4_pot_act['NODO50'] = n4_pot_act['NODO50'].clip(lower=0, upper=1.9)    #    
        n4_pot_act['NODO60'] = n4_pot_act['NODO60'].clip(lower=0, upper=0.4)    #
        n4_pot_act['NODO69'] = n4_pot_act['NODO69'].clip(lower=1, upper=6)
        n4_pot_act['NODO70'] = n4_pot_act['NODO70'].clip(lower=1, upper=6)
        n4_pot_act['NODO71'] = n4_pot_act['NODO71'].clip(lower=4, upper=16)    #
        n4_pot_act['NODO72'] = n4_pot_act['NODO72'].clip(lower=1, upper=7)    #27       
        # n4_pot_act['NODO76'] = n4_pot_act['NODO76'].clip(lower=, upper=) #28
        # n4_pot_act['NODO86'] = n4_pot_act['NODO86'].clip(lower=, upper=)
        n4_pot_act['NODO88'] = n4_pot_act['NODO88'].clip(lower=0, upper=2.3)
        # n4_pot_act['NODO92'] = n4_pot_act['NODO92'].clip(lower=, upper=)    #
     
        
    if rad_prox == 0.75:
        
        n4_pot_act['NODO3'] = n4_pot_act['NODO3'].clip(lower=0.6, upper=3)
        n4_pot_act['NODO4'] = n4_pot_act['NODO4'].clip(lower=0, upper=2.3)
        # n4_pot_act['NODO5'] = n4_pot_act['NODO5'].clip(lower=, upper=)   #
        # n4_pot_act['NODO7'] = n4_pot_act['NODO7'].clip(lower=, upper=)
        n4_pot_act['NODO10'] = n4_pot_act['NODO10'].clip(lower=0, upper=6.4)    #
        n4_pot_act['NODO11'] = n4_pot_act['NODO11'].clip(lower=0, upper=4.8)    #
        n4_pot_act['NODO12'] = n4_pot_act['NODO12'].clip(lower=0, upper=6)    #
        # n4_pot_act['NODO13'] = n4_pot_act['NODO13'].clip(lower=2.1, upper=11.3)    #
        n4_pot_act['NODO16'] = n4_pot_act['NODO16'].clip(lower=0, upper=1.4)    #
        n4_pot_act['NODO17'] = n4_pot_act['NODO17'].clip(lower=0, upper=4)
        n4_pot_act['NODO18'] = n4_pot_act['NODO18'].clip(lower=0, upper=12.7)    #
        # n4_pot_act['NODO20'] = n4_pot_act['NODO20'].clip(lower=0, upper=8.2)
        # n4_pot_act['NODO21'] = n4_pot_act['NODO21'].clip(lower=, upper=)    #
        # n4_pot_act['NODO22'] = n4_pot_act['NODO22'].clip(lower=1.4, upper=4)    #
        # n4_pot_act['NODO23'] = n4_pot_act['NODO23'].clip(lower=0, upper=12.1)    #
        # n4_pot_act['NODO27'] = n4_pot_act['NODO27'].clip(lower=0, upper=)       #
        # n4_pot_act['NODO28'] = n4_pot_act['NODO28'].clip(lower=0, upper=3.25)
        n4_pot_act['NODO30'] = n4_pot_act['NODO30'].clip(lower=0, upper=6)    #
        n4_pot_act['NODO32'] = n4_pot_act['NODO32'].clip(lower=0, upper=3.4)
        n4_pot_act['NODO33'] = n4_pot_act['NODO33'].clip(lower=0, upper=12.7)    #
        # n4_pot_act['NODO33'] = n4_pot_act['NODO35'].clip(lower=0, upper=2.2)    #        
        # n4_pot_act['NODO40'] = n4_pot_act['NODO40'].clip(lower=, upper=)    #16  
        n4_pot_act['NODO42'] = n4_pot_act['NODO42'].clip(lower=0, upper=1.2)    #        
        n4_pot_act['NODO46'] = n4_pot_act['NODO46'].clip(lower=13, upper=41.8)    #
        n4_pot_act['NODO49'] = n4_pot_act['NODO49'].clip(lower=0, upper=3.1)    #19  
        n4_pot_act['NODO50'] = n4_pot_act['NODO50'].clip(lower=0, upper=1.9)    #    
        n4_pot_act['NODO60'] = n4_pot_act['NODO60'].clip(lower=0, upper=0.4)    #
        n4_pot_act['NODO62'] = n4_pot_act['NODO62'].clip(lower=3.3, upper=6.8)  #
        # n4_pot_act['NODO68'] = n4_pot_act['NODO68'].clip(lower=, upper=)
        n4_pot_act['NODO69'] = n4_pot_act['NODO69'].clip(lower=1, upper=6)
        n4_pot_act['NODO70'] = n4_pot_act['NODO70'].clip(lower=1, upper=6)
        n4_pot_act['NODO71'] = n4_pot_act['NODO71'].clip(lower=4, upper=16)    #
        n4_pot_act['NODO72'] = n4_pot_act['NODO72'].clip(lower=1, upper=7)    #27       
        # n4_pot_act['NODO76'] = n4_pot_act['NODO76'].clip(lower=, upper=) #28
        # n4_pot_act['NODO79'] = n4_pot_act['NODO79'].clip(lower=, upper=)        #
        n4_pot_act['NODO80'] = n4_pot_act['NODO80'].clip(lower=0.3, upper=3.2)       #
        # n4_pot_act['NODO86'] = n4_pot_act['NODO86'].clip(lower=, upper=)
        n4_pot_act['NODO88'] = n4_pot_act['NODO88'].clip(lower=0, upper=2.3)
        # n4_pot_act['NODO92'] = n4_pot_act['NODO92'].clip(lower=, upper=)    #
        n4_pot_act['NODO93'] = n4_pot_act['NODO93'].clip(lower=0, upper=30.8)    #
        n4_pot_act['NODO94'] = n4_pot_act['NODO94'].clip(lower=0.4, upper=4.3)
        
    
    #%% FIGURA    
    if ima == 1:
        
        fig1, axs = plt.subplots(5,1,figsize=(12,8)) 
        
        axs[0].plot(n4_pot_act.loc[:, 'NODO69'])
        axs[1].plot(n4_pot_act.loc[:, 'NODO70'])
        axs[2].plot(n4_pot_act.loc[:, 'NODO71'])
        axs[3].plot(n4_pot_act.loc[:, 'NODO72'])
        axs[4].plot(n4_pot_act.loc[:, 'NODO4'])
        
        axs[0].set_title('MR POLLO LA 17')
        axs[1].set_title('MR POLLO LA COLINA')
        axs[2].set_title('MR POLLO NORTE')
        axs[3].set_title('MR POLLO SUR')
        axs[4].set_title('OTRO') 
        
        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)
        axs[3].grid(True)
        axs[4].grid(True)
        
        plt.tight_layout()
        
        plt.show()
    
    #%%
    
    # print(np.min(n4_pot_act[0]))
    # print(np.min(n4_pot_act[:, 0]))
    # print(n4_pot_act.iloc[:, 0].min())
    # a = n4_pot_act_nor[n4_pot_act.iloc[:,0]]
    
    # nombres_columnas = n4_pot_act.columns
    # print(nombres_columnas[0])
    
    #%%#%%NORMALIZACION DE DATOS POR SERIE TEMPORAL
    
    # print(n4_pot_act.shape[1])
    n4_pot_act_nor = n4_pot_act.copy()
    n_col = n4_pot_act.columns
    
    for i in range(n4_pot_act.shape[1]):
        nc = n_col[i]
        n4_pot_act_nor[nc] = (n4_pot_act.iloc[:,i] - n4_pot_act.iloc[:,i].min()) / (n4_pot_act.iloc[:,i].max() - n4_pot_act.iloc[:,i].min())
    
    #%%NORMALIZACION DE DATOS POR SERIE TEMPORAL
    
    # n4_pot_act_nor = n4_pot_act.copy()
    
    # n4_pot_act_nor[n4_pot_act.columns[-5]] = (n4_pot_act_nor[n4_pot_act.columns[-5]] - n0_mm[1]) / (n0_mm[0] - n0_mm[1])
    # n4_pot_act_nor[n4_pot_act.columns[-4]] = (n4_pot_act_nor[n4_pot_act.columns[-4]] - n1_mm[1]) / (n1_mm[0] - n1_mm[1])
    # n4_pot_act_nor[n4_pot_act.columns[-2]] = (n4_pot_act_nor[n4_pot_act.columns[-2]] - n2_mm[1]) / (n2_mm[0] - n2_mm[1])
    # n4_pot_act_nor[n4_pot_act.columns[-1]] = (n4_pot_act_nor[n4_pot_act.columns[-1]] - n3_mm[1]) / (n3_mm[0] - n3_mm[1])
    
    #%% FIGURA
    
    if ima == 1:
                
        fig1, axs = plt.subplots(4,1,figsize=(12,8))
        # axs[0].plot(n4_pot_act.iloc[:,'NODO92']) 
        axs[0].plot(n4_pot_act_nor.loc[:, 'NODO69'])
        axs[1].plot(n4_pot_act_nor.loc[:, 'NODO70'])
        axs[2].plot(n4_pot_act_nor.loc[:, 'NODO71'])
        axs[3].plot(n4_pot_act_nor.loc[:, 'NODO72'])
        # axs[4].plot(n4_pot_act_nor.loc[:, 'NODO4'])
        
        axs[0].set_title('MR POLLO LA 17')
        axs[1].set_title('MR POLLO LA COLINA')
        axs[2].set_title('MR POLLO NORTE')
        axs[3].set_title('MR POLLO SUR')
        # axs[4].set_title('OTRO')        
        
        axs[0].grid(True)
        axs[1].grid(True)
        axs[2].grid(True)
        axs[3].grid(True)
        # axs[4].grid(True)
        
        plt.tight_layout()
        
        plt.show()
    
    #%%CARGAR DATOS DE 4 NODOS 1 MES
    
    # Se tiene 2 parametros:
    # 1ro es el lag de la serie, esto significa cuantos datos toma la serie para
    # hacer la prediccion
    # 2do es los pasos de prediccion, cuantos pasos en el futuro se desea predecir,
    # parametro importante, puesto que es el valor que se delizara la ventala
    
    n4_pot_act_arr = np.array(n4_pot_act_nor)
    input_seq_length = 4*24*30
    output_seq_length = 4*24
    sliding_steps = output_seq_length
    
    X_train, Y_train = split_sequences(n4_pot_act_arr,
                                        n4_pot_act_arr,
                                        n_steps_in = input_seq_length,
                                        n_steps_out = output_seq_length,
                                        n_sliding_steps = sliding_steps,
                                        window_type='sliding')
    
#%% INDICES DE NODOS OBJETIVOS

    n1_ind = n4_pot_act_nor.columns.get_loc('NODO69')
    n2_ind = n4_pot_act_nor.columns.get_loc('NODO70')
    n3_ind = n4_pot_act_nor.columns.get_loc('NODO71')
    n4_ind = n4_pot_act_nor.columns.get_loc('NODO72')
    
    print(f"Indice de nodos objetivos: {n1_ind}, {n2_ind}, {n3_ind}, {n4_ind}")
    
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
    y = y.to(device)
    
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
    # print("Forma del conjunto de entrenamiento:", feature_train_set.shape)
    # print("Forma del conjunto de validación:", feature_val_set.shape)
    # print("Forma del conjunto de testeo:", feature_test_set.shape)
    
    # Divide el tensor en conjuntos de entrenamiento y validación
    y_train_set = y[:train_size].to(device)
    y_val_set = y[train_size:train_size+val_size].to(device)
    y_test_set = y[train_size+val_size:].to(device)
    
    #%% DEFINICION RED NN
    # Definición de la clase RecurrentGCN que hereda de torch.nn.Module
    class RecurrentGCN(torch.nn.Module):
        # Constructor de la clase que inicializa las capas recurrente y lineal
        
        # Celda unitaria recurrente cerrada convolucional de grafo de Chebyshev.
        # def __init__(self, node_features, cell,filters):
        #     super(RecurrentGCN, self).__init__()
        #     self.recurrent = GConvGRU(node_features, cell, filters, None)
        #     self.linear = torch.nn.Linear(cell, output_seq_length)
        
        # Celda de memoria convolucional a largo plazo a corto plazo de grafo de Chebyshev.
        # def __init__(self, node_features, filters,cell):
        #     super(RecurrentGCN, self).__init__()
        #     self.recurrent = GConvLSTM(node_features, cell, filters, None)
        #     self.linear = torch.nn.Linear(cell, output_seq_length)
        
        # #Unidad Recurrente Cerrada Convolucional de Difusión.
        # def __init__(self, node_features, cell,filters):
        #     super(RecurrentGCN, self).__init__()        
        #     self.recurrent = DCRNN(node_features, cell, filters)            
        #     self.linear = torch.nn.Linear(cell, output_seq_length)        
        
        #Unidad Recurrente Cerrada Convolucional de Difusión.
        # def __init__(self, node_features):
        #     super(RecurrentGCN, self).__init__()
        #     # Inicialización de la capa recurrente DCRNN con el número de 
        #     #características de nodo
        #     self.recurrent_1 = DCRNN(node_features, 64, 2)
        #     self.recurrent_2 = DCRNN(64, 32, 2)
        #     # Inicialización de la capa lineal con entrada de 32 y salida de 1
        #     self.linear = torch.nn.Linear(32, 1)        
        
        #Red neuronal de paso de mensajes con memoria a largo plazo.
        # def __init__(self, node_features,cell, num_nodes):
        #     super(RecurrentGCN, self).__init__()
        #     self.recurrent = MPNNLSTM(node_features, cell, num_nodes, 1, 0.2)
        #     self.linear = torch.nn.Linear(2*cell + node_features, output_seq_length)
        
        # def __init__(self, node_features, filters):
        #     super(RecurrentGCN, self).__init__()
        #     self.recurrent = STConv(4,node_features, 32, output_seq_length, 4, filters, None)
        #     self.linear = torch.nn.Linear(32, output_seq_length)
            
        def __init__(self, node_features,cell, filters):
            super(RecurrentGCN, self).__init__()
            # self.recurrent = GCLSTM(node_features, cell, output_seq_length, None)
            self.recurrent = GCLSTM(node_features, cell, filters, None)
            self.linear = torch.nn.Linear(cell, output_seq_length)
            
        # def __init__(self, node_features):
        #     super(RecurrentGCN, self).__init__()
        #     self.recurrent = EvolveGCNO(node_features)
        #     self.linear = torch.nn.Linear(node_features, output_seq_length)
        
        # def __init__(self, node_features):
        #     super(RecurrentGCN, self).__init__()
        #     self.recurrent = EvolveGCNH(4, node_features)
        #     self.linear = torch.nn.Linear(node_features, output_seq_length)
    
            
        # def __init__(self, node_features,cell):
        #     super(RecurrentGCN, self).__init__()
        #     self.recurrent = LRGCN(node_features, cell, 1, 1)
        #     self.linear = torch.nn.Linear(cell, output_seq_length)
            
        # def __init__(self, node_features,cell, filters):
        #     super(RecurrentGCN, self).__init__()
        #     self.recurrent = AGCRN(number_of_nodes = 4,
        #                           in_channels = node_features,
        #                           out_channels = cell,
        #                           K = filters,
        #                           embedding_dimensions = 4)
        #     self.linear = torch.nn.Linear(cell, output_seq_length)
    
        
        # def forward(self, x, edge_index, edge_weight):
        #     h = self.recurrent(x, edge_index, edge_weight)
        #     h = F.relu(h)
        #     h = self.linear(h)
            # return h
        
        def forward(self, x, edge_index, edge_weight, h, c):
            h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h, c)
            h = F.relu(h_0)
            h = self.linear(h)
            return h, h_0, c_0
            
        # def forward(self, x, e, h):
        #     h_0 = self.recurrent(x, e, h)
        #     y = F.relu(h_0)
        #     y = self.linear(y)
        #     return y, h_0
        
        # Método forward que realiza el pase hacia adelante
        # def forward(self, x, edge_index, edge_weight):
        #     # Pase hacia adelante en la capa recurrente DCRNN con las características de nodo, 
        #     # índices de aristas y pesos de aristas
        #     h = self.recurrent(x, edge_index, edge_weight)
        #     # h = self.recurrent_1(x, edge_index, edge_weight)
        #     # Aplicar la función de activación ReLU a la salida de la capa recurrente
        #     # h = F.relu(h[0])
        #     h = F.relu(h)
        #     # h = self.recurrent_2(h, edge_index, edge_weight)
        #     # h = F.relu(h)
        #     # h = F.dropout(h, training=self.training)
        #     # Pase hacia adelante en la capa lineal para obtener la salida final
        #     h = self.linear(h)
        #     # Devolver la salida final
        #     return h
        
    #%% --------> ENTRENAMIENTO
    # mod = 'DCRNN'
    # mod = 'DCRNN_2'
    # mod = 'GConvGRU'
    # mod = 'GConvLSTM'
    # mod = 'MPNNLSTM'
    mod = 'GCLSTM'    
    # mod = 'EvolveGCNH'
    # mod = 'EGCNO'
    
    # mod = 'AGCRN'
    # mod = 'LRGCN'
    
    # mod = 'STConv'
    
    
    if mod == 'MPNNLSTM':    
        K = [0]
        # CELL = [16, 32, 64, 128]
        CELL = [16, 32]
    else:    
        K =[1]
        CELL = [64, 128]
    
    for ind, k in enumerate(K):
        for ind, cell in enumerate(CELL):
            
            # Definir el número máximo de épocas sin mejora permitidas antes de detener el entrenamiento
            max_epochs_without_improvement = 35
            best_val_loss = float('inf')
            epochs_without_improvement = 0
            train_cost = []
            val_cost = []
            epocas = 1000
            nodos ='model_4N_'
            num_nodes = len(node_val)
            # Crear una instancia del modelo RecurrentGCN con 4 características de nodo            
            if mod == 'MPNNLSTM':
                model = RecurrentGCN(node_features = input_seq_length, cell=cell, num_nodes = num_nodes)  
            elif mod == 'GCLSTM':
                model = RecurrentGCN(node_features = input_seq_length, cell=cell,filters = k)
            elif mod == 'EGCNO':
                model = RecurrentGCN(node_features = input_seq_length)
                
            # model = RecurrentGCN(node_features = input_seq_length, window = input_seq_length)
            model.to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            opt = '_Adam'
            # optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005, alpha=0.9)
            # opt = '_RMSprop'
            # optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
            # opt = '_SGD'
            
            out_txt = nodos+str(mod)+'_Out'+str(output_seq_length)+'_In'+str(input_seq_length)+opt+'_Cell'+str(cell)+'_K'+str(k)+'_E'+str(epocas)+'_R'+str(rad_prox)
            # out_txt = 'TEST'+nodos+str(mod)+'_Out'+str(output_seq_length)+'_In'+str(input_seq_length)+opt+'_Cell'+str(cell)+'_K'+str(k)+'_E'+str(epocas)+'_R'+str(rad_prox)
            print(out_txt)
            print('-------------------------------')
            # print(model)
            
            #%% TRAIN 1
            # Guarda el tiempo de inicio
            inicio = time.time()            
            
            # Configurar el modelo en modo de entrenamiento
            model.train()
            
            # Bucle de entrenamiento a lo largo de 200 épocas
            for epoch in tqdm(range(epocas)):
                # Inicializar la variable de costo para la época actual
                cost = 0
            
                # Iterar sobre los instantes de tiempo y capturas en el conjunto de datos 
                # de entrenamiento
                # for times in enumerate(feature_train_set):
                h, c = None, None
                for times in range(len(feature_train_set)):
                    # Obtener las predicciones del modelo para las características de nodo, 
                    # índices de arista y atributos de arista en el instante de tiempo actual
                    # print(feature_train_set[times])
                    
                    # y_hat = model(feature_train_set[times].to(device), edge_index.to(device), edge_attr.to(device))
                    
                    if mod == 'MPNNLSTM' or mod == 'EGCNO':
                        y_hat = model(feature_train_set[times], edge_index, edge_attr)
                    elif mod == 'GCLSTM':
                        y_hat, h, c = model(feature_train_set[times], edge_index, edge_attr, h, c)  
                            
                    y_hat_p = y_hat[[n1_ind, n2_ind, n3_ind, n4_ind]]
                    y_train_set_p = y_train_set[times, [n1_ind, n2_ind, n3_ind, n4_ind]]
                    
                    # y_hat_p = y_hat[[n1_ind, n2_ind, n3_ind]]
                    # y_train_set_p = y_train_set[times, [n1_ind, n2_ind, n3_ind]]
                    
                    # y_hat_p = y_hat[[n2_ind]]
                    # y_train_set_p = y_train_set[times, [n2_ind]]
                
                    # Calcular el costo cuadrático medio entre las predicciones y las etiquetas reales   
                    # loss = F.mse_loss(y_hat, y_train_set[times].unsqueeze(1))
                    # loss = F.mse_loss(y_hat, y_train_set[times])
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
                        if mod == 'MPNNLSTM' or mod == 'EGCNO':
                            val_y_hat = model(feature_val_set[val_time], edge_index, edge_attr)
                        elif mod == 'GCLSTM':
                            val_y_hat, h, c = model(feature_val_set[val_time], edge_index, edge_attr, h, c)  
                            
                        # val_y_hat = model(feature_val_set[val_time], edge_index, edge_attr)
                        # val_y_hat, h, c = model(feature_val_set[val_time], edge_index, edge_attr, h, c)            
                        # val_loss += F.mse_loss(val_y_hat, y_val_set[val_time].unsqueeze(1))
                        
                        val_y_hat_p = val_y_hat[[n1_ind, n2_ind, n3_ind, n4_ind]]
                        y_val_set_p = y_val_set[val_time, [n1_ind, n2_ind, n3_ind, n4_ind]]   
                        
                        # val_y_hat_p = val_y_hat[[n1_ind, n2_ind, n3_ind]]
                        # y_val_set_p = y_val_set[val_time, [n1_ind, n2_ind, n3_ind]]                        
                        
                        # val_y_hat_p = val_y_hat[[n2_ind]]
                        # y_val_set_p = y_val_set[val_time, [n2_ind]]                                                      
                                                
                        # val_loss += F.mse_loss(val_y_hat, y_val_set[val_time])
                        val_loss += F.mse_loss(val_y_hat_p, y_val_set_p)
                        
                    val_loss = val_loss / (len(feature_val_set) + 1)
                    print(" - MSE: {:.4f}".format(cost)," - Val MSE: {:.5f}".format(val_loss.item()))
                    
                    # Verificar si la pérdida en el conjunto de validación ha disminuido
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        epochs_without_improvement = 0
                        torch.save(model.state_dict(), r'C:\Users\GIIEE\Desktop\MTE_GRAPHS\MODELOS\COMERCIAL\\'+out_txt+'.pth')
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
            # print("El bloque de código tomó {} segundos en ejecutarse.".format(tiempo_transcurrido))
                
            #%% GRAFICA DE MEAN SQUARED ERROR
            
            if ima == 1:
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
                plt.savefig(r'C:\Users\GIIEE\Desktop\MTE_GRAPHS\MODELOS\COMERCIAL\FIGURAS\Train_'+out_txt+'.png')
                
                # Mostrar la gráfica
                plt.show()
            
            #%%CARGAR MODELO
            # Crear una nueva instancia del modelo
            # model = RecurrentGCN(node_features=input_seq_length,cell=cell)
            if mod == 'MPNNLSTM':
                model = RecurrentGCN(node_features=input_seq_length,cell=cell, num_nodes = len(node_val))
            elif mod == 'GCLSTM':
                model = RecurrentGCN(node_features=input_seq_length,cell=cell,filters=k)
            elif mod == 'EGCNO':
                model = RecurrentGCN(node_features = input_seq_length)
                
            
            path = r'C:\Users\GIIEE\Desktop\MTE_GRAPHS\MODELOS\COMERCIAL\\'+out_txt+'.pth'
            # print(path)
            # # Cargar los parámetros guardados en el modelo
            model.load_state_dict(torch.load(path))
            # print('-----------> MODELO CARGADO')
            
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
                #print(times)   
                # aux_x = snapshot.x
                # aux_ed = snapshot.edge_index
                # aux_ea = snapshot.edge_attr
                
                # aux = 
                # print(aux.shape)
                # print(aux.dtype)
                # print(aux.dim())
            
                # print(snapshot.x)
                # print(snapshot.edge_attr)
                # print(snapshot.edge_index)
                # Realizar la predicción utilizando el modelo
                if mod == 'MPNNLSTM' or mod == 'EGCNO':
                    y_hat = model(feature_test_set[times], edge_index, edge_attr)
                elif mod == 'GCLSTM':
                    y_hat, h, c = model(feature_test_set[times], edge_index, edge_attr, h, c)  
                # y_hat = model(feature_test_set[times], edge_index, edge_attr)
                # y_hat, h, c = model(feature_test_set[times], edge_index, edge_attr, h, c)
                
                y_out.append(y_hat.cpu().detach().numpy())
                
                # Calcular el costo acumulado sumando el error cuadrático medio (MSE)
                # cost = cost + torch.mean((y_hat - y_test_set[times]) ** 2)
                # cost = cost + F.mse_loss(y_hat, y_test_set[times])
                
                # loss = F.mse_loss(y_hat, y_test_set[times].unsqueeze(1))
                loss = F.mse_loss(y_hat, y_test_set[times])
                cost += loss 
                
            
            #print('PREDICCION = ',y_hat)
            #print(dataset.targets[-1])
            #print(test_dataset.)
            
            # Calcular el costo promedio dividiendo por el número de instantes de tiempo
            cost = cost / (times + 1)
            
            # Obtener el valor numérico del costo
            cost = cost.item()
            
            # Imprimir el MSE (Mean Squared Error)
            # print("MSE: {:.4f}".format(cost))
            
            #%% CREA LISTAS VACIAS PARA ALMACENAR RESULTADOS VERDADEROS
            
            y_out_test = y_test_set.numpy()
            
            # print(y_out_test[0][0])
            # print(len(y_out_test))
            
            vacio = []
            
            node_0_test = np.array(vacio)
            node_1_test = np.array(vacio)
            node_2_test = np.array(vacio)
            node_3_test = np.array(vacio)
            
            
            
            for i, data in enumerate(y_out_test):
                # print(data)
                # print(i)                
                node_0_test = np.append(node_0_test,data[n1_ind])
                node_1_test = np.append(node_1_test,data[n2_ind])
                node_2_test = np.append(node_2_test,data[n3_ind])
                node_3_test = np.append(node_3_test,data[n4_ind])                    
                
            #%%CREA LISTAS VACIAS PARA ALMACENAR RESULTADOS PREDICCION
            
            
            y_out_pre = np.array(y_out)
            # print(y_out_pre[0][0])
            
            node_0 = np.array(vacio)
            node_1 = np.array(vacio)
            node_2 = np.array(vacio)
            node_3 = np.array(vacio)
            
            
            for i, data in enumerate(y_out_pre):
                # print(data)                
                node_0 = np.append(node_0,data[n1_ind])
                node_1 = np.append(node_1,data[n2_ind])
                node_2 = np.append(node_2,data[n3_ind])
                node_3 = np.append(node_3,data[n4_ind])

            
            #%% METRICAS DE ERROR
            
            # Crear listas para almacenar las métricas
            mse_list = []
            mae_list = []
            r2_list = []
            
            # Calcular métricas para cada par de variables
            for i in range(4):  # Suponiendo que tienes 4 nodos
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
                
            # y_real = node_1 #globals()[f'node_{i}']
            # y_pred = node_1_test#globals()[f'node_{i}_test']
        
            # # Calcular el Error Cuadrático Medio (MSE) 
            # mse = mean_squared_error(y_real, y_pred)
            # mse_list.append(mse)
        
            # # Calcular el Error Absoluto Medio (MAE)
            # mae = mean_absolute_error(y_real, y_pred)
            # mae_list.append(mae)
        
            # # Calcular el Coeficiente de Determinación (R²)
            # r2 = r2_score(y_real, y_pred)
            # r2_list.append(r2)
        
            # # Imprimir las métricas para cada nodo
            # print(f'\nMétricas para node_{i}:')
            # print(f'MSE: {mse}')
            # print(f'MAE: {mae}')
            # print(f'R²: {r2}')
            
            # Imprimir métricas promedio
            print('\nMétricas promedio:')
            print(f'MSE promedio: {np.mean(mse_list)}')
            print(f'MAE promedio: {np.mean(mae_list)}')
            print(f'R² promedio: {np.mean(r2_list)}')
            
            #%%GRAFICA COMPARACION
            # if ima == 1:
            fig, axs= plt.subplots(4,1,figsize=(12,8))
            
            # Título global para los subgráficos
            plt.suptitle('RESULTADOS MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(cost,np.mean(mae_list),np.mean(r2_list)))
            
            axs[0].plot(node_0_test)
            axs[0].plot(node_0,'--')
            axs[1].plot(node_1_test)
            axs[1].plot(node_1,'--')
            axs[2].plot(node_2_test)
            axs[2].plot(node_2,'--')
            axs[3].plot(node_3_test)
            axs[3].plot(node_3,'--')
            
            axs[0].set_title('LA 17 MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[0],mae_list[0],r2_list[0]))
            axs[1].set_title('COLINA MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[1],mae_list[1],r2_list[1]))
            axs[2].set_title('NORTE MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[2],mae_list[2],r2_list[2]))
            axs[3].set_title('SUR MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[3],mae_list[3],r2_list[3]))
            
            axs[0].grid(True)
            axs[1].grid(True)
            axs[2].grid(True)
            axs[3].grid(True)            
                        
            # axs[0,0].set_title('BLETHLEMITAS MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[0],mae_list[0],r2_list[0]))
            # axs[1,0].set_title('CHAMPAGNAT MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[1],mae_list[1],r2_list[1]))
            # axs[2,0].set_title('JAVERIANITO MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[2],mae_list[2],r2_list[2]))
            # axs[3,0].set_title('JAVERIANO MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[3],mae_list[3],r2_list[3]))
            # axs[0,1].set_title('U COOP MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[4],mae_list[4],r2_list[4]))
            # axs[1,1].set_title('UDENAR MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[5],mae_list[5],r2_list[5]))
            # axs[2,1].set_title('UNIMAR MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[6],mae_list[6],r2_list[6]))
            # axs[3,1].set_title('VIPRI MSE = {:.5f} - MAE = {:.5f} - R² = {:.5f}'.format(mse_list[7],mae_list[7],r2_list[7]))
            
            plt.tight_layout()
            
            # Guardar la figura en un archivo (puedes especificar el formato, como PNG, PDF, SVG, etc.)
            plt.savefig(r'C:\Users\GIIEE\Desktop\MTE_GRAPHS\MODELOS\COMERCIAL\FIGURAS\Res_'+out_txt+'.png')
            
            # plt.show()
            
            # Cerrar la figura
            plt.close()
            
            #%% GUARDA METRICAS
            
            # print(out_txt)
                
            datos_modelo['NAME MODEL'].append(out_txt)
            datos_modelo["RAD"].append(rad_prox)
            datos_modelo["K"].append(k)
            datos_modelo["CELL"].append(cell)
            datos_modelo["MSE"].append(np.mean(mse_list))
            datos_modelo["MAE"].append(np.mean(mae_list))
            datos_modelo["R2"].append(np.mean(r2_list))
            datos_modelo["TIME"].append(tiempo_transcurrido)
            
            # Nombre del archivo CSV
            # input_seq_length = 4*24*30
            # output_seq_length = 4*24
            nombre_archivo = r"C:\Users\GIIEE\Desktop\MTE_GRAPHS\MODELOS\COMERCIAL\METRICAS\DATA_"+mod+"_OUT"+str(output_seq_length)+"_IN"+str(input_seq_length)+".csv"
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
        




