# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 03:28:42 2020

@author: Alejandro
"""

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importar base de datos
bd1 = pd.read_csv('electrodb.csv', encoding = 'unicode_escape')
bd1.head()
bd1

#ver tamaño de dataset
bd1.shape

#Ver distribución por distrito
dist_ciudad=bd1[['CIUDAD','COD_CLIENTE']].drop_duplicates()
dist_ciudad.groupby(['CIUDAD'])['COD_CLIENTE'].aggregate('count').reset_index().sort_values('COD_CLIENTE', ascending=False)

#Solo utilizamos la información de JM
bd1 = bd1.query("CIUDAD=='LIMA METROPOLITANA'").reset_index(drop=True)

#Se comprueba los vablores
bd1.isnull().sum(axis=0)

#Quitar valores donde la columna de cliente este vacía
bd1 = bd1[pd.notnull(bd1['COD_CLIENTE'])]

#Validar que no haya valores negativos en la cantidad
bd1.CANTIDAD.min()

#validar que no haya valores negativos en el precio unitario
bd1.PRECIO_UNIT.min()

#Quitar entradas donde existan valroes negativos
bd1 = bd1[(bd1['CANTIDAD']>0)]

#Convertir FECHA_FACT en formato fecha
bd1['FECHA_FACT'] = pd.to_datetime(bd1['FECHA_FACT'])

#Creacion de nueva columna precio total

bd1['PRECIO_UNIT'] = bd1['PRECIO_UNIT'].astype(int)
bd1['PRECIO_TOTAL'] = bd1['CANTIDAD'] * bd1['PRECIO_UNIT']

#comprobamos los filtros anteriores y que se haya crado nva columna
bd1.shape

bd1.head()

#bd1.to_csv(r'C:\Users\Alejandro\Documents\PYTHON\prueba0\st2_tabla2.csv')
#Modelo RFM

#Recency = Ultima fecha de factura - Ultima información de compra, Frecuency = Conteo de número de facturas, Monetary= Suma total de gastos por cliente

import datetime as dt

#Seleccionar ultima fecha + +1 como el la utima fecha de factura para calcular recency
Latest_Date = dt.datetime(2019,12,31)

#Crear scores de RFM para cada cliente
RFMScores = bd1.groupby('COD_CLIENTE').agg({'FECHA_FACT': lambda x: (Latest_Date - x.max()).days, 'NRO_FACT': lambda x: len(x), 'PRECIO_TOTAL': lambda x: x.sum()})

#Convertir fecha de factura a entero
RFMScores['FECHA_FACT'] = RFMScores['FECHA_FACT'].astype(int)

#Cambio de nombre de columnas a Recency, Frequency y Monetary
RFMScores.rename(columns={'FECHA_FACT': 'Recency', 
                         'NRO_FACT': 'Frequency', 
                         'PRECIO_TOTAL': 'Monetary'}, inplace=True)

RFMScores.reset_index().head()

#Estadisticas descriptivas para Recency
RFMScores.Recency.describe()

#graficas para recency
import seaborn as sns
x = RFMScores['Recency']
ax = sns.distplot(x)

#Estadisticas descriptivas para Frecuency
RFMScores.Frequency.describe()

#Graficas para Frecuency tomando registros que tengan un valor menor a 1000
import seaborn as sns
x = RFMScores.query('Frequency < 1000')['Frequency']

ax = sns.distplot(x)

#Estadisticas descriptivas para Monetary
RFMScores.Monetary.describe()

#Graficas para Monetary con valores menores a 10000

import seaborn as sns
x = RFMScores.query('Monetary < 10000')['Monetary']

ax = sns.distplot(x)

#Crear 4 segmentos usando quartiles
quantiles = RFMScores.quantile(q=[0.25,0.5,0.75])
quantiles = quantiles.to_dict()

quantiles

#FUnciones para creacion de segmentos R F M
def RScoring(x,p,d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.50]:
        return 2
    elif x <= d[p][0.75]: 
        return 3
    else:
        return 4
    
def FnMScoring(x,p,d):
    if x <= d[p][0.25]:
        return 4
    elif x <= d[p][0.50]:
        return 3
    elif x <= d[p][0.75]: 
        return 2
    else:
        return 1 

#Calculo de los segmentos y adicion en dataset
RFMScores['R'] = RFMScores['Recency'].apply(RScoring, args=('Recency',quantiles,))
RFMScores['F'] = RFMScores['Frequency'].apply(FnMScoring, args=('Frequency',quantiles,))
RFMScores['M'] = RFMScores['Monetary'].apply(FnMScoring, args=('Monetary',quantiles,))
RFMScores.head()
RFMScores
#RFMScores.to_csv(r'C:\Users\Alejandro\Documents\PYTHON\prueba0\RFM2st2.csv')
#Calcular y agregar valor de RFM en una columna enseñando la suma de los valores
RFMScores['RFMGroup'] = RFMScores.R.map(str) + RFMScores.F.map(str) + RFMScores.M.map(str)

#Calcular una columna de RFM score mostrando el total de las variables de grupo RFM
RFMScores['RFMScore'] = RFMScores[['R', 'F', 'M']].sum(axis = 1)
RFMScores.head()

#Asignación de nivel de fidelidad para c/ cliente
Loyalty_Level = ['Platino', 'Oro', 'Plata', 'Bronce']
Score_cuts = pd.qcut(RFMScores.RFMScore, q = 4, labels = Loyalty_Level)
RFMScores['RFM_Loyalty_Level'] = Score_cuts.values
RFMScores.reset_index().head()
RFMScores[RFMScores['RFMGroup']=='123'].sort_values('Monetary', ascending=False).reset_index().head(10)
#RFMScores.to_csv(r'C:\Users\Alejandro\Documents\PYTHON\prueba0\2st2_FIDELIDAD.csv')

import chart_studio as cs
import plotly.offline as po
import plotly.graph_objs as gobj
#conda install -c plotly chart-studio
#Recency Vs Frequency
graph = RFMScores.query("Monetary < 50000 and Frequency < 2000")

plot_data = [
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Bronce'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Bronce'")['Frequency'],
        mode='markers',
        name='Bronce',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Plata'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Plata'")['Frequency'],
        mode='markers',
        name='Plata',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Oro'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Oro'")['Frequency'],
        mode='markers',
        name='Oro',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Platino'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Platino'")['Frequency'],
        mode='markers',
        name='Platino',
        marker= dict(size= 13,
            line= dict(width=1),
            color= 'black',
            opacity= 0.9
           )
    ),
]

plot_layout = gobj.Layout(
        yaxis= {'title': "Frequency"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = gobj.Figure(data=plot_data, layout=plot_layout)
po.iplot(fig)

#Frequency Vs Monetary
graph = RFMScores.query("Monetary < 50000 and Frequency < 2000")

plot_data = [
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Bronce'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'Bronce'")['Monetary'],
        mode='markers',
        name='Bronce',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Plata'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'Plata'")['Monetary'],
        mode='markers',
        name='Plata',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Oro'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'Oro'")['Monetary'],
        mode='markers',
        name='Oro',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Platino'")['Frequency'],
        y=graph.query("RFM_Loyalty_Level == 'Platino'")['Monetary'],
        mode='markers',
        name='Platino',
        marker= dict(size= 13,
            line= dict(width=1),
            color= 'black',
            opacity= 0.9
           )
    ),
]

plot_layout = gobj.Layout(
        yaxis= {'title': "Monetary"},
        xaxis= {'title': "Frequency"},
        title='Segments'
    )

fig = gobj.Figure(data=plot_data, layout=plot_layout)
po.iplot(fig)

#Recency Vs Monetary
graph = RFMScores.query("Monetary < 50000 and Frequency < 2000")

plot_data = [
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Bronze'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Bronze'")['Monetary'],
        mode='markers',
        name='Bronze',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Silver'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Silver'")['Monetary'],
        mode='markers',
        name='Silver',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Gold'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Gold'")['Monetary'],
        mode='markers',
        name='Gold',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
    gobj.Scatter(
        x=graph.query("RFM_Loyalty_Level == 'Platinum'")['Recency'],
        y=graph.query("RFM_Loyalty_Level == 'Platinum'")['Monetary'],
        mode='markers',
        name='Platinum',
        marker= dict(size= 13,
            line= dict(width=1),
            color= 'black',
            opacity= 0.9
           )
    ),
]

plot_layout = gobj.Layout(
        yaxis= {'title': "Monetary"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = gobj.Figure(data=plot_data, layout=plot_layout)
po.iplot(fig)

#KMEANS

#Manejo de valores negativos y de 0 como numeros infinitos
def handle_neg_n_zero(num):
    if num <= 0:
        return 1
    else:
        return num
    
#Aplicar función anterior a valores de Recency y Monetary 
RFMScores['Recency'] = [handle_neg_n_zero(x) for x in RFMScores.Recency]
RFMScores['Monetary'] = [handle_neg_n_zero(x) for x in RFMScores.Monetary]

#realizar transformada para aplicar una distribución normal o semi normal al dataset
Log_Tfd_Data = RFMScores[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)

#Distribución de data después de normalización de Recency
Recency_Plot = Log_Tfd_Data['Recency']
ax = sns.distplot(Recency_Plot)

#Distribución de data despues de normalización de Frecuency
Frequency_Plot = Log_Tfd_Data.query('Frequency < 1000')['Frequency']
ax = sns.distplot(Frequency_Plot)

#Distribución de data despues de normalización de Monetary
Monetary_Plot = Log_Tfd_Data.query('Monetary < 10000')['Monetary']
ax = sns.distplot(Monetary_Plot)

from sklearn.preprocessing import StandardScaler

#Escalar la data
scaleobj = StandardScaler()
Scaled_Data = scaleobj.fit_transform(Log_Tfd_Data)

#Transformar de nuevo al dataframe antiguo
Scaled_Data = pd.DataFrame(Scaled_Data, index = RFMScores.index, columns = Log_Tfd_Data.columns)

from sklearn.cluster import KMeans

sum_of_sq_dist = {}
for k in range(1,15):
    km = KMeans(n_clusters= k, init= 'k-means++', max_iter= 1000)
    km = km.fit(Scaled_Data)
    sum_of_sq_dist[k] = km.inertia_
    
#Grafico de las suma de valores de square distance y numero de clusters
sns.pointplot(x = list(sum_of_sq_dist.keys()), y = list(sum_of_sq_dist.values()))
plt.xlabel('Number of Clusters(k)')
plt.ylabel('Sum of Square Distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#Segmentacion k-means
KMean_clust = KMeans(n_clusters= 4, init= 'k-means++', max_iter= 1000)
KMean_clust.fit(Scaled_Data)

#Encontrar clusters ideales para el dataset
RFMScores['Cluster'] = KMean_clust.labels_
RFMScores.head()
print(RFMScores)
#pip install tabulate
print(KMean_clust.labels_)
print(KMean_clust.cluster_centers_)
print(KMean_clust.inertia_)



##Grafico plot de Frecuency vs Recency
Colors = ["red", "green", "blue", "yellow"]
RFMScores['Color'] = RFMScores['Cluster'].map(lambda p: Colors[p])
ax = RFMScores.plot(    
    kind="scatter", 
    x="Recency", y="Frequency",
    figsize=(10,8),
    c = RFMScores['Color']
)

RFMScores.head()
RFMScores
RFMScores.to_csv(r'C:\Users\Alejandro\Documents\PYTHON\prueba0\clusterfinal.csv')
RFMScores['Color'] = "red"
print(RFMScores)
RFMScores.to_csv(r'C:\Users\Alejandro\Documents\PYTHON\prueba0\clusterred.csv')
