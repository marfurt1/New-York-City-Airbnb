from utils import db_connect
engine = db_connect()

# your code here
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
import folium
from folium import plugins
from folium.plugins import MarkerCluster
from folium.plugins import FastMarkerCluster
from folium.plugins import HeatMap

airbnb = pd.read_csv('../data/raw/AB_NYC_2019.csv')

len(airbnb)

warnings.filterwarnings('ignore')
airbnb.dtypes
airbnb.info()
airbnb.head(3)
airbnb.isnull().sum()
top_host=airbnb.host_id.value_counts().head(10)
top_host
airbnb.sample(5)

airbnb['last_review'] = airbnb['last_review'].astype('datetime64[ns]')
airbnb=airbnb.astype({'name':'str','host_name':'str','neighbourhood_group':'category','neighbourhood':'category','room_type':'category'})
airbnb.describe()

sns.set(rc={'figure.figsize':(10,8)})
sns.set_style('white')

top_host_df=pd.DataFrame(top_host)
top_host_df.reset_index(inplace=True)
top_host_df.rename(columns={'index':'Host_ID', 'host_id':'P_Count'}, inplace=True)
top_host_df
viz_1=sns.barplot(x="Host_ID", y="P_Count", data=top_host_df,
                 palette='Blues_d')
viz_1.set_title('Hosts with the most listings in NYC')
viz_1.set_ylabel('Count of listings')
viz_1.set_xlabel('Host IDs')
viz_1.set_xticklabels(viz_1.get_xticklabels(), rotation=45)


sub_1=airbnb.loc[airbnb['neighbourhood_group'] == 'Brooklyn']
price_sub1=sub_1[['price']]

sub_2=airbnb.loc[airbnb['neighbourhood_group'] == 'Manhattan']
price_sub2=sub_2[['price']]

sub_3=airbnb.loc[airbnb['neighbourhood_group'] == 'Queens']
price_sub3=sub_3[['price']]

sub_4=airbnb.loc[airbnb['neighbourhood_group'] == 'Staten Island']
price_sub4=sub_4[['price']]

sub_5=airbnb.loc[airbnb['neighbourhood_group'] == 'Bronx']
price_sub5=sub_5[['price']]

price_list_by_n=[price_sub1, price_sub2, price_sub3, price_sub4, price_sub5]

#creating an empty list that we will append later with price distributions for each neighbourhood_group
p_l_b_n_2=[]
#creating list with known values in neighbourhood_group column
nei_list=['Brooklyn', 'Manhattan', 'Queens', 'Staten Island', 'Bronx']
#creating a for loop to get statistics for price ranges and append it to our empty list
for x in price_list_by_n:
    i=x.describe(percentiles=[.25, .50, .75])
    i=i.iloc[3:]
    i.reset_index(inplace=True)
    i.rename(columns={'index':'Stats'}, inplace=True)
    p_l_b_n_2.append(i)
#changing names of the price column to the area name for easier reading of the table    
p_l_b_n_2[0].rename(columns={'price':nei_list[0]}, inplace=True)
p_l_b_n_2[1].rename(columns={'price':nei_list[1]}, inplace=True)
p_l_b_n_2[2].rename(columns={'price':nei_list[2]}, inplace=True)
p_l_b_n_2[3].rename(columns={'price':nei_list[3]}, inplace=True)
p_l_b_n_2[4].rename(columns={'price':nei_list[4]}, inplace=True)
#finilizing our dataframe for final view    
stat_df=p_l_b_n_2
stat_df=[df.set_index('Stats') for df in stat_df]
stat_df=stat_df[0].join(stat_df[1:])
stat_df
#creating a sub-dataframe with no extreme values / less than 500
sub_6=airbnb[airbnb.price < 500]
#using violinplot to showcase density and distribtuion of prices 
viz_2=sns.violinplot(data=sub_6, x='neighbourhood_group', y='price')
viz_2.set_title('Density and distribution of prices for each neighberhood_group')

airbnb.neighbourhood.value_counts().head(10)

sub_7=airbnb.loc[airbnb['neighbourhood'].isin(['Williamsburg','Bedford-Stuyvesant','Harlem','Bushwick',
                 'Upper West Side','Hell\'s Kitchen','East Village','Upper East Side','Crown Heights','Midtown'])]
#using catplot to represent multiple interesting attributes together and a count
viz_3=sns.catplot(x='neighbourhood', hue='neighbourhood_group', col='room_type', data=sub_7, kind='count')
viz_3.set_xticklabels(rotation=20)

Long = -73.95
Lat = 40.73
locations = list(zip(airbnb.latitude, airbnb.longitude))

map1 = folium.Map(location=[Lat,Long], zoom_start=10)
FastMarkerCluster(data=locations).add_to(map1)
map1

# Obtener el precio promedio (excluyendo los valores nulos) por zona y tipo de alojamiento
df_precio_no_nulo = airbnb[airbnb['price'] > 0]
df_precio_promedio = df_precio_no_nulo.groupby(['room_type','neighbourhood_group'])['price'].mean().sort_values(ascending=True)
pd.DataFrame(df_precio_promedio).sort_values(by='room_type')
dict_precios_promedio = df_precio_promedio.to_dict()

def precio_nulo_a_promedio(fila):
	if fila['price'] > 0:
		return fila['price']
	else:
		return dict_precios_promedio[fila['room_type'], fila['neighbourhood_group']]

airbnb['price'] = airbnb.apply(precio_nulo_a_promedio, axis=1)

fig = px.box(airbnb, x='price')
fig.show()

q1_pr = airbnb['price'].quantile(0.25)
q3_pr =airbnb['price'].quantile(0.75)
IQR_pr = q3_pr - q1_pr
min_pr = q1_pr - 1.5*IQR_pr
max_pr = q3_pr + 1.5*IQR_pr
df_sin_outliers = airbnb[(airbnb['price']>=min_pr) & (airbnb['price']<=max_pr)]
plt.hist(df_sin_outliers.price, bins=15)
plt.title('Histograma de precios (excluyendo outliers)')
plt.show()

df_precio_promedio = airbnb.groupby(['room_type','neighbourhood_group'])['price'].mean().sort_values(ascending=True)
pd.DataFrame(df_precio_promedio).sort_values(by=['room_type', 'price'])

plt.figure(figsize=(12,6))
df_precio_promedio.plot(kind='bar')
plt.title('Precio promedio por tipo de alojamiento y distrito')
plt.ylabel('Precio promedio')
plt.xlabel('Tipo de alojamiento - Distrito')
plt.show()

q1_pr = airbnb['price'].quantile(q=0.25)
q2_pr = airbnb['price'].quantile(q=0.5)
q3_pr = airbnb['price'].quantile(q=0.75)

def categoria_precio(precio):
  if precio <= q1_pr:
    return 'min - q1'
  elif precio <= q2_pr:
    return 'q1 - q2'
  elif precio <= q3_pr:
    return 'q2 - q3'
  else:
    return 'q3 - max'

airbnb['Price_category'] = airbnb['price'].apply(categoria_precio).astype('category')

plt.subplots(1, 2, sharex=True, sharey=True, figsize=(16, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=airbnb.longitude, y=airbnb.latitude, hue=airbnb.neighbourhood_group, s=3, palette='Paired')
plt.title('Distritos de Nueva York')
plt.subplot(1, 2, 2)
sns.scatterplot(x=airbnb.longitude, y=airbnb.latitude, hue=airbnb.Price_category, hue_order=['min - q1', 'q1 - q2', 'q2 - q3', 'q3 - max'], s=3)
plt.title('Distribución geográfica por rangos de precios')
plt.show()

df_manhattan = airbnb[airbnb['neighbourhood_group']=='Manhattan']
df_manhattan['Price_category'] = df_manhattan['price'].apply(categoria_precio).astype('category')

plt.figure(figsize=(6, 6))
sns.scatterplot(x=df_manhattan.longitude, y=df_manhattan.latitude, hue=df_manhattan.Price_category, hue_order=['min - q1', 'q1 - q2', 'q2 - q3', 'q3 - max'], s=3, palette='dark')
plt.title('Distribución geográfica por rangos de precios en Manhattan')
plt.show()

df_outliers = airbnb[(airbnb['price']>=max_pr) | (airbnb['price']<=min_pr)]

df_outliers['Categoria_outliers_precio'] = 'Precios más altos (outliers)'
df_outliers['Categoria_outliers_precio'] = df_outliers['Categoria_outliers_precio'].astype('category')

color = ['k']

plt.figure(figsize=(9,6))
sns.scatterplot(x=airbnb.longitude, y=airbnb.latitude, hue=df.neighbourhood_group, s=5, palette='Paired')
sns.scatterplot(x=df_outliers.longitude, y=df_outliers.latitude, hue=df_outliers.Categoria_outliers_precio, palette=color, s=8, marker='D', legend=False)
plt.title('Ubicación de los alojamientos con precios más altos (atípicos)')
plt.show()

plt.subplots(1, 2, figsize=(12,4))
plt.subplot(1, 2, 1)
df_outliers['room_type'].value_counts().plot(kind='bar')
plt.title('Cantidad de alojamientos por tipo (solo datos atípicos)')
plt.subplot(1, 2, 2)
df_sin_outliers['room_type'].value_counts().plot(kind='bar')
plt.title('Cantidad de alojamientos por tipo (sin datos atípicos)')
plt.show()

fig = px.box(airbnb, x='minimum_nights')
fig.show()

print(f"Hay {len(airbnb.loc[airbnb['number_of_reviews']==0])} filas con 0 reviews y {airbnb['reviews_per_month'].isna().sum()} filas con nan en reviews por mes")

airbnb.loc[airbnb['number_of_reviews']==0]['reviews_per_month'].isna().sum()

airbnb['reviews_per_month'] = airbnb['reviews_per_month'].fillna(0)

np.all(np.isnat(np.array(airbnb.loc[airbnb['number_of_reviews']==0]['last_review'])))

np.size(np.isnat(np.array(airbnb.loc[airbnb['number_of_reviews']==0]['last_review']))) == len(airbnb.loc[airbnb['number_of_reviews']==0])

df_reviews = df.groupby('host_id').agg({'number_of_reviews': np.sum, 'id': pd.Series.nunique})
pd.DataFrame(df_reviews)

plt.hist(df_reviews['number_of_reviews'], bins=20)
plt.title('Histograma de total de reviews por host')
plt.show()

df_reviews_month = df.groupby('host_id').agg({'reviews_per_month': np.sum, 'id': pd.Series.nunique})
pd.DataFrame(df_reviews_month)

plt.hist(df_reviews_month['reviews_per_month'], bins=20)
plt.title('Histograma de total de reviews por mes por host')
plt.show()

f_no_disponible = airbnb[airbnb['availability_365']==0]
df_no_disponible['Categoria_disponibilidad'] = 'No disponible'
df_no_disponible['Categoria_disponibilidad'] = df_no_disponible['Categoria_disponibilidad'].astype('category')

color = ['k']

plt.figure(figsize=(9,6))
sns.scatterplot(x=airbnb.longitude, y=airbnb.latitude, hue=airbnb.neighbourhood_group, s=5, palette='Paired')
sns.scatterplot(x=df_no_disponible.longitude, y=df_no_disponible.latitude, hue=df_no_disponible.Categoria_disponibilidad, palette=color, s=8, marker='D', legend=False)
plt.title('Ubicación de los alojamientos no disponibles')
plt.show()

color = ['k']
color2 = ['c']

plt.figure(figsize=(9,6))
sns.scatterplot(x=df_no_disponible.longitude, y=df_no_disponible.latitude, hue=df_no_disponible.Categoria_disponibilidad, palette=color, s=5, legend=True)
sns.scatterplot(x=df_outliers.longitude, y=df_outliers.latitude, hue=df_outliers.Categoria_outliers_precio, palette=color2, s=5, legend=True)
plt.title('Comparación de la ubicación de los alojamientos más caros con la ubicación de los no disponibles')
plt.show()

df_no_disponible = df[df['availability_365']==0]
df_precio_promedio_no_disponible = df_no_disponible.groupby(['room_type','neighbourhood_group'])['price'].mean().sort_values(ascending=True)
pd.DataFrame(df_precio_promedio_no_disponible).sort_values(by=['room_type', 'price'])

df_disponible =airbnb[airbnb['availability_365']>0]
df_precio_promedio_disponible = df_disponible.groupby(['room_type','neighbourhood_group'])['price'].mean().sort_values(ascending=True)
pd.DataFrame(df_precio_promedio_disponible).sort_values(by=['room_type', 'price'])

plt.figure(figsize=(12,6))
df_precio_promedio_disponible.plot(kind='bar', width=-0.4, align='edge')
df_precio_promedio_no_disponible.plot(kind='bar', color='k', width=0.4, align='edge')
plt.legend(['Disponible', 'No disponible'])
plt.title('Precio promedio por tipo de alojamiento y distrito según disponibilidad')
plt.ylabel('Precio promedio')
plt.xlabel('Tipo de alojamiento - Distrito')
plt.show()

df_no_disponible['last_review'].sample(30)

