# ## Inicialización
# 
# Cargamos las librerias que se van a utilizar en el proyecto

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Importamos el archivo con los datos
# 
# 

# In[2]:


data = pd.read_csv("/datasets/games.csv")


# Estudiamos la informacion general

# In[3]:


data.info()


# In[4]:


data.head()


# ## Preparar los datos
# 

# Reemplazar los nombres de las columnas

# In[5]:


data.columns = data.columns.str.lower()


# In[6]:


data.head()


# Convertimos los datos en los tipos necesarios
# 

# Revisamos que datos tenemos en la columna rating. Vemos que son Letras entonces las dejamos del tipo object

# In[7]:


rating = data['rating'].value_counts()
print(rating)


# Convertimos la columna user_score a float ya que no puede ser de tipo object al tener numeros y decimales
# 
# Primero revisamos que datos no numericos pueden haber

# In[8]:


user_score = data['user_score'].value_counts()
print(user_score)


# Ya que tenemos campos con tdb que son del tipo object lo cambiaremos por NaN para poder luego modificar los tipos de datos a float

# In[9]:


data['user_score'] = data['user_score'].replace('tbd', np.nan)
data['user_score'] = data['user_score'].astype(float)


# Verificamos que todos los cambios se han aplicado correctamente

# In[10]:


data.info()


# In[11]:


data.head()


# Tratamos los valores ausentes

# In[12]:


data.isna().sum()


# Hay dos filas que no tienen nombre de videojuego y las eliminamos

# In[13]:


data = data.dropna(subset=['name'])


# Calculamos las ventas totales y agregamos una columna

# In[14]:


data['total_sales'] = data['na_sales'] + data['eu_sales'] + data['jp_sales'] + data['other_sales']


# In[15]:


data['total_sales'].max()


# Verificamos

# In[16]:


data


# Verificamos si hay duplicados en name asi saber si hay juegos en varias plataformas

# In[17]:


data['name'].duplicated().sum()


# Verificamos los valores nulos de años en funcion de la plataforma

# In[19]:


años_nulos = data['year_of_release'].isna().groupby(data['platform']).sum().reset_index()
años_nulos = años_nulos[años_nulos['year_of_release'] != 0].sort_values(by='year_of_release', ascending=False)
años_nulos = años_nulos['platform']
años_nulos


# Tenemos bastantes años sin valores asi que vamos a buscar la moda que es el valor que mas se repite por plataforma para modificar estos valores
# Posteriormente filtramos todas las plataformas con solo las que necesitamos rellenar con valores nulos

# In[20]:


moda_years = data.groupby('platform')['year_of_release'].apply(lambda x: x.mode().iloc[0]).reset_index()

moda_years_filtrado = moda_years[moda_years['platform'].isin(años_nulos)]

moda_years_filtrado


# Con una funcion rellenamos los años en funcion de la moda de cada plataforma calculado previamente

# In[21]:


def year_filled (row):
    if row['platform'] == '2600' and pd.isna(row['year_of_release']):
        return 1981.0
    if row['platform'] == '3DS' and pd.isna(row['year_of_release']):
        return 2011.0
    if row['platform'] == 'DS' and pd.isna(row['year_of_release']):
        return 2008.0
    if row['platform'] == 'GB' and pd.isna(row['year_of_release']):
        return 2000.0
    if row['platform'] == 'GBA' and pd.isna(row['year_of_release']):
        return 2002.0
    if row['platform'] == 'GC' and pd.isna(row['year_of_release']):
        return 2002.0
    if row['platform'] == 'N64' and pd.isna(row['year_of_release']):
        return 1999.0
    if row['platform'] == 'PC' and pd.isna(row['year_of_release']):
        return 2011.0
    if row['platform'] == 'PS' and pd.isna(row['year_of_release']):
        return 1998.0
    if row['platform'] == 'PS2' and pd.isna(row['year_of_release']):
        return 2002.0
    if row['platform'] == 'PS3' and pd.isna(row['year_of_release']):
        return 2011.0
    if row['platform'] == 'PSP' and pd.isna(row['year_of_release']):
        return 2006.0
    if row['platform'] == 'PSV' and pd.isna(row['year_of_release']):
        return 2015.0
    if row['platform'] == 'Wii' and pd.isna(row['year_of_release']):
        return 2009.0
    if row['platform'] == 'X360' and pd.isna(row['year_of_release']):
        return 2011.0
    if row['platform'] == 'XB' and pd.isna(row['year_of_release']):
        return 2003.0
    return row['year_of_release']

data['year_of_release'] = data.apply(year_filled, axis=1)


# Verificamos si quedaron valores nulos y volvemos a revisar los datos completos

# In[22]:


data[data['year_of_release'].isna()]
data.info()
data.head()


# Ahora cambiamos la columna de years_of_release para que no aparezca el .0 y lo dejamos como enteros

# In[23]:


data['year_of_release'] = data['year_of_release'].replace([np.inf, -np.inf], np.nan)
data['year_of_release'] = data['year_of_release'].astype(int)


# In[24]:


data.head()


# In[25]:


data.info()


# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Genial, buen comienzo Paolo.
#     
# Hiciste un analisis preeliminar exploratorio muy bueno, identificaste cada campo del dataset y su data type, agregaste una columna de ventas totales y chequeaste duplicados
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Necesita correcion. 
#     
# Ahora el tratamiento de los ausentes en las diferentes columnas precisa algunas correcciones
#     
# Las columnas nombre y genero traen muy pocos ausentes, no vale la pena trabajarlas con dropearlas esta bien
#     
# La columna de year of release es una con muchos ausentes, por un lado puedes chequear algunos juegos se lanzan en mas de una plataforma entonces ves en que ano fue y lo imputas. Pero algunos otros te recomiendo que tomes un estadistico como la moda agrupado por plataforma y tomes eso para imputar como ya han hecho en anteriores proyectos
#     
# Las columnas critic y user score, esta muy bien cambiar los tbd o 0s por np.nan, pero dada la cantidad y la poca correlacion con demas variables del dataset es mejor dejarlas como estas sin sesgar la distribucion. Ademas los datos faltantes son de un periodo de tiempo especifico o al menos hay un periodo de tiempo con una gran cantidad de valores de score ausentes en relacion a optros periodos
# </div>
# 
# <div class="alert alert-block alert-info">
# Gracias Matias. He cambiado mi codigo y he rellenado los valores ausentes de year_of_release calculando la moda como comentas. Se que hay varios videojuegos en distintas plataforma pero no he logrado buscar rellenar con esa informacion. 
# Las funciones de critic y user score las he eliminado, como bien dices no hay relacion con el resto de datos y no se deberia rellenar
# </div>
# 
# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor - 2da ITERACION</b> <a class="tocSkip"></a>
# 
# Buenas correcciones en esta seccion, bien hecho
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario de Revisor - 2da ITERACION</b> <a class="tocSkip"></a>
# 
# Te comparto una plantilla de codigo para simplificarte en proximos proyectos la imputacion
#     
# <code>
#     
# *calcula la moda de 'rating' para cada género*
#     - genre_rating_mode = df_datos.groupby('genre')['rating'].apply(lambda x: x.mode().iloc[0])
# 
# *reemplaza los valores ausentes en 'rating' con las modas correspondientes*
#     - for genre, mode_rating in genre_rating_mode.iteritems():
#         df_datos.loc[(df_datos['genre'] == genre) & (df_datos['rating'].isnull()), 'rating'] = mode_rating
# 
# </code>
# </div>

# ## Analizar los datos
# 
# 3.1. Graficamos los juegos lanzados por año y notamos que hay un pico en el año 2008 / 2009.
# 
# Los datos significativos son a aprtir del año 2000 con mas de 300 juegos lanzados a partir de ese año
# 

# In[26]:


juegos_lanzados_poraño = data.groupby(['year_of_release']).size().plot(kind='line')

juegos_lanzados_poraño

juegos_lanzados_poraño_singrafico = data.groupby(['year_of_release']).size()

juegos_lanzados_poraño_singrafico


# Creamos un nuevo dataset con los videojuegos a partir del año 2000

# In[27]:


data_significante = data[data['year_of_release'] > 2000]


# 3.2. Buscamos las plataformas mas populares en funcion de las ventas y creamos una lista con ella que se utilizara luego para filtra el dataset
# 

# In[28]:


ventas_plataformas_populares = list(data_significante.groupby(['platform'])['total_sales'].sum().reset_index().sort_values(by='total_sales', ascending=False).head(10)['platform'])
display(ventas_plataformas_populares)


# In[29]:


ventas_plataformas_populares
columna_a_comparar = 'platform'

data_filtrado = data_significante[data_significante['platform'].isin(ventas_plataformas_populares)]

columnas_seleccionadas = ['year_of_release', 'platform', 'total_sales']
data_filtrado = data_filtrado[columnas_seleccionadas]
data_filtrado


# Graficamos los histrogramas de las plataformas mas populares en funcion del año de lanzamiento y ventas
# 

# In[30]:


for platform, group in data_filtrado.groupby('platform'):
    plt.figure(figsize=(10, 4))
    plt.hist(group['year_of_release'], bins=10, alpha=0.6)
    plt.title(f"Histograma de Ventas Totales por Año para la Plataforma {platform}")
    plt.xlabel("Año")
    plt.ylabel("Ventas Totales")
    plt.grid(True)
    plt.show()


# Notamos que la unica plataforma de las populares en funcion de sus ingresos historicos que tienen una tendencia alcista es las PS4. El resto de plataformas tienden en los ultimos años a decrecer en funcion del tiempo.
# Por otro lado, es posible notar que el tiempo en años que se mantiene una plataforma varia entre 5 y 10 años. Siendo Play Station y Xbox las que tienden a mantenerse por el mayor tiempo y DS por ejemplo la que ha mantenido ingresos relevenates por 5 años.
# 

# 3.3. Analizando todos los graficos anteriores podemos decir que los datos relevenates son a partir del año 2000 donde empezo a hacerse popular la PS

# 3.4. Plataformas populares que en años recientes ya no tienen ventas o ventas casi nulas son GBA, PS, XB
# y Nintendo Wii.

# 3.5. La plataforma que ha mantenido ingresos altos contsantes en el tiempo ha sido la PS2 hasta el año 2008 aproximadamente en donde empezo a caer, lo mas evidente se puede atribuir al lanzamiento de la PS3 que deberia ser su sutitucion

# 3.6. Diagrama de caja para las ventas globales de todos los juegos, desglosados por plataforma.

# In[31]:


data_significante.query("platform in @ventas_plataformas_populares")[['platform','total_sales']].boxplot(by='platform', figsize=(10,8))
plt.ylim(0,2)


# Podemos ver en los diagramas de caja que las medianas de la PS3 y X360 son mayores al resto lo que significa que tenemos mayores valores aitpicos en el extremo superior, ventas mas dispersas. En cuanto a los cuatiles vemos que el 75% es mayor para X360 PS3 y PS4 al resto y casi iguales entre ellos. 

# 3.7. Grafico dispersion Reseñas de usuario VS Ventas Totales de una plataforma mas popular (PS2)

# In[32]:


ps_dos = ['PS2']

data_ps_dos = data_significante[data_significante['platform'].isin(ps_dos)]

data_ps_dos.plot(kind='scatter', x='critic_score', y='total_sales', figsize=(10, 8))  
plt.ylim(0,5)
plt.title('Gráfico de Dispersión: Reseñas Usuarios vs. Ventas Totales')
plt.xlabel('Reseñas')
plt.ylabel('Ventas')
plt.show()


# In[33]:


data_ps_dos[['critic_score','total_sales']].corr()


# En este caso podemos notar que hay una correlacion baja entre las dos variables

# 3.8. Analizamos las ventas de los videojuegos populares y contamos en cuantas plataformas estan

# In[34]:


ventas_totales_por_juego = data_significante.groupby('name')['total_sales'].sum().reset_index()
plataformas_por_juego = data_significante.groupby('name')['platform'].nunique().reset_index()
juegos_plataformas = pd.merge(ventas_totales_por_juego, plataformas_por_juego, on='name')
juegos_plataformas = juegos_plataformas[juegos_plataformas['platform'] > 1]
juegos_plataformas = juegos_plataformas.sort_values(by='total_sales', ascending=False)
juegos_plataformas.head()


# Seleccionamos el  juego con mas ventas en toda la historia y comparamos las ventas que ha tenido en las distintas plataformas

# In[35]:


juego_popular = data_significante.query("name == 'Grand Theft Auto V'")
juego_popular


# Grand Theft Auto V es el juego con mayores ventas de todos y su mayor parte de ingresos proviene de la PS3. Por otro lado los menores ingresos provienen de la plataforma PC

# In[36]:


juegos_mas_populares = data_significante[data_significante['name'].isin(['Grand Theft Auto V', 'Call of Duty: Black Ops', 'Call of Duty: Modern Warfare 3','Call of Duty: Ghosts','Call of Duty: Black Ops II'  ])]

juegos_mas_populares


# Grand Theft Auto Ves lider de ventas en dos plataformas y le sigue Call of Duty: Modern Warfare 3 en X360

# In[41]:


generos_unicos = data_significante['genre'].unique()

for genero in generos_unicos:
    subset_genero = data_significante[data_significante['genre'] == genero]
    plt.hist(subset_genero['total_sales'], bins=5, alpha=0.5, label=genero)

plt.xlim(0, 20)
plt.xlabel('Ventas Totales')
plt.ylabel('Frecuencia')
plt.legend()
plt.title('Histograma de Ventas Totales de Juegos Populares por Género')
plt.show()







# Vemos que el genero Action es el que mayor se repite dentro del rango de ventas hasta 5 millones de dolares. 
# El genero deportes es el que tiene mayor frecuencia y con mayores ingresos por ventas (16 millones)

# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Muy bien Paolo, analisaste de forma correcta la venta a traves del tiempo, las plataformas y juegos con mas ventas y la relacion criticas/ventas
#     
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Aun hace falta algunas correcciones
#     
# Principalmente en funcion del primer interrogante de las ventas a traves del tiempo desde los 1990 hasta la actualidad, deberias seleccionar un periodo representativo de la actualidad para solo analizar el resto de los interrogantes para ese subconjunto de datos
#     
# Luego, una vez filtrado el dataset o creado uno nuevo solo con esos datos, ahondar un poco mas en el ciclo de vida de las plataformas cuando duran vigentes, cuales son las que mas ventas tienen, que generos y que juegos, como se relacionan las criticas con las ventas ademas del scatterplot un calculo de coef de correlacion
#     
# Recuerda sumar a cada una de estos puntos visualizaciones para soportar tus hallazgos y docuentar tus conclusiones en celdas de markdown para detallar tu analisis
# </div>
# 
# <div class="alert alert-block alert-info">
# Gracias Matias. He cambiado mi codigo y he filtrado el dataset original a partir del año 2000, llamandolo data_significante y con ese nuevo he hecho los analisis.
# </div>
# 
# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor - 2da ITERACION</b> <a class="tocSkip"></a>
# 
# Buenas correcciones en esta seccion, bien hecho
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario de Revisor - 2da ITERACION</b> <a class="tocSkip"></a>
# 
# Si bien tu correccion es acertada, hubiera sido mejor tomar un periodo de anlisis aun mas acotado de no mas de 4 o 5 anos, eso serian datos suficientes
#     
# Ademas, si ves de 2000 a 2011 es un periodo con ventas muy altas en comparacion al periodo 2012-2016 incluso las plataformas vigentes en ambos periodos son diferentes
# </div>

# ## Analisis por region
# 
# Estudio por region (NA, UE, JP)
# 

# In[42]:


regiones = ['na_sales', 'eu_sales', 'jp_sales']

for region in regiones:
    plataformas_populares = data_significante.groupby('platform')[region].sum().reset_index().sort_values(by=region, ascending=False).head(5)
    print(f'Plataformas populares en {region}:')
    display(plataformas_populares[['platform', region]])
    print()


# 4.1. En el mercado americano la plataforma con mayores ventas es el X360, que en europa baja al 3 lugar y en japon no se observa dentro de los 5 primeros. A su vez vemos que la PS2 esta en el podio en las 3 regiones. las PS3 lidera en europa muy de cerca con la PS2 y el nintendo DS esta de primero en japon. Tambien destaca en general para los 5 juegos mas populares las mayores ventas se centran en america y las menores en japon.
# 

# In[43]:


for region in regiones:
    plataformas_populares = data_significante.groupby('genre')[region].sum().reset_index().sort_values(by=region, ascending=False).head(5)
    print(f'Generos populares en {region}:')
    display(plataformas_populares[['genre', region]])
    print()


# 4.2. Para los generos encontramos que el mas popular en america y europa coinciden con el de Action, siendo Role-Playing el mas popular en Japon el cual no aparece en la lista top de los dos anteriores y Action en 2do lugar
# 

# In[44]:


for region in regiones:
    plataformas_populares = data_significante.groupby('rating')[region].mean().reset_index().sort_values(by=region, ascending=False).head(5)
    print(f'Generos populares en {region}:')
    display(plataformas_populares[['rating', region]])
    print()


# 4.3. El promedio de los rating por region revela que en america y europa la califiacion AO genera mayores ingresos por juego. En cambio en japon E es la que mayor ingresos genera
# 

# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Buen trabajo caracterizando los perfiles de usuarios en cada region, sus preferencias basados en los datos
#     
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Primero, deberiamos usar solo los datos del periodo seleccionado en la seccion anterior para este analisis
#     
# Ademas, seria bueno sumar visualizaciones en vez de tablas sin formatos y quizas explayarte un poco mas en tu documentacion de hallazgos en cada analisis de la seccion
# </div>
# 
# <div class="alert alert-block alert-info">
# Gracias Matias. Actualizado la parte del dataset filtrado anteriormente. En caunto a la visualizacion he cambiado los prints por display asi se ve mejor la tabla
# </div>
# 
# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor - 2da ITERACION</b> <a class="tocSkip"></a>
# 
# Buenas correcciones en esta seccion, bien hecho
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario de Revisor - 2da ITERACION</b> <a class="tocSkip"></a>
# 
# Mas que mejorar la visualizacion de las tablas, que esta muy bien hayas modificado, me referia a sumar algunos graficos de barras apiladas por ejemplo
# </div>

# ## Prueba de Hipotesis
# 
# Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas y las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son diferentes.
# 
# Primero rellenamos los NaN de la columna user_score con cero ya que no tenemos formas de rellenarlo con el analisis de otras columnas
# 

# In[55]:


data_significante['user_score'].fillna(0, inplace=True)


# In[56]:


from scipy.stats import ttest_ind

alpha = 0.01

t_stat, p_value = ttest_ind(
    data_significante.query("platform == 'XOne'")['user_score'],
    data_significante.query("platform == 'PC'")['user_score']
)

print("Valor p:", p_value)

if p_value < alpha:
    print("Se rechaza la hipótesis nula")
else:
    print("No se rechaza la hipótesis nula")


# Para la primera hipotesis en donde buscamos validar que el promedio de las calificaciones de los usuarios para las plataformas Xbox One y PC son las mismas filtramos los datos con dos query por cada una de las plataformas para realizar el test con el promedio del user_score. El resultado resulto ser mucho menor a 0,01 y por eso descartamos la hipotesis nula.
# 

# In[57]:


from scipy.stats import ttest_ind

alpha = 0.01

t_stat, p_value = ttest_ind(
    data_significante.query("genre == 'Action'")['user_score'],
    data_significante.query("genre == 'Sports'")['user_score']
)

print("Valor p:", p_value)

if p_value < alpha:
    print("Se rechaza la hipótesis alternativa")
else:
    print("No se rechaza la hipótesis alternativa")


# Para las calificaciones promedio de los usuarios para los géneros de Acción y Deportes se buscaba validar la hipotesis de diferencia. El p value en este caso tambien fue mayor a 0.01 y no se descarta la hipotesis alternativa por lo que la calificacion de usuarios promedio para estos dos generos no es igual.

# <div class="alert alert-block alert-danger">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# Paolo preparaste bien las muestras para los test, pero deberias usar solo los scores para el periodo de analisis de la seccion anterior
#     
# Debes explicitamente definir las hipotest H0 y H1, y reevaluar el resultado del pvalue para concluir en ambos casos
#     
# Te comparto un recurso de bajo para que ahondes en este tema    
# </div>
# 
# <div class="alert alert-block alert-info">
# Gracias Matias. He actualizado mi codigo en base al comentario
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario de Revisor</b> <a class="tocSkip"></a>
# 
# **Ten en cuenta para futuros proyectos**    
#     
# Al hacer un test de hipotesis con T test estas preasumiendo 2 cuestiones claves
#     
#     - las varianzas de las muestras son iguales [si bien puedes setear el param eqal-var en false como hiciste]
#     
#     - ambas muestras tienen un distribucion normal
#     
# Ambas cosas pueden corroborarse con tests, y en caso de no cumplirse en vez de T test se puede usar algun test como el de MannWhitney
#     
# Te dejo este <a href='https://towardsdatascience.com/hypothesis-testing-with-python-step-by-step-hands-on-tutorial-with-practical-examples-e805975ea96e?gi=a640c3136da8'> recurso </a> para que revises para futuros proyectos
# </div>
# 
# <div class="alert alert-block alert-success">
# <b>Comentario de Revisor - 2da ITERACION</b> <a class="tocSkip"></a>
# 
# Buenas correcciones en esta seccion, muy bien hecho
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario de Revisor - 2da ITERACION</b> <a class="tocSkip"></a>
# 
# Solo a modo de comentario, un alpha de 0.01 es muuy exigente osea es un intervalo de confianza muy duro. En la practica se da mas usar 0.05 a menos que sea muy critica la distincion en ese caso si se justifica usar 0.01
# </div>

# ## Conclusion general
# 
# Para la elaboraci{on de este proyecto en primer lugar verificamos los datos suministrados para poder realizar los ajustes necesarios para su analisis, tales como modificar los nombres de las columnas, modificar los tipos de datos principalemte cambiar los de tipo objetvo a fecha y numeros, como tambien validar los datos nulos y duplicados. Se eliminaron algunas filas y se rellenaron datos en las columnas de year_of_release en base al promedio por categoria.
# 
# Posteriomente realizamos el analisis de datos en donde encontramos el pico de lanzamiento de videojuegos que fue duarnte los años 2008 y 2009. Seguidamente identificamos las plataformas mas populares en base a las ventas totales historicas y comparamos sus graficos. Destacamos que la data es relevente desde el año 2000 que empezaron a hacerse popular los videojuegos a nivel masivo. A lo largo de los años las diferentes versiones de Play Station han sido mucho mas populares que el resto, empezando por la PS y de los datos mas reciente la unica plataforma de las populares que tiene tendencia alsista es la PS4, que tiene logica al ser las mas nueva y que reemplaza las anteriores versiones.
# 
# De la PS2 que notamos mayores ventas sostenidas en los años analizamos la correlacion entre los ingresos y las calificaciones de los usuarios que nos dio un valor bajo. En este sentido podemos decir que los ingresos no necesariamente corresponden con las buenas o malas criticas. Es posible que esto se deba a cierta cantidad de videojuegos y no en la plataforma en si.
# 
# Tambien encontramos que Grand Theft Auto V es el juego con mayores ventas de todos y su mayor parte de ingresos proviene de la plataforma PS3. Por otro lado los menores ingresos provienen de la plataforma PC.
# 
# Del analisis de ingreso por region destaca que en america domina el XBOX 360, es europa la PS3 / PS2 y en japon la Nintendo DS. En cuanto a los generos de videojuegos america y europa prefieren los de action y japon role-playing.
# 
# Por ultimo, realizamos las pruebas de hipotesis de las calificaciones promedios entre dos plataformas (XOne y PC) y dos generos (Action y Sports). Para el primer caso se ha descartado la hipotesis de nula y para el segundo caso se ha aceptado la hipotesis alternativa.

# In[ ]:




