import streamlit as st
import streamlit.components.v1 as components
from scipy.spatial import distance
import avanzadas

def cargaMD():
	st.title('📁 Carga tu archivo')
	st.markdown('##### Características del archivo:')
	st.markdown('Para el funcionamiento eficiente del algoritmo, carga un archivo desde tu computadora que cumpla con las siguientes características')
	st.markdown('🟣 **Archivo con extensión .csv**')
	st.markdown('Esta extensión corresponde a un archivo con valores separados por comas')
	st.markdown('🟣 **Archivo con datos numéricos**')
	st.markdown('Estos deben ser datos numéricos de los elementos de los cuales deseas obtener las distancias')
	file = st.file_uploader("Carga tu archivo", type=["csv", "txt"], key='Metricas')
	return file

def calculaAcotada():
	st.success('Yay')


def start():
	import pandas as pd                         # Para la manipulación y análisis de datos
	import numpy as np                          # Para crear vectores y matrices n dimensionales
	import matplotlib.pyplot as plt             # Para generar gráficas a partir de los datos
	from scipy.spatial.distance import cdist    # Para el cálculo de distancias

	st.title('Módulo: Métricas de Distancia')
	st.markdown('##### Obten la matriz de distancias a partir de un conjunto de elementos, con diferentes medidas de distancia:')
	st.markdown('🟣 **Distancia Euclidiana (Euclídea)**')
	st.markdown('🟣 **Distancia de Chebyshev**')
	st.markdown('🟣 **Distancia de Manhattan (Geometría del taxista)**')
	st.markdown('🟣 **Distancia de Minkowsky**')
	st.markdown('**Descripción**: Las métricas de distancia son una herramienta muy útil que implementan muchos de los algoritmos de IA para poder identificar elementos que comparten características en común y sus diferencias. Para obtener una mayor eficiencia, es necesario identificar qué medida de distancia utilizar para obtener modelos más precisos. Este módulo te permite obtener la diferencia entre estas métricas, a forma de herramienta que te permita elegir fundamentadamente. ')
	st.image("https://blogs.iadb.org/conocimiento-abierto/wp-content/uploads/sites/10/2018/05/calcular-distancias-banner-2.jpg",width=1000)
	datosMD = cargaMD()



	st.header('📐 Distancia Euclidiana (Euclídea)')
	st.markdown('**Descripción**: La distancia euclideana es una de las métricas más utilizadas para calcular la distancia entre dos puntos. Sus bases se encuentran en la aplicación del Teorema de Pitágoras, donde la distancia viene a ser la longitud de la hipotenusa.')
	if datosMD is not None:		
		Hipoteca=pd.read_csv(datosMD)

		st.header("Datos cargados: ")
		st.dataframe(Hipoteca)

		DstEuclidiana = cdist  (Hipoteca, Hipoteca, metric = 'euclidean')
		#MEuclidiana = pd.DataFrame(DstEuclidiana)

		st.header("Matriz de distancia euclideana: ")
		st.dataframe(DstEuclidiana)

		st.warning('Puedes acotar la matriz de distancia haciendo una selección de los elementos que quieres obtener de la matriz.')
		acotar = st.checkbox('Acotar matriz')
		if acotar:
			st.markdown('**Ingresa la cantidad de elementos que deseas obtener en la matriz**')
			y=0
			y=st.slider('E. Elementos:', min_value=1, max_value=202, value=0, step=1)
			st.write('Elementos: '+str(y))

			if y>0:
				DstEuclidiana = cdist  (Hipoteca.iloc[0:y], Hipoteca.iloc[0:y], metric = 'euclidean')
				st.header("Matriz de distancia euclideana acotada: ")
				st.markdown('**Elementos en el rango [0:**'+str(y)+'**]**')
				st.dataframe(DstEuclidiana)

		st.info('Puedes elegir dos elementos en específico para obtener la distancia entre ellos.')
		seleccion = st.checkbox('Elegir 2 elementos')
		if seleccion:
			st.markdown('**Selecciona los dos elementos de los cuales deseas obtener la distancia**')
			elemento1=-1
			elemento2=-1
			elemento1=st.slider('Límite inferior', min_value=0, max_value=201, value=0, step=1)
			st.write('Elemento 1: '+str(elemento1))
			elemento2=st.slider('Límite superior', min_value=1, max_value=201, value=0, step=1)
			st.write('Elemento 2: '+str(elemento2))

			if elemento1>=0 and elemento2>=0:
				Objeto1 = Hipoteca.iloc[elemento1]
				Objeto2 = Hipoteca.iloc[elemento2]
				dstEuclidiana = distance.euclidean(Objeto1,Objeto2)
				st.success('La distancia euclideana entre el elemento '+str(elemento1)+' y el elemento '+str(elemento2)+' es: '+str(dstEuclidiana))


	st.header('📐 Distancia Chebyshev')
	st.markdown('**Descripción**: La distancia Chebyshev es el valor máximo absoluto de las diferencias entre las coordenadas de un par de elementos.')
	if datosMD is not None:		
		
		DstChebyshev = cdist  (Hipoteca, Hipoteca, metric = 'chebyshev')

		st.header("Matriz de distancia Chebyshev: ")
		st.dataframe(DstChebyshev)

		st.warning('Puedes acotar la matriz de distancia haciendo una selección de los elementos que quieres obtener de la matriz.')
		acotarC = st.checkbox('Acotar matriz Chebyshev')
		if acotarC:
			st.markdown('**Ingresa la cantidad de elementos que deseas obtener en la matriz**')
			yC=0
			yC=st.slider('C. Elementos:', min_value=1, max_value=202, value=0, step=1)
			st.write('Elementos: '+str(yC))

			if yC>0:
				DstChebyshev = cdist  (Hipoteca.iloc[0:yC], Hipoteca.iloc[0:yC], metric = 'chebyshev')
				st.header("Matriz de distancia Chebyshev acotada: ")
				st.markdown('**Elementos en el rango [0:**'+str(yC)+'**]**')
				st.dataframe(DstChebyshev)


		st.info('Puedes elegir dos elementos en específico para obtener la distancia entre ellos.')
		seleccion = st.checkbox('Elegir 2 elementos para obtener distancia Chebyshev')
		if seleccion:
			st.markdown('**Selecciona los dos elementos de los cuales deseas obtener la distancia Chebyshev**')
			elemento1C=-1
			elemento2C=-1
			elemento1C=st.slider('Límite inferior ', min_value=0, max_value=201, value=5, step=1)
			st.write('Elemento 1: '+str(elemento1C))
			elemento2C=st.slider('Límite superior ', min_value=1, max_value=201, value=3, step=1)
			st.write('Elemento 2: '+str(elemento2C))

			if elemento1C>=0 and elemento2C>=0:
				Objeto1 = Hipoteca.iloc[elemento1C]
				Objeto2 = Hipoteca.iloc[elemento2C]
				dstChebyshev = distance.chebyshev(Objeto1,Objeto2)
				st.success('La distancia Chebyshev entre el elemento '+str(elemento1C)+' y el elemento '+str(elemento2C)+' es: '+str(dstChebyshev))


	st.header('📐 Distancia Manhattan')
	st.markdown('**Descripción**: La distancia euclidiana es una buena métrica. Sin embargo, en la vida real, por ejemplo es imposible moverse siempre de un punto a otro de manera recta. Se utiliza la distancia de Manhattan si se necesita calcular la distancia entre dos puntos en una ruta similar a una cuadrícula.')
	if datosMD is not None:		
		
		DstManhattan = cdist  (Hipoteca, Hipoteca, metric = 'cityblock')

		st.header("Matriz de distancia Manhattan: ")
		st.dataframe(DstManhattan)

		st.warning('Puedes acotar la matriz de distancia haciendo una selección de los elementos que quieres obtener de la matriz.')
		acotarM = st.checkbox('Acotar matriz Manhattan')
		if acotarM:
			st.markdown('**Ingresa la cantidad de elementos que deseas obtener en la matriz**')
			yM=0
			yM=st.slider('M. Elementos:', min_value=1, max_value=202, value=0, step=1)
			st.write('Elementos: '+str(yM))

			if yM>0:
				DstManhattan = cdist  (Hipoteca.iloc[0:yM], Hipoteca.iloc[0:yM], metric = 'cityblock')
				st.header("Matriz de distancia Manhattan acotada: ")
				st.markdown('**Elementos en el rango [0:**'+str(yM)+'**]**')
				st.dataframe(DstManhattan)

		st.info('Puedes elegir dos elementos en específico para obtener la distancia entre ellos.')
		seleccion = st.checkbox('Elegir 2 elementos para obtener distancia Manhattan')
		if seleccion:
			st.markdown('**Selecciona los dos elementos de los cuales deseas obtener la distancia Manhattan**')
			elemento1M=-1
			elemento2M=-1
			elemento1M=st.slider('M. Límite inferior ', min_value=0, max_value=201, value=5, step=1)
			st.write('Elemento 1: '+str(elemento1M))
			elemento2M=st.slider('M. Límite superior ', min_value=1, max_value=201, value=3, step=1)
			st.write('Elemento 2: '+str(elemento2M))

			if elemento1M>=0 and elemento2M>=0:
				Objeto1 = Hipoteca.iloc[elemento1M]
				Objeto2 = Hipoteca.iloc[elemento2M]
				dstManhattan = distance.cityblock(Objeto1,Objeto2)
				st.success('La distancia Manhattan entre el elemento '+str(elemento1M)+' y el elemento '+str(elemento2M)+' es: '+str(dstManhattan))

	st.header('📐 Distancia Minkowski')
	st.markdown('**Descripción**: La distancia Minkowski es una distancia entre dos puntos en un espacio n-dimensional. Es una métrica de distancia generalizada: Euclidiana, Manhattan y Chebyshev. ')
	st.markdown('Esta métrica permite calcular la distancia de tres formas diferentes, en función del valor de **lambda**, que define el orden para las 3 diferentes métricas que conocemos. Los valores se definen de la siguiente forma')
	st.markdown('🟣 **λ=1. Distancia Manhattan**')
	st.markdown('🟣 **λ=2. Distancia Euclidiana**')
	st.markdown('🟣 **λ=3. Distancia de Chebyshev**')
	st.markdown('**Actualmente se suelen emplear valores intermedios, como **λ=1.5** que proporciona un equilibrio entre las medidas. Este es el valor por default en el programa.**')
	st.info('**Nota**: El algoritmo está configurado de forma genérica para ofrecerte un resultado útil, pero si deseas cambiar la configuración predeterminada para los parámetros con que trabajan los algoritmos, dirigete a la sección de configuaciones avanzadas antes de seguir. **Esta opción es recomendada únicamente para usuarios expertos**, ya que puede influir en el funcionamiento y desempeño del algoritmo, y si se realiza de forma incorrecta puede afectar a los resultados. ')
	
	if datosMD is not None:		
		
		DstMinkowski = cdist (Hipoteca, Hipoteca, metric = 'minkowski', p=avanzadas.lamb)

		st.header("Matriz de distancia Minkowski: ")
		st.dataframe(DstMinkowski)

		st.warning('Puedes acotar la matriz de distancia haciendo una selección de los elementos que quieres obtener de la matriz.')
		acotarMk = st.checkbox('Acotar matriz Minkowski')
		if acotarMk:
			st.markdown('**Ingresa la cantidad de elementos que deseas obtener en la matriz**')
			yMk=0
			yMk=st.slider('Mk. Elementos:', min_value=1, max_value=202, value=0, step=1)
			st.write('Elementos: '+str(yMk))

			if yMk>0:
				DstMinkowski = cdist  (Hipoteca.iloc[0:yMk], Hipoteca.iloc[0:yMk], metric = 'minkowski', p=1.5)
				st.header("Matriz de distancia Minkowski acotada: ")
				st.markdown('**Elementos en el rango [0:**'+str(yMk)+'**]**')
				st.dataframe(DstMinkowski)

		st.info('Puedes elegir dos elementos en específico para obtener la distancia entre ellos.')
		seleccion = st.checkbox('Elegir 2 elementos para obtener distancia Minkowski')
		if seleccion:
			st.markdown('**Selecciona los dos elementos de los cuales deseas obtener la distancia Minkowski**')
			elemento1Mk=-1
			elemento2Mk=-1
			elemento1Mk=st.slider('M. Elemento 1 ', min_value=0, max_value=201, value=5, step=1)
			st.write('Elemento 1: '+str(elemento1Mk))
			elemento2Mk=st.slider('M. Elemento 2 ', min_value=1, max_value=201, value=3, step=1)
			st.write('Elemento 2: '+str(elemento2Mk))

			if elemento1Mk>=0 and elemento2Mk>=0:
				Objeto1 = Hipoteca.iloc[elemento1Mk]
				Objeto2 = Hipoteca.iloc[elemento2Mk]
				dstMinkowski = distance.minkowski(Objeto1,Objeto2, p=lamb)
				st.success('La distancia Minkowski entre el elemento '+str(elemento1Mk)+' y el elemento '+str(elemento2Mk)+' es: '+str(dstMinkowski))
		

	
