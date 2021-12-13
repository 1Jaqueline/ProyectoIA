import streamlit as st
import streamlit.components.v1 as components
from matplotlib import text
import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
import streamlit as st            # Para la generación de gráficas interactivas
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Para escalar los datos
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from mpl_toolkits.mplot3d import Axes3D
import avanzadas

def cargaCR():
	st.title('📁 Carga tu archivo')
	st.markdown('##### Características del archivo:')
	st.markdown('Para el funcionamiento eficiente del algoritmo, carga un archivo desde tu computadora que cumpla con las siguientes características')
	st.markdown('🟣 **Archivo con extensión .csv**')
	st.markdown('Esta extensión corresponde a un archivo con valores separados por comas')
	st.markdown('🟣 **Archivo con registros**')
	st.markdown('Estos deben ser registros que contengan las características que deseas evaluar')
	file = st.file_uploader("Carga tu archivo", type=["csv", "txt"], key='Clustering')
	return file


def start():
	st.set_option('deprecation.showPyplotGlobalUse', False) 
	st.title('Módulo de clustering')
	st.markdown('La IA aplicada en el análisis clústeres consiste en la segmentación y delimitación de grupos de objetos (elementos), que son unidos por características comunes que éstos comparten por medio de aprendizaje no supervisado). Su objetivo es dividir una población heterogénea de elementos en un número de grupos homogéneos, de acuerdo a sus similitudes. Los grupos nacen a partir de los datos y se proveen una serie de patrones ocultos en éstos.')

	st.image("https://www.researchgate.net/profile/Rafael-Torrealba/publication/278668507/figure/fig5/AS:668640991010834@1536427843576/3D-plot-of-clusters-obtained-from-the-k-means-algorithm.ppm",width=1000)
	st.markdown('#### **Este módulo cuenta con 2 algoritmos**')
	st.markdown('🟣 **Algoritmo de clúster jerárquico Ascendente**')
	st.markdown('🟣 **Algoritmo de clúster particional K-means**')


	st.header('🟣 Algoritmo de clúster jerárquico Ascendente')
	st.markdown('El algoritmo de **clustering jerárquico** organiza los elementos, de manera recursiva, en una estructura en forma de árbol que representa las relaciones de similitud entre los distintos elementos.')
	st.markdown('El algoritmo **ascendente jerárquico** consiste en agrupar en cada iteración aquellos 2 elementos más cercanos (clúster). De forma que se va construyendo una estructura en forma de árbol y el proceso concluye cuando se forma un único clúster.')
	datos = cargaCR()

	if datos is not None:		
		DatosTransacciones=pd.read_csv(datos)
		st.header("Datos cargados: ")
		st.dataframe(DatosTransacciones)

		st.header('Selección de características ')
		st.markdown('Para poder realizar un modelo eficiente es necesario trabajar con variables que aporten información significativa al modelo, para evitar redundancia innecesaria. Para poder realizar una selección de características de forma fundamentada empleamos una matriz de correlaciones, con el objetivo de seleccionar variables significativas')

		st.header('Evaluación visual')
		st.markdown('Para una apreciación visual de la correlación de las variables manejamos uná gráfica de dispersión que evalúa el conjunto de variables que tienen los datos.')
		variables=DatosTransacciones.columns
		variablesDF=st.dataframe(variables)
		option = st.selectbox('Elige la variable que deseas evaluar en el gráfico de dispersión',variables,index=9)
		st.write('**Variable seleccionada:**', option)

		verDisp = st.checkbox('Ver gráfico de dispersión de todas las variables')
		st.error('Considera que esto consume una cantidad considerable de tiempo. Se recomienda desactivar la opción para continuar con las siguientes instrucciones.')
		if verDisp:
			st.header('Gráfico de dispersión de todas las variables')
			disp=sns.pairplot(DatosTransacciones, hue=option)
			st.pyplot(disp)

		st.warning('También puedes obtener la gráfica de dispersión para dos diferentes variables, lo que tiene un menor costo de tiempo.')
		dos = st.checkbox('Obtener gráfica de dispersión de dos variables')
		if dos:
			st.header('Gráfico de dispersión de dos variables')
			optionDos = st.multiselect('Selecciona las dos variables que deseas comparar',variables)
			if len(optionDos)==2:
				st.set_option('deprecation.showPyplotGlobalUse', False) 
				st.write('**Variables seleccionadas:**', optionDos[0], optionDos[1])
				d1=optionDos[0][:]
				d2=optionDos[1][:]

				sns.scatterplot(x=d1, y =d2, data=DatosTransacciones, hue=option)
				plt.title('Gráfico de dispersión')
				plt.xlabel(optionDos[0])
				plt.ylabel(optionDos[1])
				st.pyplot()
			else:
				st.error('Selecciona unicamente 2 opciones')

		st.header('Coeficiente de Pearson')
		st.markdown('Este método nos permite obtener una matriz de correlación que nos brinda información para realizar una selección de variables de forma fundamentada.')
		st.markdown('Los intervalos sugeridos para la consideración de la correlación con el método de Pearson son los siguientes:')
		st.markdown('🟣 **De -1.0 a -0.67 y 0.67 a 1.0 se conocen como correlaciones fuertes o altas.**')
		st.markdown('🟣 **De -0.66 a -0.34 y 0.34 a 0.66 se conocen como correlaciones moderadas o medias.**')
		st.markdown('🟣 **De -0.33 a 0.0 y 0.0 a 0.33 se conocen como correlaciones débiles o bajas.**')

		st.header('Matriz de correlaciones')
		CorrDatos = DatosTransacciones.corr(method='pearson')
		st.dataframe(CorrDatos)

		MatrizInf = np.triu(CorrDatos)
		sns.heatmap(CorrDatos, cmap='RdBu_r', annot=True, mask=MatrizInf)

		fig=plt.figure(figsize=(14,7))
		MatrizInf = np.triu(CorrDatos)
		sns.heatmap(CorrDatos, cmap='RdBu_r', annot=True, mask=MatrizInf)
		st.header('Mapa de calor')
		plt.title('Mapa de calor')
		st.pyplot(fig)

		st.markdown('#### Con base en la información obtenida, ahora puedes seleccionar las variables con las que NO quieres trabajar.')
		st.info('Como recomendación, considera las variables que son indispensables para tu caso de estudio, sin que importe mucho si tiene o no una alta correlación. La selección de características no es arbitaria, analiza la importancia de las variables en tu caso de estudio y descarta aquellas que no aportan valor y tienen alta correlación.')
		variablesEliminadas = st.multiselect('Selecciona las variables con las que NO deseas trabajar',variables)

		listo = st.checkbox('Listo. He seleccionado las variables')
		if listo:

			for item in variablesEliminadas:
				DatosTransacciones = DatosTransacciones.drop(columns=[item])

			st.markdown('### **Variables seleccionadas**')
			st.dataframe(DatosTransacciones)

			st.markdown('Cuando se trabaja con clustering, dado que son algoritmos basados en distancias, es fundamental escalar los datos para que cada una de las variables contribuyan por igual en el análisis.')
			estandarizar = StandardScaler()
			MEstandarizada = estandarizar.fit_transform(DatosTransacciones)
			st.markdown('### **Matriz Estandarizada**')
			st.dataframe(MEstandarizada)

			algoJerarquico = st.checkbox('Aplicar el algoritmo Jerárquico Ascendente')
			if algoJerarquico:
				st.subheader('Aplicación del algoritmo Jerárquico Ascendente')

				metricas=['euclidean','chebyshev','cityblock','minkowski']
				metric = st.selectbox('Elige la métrica que deseas implementar en el algoritmo',metricas,index=0)

				arbol = st.checkbox('Obtener árbol')
				st.warning('Considera que la generación del árbol puede llevar un rato.')
				if arbol:
					plt.figure(figsize=(10, 7))
					plt.title("Caso de estudio")
					shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric=metric))
					st.pyplot()

				st.subheader('Obtención de los clústeres')
				st.markdown('Los clústeres son conjuntos de elementos que comparten características entre sí después del proceso de los datos de un conjunto general.')

				totalCluster=avanzadas.totalClusterC

				MJerarquico = AgglomerativeClustering(n_clusters=totalCluster, linkage='complete', affinity='euclidean')
				MJerarquico.fit_predict(MEstandarizada)

				DatosTransacciones['clusterH'] = MJerarquico.labels_
				st.dataframe(DatosTransacciones)

				st.subheader('La cantidad de elementos que tiene cada clúster identificado')
				#Cantidad de elementos en los clusters
				cuenta=st.dataframe(DatosTransacciones.groupby(['clusterH'])['clusterH'].count())

				verCluster = st.checkbox('Ver elementos que pertenecen a cada clúster')
				if verCluster:
					i=0
					while i<totalCluster:
						st.info('Elementos del clúster número '+str(i))
						st.dataframe(DatosTransacciones[DatosTransacciones.clusterH==i])
						i=i+1

				
				obtCentroidesJ = st.checkbox('Obtener centroides')
				if obtCentroidesJ:
					st.subheader('Obtención de los centroides')
					st.markdown('El centroide es el punto que ocupa la posición media en un cluster, y nos brindan un valor promedio de las características de un conjunto de elementos que pertenecen al mismo clúster')
					st.dataframe(DatosTransacciones.groupby('clusterH').mean())

				infoCluster = st.checkbox('Ver interpretación de cada clúster')
				if infoCluster:
					CentroidesH = DatosTransacciones.groupby('clusterH').mean()
					for i in range(totalCluster):
						st.subheader("Clúster "+str(i))
						st.info('El clúster número '+str(i)+ ' se conforma por un total de '+str(DatosTransacciones.groupby(['clusterH'])['clusterH'].count()[i])+' elementos con las siguientes características en promedio:')
						st.table(CentroidesH.iloc[i])

						st.info('Con esta información es posible que tú realices una interpretación más particular y especializada')

			algoKmeans = st.checkbox('Aplicar el algoritmo K-means')
			if algoKmeans:
				st.subheader('Aplicación del algoritmo K-means')
				st.markdown('Un algoritmo particional organiza los elementos dentro de k clústeres. El algoritmo K-means es uno de los algoritmos utilizados en la industria para crear k clústeres a partir de un conjunto de elementos, de modo que los miembros de un grupo sean similares.')

				st.subheader('Elbow Method. Definición de k clusters para K-means')
				st.markdown('El método del codo es un método para decidir la cantidad de grupos que se pueden crear para la división de los datos. En el método del codo se traza una curva en la que el punto de inflexión se considera como un indicador del número adecuado de grupos. Con ayuda de este mecanismo se realiza la selección de los grupos que utiliza este programa')

				#Se utiliza random_state para inicializar el generador interno de números aleatorios

				SSE = []
				for i in range(2, 12):
				    km = KMeans(n_clusters=i, random_state=0)
				    km.fit(MEstandarizada)
				    SSE.append(km.inertia_)

				#Se grafica SSE en función de k
				plt.figure(figsize=(10, 7))
				plt.plot(range(2, 12), SSE, marker='o')
				plt.xlabel('Cantidad de clusters *k*')
				plt.ylabel('SSE')
				plt.title('Elbow Method')
				st.pyplot()

				st.warning('Con esta información puedes elegir el número de clústeres en el lugar donde aprecies que existe un codo afilado en la gráfica por el método del codo.')
				numClus=st.slider('Ingresa el número de clústeres que quieres generar', min_value=1, max_value=15, value=4, step=1)
				st.write('Cantidad de clústeres: '+str(numClus))

				st.error('En caso de que la gráfica del método del codo no sea de muy clara, puedes pedir ayuda.')
				ayuda = st.checkbox('Ayuda para identificar el codo')
				

				if ayuda:
					kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
					codo=kl.elbow
					st.success('La posición donde se encuentra el codo es: '+str(codo)+'. Se recomienda emplear esa cantidad de clústeres')	
					plt.style.use('ggplot')
					kl.plot_knee()
					st.pyplot()



				aply = st.checkbox('Listo. He seleccionado el número de clústeres')
				if aply:

					MParticional = KMeans(n_clusters=numClus, random_state=0).fit(MEstandarizada)
					MParticional.predict(MEstandarizada)

					DatosTransacciones['clusterP']=MParticional.labels_
					st.dataframe(DatosTransacciones)
					st.subheader('La cantidad de elementos que tiene cada clúster identificado')
					#Cantidad de elementos en los clusters
					st.dataframe(DatosTransacciones.groupby(['clusterP'])['clusterP'].count())
					verClusterP = st.checkbox('P. Ver elementos que pertenecen a cada clúster')
					if verClusterP:
						i=0
						while i<numClus:
							st.info('Elementos del clúster número '+str(i))
							st.dataframe(DatosTransacciones[DatosTransacciones.clusterP==i])
							i=i+1
					
					obtCentroides = st.checkbox('P. Obtener centroides')
					if obtCentroides:
						st.subheader('Obtención de los centroides')
						st.markdown('El centroide es el punto que ocupa la posición media en un cluster, y nos brindan un valor promedio de las características de un conjunto de elementos que pertenecen al mismo clúster')
						st.dataframe(DatosTransacciones.groupby('clusterP').mean())

					infoClusterP = st.checkbox('P. Ver interpretación de cada clúster')
					if infoClusterP:
						CentroidesP = DatosTransacciones.groupby('clusterP').mean()
						for i in range(numClus):
							st.subheader("Clúster "+str(i))
							st.info('El clúster número '+str(i)+ ' se conforma por un total de '+str(DatosTransacciones.groupby(['clusterP'])['clusterP'].count()[i])+' elementos con las siguientes características en promedio:')
							st.table(CentroidesP.iloc[i])

						st.warning('Con esta información es posible que tú realices una interpretación más particular y especializada')

					g3d = st.checkbox('Ver gráfica de los elementos en los clústeres en 3D')
					if g3d:
						try:
							st.header("Gráfica 3D de los clústeres generados: ")
							plt.rcParams['figure.figsize'] = (10, 7)
							plt.style.use('ggplot')

							if numClus == 3:
								colores=['red', 'blue', 'green']
							elif numClus == 4:
								colores=['red', 'blue', 'green', 'yellow']
							else:
								colores=['red', 'blue', 'green', 'yellow', 'pink']
		
							asignar=[]
							for row in MParticional.labels_:
								asignar.append(colores[row])

							fig = plt.figure()
							ax = Axes3D(fig)
							ax.scatter(MEstandarizada[:, 0], 
                                    MEstandarizada[:, 1], 
                                    MEstandarizada[:, 2], marker='o', c=asignar, s=60)
							ax.scatter(MParticional.cluster_centers_[:, 0], 
                                    MParticional.cluster_centers_[:, 1], 
                                    MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
							st.pyplot()
						except:
							st.error("No fue posible obtener la gráfica en 3D")
					








						










			

