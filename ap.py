import streamlit as st
import streamlit.components.v1 as components
import Apriori
import metricas
import clustering
import clasificacion
import arboles
import avanzadas

def main(): 
	st.set_page_config(
	        page_title="AlGoRithm",
	        page_icon="🌼",
	        layout="wide",
	        initial_sidebar_state="auto"
	    )



	st.sidebar.title('Elige la sección que deseas probar')
	option=st.sidebar.selectbox('Modulos',('Página principal','Algoritmo Apriori',
			'Métricas de distancia','Módulo de Clustering','Clasificación con Regresión Logística',
			 'Módulo Árboles de decisión','Configuraciones avanzadas'))
	
	if option=='Página principal':

		st.title('AlGoRithm')
		st.markdown('Creada por Jaqueline Arias Sanguino ⭐')
		st.image("https://918429.smushcdn.com/2325059/wp-content/uploads/2021/03/mano-de-humano-toca-una-pantalla-que-tambien-toca-mano-robotica.jpg?lossy=1&strip=1&webp=1",width=1000)

		st.title('AlGoRithm')
		st.markdown('Bienvenido a la aplicación AlGoRithm, la aplicación que te permite obtener los beneficios de los algoritmos de Inteligencia Artificial de una manera sencilla, amigable e intuitiva. En esta aplicación tendrás acceso a un conjunto de módulos diferentes que te permiten probar distintos algoritmos')

		st.title('Módulos')
		st.subheader('**Algoritmo Apriori**')
		st.markdown('🟣 **Reglas de asociación**')
		st.subheader('**Módulo de Métricas de distancia**')
		st.markdown('🟣 **Métricas de distancia**')
		st.subheader('**Módulo de Clustering**')
		st.markdown('🟣 **Algoritmo de clúster jerárquico Ascendente**')
		st.markdown('🟣 **Algoritmo de clúster particional K-means**')
		st.subheader('**Módulo Clasificación**')
		st.markdown('🟣 **Regresión Logística**')
		st.subheader('**Módulo Árboles de decisión**')
		st.markdown('🟣 **Pronóstico**')
		st.markdown('🟣 **Clasificación**')

	if option=='Algoritmo Apriori':
		Apriori.start()
	if option=='Métricas de distancia':
		metricas.start()
	if option=='Módulo de Clustering':
		clustering.start()
	if option=='Clasificación con Regresión Logística':
		clasificacion.start()
	if option=='Módulo Árboles de decisión':
		arboles.start()
	if option=='Configuraciones avanzadas':
		avanzadas.start()



main()


  
