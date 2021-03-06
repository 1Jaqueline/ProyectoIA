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
	        page_icon="馃尲",
	        layout="wide",
	        initial_sidebar_state="auto"
	    )



	st.sidebar.title('Elige la secci贸n que deseas probar')
	option=st.sidebar.selectbox('Modulos',('P谩gina principal','Algoritmo Apriori',
			'M茅tricas de distancia','M贸dulo de Clustering','Clasificaci贸n con Regresi贸n Log铆stica',
			 'M贸dulo 脕rboles de decisi贸n','Configuraciones avanzadas'))
	
	if option=='P谩gina principal':

		st.title('AlGoRithm')
		st.markdown('Creada por Jaqueline Arias Sanguino 猸?')
		st.image("https://918429.smushcdn.com/2325059/wp-content/uploads/2021/03/mano-de-humano-toca-una-pantalla-que-tambien-toca-mano-robotica.jpg?lossy=1&strip=1&webp=1",width=1000)

		st.title('AlGoRithm')
		st.markdown('Bienvenido a la aplicaci贸n AlGoRithm, la aplicaci贸n que te permite obtener los beneficios de los algoritmos de Inteligencia Artificial de una manera sencilla, amigable e intuitiva. En esta aplicaci贸n tendr谩s acceso a un conjunto de m贸dulos diferentes que te permiten probar distintos algoritmos')

		st.title('M贸dulos')
		st.subheader('**Algoritmo Apriori**')
		st.markdown('馃煟 **Reglas de asociaci贸n**')
		st.subheader('**M贸dulo de M茅tricas de distancia**')
		st.markdown('馃煟 **M茅tricas de distancia**')
		st.subheader('**M贸dulo de Clustering**')
		st.markdown('馃煟 **Algoritmo de cl煤ster jer谩rquico Ascendente**')
		st.markdown('馃煟 **Algoritmo de cl煤ster particional K-means**')
		st.subheader('**M贸dulo Clasificaci贸n**')
		st.markdown('馃煟 **Regresi贸n Log铆stica**')
		st.subheader('**M贸dulo 脕rboles de decisi贸n**')
		st.markdown('馃煟 **Pron贸stico**')
		st.markdown('馃煟 **Clasificaci贸n**')

	if option=='Algoritmo Apriori':
		Apriori.start()
	if option=='M茅tricas de distancia':
		metricas.start()
	if option=='M贸dulo de Clustering':
		clustering.start()
	if option=='Clasificaci贸n con Regresi贸n Log铆stica':
		clasificacion.start()
	if option=='M贸dulo 脕rboles de decisi贸n':
		arboles.start()
	if option=='Configuraciones avanzadas':
		avanzadas.start()



main()


  
