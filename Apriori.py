import streamlit as st
import streamlit.components.v1 as components

def cargaRA():
	st.title('📁 Carga tu archivo')
	st.markdown('##### Características del archivo:')
	st.markdown('Para el funcionamiento eficiente del algoritmo, carga un archivo desde tu computadora que cumpla con las siguientes características')
	st.markdown('🟣 **Archivo con extensión .csv**')
	st.markdown('Esta extensión corresponde a un archivo con valores separados por comas')
	st.markdown('🟣 **Archivo con datos transaccionales**')
	st.markdown('Estos deben ser datos transaccionales obtenidos de operaciones realizadas en tu negocio')
	file = st.file_uploader("Carga tu archivo", type=["csv", "txt"], key='RA')
	return file

def start():
	import pandas as pd                 # Para la manipulación y análisis de los datos
	import numpy as np                  # Para crear vectores y matrices n dimensionales
	import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
	from apyori import apriori
	st.title('Algoritmo Apriori: Reglas de asociación')
	st.markdown('##### Ofrece a tus clientes lo que están buscando')
	st.markdown('**Descripción**: Este algoritmo es muy útil porque ofrece una herramienta para ofrecerte un conjunto de reglas de asociación para poder identificar si una compra se puede relacionar con otras y así poder tener una mejor experiencia en tu negocio. Con las estrategias que puedes obtener implementando este algoritmo puedes incrementar tus ventas considerablente, siempre tratando de hacer ventas inteligentes que le interesen a tus clientes. Recuerda que mantener su interés en tus ventas es el camino a seguir para mantener el éxito de tu negocio.')
	st.image("https://www.tecnologias-informacion.com/das.png",width=1000)
	datos = cargaRA()

	if datos is not None:		
		DatosTransacciones=pd.read_csv(datos, header=None)
		st.header("Datos cargados: ")
		st.dataframe(DatosTransacciones)
		st.markdown('**Nota**:\t **<NA>** \tindica que ese elemento no participó en esa transacción.')

		st.title('Procesamiento de datos')
		st.markdown('#### Lista de elementos:')
		Transacciones=DatosTransacciones.values.reshape(-1).tolist()
		Lista=pd.DataFrame(Transacciones)
		Lista['Frecuencia']=1
		Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
		Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
		Lista = Lista.rename(columns={0 : 'Item'})
		st.dataframe(Lista)


		graficaView = st.checkbox('Ver gráfica de elementos')
		if graficaView:
			grafica = plt.figure(figsize=(16,20), dpi=300)
			plt.title('Elementos en función de la frecuencia')
			plt.ylabel('Elementos')
			plt.xlabel('Frecuencia')
			plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='blue')
			st.pyplot(grafica)

		TransaccionesLista = DatosTransacciones.stack().groupby(level=0).apply(list).tolist()


		st.title('Aplicación del algoritmo')
		st.markdown('Para comenzar con la aplicación del algoritmo, selecciona los parámetros que deseas configurar para obtener las reglas de asociación')
		st.markdown('🟣 **Soporte.** Indica la importancia que deben de tener las reglas que quieres obtener del conjunto de transacciones')

		st.markdown('🟣 **Confianza.** Indica la fiabilidad que deben de tener las reglas que quieres obtener del conjunto de transacciones')
		

		st.markdown('🟣 **Elevación (Lift).** Indica el nivel de relación que deseas que tengan los elementos en las reglas de asociación')
		st.warning('**Nota:** Recuerda que nos interesan las reglas que tengan un lift > 1 porque eso indica que existe un aumento en la posibilidad de que se repita la transacción')

		soporte=st.slider('Soporte', min_value=0.0, max_value=5.0, value=1.0, step=0.01)
		st.write('Soporte: '+str(soporte)+' %')
		confianza=st.slider('Confianza', min_value=10, max_value=50, value=30, step=5)
		st.write('Confianza: '+str(confianza)+' %')
		lift=st.slider('Elevación', min_value=0, max_value=5, value=2, step=1)
		st.write('Elevación: '+str(lift))

		ReglasC1 = apriori(TransaccionesLista,
			min_support=soporte/100,
			min_confidence=confianza/100,
			min_lift=lift)

		ResultadosC1=list(ReglasC1)
		R1 = list(ResultadosC1[0])
		#st.write(R1)
		#st.write(ResultadosC1)
		st.write('**Cantidad de Reglas encontradas: **',str(len(ResultadosC1))+' reglas')
		cantidad=len(ResultadosC1)

		if cantidad==0:
			st.error('No se han encontrado reglas de asociación. Intenta con otra configuración de parámetros o con otro conjunto de valores.')
		else: #Para imprimir las reglas
			st.success('Reglas encontradas con éxito')
			reglasView = st.checkbox('Ver reglas')
			if reglasView:
				col1, col2, col3, col4, col5, col6 = st.columns([0.1,1,.3,.4,.34,1.5]) #Definimos el tamaño de cada columna
				with st.container():
					col1.subheader("No.")
					col2.subheader("Regla")
					col3.subheader("Soporte")
					col4.subheader("Confianza")
					col5.subheader("Elevacion")
					col6.subheader("Análisis")

					for item in ResultadosC1:
						with col1:
							#Número de la regla
							st.info(str(ResultadosC1.index(item)+1))
						with col2:
							#Contenido de la regla
							st.warning(str(", ".join(item[0])))
						with col3:
							#Soporte de la regla
							st.warning(str(round(item[1]*100,2))+" %")
						with col4:
							#Confianza de la regla
							st.warning(str(round(item[2][0][2]*100,2))+" %")
						with col5:
							#Elevación de la regla
							st.warning(str(round(item[2][0][3],2))) 
						with col6:
							#Análisis de la regla
							st.success('Existe '+str(round(item[2][0][3]))+' veces más posibilidades de que los elementos se compren juntos')

		st.subheader("Con esta información ya tienes una idea de la forma en que puedes acomodar tus productos. \nRecuerda hacerlo de forma estratégica, coloca artículos que puedan ser de interés para tus clientes basándote en los elementos que se suelen comprar juntos. \nPuedes probar con diferentes configuraciones y obtener diferentes reglas.")
                
		
                    



