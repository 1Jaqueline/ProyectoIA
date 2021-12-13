import pandas as pd               # Para la manipulación y análisis de datos
import numpy as np                # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
#%matplotlib inline 
import streamlit as st            # Para la generación de gráficas interactivas
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection
import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection
def cargaA():
	st.title('📁 Carga tu archivo')
	st.markdown('##### Características del archivo:')
	st.markdown('Para el funcionamiento eficiente del algoritmo, carga un archivo desde tu computadora que cumpla con las siguientes características')
	st.markdown('🟣 **Archivo con extensión .csv**')
	st.markdown('Esta extensión corresponde a un archivo con valores separados por comas')
	st.markdown('🟣 **Archivo con registros**')
	st.markdown('Estos deben ser registros que contengan las características que deseas evaluar')
	file = st.file_uploader("Carga tu archivo", type=["csv", "txt"], key='arboles')
	return file


def start():
	st.title('Módulo: Árboles de decisión')
	st.markdown('Los árboles de decisión son un algoritmos ampliamente usado cuyo objetivo es construir una estructura jerárquica eficiente y escalable que divide los datos en función de determinadas condiciones, trabajando con la estrategia: divide y vencerás.')

	st.image("https://static.wixstatic.com/media/a27d24_84d8e5d1d05f415e8e4945ed69409837~mv2.jpg/v1/fill/w_724,h_483,al_c,q_90/a27d24_84d8e5d1d05f415e8e4945ed69409837~mv2.jpg",width=1000)
	st.markdown('#### **Este módulo cuenta con 2 algoritmos**')
	st.markdown('🟣 **Árbol de Decisión para Pronóstico**')
	st.markdown('🟣 **Árbol de decisión para Clasificación**')
	datos = cargaA()

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
		

		st.warning('También puedes obtener la gráfica de dispersión para dos diferentes variables, lo que tiene un menor costo de tiempo.')
		dos = st.checkbox('Obtener gráfica de dispersión de dos variables')
		st.error('Considera que esto consume una cantidad considerable de tiempo. Se recomienda desactivar la opción para continuar con las siguientes instrucciones.')
		if dos:
			option = st.selectbox('Elige la variable que deseas evaluar en el gráfico de dispersión',variables,index=1)
			st.write('**Variable seleccionada:**', option)
			st.header('Gráfico de dispersión de dos variables')
			optionDos = st.multiselect('Selecciona las dos variables que deseas comparar',variables)
			if len(optionDos)==2:
				st.set_option('deprecation.showPyplotGlobalUse', False) 
				st.write('**Variables seleccionadas:**', optionDos[0], optionDos[1])
				d1=optionDos[0][:]
				d2=optionDos[1][:]

				plt.figure(figsize=(20, 5))
				plt.plot(DatosTransacciones[d1], DatosTransacciones[d2], color='green', marker='o', label=option)
				plt.xlabel(d1)
				plt.ylabel(d2)
				plt.title(option)
				plt.grid(True)
				plt.legend()
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
		Pronos = st.checkbox('Trabajar con Árbol de Decisión para Pronóstico')
		if Pronos: #11
			st.subheader('Aplicación de Árbol de Decisión para Pronóstico')
			
			st.markdown('#### Con base en la información obtenida, ahora puedes seleccionar las variables con las que quieres trabajar.')
			st.info('Como recomendación, considera las variables que son indispensables para tu caso de estudio, sin que importe mucho si tiene o no una alta correlación. La selección de características no es arbitaria, analiza la importancia de las variables en tu caso de estudio y descarta aquellas que no aportan valor y tienen alta correlación.')
			variablesPredictoras = st.multiselect('Selecciona las variables con las que deseas trabajar. Estas se convierten en las variables que van a predecir la variable deseada',variables)

			listo = st.checkbox('Listo. He seleccionado las variables')
			if listo:

				st.markdown('### **Variables seleccionadas**')
				st.dataframe(variablesPredictoras)

				st.header('Definición de variables predictoras')
				st.markdown('Las variables predictoras son el conjunto de variables conocidas que serán la base para la predicción de una variable desconocida. Es decir, obtenemos el valor de la variable clase en función de las variables predictoras')
				
				clase = st.selectbox('Elige la variable que deseas predecir',variables,index=1)
				
				done = st.checkbox('Listo. He elegido las variables predictoras y la variable a predecir.')
				if done:
					st.header('Variables predictoras')
					X= np.array(DatosTransacciones[variablesPredictoras])
					df = pd.DataFrame(X,columns=([variablesPredictoras]))
					st.dataframe(df)

					st.header('Variable clase (variable a predecir)')
					Y= np.array(DatosTransacciones[clase])
					st.warning(clase)
					tamPrueba=0.2
					seed=1234
					X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,test_size = 0.2,random_state = 1234,shuffle = True)

					st.header('Entrenando al modelo...')
					st.info('Para el entrenamiento del modelo, se utilizará un '+str(100-(tamPrueba*100))+'% de los registros como conjunto de entrenamiento para que el modelo pueda aprender de esa información. Y se utilizará un '+str((tamPrueba*100))+'% de los registros como conjunto de prueba, para comprobar la eficiencia del modelo generado')


					PronosticoAD = DecisionTreeRegressor(max_depth=8,min_samples_leaf=2,min_samples_split=8)
					PronosticoAD.fit(X_train, Y_train)
					st.success('Modelo obtenido')

					#Se genera el pronóstico
					Y_Pronostico = PronosticoAD.predict(X_test)
					pd.DataFrame(Y_Pronostico)

					Valores = pd.DataFrame(Y_test, Y_Pronostico)

					Graf = st.checkbox('Ver grafica de la comparación entre los valores reales y los obtenidos')
					if Graf: 
						option = st.selectbox('Elige la variable que deseas evaluar en el gráfico de dispersión',variables,index=1)
						st.write('**Variable seleccionada:**', option)
						st.header('Gráfico de dispersión de dos variables')
						optionD = st.multiselect('Selecciona las dos variables que deseas comparar en la gráfica',variables)
						if len(optionD)==2:
							st.set_option('deprecation.showPyplotGlobalUse', False) 
							st.write('**Variables seleccionadas:**', optionD[0], optionD[1])
							dd1=optionD[0][:]
							dd2=optionD[1][:]
							st.subheader('Gráfica de la comparación entre los valores reales y los obtenidos')
							plt.figure(figsize=(20, 5))
							plt.plot(Y_test, color='green', marker='o', label='Y_test')
							plt.plot(Y_Pronostico, color='red', marker='o', label='Y_Pronostico')
							plt.xlabel(dd1)
							plt.ylabel(dd2)
							plt.title(option)
							plt.grid(True)
							plt.legend()
							st.pyplot()

					st.header('Obteniendo el modelo...')
					#Se calcula el exactitud promedio de la validación
					score=r2_score(Y_test, Y_Pronostico)
					if score>=0.80:
						st.success('El modelo generado tiene una exactitud promedio de: '+str(score)+'. Es una exactitud buena, lo cual es útil porque significa que el modelo puede hacer predicciones de forma generalmente acertada.')
					elif score >= 0.60:
						st.warning('El modelo tiene una exactitud promedio de: '+str(score)+'. Es una exactitud moderada, esto debe considerarse al momento de trabajar con el modelo, ya que puede no acertar en muchos de los casos.')
						st.warning('Se recomienda ampliar la muestra de datos, trabajar con variables que aporten más valor al modelo, o que el usuario experto modifique los parámetros del algoritmo en la sección de configuraciones avanzadas')
					else:
						st.warning('El modelo tiene una exactitud promedio de: '+str(score)+'. Cuidado, es una exactitud bastante baja, no recomendamos trabajar con este modelo porque estaría acertando con suerte en la mitad de los casos.')
						st.warning('Se recomienda ampliar la muestra de datos, trabajar con variables que aporten más valor al modelo, o que el usuario experto modifique los parámetros del algoritmo en la sección de configuraciones avanzadas')

					masInfo = st.checkbox('Obtener más información sobre el modelo. Recomendado para usuarios expertos')
					if masInfo:
						st.write('Criterio: \n', PronosticoAD.criterion)
						st.write("MAE: %.4f" % mean_absolute_error(Y_test, Y_Pronostico))
						st.write("MSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico))
						st.write("RMSE: %.4f" % mean_squared_error(Y_test, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
						st.write('Score: %.4f' % r2_score(Y_test, Y_Pronostico))
						st.markdown('#### Importancia de las variables en el modelo obtenido:')
						Importancia = pd.DataFrame({'Variable': list(DatosTransacciones[variablesPredictoras]),
		                    'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
						st.dataframe(Importancia)

					st.header('Características del modelo')
					st.warning('El error absoluto medio (MAE) del algoritmo es '+str(mean_absolute_error(Y_test, Y_Pronostico))+'%')
					st.info('Se tiene un score de: '+str(r2_score(Y_test, Y_Pronostico))+'%, lo cual indica que el pronóstico se hace con ese porcentaje de efectividad')
					st.warning('Por otro lado, los pronósticos del modelo final se alejan en promedio '+str(mean_squared_error(Y_test, Y_Pronostico, squared=False))+'% unidades del valor real')

					verArbol = st.checkbox('Obtener la imagen del árbol generado durante el proceso')
					st.error('Considera que esto consume una cantidad considerable de tiempo. Se recomienda desactivar la opción para continuar con las siguientes instrucciones.')
					if verArbol:
						plt.figure(figsize=(16,16))  
						plot_tree(PronosticoAD, feature_names = variablesPredictoras)
						st.pyplot()
					st.markdown('#### La imagen del árbol es grande, pero se puede leer en el siguiente orden:')
					st.markdown('🟣 **La decisión que se toma para dividir el nodo.**')
					st.markdown('🟣 **El tipo de criterio que se usó para dividir cada nodo.**')
					st.markdown('🟣 **Cuantos valores tiene ese nodo.**')
					st.markdown('🟣 **Valores promedio.**')
					st.markdown('🟣 **Por último, el valor pronosticado en ese nodo.**')
					report = st.checkbox('Obtener el reporte escrito de la estructura del árbol')
					st.info('Es un texto amplio, puedes ocultarlo presionando nuevamente el botón.')
					if report:
						st.header('Reglas del árbol obtenido')
						Reporte = export_text(PronosticoAD, feature_names = variablesPredictoras)
						st.text(Reporte)

					st.warning('Ahora puedes probar el modelo ingresando tus propios datos')
					probar = st.checkbox('Deseo probar el modelo')
					if probar:
						st.info('Ingresa tu información personal y además las características de tu caso de estudio, en función de las variables predictoras seleccionadas:')
						name = st.text_input('Ingresa tu nombre', value="")
						edad = st.text_input('Ingresa tu edad', value="")
						email = st.text_input('Ingresa tu email', value="")
						if len(variablesPredictoras)==6:
							varObjeto=[["","","","","",""]]
						else:
							varObjeto=[["","","","","","","","",""]]
						z=int((len(variablesPredictoras))/2)
						x=0
						c1, c2 = st.columns(2)
						for k in range(len(variablesPredictoras)):
							if k<z:
								varObjeto[0][x] = c1.text_input(variablesPredictoras[k], value="")
							else:
								varObjeto[0][x] = c2.text_input(variablesPredictoras[k], value="")
							x=x+1

						ready = st.checkbox('Realizar pronóstico')
						if ready:
							st.markdown('#### Elemento generado:')
							elemento = pd.DataFrame(varObjeto,index=[0],columns=variablesPredictoras)
							st.dataframe(elemento)
							st.markdown('#### Generando la predicción con el modelo...')

							diagnostico=PronosticoAD.predict(elemento)
							st.success('Las características del elementos descrito corresponden a un elemento que tiene el siguiente valor en la variable que deseas pronósticar: '+str(clase)+' = '+str(diagnostico)+', con un '+str(score)+'% de exactitud, de acuerdo con las variables seleccionadas, las características descritas y el modelo obtenido.')
							st.info('Recuerda que, a pesar de la exactitud de la predicción, la decisión final debe realizarla un experto en la materia. Ofrece esta herramienta a un profesional que te pueda ayudar a confirmar la predicción.')

		Clasif = st.checkbox('Trabajar con Árbol de decisión para Clasificación')
		if Clasif: #12
			st.subheader('Aplicación de Árbol de decisión para Clasificación')

			st.markdown('#### Con base en la información obtenida, ahora puedes seleccionar las variables con las que quieres trabajar.')
			st.info('Como recomendación, considera las variables que son indispensables para tu caso de estudio, sin que importe mucho si tiene o no una alta correlación. La selección de características no es arbitaria, analiza la importancia de las variables en tu caso de estudio y descarta aquellas que no aportan valor y tienen alta correlación.')
			variablesPredictoras = st.multiselect('Selecciona las variables con las que deseas trabajar. Estas se convierten en las variables que van a predecir la variable deseada',variables)

			listo = st.checkbox('Listo. He seleccionado las variables')
			if listo:

				st.markdown('### **Variables seleccionadas**')
				st.dataframe(variablesPredictoras)

				st.header('Definición de variables predictoras')
				st.markdown('Las variables predictoras son el conjunto de variables conocidas que serán la base para la predicción de una variable desconocida. Es decir, obtenemos el valor de la variable clase en función de las variables predictoras')
				
				clase = st.selectbox('Elige la variable que deseas predecir',variables,index=1)
				st.warning('La variable a elegir debe ser una variable binaria, es decir, solamente tiene dos posibles valores.')
				st.write('**Variable seleccionada:**', clase)

				valor1=st.text_input('Ingresa el valor negativo que puede tener la variable', value="", max_chars=None)
				valor2=st.text_input('Ingresa el valor positivo que puede tener la variable', value="", max_chars=None)

				done = st.checkbox('Listo. He ingresado los posibles valores que puede tener la variable')
				if done:
					st.header('Realizando el reetiquetado...')
					try:
						DatosTransacciones=DatosTransacciones.replace({valor1: 0, valor2: 1})

						st.header('Variables predictoras')
						X= np.array(DatosTransacciones[variablesPredictoras])
						df = pd.DataFrame(X,columns=([variablesPredictoras]))
						st.dataframe(df)

						st.header('Variable clase (variable a predecir)')
						Y= np.array(DatosTransacciones[clase])
						st.warning(clase)
						tamPrueba=0.2
						seed=1234
					except:
						st.error("Los valores ingresados no corresponden a los registros que cargaste al programa. Coloca valores correctos")



					st.header('Entrenando al modelo...')
					st.info('Para el entrenamiento del modelo, se utilizará un '+str(100-(tamPrueba*100))+'% de los registros como conjunto de entrenamiento para que el modelo pueda aprender de esa información. Y se utilizará un '+str((tamPrueba*100))+'% de los registros como conjunto de prueba, para comprobar la eficiencia del modelo generado')
					X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,test_size = tamPrueba,random_state = seed,shuffle = True)

					ClasificacionAD = DecisionTreeClassifier(max_depth=8, min_samples_split=4, min_samples_leaf=2)
					X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y,test_size = 0.2,random_state = 1234,shuffle = True)
					ClasificacionAD.fit(X_train, Y_train)
					st.success('Modelo obtenido')

					#Se etiquetan las clasificaciones
					Y_Clasificacion = ClasificacionAD.predict(X_validation)
					Valores = pd.DataFrame(Y_validation, Y_Clasificacion)

					#Se calcula la exactitud promedio de la validación
					socreC=ClasificacionAD.score(X_validation, Y_validation)

					st.header('Obteniendo el modelo...')
					#Se calcula el exactitud promedio de la validación
					socreC=ClasificacionAD.score(X_validation, Y_validation)
					if socreC>=0.80:
						st.success('El modelo generado tiene una exactitud promedio de: '+str(socreC)+'. Es una exactitud buena, lo cual es útil porque significa que el modelo puede hacer predicciones de forma generalmente acertada.')
					elif socreC >= 0.60:
						st.warning('El modelo tiene una exactitud promedio de: '+str(socreC)+'. Es una exactitud moderada, esto debe considerarse al momento de trabajar con el modelo, ya que puede no acertar en muchos de los casos.')
						st.warning('Se recomienda ampliar la muestra de datos, trabajar con variables que aporten más valor al modelo, o que el usuario experto modifique los parámetros del algoritmo en la sección de configuraciones avanzadas')
					else:
						st.warning('El modelo tiene una exactitud promedio de: '+str(socreC)+'. Cuidado, es una exactitud bastante baja, no recomendamos trabajar con este modelo porque estaría acertando con suerte en la mitad de los casos.')
						st.warning('Se recomienda ampliar la muestra de datos, trabajar con variables que aporten más valor al modelo, o que el usuario experto modifique los parámetros del algoritmo en la sección de configuraciones avanzadas')

					masInfo = st.checkbox('Obtener más información sobre el modelo. Recomendado para usuarios expertos')
					if masInfo:
						#Matriz de clasificación
						Y_Clasificacion = ClasificacionAD.predict(X_validation)
						st.markdown('#### Matriz de clasificación:')
						Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(),Y_Clasificacion, rownames=['Real'],colnames=['Clasificación']) 
						st.table(Matriz_Clasificacion)

						col1, col2 = st.columns(2)
						col2.info('Falsos Positivos (FP): '+str(Matriz_Clasificacion.iloc[0,1]))
						col2.info('Verdaderos Positivos (VP): '+str(Matriz_Clasificacion.iloc[1,1]))
						col1.info('Verdaderos Negativos (VN): '+str(Matriz_Clasificacion.iloc[0,0]))
						col1.info('Falsos Negativos (FN): '+str(Matriz_Clasificacion.iloc[1,0]))

						#Reporte de la clasificación
						st.markdown('#### Reporte de la clasificación:')
						st.write("Exactitud", ClasificacionAD.score(X_validation, Y_validation))
						st.text(classification_report(Y_validation, Y_Clasificacion))

						st.markdown('#### Importancia de las variables en el modelo obtenido:')
						Importancia = pd.DataFrame({'Variable': list(DatosTransacciones[variablesPredictoras]),
			                'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)

						st.dataframe(Importancia)

					verArbolC = st.checkbox('Obtener la imagen del árbol generado durante el proceso')
					st.error('Considera que esto consume una cantidad considerable de tiempo. Se recomienda desactivar la opción para continuar con las siguientes instrucciones.')
					if verArbolC:
						plt.figure(figsize=(16,16))  
						plot_tree(ClasificacionAD,feature_names = ['Texture', 'Area', 'Smoothness','Compactness', 'Symmetry', 'FractalDimension'])
						st.pyplot()
					st.markdown('#### La imagen del árbol es grande, pero se puede leer en el siguiente orden:')
					st.markdown('🟣 **La decisión que se toma para dividir el nodo.**')
					st.markdown('🟣 **El tipo de criterio que se usó para dividir cada nodo.**')
					st.markdown('🟣 **Cuantos valores tiene ese nodo.**')
					st.markdown('🟣 **Valores promedio.**')
					st.markdown('🟣 **Por último, el valor pronosticado en ese nodo.**')
					report = st.checkbox('Obtener el reporte escrito de la estructura del árbol')
					st.info('Es un texto amplio, puedes ocultarlo presionando nuevamente el botón.')
					if report:
						st.header('Reglas del árbol obtenido')
						ReporteC = export_text(ClasificacionAD,feature_names = variablesPredictoras)
						st.text(ReporteC)

					st.warning('Ahora puedes probar el modelo ingresando tus propios datos')
					probar = st.checkbox('Deseo probar el modelo')
					if probar:
						st.info('Ingresa tu información personal y además las características de tu caso de estudio, en función de las variables predictoras seleccionadas:')
						name = st.text_input('Ingresa tu nombre', value="")
						edad = st.text_input('Ingresa tu edad', value="")
						email = st.text_input('Ingresa tu email', value="")
						if len(variablesPredictoras)==6:
							varObjeto=[["","","","","",""]]
						else:
							varObjeto=[["","","","","","","","",""]]
						z=int((len(variablesPredictoras))/2)
						x=0
						c1, c2 = st.columns(2)
						for k in range(len(variablesPredictoras)):
							if k<z:
								varObjeto[0][x] = c1.text_input(variablesPredictoras[k], value="")
							else:
								varObjeto[0][x] = c2.text_input(variablesPredictoras[k], value="")
							x=x+1

						ready = st.checkbox('Realizar pronóstico')
						if ready:
							st.markdown('#### Elemento generado:')
							elemento = pd.DataFrame(varObjeto,index=[0],columns=variablesPredictoras)
							st.dataframe(elemento)
							st.markdown('#### Generando la predicción con el modelo...')

							diagnosticoC=ClasificacionAD.predict(elemento)
							if(diagnosticoC==1):
								st.success('Las características del elementos descrito corresponden a un elemento que tiene el siguiente valor en la variable que deseas pronósticar: '+str(clase)+' = '+str(valor1)+', con un '+str(socreC*100)+'% de exactitud, de acuerdo con las variables seleccionadas, las características descritas y el modelo obtenido.')
							else:
								st.success('Las características del elementos descrito corresponden a un elemento que tiene el siguiente valor en la variable que deseas pronósticar: '+str(clase)+' = '+str(valor2)+', con un '+str(socreC*100)+'% de exactitud, de acuerdo con las variables seleccionadas, las características descritas y el modelo obtenido.')
							st.info('Recuerda que, a pesar de la exactitud de la predicción, la decisión final debe realizarla un experto en la materia. Ofrece esta herramienta a un profesional que te pueda ayudar a confirmar la predicción.')
					
						



