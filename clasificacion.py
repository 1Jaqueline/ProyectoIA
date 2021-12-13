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
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def cargaCL():
	st.title('📁 Carga tu archivo')
	st.markdown('##### Características del archivo:')
	st.markdown('Para el funcionamiento eficiente del algoritmo, carga un archivo desde tu computadora que cumpla con las siguientes características')
	st.markdown('🟣 **Archivo con extensión .csv**')
	st.markdown('Esta extensión corresponde a un archivo con valores separados por comas')
	st.markdown('🟣 **Archivo con registros**')
	st.markdown('Estos deben ser registros que contengan las características que deseas evaluar')
	file = st.file_uploader("Carga tu archivo", type=["csv", "txt"], key='Clasificacion')
	return file


def start():
	st.title('Módulo de Clasificación (Regresión Logística)')
	st.markdown('La regresión logística es otro tipo de algoritmo de aprendizaje supervisado cuyo objetivo es predecir valores binarios (0 o 1). Este algoritmo permite trabajar con datos que no son linealmente separables, empleando la función logística o sigmoide.')

	st.image("https://www.iic.uam.es/wp-content/uploads/2020/07/etiquetado-textos.jpg",width=1000)
	st.markdown('#### **Este módulo aplica el siguiente algoritmo**')
	st.header('🟣 Regresión Logística')
	datos = cargaCL()

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
		option = st.selectbox('Elige la variable que deseas evaluar en el gráfico de dispersión',variables,index=1)
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
					X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y,test_size = tamPrueba,random_state = seed,shuffle = True)

					st.header('Entrenando al modelo...')
					st.info('Para el entrenamiento del modelo, se utilizará un '+str(100-(tamPrueba*100))+'% de los registros como conjunto de entrenamiento para que el modelo pueda aprender de esa información. Y se utilizará un '+str((tamPrueba*100))+'% de los registros como conjunto de prueba, para comprobar la eficiencia del modelo generado')

					#Se entrena el modelo a partir de los datos de entrada
					Clasificacion = linear_model.LogisticRegression()
					Clasificacion.fit(X_train, Y_train)

					#Predicciones probabilísticas de los datos de prueba
					Probabilidad = Clasificacion.predict_proba(X_validation)

					#Predicciones con clasificación final 
					Predicciones = Clasificacion.predict(X_validation)

					st.header('Obteniendo el modelo...')
					#Se calcula el exactitud promedio de la validación
					score=Clasificacion.score(X_validation, Y_validation)
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
						#Matriz de clasificación
						Y_Clasificacion = Clasificacion.predict(X_validation)
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
						st.write("Exactitud: "+str(score*100)+'%')
						st.write("Precisión: "+ str(float(classification_report(Y_validation, Y_Clasificacion).split()[10])*100)+ " %")
						st.write("Tasa de error: "+str((1-Clasificacion.score(X_validation, Y_validation))*100)+" %")
						st.write("Sensibilidad: "+ str(float(classification_report(Y_validation, Y_Clasificacion).split()[11])*100)+ " %")
						st.write("Especificidad: "+ str(float(classification_report(Y_validation, Y_Clasificacion).split()[6])*100)+" %")

						st.markdown('#### Ecuación del modelo de clasificación:')
						intercepto=Clasificacion.intercept_
						coeficientes=Clasificacion.coef_

						st.write(coeficientes)

						if len(variablesPredictoras)==6:
							st.write("a+bX="+str(intercepto)+"+ ("+str(Clasificacion.coef_[0][0].round(6))+") ["+str(variablesPredictoras[0])+"]"+"+ ("+str(Clasificacion.coef_[0][1].round(6))+") ["+str(variablesPredictoras[1])+"]"+"+ ("+str(Clasificacion.coef_[0][2].round(6))+") ["+str(variablesPredictoras[2])+"]"+"+ ("+str(Clasificacion.coef_[0][3].round(6))+") ["+str(variablesPredictoras[3])+"]"+"+ ("+str(Clasificacion.coef_[0][4].round(6))+") ["+str(variablesPredictoras[4])+"]"+"+ ("+str(Clasificacion.coef_[0][5].round(6))+") ["+str(variablesPredictoras[5])+"]")
						else:
							st.error('No fue posible obtener la ecuación del modelo.')

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

						
						st.markdown('#### Elemento generado:')
						elemento = pd.DataFrame(varObjeto,index=[0],columns=variablesPredictoras)
						st.dataframe(elemento)
						st.markdown('#### Generando la predicción con el modelo...')

						diagnostico=Clasificacion.predict(elemento)
						if diagnostico[0]==0:
							st.error('Las características del elementos descrito corresponden a un caso negativo, con un '+str(score*100)+'% de exactitud, de acuerdo con las variables seleccionadas y el modelo generado. Este valor corresponde al valor '+str(clase)+' : '+str(valor1)+' de los registos ingresados.')
						else:
							st.success('Las características del elementos descrito corresponden a un caso positivo, con un '+str(score*100)+'% de exactitud, de acuerdo con las variables seleccionadas y el modelo generado. Este valor corresponde al valor '+str(clase)+' : '+str(valor2)+' de los registos ingresados.')

						st.info('Recuerda que, a pesar de la exactitud de la predicción, la decisión final debe realizarla un experto en la materia. Ofrece esta herramienta a un profesional que te pueda ayudar a confirmar la predicción.')

				except:
					st.error("Los valores ingresados no corresponden a los registros que cargaste al programa. Coloca valores correctos")
