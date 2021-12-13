import streamlit as st
import streamlit.components.v1 as components

#Valores por default
#Definiendo variables globales para las configuraciones avanzadas

#métricas
lamb=1.5

#Clustering
totalClusterC=7

#ArbolesDeDecisión y Clasificación Regresión Logística
tamPrueba=0.2
seed=0
maxProfundidad=8
minMuestraSplit=8
minMuestraLeaf=2

def start():
	st.title('Configuraciones Avanzadas')
	st.warning('**Nota**: Los algoritmos están configurado de forma genérica para ofrecerte un resultado útil, pero si deseas cambiar la configuración predeterminada para los parámetros con que trabajan los algoritmos, en esta sección podrás hacerlo para algunos de ellos. **Esta opción es recomendada únicamente para usuarios expertos**, ya que puede influir en el funcionamiento y desempeño del algoritmo, y si se realiza de forma incorrecta puede afectar a los resultados.')
	st.success('Puedes regresar a los valores por default con la opción correspondiente a cada sección cuando lo desees.')

	st.image("https://dinahosting.com/blog/cont/uploads/2019/01/Post-configurar-WordPress.jpg",width=1000)
	st.markdown('#### **En esta sección puedes modificar algunos parámetros comunes de los siguientes módulos**')
	st.markdown('🟣 **Módulo métricas de distancia**')
	st.markdown('🟣 **Módulo de Clustering**')
	st.markdown('🟣 **Módulo de Clasificación por Regresión Logística**')
	st.markdown('🟣 **Módulo de Árboles de decisión**')

	#Metricas
	MD = st.checkbox('Modificar parámetros para el Módulo métricas de distancia')
	if MD:
		st.subheader('📐 Distancia Minkowski')
		st.markdown('En esta sección puedes moficar el valor de lambda para la distancia Minkowski')
		st.markdown('**Descripción**: La distancia Minkowski es una distancia entre dos puntos en un espacio n-dimensional. Es una métrica de distancia generalizada: Euclidiana, Manhattan y Chebyshev. ')
		st.markdown('Esta métrica permite calcular la distancia de tres formas diferentes, en función del valor de **lambda**, que define el orden para las 3 diferentes métricas que conocemos. Los valores se definen de la siguiente forma')
		st.markdown('🟣 **λ=1. Distancia Manhattan**')
		st.markdown('🟣 **λ=2. Distancia Euclidiana**')
		st.markdown('🟣 **λ=3. Distancia de Chebyshev**')
		st.warning('Actualmente se suelen emplear valores intermedios, como **λ=1.5** que proporciona un equilibrio entre las medidas')
		st.markdown('**Selecciona el valor de λ que deseas:**')
		lamb=st.slider('Lambda λ:', min_value=1.0, max_value=3.0, value=1.5, step=0.5)
		st.write('**λ: **'+str(lamb))

		resMD = st.checkbox('MD. Volver al valor por default')
		if resMD:
			lamb=1.5


	#Clustering
	MC = st.checkbox('Modificar parámetros para el Módulo de Clustering')
	if MC:
		st.subheader('Módulo de Clustering')
		st.markdown('Para trabajar con ese algoritmo se debe definir un número de clústeres')
		st.warning('Actualmente se emplea un valor por defecto de '+str(7)+' clústeres')
		totalClusterC=st.slider('J. Ingresa el número de clústeres que quieres generar', min_value=1, max_value=15, value=7, step=1)
		st.write('J. Cantidad de clústeres: '+str(totalClusterC))

		resMC = st.checkbox('MC. Volver al valor por default')
		if resMC:
			totalClusterC=7

	#Clasificación Regresión Logística
	RL = st.checkbox('Modificar parámetros para el Módulo de Clasificación por Regresión Logística')
	if RL:
		st.subheader('Módulo de Clasificación por Regresión Logística')
		st.markdown('Para trabajar con ese algoritmo se debe definir las características de la estructura jerárquica a generar, puedes seleccionar entre el tamaño de prueba para la división de los datos, la semilla para la generación de los números random, la muestra mínima de división y la muestra mínima de los nodos hoja')
		st.markdown('🟣 **Profundidad máxima.** Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el overfitting, pero también puede provocar underfitting.')
		st.markdown('🟣 **Cantidad mínima de datos por hoja.** Indica la cantidad mínima de datos que debe tener un nodo hoja.')
		st.markdown('🟣 **Cantidad mínima de división.** Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.')

		tamPruebaIn=st.slider('Ingresa el porcentaje del tamaño del conjunto de prueba', min_value=0, max_value=100, value=20, step=2)
		st.write('Porcentaje: '+str(tamPruebaIn)+'%')
		tamPrueba=tamPruebaIn/100

		seed=number = st.number_input('Inserta la semilla deseada')

		maxProfundidad = st.slider('Ingresa la profundidad máxima deseada', min_value=0, max_value=30, value=8, step=1)
		st.write('Profundidad máxima: '+str(maxProfundidad))

		minMuestraLeaf = st.slider('Ingresa la cantidad mínima de datos por hoja', min_value=0, max_value=30, value=2, step=1)
		st.write('Cantidad mínima de datos por hoja: '+str(minMuestraLeaf))

		minMuestraSplit = st.slider('Ingresa la cantidad mínima de división', min_value=0, max_value=30, value=2, step=1)
		st.write('Cantidad mínima de división: '+str(minMuestraSplit))

		resRL = st.checkbox('RL. Volver al valor por default')
		if resRL:
			tamPrueba=0.2
			seed=0
			maxProfundidad=8
			minMuestraSplit=8
			minMuestraLeaf=2

	#Árboles de Decisión
	AA = st.checkbox('Modificar parámetros para el Módulo de Árboles de Decisión')
	if AA:
		st.subheader('Módulo de Árboles de Decisión')
		st.markdown('Para trabajar con ese algoritmo se debe definir las características del árbol a generar, puedes seleccionar entre el tamaño de prueba para la división de los datos, la semilla para la generación de los números random, la muestra mínima de división y la muestra mínima de los nodos hoja')
		st.markdown('🟣 **Profundidad máxima.** Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el overfitting, pero también puede provocar underfitting.')
		st.markdown('🟣 **Cantidad mínima de datos por hoja.** Indica la cantidad mínima de datos que debe tener un nodo hoja.')
		st.markdown('🟣 **Cantidad mínima de división.** Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.')

		tamPruebaIn=st.slider('AA. Ingresa el porcentaje del tamaño del conjunto de prueba', min_value=0, max_value=100, value=20, step=2)
		st.write('Porcentaje: '+str(tamPruebaIn)+'%')
		tamPrueba=tamPruebaIn/100

		seed=number = st.number_input('AA. Inserta la semilla deseada')

		maxProfundidad = st.slider('AA. Ingresa la profundidad máxima deseada', min_value=0, max_value=30, value=8, step=1)
		st.write('Profundidad máxima: '+str(maxProfundidad))

		minMuestraLeaf = st.slider('AA. Ingresa la cantidad mínima de datos por hoja', min_value=0, max_value=30, value=2, step=1)
		st.write('Cantidad mínima de datos por hoja: '+str(minMuestraLeaf))

		minMuestraSplit = st.slider('AA. Ingresa la cantidad mínima de división', min_value=0, max_value=30, value=2, step=1)
		st.write('Cantidad mínima de división: '+str(minMuestraSplit))

		resAA = st.checkbox('AA. Volver al valor por default')
		if resAA:
			tamPrueba=0.2
			seed=0
			maxProfundidad=8
			minMuestraSplit=8
			minMuestraLeaf=2





	
	

