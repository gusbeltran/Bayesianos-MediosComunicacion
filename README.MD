Deteccion de Noticias Falsas con Naive Bayes
Descripcion
Este proyecto implementa un modelo de Machine Learning utilizando el algoritmo Multinomial Naive Bayes para detectar noticias falsas y reales a partir de su contenido textual.

El objetivo es transformar textos en datos numericos mediante la tecnica Bag of Words y entrenar un modelo capaz de identificar patrones asociados a noticias verdaderas y falsas.

Se utilizo la libreria Scikit-Learn para el procesamiento de texto, entrenamiento del modelo y evaluacion de resultados.

Tecnologias utilizadas
Python
Pandas
Scikit-Learn
CountVectorizer
Multinomial Naive Bayes
Procedimiento
Se construyo un conjunto de noticias etiquetadas como reales o falsas.
Los textos fueron convertidos en vectores numericos mediante CountVectorizer.
Se entreno un modelo MultinomialNB.
Se evaluo el rendimiento mediante Accuracy y Matriz de Confusion.
Se clasificaron nuevas noticias no vistas previamente.
Resultados
Precision total (Accuracy):

100%

Matriz de confusion:

Pred Fake	Pred Real
Fake	5	0
Real	0	5
Clasificacion de nuevas noticias:

Nuevo estudio demuestra que el cafe mejora la memoria -> Real
Expertos afirman que los gatos pueden hablar con humanos -> Fake
Conclusion
El algoritmo Multinomial Naive Bayes logro clasificar correctamente todas las noticias del conjunto de entrenamiento, obteniendo una precision del 100%. Esto demuestra que el modelo pudo identificar patrones relevantes en el texto para diferenciar noticias reales de noticias falsas.

Sin embargo, el conjunto utilizado es muy pequeño y los resultados pueden ser optimistas debido a que la evaluacion se realizo sobre los mismos datos empleados para entrenar el modelo. En aplicaciones reales se recomienda utilizar una mayor cantidad de ejemplos y realizar una division entre datos de entrenamiento y prueba para obtener una evaluacion mas confiable del rendimiento del modelo.

Conclusion corta para un informe academico:

El modelo Multinomial Naive Bayes obtuvo una precision del 100% sobre el conjunto analizado, clasificando correctamente todas las noticias. Los resultados muestran que el algoritmo puede identificar patrones textuales asociados a noticias reales y falsas. No obstante, debido al tamaño reducido del dataset y a que la evaluacion se realizo sobre los mismos datos de entrenamiento, los resultados deben considerarse demostrativos y no concluyentes para un escenario real.
