# grade-prediction_deeplearning
Estudio de notas futuras en alumnos del Colegio Mayor.

En el estado actual existe solo el subproyecto Embeddings. Este se dedica de vectorizar todas loas variables categoricas relevante y/o disponibles.

La intención del vectorizado es entregar a un futuro modelo de Forecasting un espacio vectorial representativo, vale decir, al igual que en el caso de W2V, poder tener representaciones de alumnos, por ejemplo, que vaya en concordacia con su rendimiento a largo plazo (aptitudes si se quiere) y que estas representaciones sean similares entre alumnos con parecidas aptitudes.

Con esto el próximo modelo tendrá una base sólida desde la cual aprender patrones mas complejos y puntuales, como por ejemplo su actividad en la plataforma de aprendizaje Moodle.

Este método se aplica para cada v. categorica encontrada en el dataset 'archivos_mayor_cleanframe-csv'.
