# AI-image-classifier
Web application to classify a given image in two main classes.

Based on Django framework, using Keras and TensorFlow for predictions.
The application receives an input image from users returning the class with the greater probability of belonging to that class.
Currently configured to classify images in classes: "Doberman" and "Belgian Shepherd Groenendael" (dog breeds).
It is also able to, in cases when probability is less than 1.5% for both classes, return the classification for the others model's labels.

Developed as an internship test for Pickcells.

######################################################################################################

Aplicação web para classificar imagens em duas classes principais.

Baseado no framework Django, utilizando Keras e TensorFlow para as predições. 
A aplicação recebe uma imagem do usuário devolvendo a classe com qual mais se identifica e probabilidade da imagem pertencer aquela classe.
Atualmente configurada para classificar imagens nas classes: "Doberman" e "Pastor Belga Groenendael"(raças de cães).
Também é capaz de caso a probabilidade seja menor que 1,5% para ambas as classes, retornar a classificação nos outros labels do modelo.

Desenvolvido como teste para estágio na empresa Pickcells.


Luiz Antonio V de M. M. C. Negrinho.
