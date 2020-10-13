from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf 
import json
from tensorflow import Graph

# Define o tamanho padrão da imagem a ser carregada.
# Define o arquivo json com as classes de imagens.
# Adicionado o parâmetro encoding que resolve problemas de acentuação quando retornando dados do json.


img_height, img_width = 224,224                         
with open('./models/imagenet_classes.json','r', encoding='utf8') as f:   
    labelInfo = f.read()
labelInfo = json.loads(labelInfo) 

# model = load_model('./models/MobileNetModelImagenet.h5') # Carrega o modelo treinado do keras, desse modo funciona apenas nos notebooks

# Carrega o modelo.
model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session() # Usa o Session do tensorflow anterior a versão 2.0
    with tf_session.as_default():
        model=load_model('./models/MobileNetModelImagenet.h5')


def index(request):
    context = {'a':1}
    
    return render(request,'index.html', context)

# Função de detecção.
def predictImage(request):
    print(request)
    print(request.POST.dict())
    fileObj = request.FILES['filePath']
    # Define o banco de dados
    fs = FileSystemStorage()
    # Salva o arquivo enviado com o mesmo nome do original e define o caminho pra imagem.
    filePathName = fs.save(fileObj.name,fileObj) 
    filePathName = fs.url(filePathName)
    testimage = '.'+ filePathName
    # Carrega a imagem e lhe atribui o tamanho definido anteriormente.    
    img = image.load_img(testimage, target_size=(img_height, img_width))
    # Transforma a imagem em uma array.
    x = image.img_to_array(img)   
    x = x/255
    x = x.reshape(1,img_height, img_width, 3)
    # Executa a classificação.
    with model_graph.as_default():
       with tf_session.as_default():
           predi=model.predict(x)
    
    import numpy as np
    # axis=-1 é o id do label.
    # predi[0] é a array com as probabilides de todas as classes do modelo.
    # Pega o label da maior probabilidade dentro da array de probabilidades.
    predictedLabel = labelInfo[str(np.argmax(predi[0]))]
    # Retorna a probabilidade de a Imagem pertencer a classe detectada(a com maior probabilidade) e transforma em porcentagem.
    score = f'{(max(predi[0]) * 100):.2f}'
    # Normaliza as probabilidades das classes selecionadas, arredondando em 2 casas decimais, 
    # assim quando a chance de ser uma das duas classes principais for menor que 0,1% serão igualhadas em 0 e a condição retorna o else. 
    scoreDobraw = round(predi[0][236],3) 
    scoreGroraw = round(predi[0][224],3)
    scoreDob = scoreDobraw*100
    scoreGro = scoreGroraw*100
    # Analisa a classe com maior grau de compatibilidade e normaliza em porcentagem com 1 casa decimal. 
    myclass = 0
    if scoreDob > scoreGro:
        myclass = scoreDob
    else:
        myclass = scoreGro
    
    if myclass == scoreDob and myclass > 1.5:
        retScore = f'A imagem analisada tem {scoreDob:.1f}% de chances de pertencer a classe: Doberman' # Probabilidade de Pertencer a Classe Doberman
    elif myclass == scoreGro and myclass > 1.5:
        retScore = f'A imagem analisada tem {scoreGro:.1f}% de chances de pertencer a classe: Pastor Belga Groenendael' # Probabilidade de Pertencer a classe Groenendael
    # Se as duas classes tiverem chance menor que 0.05%, retorna a maior probabilidade de todas as classes do modelo traduzida para português.
    else:
        from googletrans import Translator
        maisProvavel = str((predictedLabel[1]).replace('_',' '))
        translator = Translator()
        traduz = translator.translate(maisProvavel,dest='pt')
        maisProvavelOut = traduz.text
        retScore = f'Não parece ser um Doberman, muito menos um Pastor Belga Groenendael, seria um(a) {maisProvavelOut}?'
    

    context = {'filePathName':filePathName, 'predictedLabel':predictedLabel[1], 'retScore':retScore}
    return render(request,'index.html', context)

# Define view da galeria
def viewDataBase(request):
    import os
    # Cria uma lista com os arquivos dentro do diretório ./media/
    listOfImages = os.listdir('./media/') 
    # Pega o caminho de cada arquivo e adiciona em uma lista
    listOfImagesPath = ['./media/'+ i for i in listOfImages] 
    context = {'listOfImagesPath':listOfImagesPath}
    

    return render(request,'viewDB.html',context) #envia context a viewDB.html
