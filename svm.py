#MODELO SVM - Máquinas de Vetores de Suporte
#Base de Dados mnist (todas imagens sao 8x8 =64px para cada imagem)

import matplotlib.pyplot as plt #plotar os graficos
from sklearn import datasets, svm, metrics #bibliotecas(ML) svm e metrics para matriz de confusao
import pickle #salvar no disco para usar/acessar 

def treinarSVM():
    #carrega imagens da base de dados para variavel
    digits = datasets.load_digits()
    #plt.gray()
    #plt.matshow(digits.images[0]) #exemplo pegar primeira imagem data
    #plt.show()
    #exibe quantas imagens contém base de dados(1797), e tamanho matriz (8,8)
    #print(digits.images.shape)
    #variavel que guarda tamanho do container
    numeroDeSamples = len(digits.images)
    #converter para svm enteder , de matriz 8x8 para vetor de 64 posicoes 
    data = digits.images.reshape((numeroDeSamples, -1))
    #print(data.shape)    
    
    #10 classes(de 0-9) gamma ajusta o conjunto
    classificador = svm.SVC(gamma= "scale")
    
    #treinar classificador com 50% das imagens da base de dados
    # se o metodo classificar com o msm valor de target , significa que esta correto
    classificador.fit(data[:numeroDeSamples // 2], digits.target[:numeroDeSamples // 2])
    
    #salva em disco o modelo ja treinado pra nao precisar treinar toda vez
    with open('filename.treinosvm', 'wb') as filePointer:
        pickle.dump(classificador, filePointer)
    
def carregaSVM():
    digits = datasets.load_digits()
    
    #transforma para vetor
    numeroDeSamples = len(digits.images)
    data = digits.images.reshape((numeroDeSamples, -1))
    
    #carregar em disco o modelo ja treinado
    with open('filename.treinosvm', 'rb') as filePointer:
        classificador = pickle.load(filePointer)
        
        
    #teste, vamos predizer os rotulos da segunda metade das imagens
    rotulosEsperados = digits.target[numeroDeSamples //2 :]
    #faz a predicao do modelo treinado
    rotulosPreditos = classificador.predict(data[numeroDeSamples // 2:])
    
    #mostra matrizes de confusao para verificar se classificacao foi good
    print("Relatorio de Classificacao %s:\n%s\n" % (classificador, metrics.classification_report(rotulosEsperados, rotulosPreditos)))
    print("Matrizes de Confusao: \n%s" % metrics.confusion_matrix(rotulosEsperados, rotulosPreditos))

    #visualizar os 12 imagens do dataset com suas classificacoes
    imagensPredicoes = list(zip(digits.images[numeroDeSamples // 2 :], rotulosPreditos)) 
    for index, (imagem, predicao) in enumerate(imagensPredicoes[: 20]):  #12
        plt.subplot(5, 4, index + 1)  #3,4
        plt.axis('off')
       #plt.imshow(imagem, interpolation='nearest')
        
        #teste outra cor
        plt.imshow(imagem, interpolation='nearest', cmap='gray')


        plt.title("Predicao: %i" % predicao)
        #teste
        plt.text(0, 7, "Target: %i" % rotulosEsperados[index], color='red')

        
    plt.show()

if __name__ == "__main__":
    print("Salvando o modelo..")
    treinarSVM()
    print("ok! Modelo salvo.")
    
    print("Carregando o modelo..")
    carregaSVM()
    print("ok! Modelo carregado.") 
