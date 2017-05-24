##################################################################
##                                                              ##
##      Codigo criado por Alexandre Xavier Ciuffatelli          ##
##      https://www.facebook.com/alexandre.ciuffatelli          ##
##      Todos podem usar este codigo livremente                 ##
##      Não comercialize esse código, grato                     ##
##                                                              ##
##################################################################

import RedeNeuralArtificial_lucas as RNA
import pega_dados_planilha as Dados

TAXA_DE_APRENDIZADO = 0.6

'''
Funcoes do RNA usadas:
    - feedfoward()
    - backpropagation(esperado, saida_output_layer, saida_hidden_layer, taxa_aprendizado, rede, entrada)
    - cria_rede()
'''

def pegaBD(arq_xls):
    BD = []
    for linha in Dados.xlread_linha(arq_xls):#pega valores do BD
        BD.append(linha)
    for i in range(0,len(BD)):#transforma valores de str para float
        for j in range(0,len(BD[i])):
            BD[i][j] = float(BD[i][j])
    return BD

def treinamento(rede, arq_xls):

    BD = pegaBD(arq_xls)
    print(BD)
    if(rede == [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]):
        rede = RNA.cria_rede()


    for i in range(0,len(BD)):
        entrada = []

        for j in range(0,len(BD[i])-1):
            entrada.append(BD[i][j])

        saida_esperada = BD[i][len(BD[i])-1]
        if(saida_esperada == 1.0):
            saida_esperada = [1.0, 0.01]
        elif(saida_esperada == 2.0):
            saida_esperada = [0.01, 1.0]
        else:
            saida_esperada = [0.01, 0.01]


    # Condições de parada:
    treinando = True
    epochs = 0
    while (treinando == True and epochs < 600):
        epochs += 1
        saida_hidden_layer, saida_output_layer = RNA.feedfoward(entrada, rede)

        if (RNA.Erro(saida_esperada, saida_output_layer) >= 0.005):
           treinando = True
           rede = RNA.backpropagation(saida_esperada, saida_output_layer, saida_hidden_layer, rede, entrada, TAXA_DE_APRENDIZADO)
        else:
            treinando = False
        if( (epochs%10) == 0):
            print("Época:     ", epochs)
            print("Entrada:   ", entrada)
            print("Classe 1:  ", saida_output_layer[0], "%")
            print("Classe 2:  ", saida_output_layer[1], "%")
            print("Esperado:  ", saida_esperada)
            print("Erro:      ", RNA.Erro(saida_esperada, saida_output_layer))
            print("")
    return rede

def treinamento_site():
    #teste do site https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    rede = [[0.15,0.20,0.35],[0.25,0.30,0.35],[0.40,0.45,0.60], [0.50,0.55,0.60]]
    entrada = [0.05,0.10,1]
    saida_hidden_layer, saida_output_layer = RNA.feedfoward(entrada, rede)

    prints(rede, saida_output_layer, entrada)

    rede = RNA.backpropagation([0.01, 0.99], saida_output_layer, saida_hidden_layer, rede, entrada, TAXA_DE_APRENDIZADO)
    saida_hidden_layer, saida_output_layer = RNA.feedfoward(entrada, rede)

    prints(rede, saida_output_layer, entrada)

    for i in range(0,10000):
        saida_hidden_layer, saida_output_layer = RNA.feedfoward(entrada, rede)
        rede = RNA.backpropagation([0.01, 0.99], saida_output_layer, saida_hidden_layer, rede, entrada, .5)

    prints(rede, saida_output_layer, entrada)


def prints(rede, saida_output_layer, entrada):
    print(rede[0])
    print(rede[1])
    print(rede[2])
    print("Classe 1: ", saida_output_layer[0], "%")
    print("Classe 2: ", saida_output_layer[1], "%")
    print("Esperado: ", entrada)
    print("Erro: ", RNA.Erro(entrada, saida_output_layer))
    print("")

print("###### SITE #####")
treinamento_site()

print("###### Banco de Dados 2 #####")
rede = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
rede = treinamento(rede, "BD.xlsx")
