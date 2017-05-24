##################################################################
##                                                              ##
##      Codigo criado por Alexandre Xavier Ciuffatelli          ##
##      https://www.facebook.com/alexandre.ciuffatelli          ##
##      Todos podem usar este codigo livremente                 ##
##      Não comercialize esse código, grato                     ##
##                                                              ##
##################################################################

import random
from math import sqrt

'''
Algumas coisas q vc deve saber:
    - As saídas esperadas são 0 ou 1, ou seja, se eh da classe 1, output1 deve ser 1 e output2 deve ser 0 e se eh da classe 2, output1 deve ser 0 e output2 deve ser 1
    - O desenho da rede pode ser visto na figura Rede_desenho.png, mas eh basicamente uma rede de 2 entradas, 1 hidden layer com 2 neuronios e a output layer com 2 neuronios
    - a rede classifica uma entrada como sendo da classe 1 ou 2, ou seja, ela devolve a porcentagem de ser da classe 1 ou 2
'''

'''
exemplo:
https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
entrada = [0.05,0.10,1]
rede = [[0.15,0.20,0.35],[0.25,0.30,0.35],[0.40,0.45,0.60], [0.50,0.55,0.60]]

rede = [[W00, W01, W02],[W10, W11, W12],[W20, W21, W22],[W30, W31, W32]] do desenho, ou seja,
[[Pesos do Neuronio 0],[Pesos do neuronio 1],[Pesos do neuronio 2],[Pesos do neuronio 3]]
'''

def calcula_somatoria(id_neuronio, lista_entrada, rede):
    somatoria = 0 #Y do desenho
    for i in range (0, len(lista_entrada)):
        somatoria += lista_entrada[i]*rede[id_neuronio][i]
    return somatoria

def funcao_ativacao(somatoria):
    return (1/(1+2.718281**(-somatoria)))#1/[1+e^(-somatoria)]
'''
Usei a função sigmoidal na funcao_ativacao(). Ela calcula F(Y) do desenho
'''

def feedfoward(entrada, rede):#calcula o output1 e output2

    soma_neuronio0 = calcula_somatoria(0, entrada, rede)#tambem chamado de net0
    saida_neuronio0 = funcao_ativacao(soma_neuronio0)

    soma_neuronio1 = calcula_somatoria(1, entrada, rede)#tambem chamado de net1
    saida_neuronio1 = funcao_ativacao(soma_neuronio1)

    entrada_output_layer = [saida_neuronio0, saida_neuronio1, 1]#1 eh o bias

    soma_neuronio2 = calcula_somatoria(2, entrada_output_layer, rede)#tambem chamado de net2
    saida_neuronio2 = funcao_ativacao(soma_neuronio2)#tambem chamado de out1

    soma_neuronio3 = calcula_somatoria(3, entrada_output_layer, rede)#tambem chamado de net3
    saida_neuronio3 = funcao_ativacao(soma_neuronio3)#tambem chamado de out2

    saida_hidden_layer = [saida_neuronio0, saida_neuronio1, 1]#1 do bias
    saida_output_layer = [saida_neuronio2, saida_neuronio3]

    return saida_hidden_layer, saida_output_layer

def backpropagation_camada_saida(esperado, saida_output_layer, saida_hidden_layer, rede, taxa_aprendizado):
    '''A partir do erro (saida esperada - saida obtida), corrige os pesos da rede, quase como W = W - N * E,         ###### tem que diferenciar o erro relativo da derivada do erro (geralmente chamam de delta)
    ou seja, peso = peso - taxa_de_aprendizado * erro'''
    #esperado = [target1, target2]
    #saida_output_layer = [output1, output2] ou [saida_neuronio2, saida_neuronio3]                                   ###### a operação que a função faz parece correta, mas eu acho que os nomes não estão precisos
    #saida_hidden_layer = [saida_neuronio0, saida_neuronio1, 1]

    for i in range (2,4):
        for j in range(0,2):
            Erro_Relativo = (saida_output_layer[i-2] - esperado[i-2]) * saida_output_layer[i-2]*(1-saida_output_layer[i-2]) * saida_hidden_layer[j]             ###### aqui é um calculo de delta
            rede[i][j] = rede[i][j] - taxa_aprendizado * Erro_Relativo
    return rede

def backpropagation_camada_escondida(esperado, saida_output_layer, saida_hidden_layer, rede, entrada, taxa_aprendizado):       ###### sugiro quebrar a função em funções menores e chamar as coisas aos poucos pra analisar por pedaços
    '''A partir do erro (saida esperada - saida obtida), corrige os pesos da rede, quase como W = W - N * P_E,
    ou seja, peso = peso - taxa_de_aprendizado * propagação do erro'''
    for i in range (0,2):
        for j in range(0,2):#len(entrada)
            Erro_Relativo = ((saida_output_layer[0]-esperado[0])*saida_output_layer[0]*(1-saida_output_layer[0])*rede[2][i] + (saida_output_layer[1]-esperado[1])*saida_output_layer[1]*(1-saida_output_layer[1])*rede[3][i]) * saida_hidden_layer[i]*(1-saida_hidden_layer[i]) * entrada[j]
            rede[i][j] = rede[i][j] - taxa_aprendizado * Erro_Relativo
    return rede

def backpropagation(esperado, saida_output_layer, saida_hidden_layer, rede , entrada, taxa_aprendizado):
    rede = backpropagation_camada_escondida(esperado, saida_output_layer, saida_hidden_layer, rede, entrada, taxa_aprendizado)
    rede = backpropagation_camada_saida(esperado, saida_output_layer, saida_hidden_layer, rede, taxa_aprendizado)
    return rede


def cria_rede():
    rede = [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
    for i in range(0,4):
        for j in range(0,3):
            rede[i][j] = random.uniform(-1.0, 1.0)
    return rede

def Erro(esperado, saida_output_layer):
    return (esperado[0]-saida_output_layer[0])**2 + (esperado[1]-saida_output_layer[1])**2



''' Sugestões:
    Quebrar as etapas das funções
'''
