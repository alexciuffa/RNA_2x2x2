##################################################################
##                                                              ##
##      Codigo criado por Alexandre Xavier Ciuffatelli          ##
##      https://www.facebook.com/alexandre.ciuffatelli          ##
##      Todos podem usar este codigo livremente                 ##
##      Não comercialize esse código, grato                     ##
##                                                              ##
##################################################################

import xlrd

def xlread_linha(arq_xls):#le a linha
    """
    Gerador que le arquivo .xls
    """

    # Abre o arquivo
    xls = xlrd.open_workbook(arq_xls)
    # Pega a primeira planilha do arquivo
    plan = xls.sheets()[0]

    # Para i de zero ao numero de linhas da planilha
    for i in range(plan.nrows):
        # Le os valores nas linhas da planilha
        yield plan.row_values(i)
        
#for linha in xlread_linha('BD.xlsx'):
#    print(linha)

def xlread_coluna(arq_xls):#le a coluna
    """
    Gerador que le arquivo .xls
    """

    # Abre o arquivo
    xls = xlrd.open_workbook(arq_xls)
    # Pega a primeira planilha do arquivo
    plan = xls.sheets()[0]

    # Para i de zero ao numero de linhas da planilha
    for i in range(plan.ncols):
        # Le os valores nas linhas da planilha
        yield plan.col_values(i)

#for linha in xlread_coluna('arquivo.xlsx'):
#    print(linha)
