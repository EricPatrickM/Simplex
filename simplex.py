"""
    Equipe: Aqui acabo
    Aluno 1: Eric Patrick Militão
"""
import cplex
import os
import np

class Sistema:
    def __init__(self, num_Variaveis, num_Restricoes):
        self.numVariaveis = int(num_Variaveis)
        self.numRestricoes = int(num_Restricoes)
        self.numIgualdade = 0

        self.matrizB = [[0 for y in range(num_Variaveis)] for x in range(num_Restricoes)]
        self.matrizCB = []
        self.matrizCN = []
        self.resultado = [0 for x in range(num_Restricoes)]
        self.objetivo = []

        self.sentidoOriginal=""
        self.nomeVariaveis = []
        self.objetivo_copia = []
        
def lerArquivoLP(lp_file_path):
    cpx = cplex.Cplex()
    cpx.read(lp_file_path)
    artificial_necessario = False

    num_Variaveis = cpx.variables.get_num()

    #Bounds
    for i in range(num_Variaveis):
        if cpx.variables.get_lower_bounds(i) != 0:
            cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=[cpx.variables.get_names()[i]],
                val=[1])],
                senses=["G"],
                rhs=[cpx.variables.get_lower_bounds(i)])
        if cpx.variables.get_upper_bounds(i) != 1e+20:
            cpx.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=[cpx.variables.get_names()[i]],
                val=[1])],
                senses=["L"],
                rhs=[cpx.variables.get_upper_bounds(i)])
    
    num_Restricoes = cpx.linear_constraints.get_num()
    sistema = Sistema(num_Variaveis, num_Restricoes)
    sistema.nomeVariaveis = cpx.variables.get_names()
    #Pegar funcao objetivo
    for j in range(num_Variaveis):
        sistema.objetivo.append(cpx.objective.get_linear(j))

    # Verifica se o modelo é de maximização ou minimização
    if cpx.objective.get_sense() == cpx.objective.sense.maximize:
        sistema.sentidoOriginal = "maximize"
        for j in range(num_Variaveis):
            sistema.objetivo[j] *= -1
    else:
        sistema.sentidoOriginal= "minimize"

    #Pegar o resultado
    for i in range(num_Restricoes):
        sistema.resultado[i] = cpx.linear_constraints.get_rhs(i)

    #Pegar restricoes tecnicas
    for i in range(num_Restricoes):
        lin_expr = cpx.linear_constraints.get_rows(i)

        lin_expr = [(n,m) for n,m in zip(lin_expr.ind,lin_expr.val)]
        for j in lin_expr:
            sistema.matrizB[i][j[0]]=j[1]

        sentido = cpx.linear_constraints.get_senses(i)
        if sistema.resultado[i] < 0:
            if sentido == "L":
                sentido = "G"
            elif sentido == "G":
                sentido == "L"
            sistema.resultado[i] = sistema.resultado[i] * -1
            for j in range(len(sistema.matrizB[i])):
                sistema.matrizB[i][j] *= -1

        if sentido== "E":
            sistema.numIgualdade += 1
            artificial_necessario = True
            continue
        if sentido == "G":
            artificial_necessario = True

        #folga
        for x in range(num_Restricoes):
            sistema.matrizB[x].append(0)
        if sentido == "L":
            sistema.matrizB[i][-1]=1 
        else:
            sistema.matrizB[i][-1]=-1
        sistema.objetivo.append(0)

    return sistema, artificial_necessario

def definirBasicoNaoBasico(sistema):
    sistema.matrizCB = [x for x in range(sistema.numVariaveis, sistema.numVariaveis + sistema.numRestricoes)]
    sistema.matrizCN = [x for x in range(sistema.numVariaveis)]

def pegarMatriz(sistema, A):
    return np.take(sistema.matrizB, A, axis=1)

def pegarObjetivo(sistema, A):
    return np.take(sistema.objetivo, A)

def passo_1(c_B, inversa, sistema):
    solucaoBasica = np.dot(inversa, sistema.resultado)
    soma = sum(x * y for x, y in zip(c_B, solucaoBasica))
    return solucaoBasica, soma

def passo_2_1(c_B, inversa):
    return np.dot(c_B, inversa)

def passo_2_2e3(c_NB, VMSimplex, matriz_nao_basica, sistema):
    custoRelativo=[]

    for x in range(len(sistema.matrizCN)):
        custoRelativo.append(np.subtract(c_NB[x], np.dot(VMSimplex, matriz_nao_basica[:, x])))

    return np.argmin(custoRelativo), custoRelativo

def passo_4(inversa, matriz_nao_basica, k):
    return list(np.dot(inversa, matriz_nao_basica[:, k]))

def passo_5(solucaoBasica, direcaoSimplex):
    aux = []
    for x in range(len(solucaoBasica)):
        if direcaoSimplex[x] > 0 :
            if direcaoSimplex[x]==0:
                aux.append(np.inf)
            else:
                aux.append(solucaoBasica[x]/direcaoSimplex[x])
        else:
            aux.append(np.inf)
    return np.argmin(aux)

def passo_6(c_B, c_NB, sistema, matriz_basica, matriz_nao_basica, k, l):
    c_B[l], c_NB[k] = c_NB[k], c_B[l]
    sistema.matrizCB[l], sistema.matrizCN[k] = sistema.matrizCN[k], sistema.matrizCB[l]
    for x in range(len(matriz_basica)):
        matriz_basica[x][l], matriz_nao_basica[x][k] = matriz_nao_basica[x][k], matriz_basica[x][l]

def artificial(sistema):
    sistema.objetivo_copia = sistema.objetivo.copy()

    for x in range(sistema.numVariaveis+sistema.numRestricoes-sistema.numIgualdade, sistema.numVariaveis+2*sistema.numRestricoes-sistema.numIgualdade):
        for y in range(len(sistema.matrizB)):
            sistema.matrizB[y].append(0)
        sistema.matrizCB.append(x)
    sistema.matrizCN = [x for x in range(len(sistema.matrizB[0])-sistema.numRestricoes)]
    
    for x in range(len(sistema.matrizB)):
        sistema.matrizB[x][sistema.numVariaveis+sistema.numRestricoes-sistema.numIgualdade+x] = 1

    for x in range(len(sistema.objetivo)):
        sistema.objetivo[x] = 0
    for x in range(sistema.numRestricoes):
        sistema.objetivo.append(1)

    matriz_basica = pegarMatriz(sistema, sistema.matrizCB)
    matriz_nao_basica = pegarMatriz(sistema, sistema.matrizCN)
    c_B = pegarObjetivo(sistema, sistema.matrizCB)
    c_NB = pegarObjetivo(sistema, sistema.matrizCN)

    while(1):
        inversa = np.linalg.inv(matriz_basica)

        #Passo 1 Calculo solucao basica
        solucaoBasica, soma = passo_1(c_B, inversa, sistema)

        #Passo 2.1 V.M. Simplex
        VMSimplex = passo_2_1(c_B, inversa)
        
        #Passo 2.2 Custos Relativos
        k, custoRelativo = passo_2_2e3(c_NB, VMSimplex, matriz_nao_basica, sistema)

        #Passo 3
        if custoRelativo[k] >= 0:
            if len([x for x in sistema.matrizCB if x >= sistema.numVariaveis+sistema.numRestricoes-sistema.numIgualdade]) > 0:
                print("Problem infeasible")
                return False
            return True

        #Passo 4
        direcaoSimplex = passo_4(inversa, matriz_nao_basica, k)

        #Passo 5
        if len([x for x in direcaoSimplex if x > 0]) == 0:
            print("Solution unbounded")
            return False
        l= passo_5(solucaoBasica, direcaoSimplex)

        #Passo 6
        passo_6(c_B, c_NB, sistema, matriz_basica, matriz_nao_basica, k, l)
        
        if len([x for x in sistema.matrizCB if x >= sistema.numVariaveis+sistema.numRestricoes-sistema.numIgualdade]) == 0:
            return True

def main():
    sistema, artificial_necessario = lerArquivoLP('./entrada.lp')

    if(artificial_necessario):
        if not artificial(sistema):
            return
        sistema.objetivo = sistema.objetivo_copia.copy()
        for x in range(len(sistema.matrizCN)-1,-1,-1):
            if sistema.matrizCN[x] >= sistema.numRestricoes+sistema.numVariaveis-sistema.numIgualdade:
                sistema.matrizCN.pop(x)
        sistema.matrizB = [linha[:-sistema.numRestricoes] for linha in sistema.matrizB]
    else:
        definirBasicoNaoBasico(sistema)


    matriz_basica = pegarMatriz(sistema, sistema.matrizCB)
    matriz_nao_basica = pegarMatriz(sistema, sistema.matrizCN)
    c_B = pegarObjetivo(sistema, sistema.matrizCB)
    c_NB = pegarObjetivo(sistema, sistema.matrizCN)

    while(1):
        inversa = np.linalg.inv(matriz_basica)

        #Passo 1 Calculo solucao basica
        solucaoBasica, soma = passo_1(c_B, inversa, sistema)

        #Passo 2.1 V.M. Simplex
        VMSimplex = passo_2_1(c_B, inversa)
        
        #Passo 2.2 Custos Relativos
        k, custoRelativo = passo_2_2e3(c_NB, VMSimplex, matriz_nao_basica, sistema)

        #Passo 3
        if custoRelativo[k] >= 0:
            print("z: ", end="")
            print(soma*-1) if sistema.sentidoOriginal=="maximize" else print(soma)
            ordenacao = []
            for x in range(len(sistema.matrizCB)):
                if sistema.matrizCB[x] < sistema.numVariaveis:
                    ordenacao.append({"variavel":sistema.nomeVariaveis[sistema.matrizCB[x]], "resultado":solucaoBasica[x]})
            ordenacao = sorted(ordenacao, key=lambda x: x["variavel"])
            for x in ordenacao:
                print(x["variavel"], ": ", x["resultado"])
            return
 
        #Passo 4
        direcaoSimplex = passo_4(inversa, matriz_nao_basica, k)

        #Passo 5
        if len([x for x in direcaoSimplex if x > 0]) == 0:
            print("Solution unbounded")
            return False
        l= passo_5(solucaoBasica, direcaoSimplex)

        #Passo 6
        passo_6(c_B, c_NB, sistema, matriz_basica, matriz_nao_basica, k, l)


with open("entrada.lp", "w") as arquivo_saida:
    while True:
        try:
            linha = input()
            arquivo_saida.write(linha + "\n")
        except EOFError:
            break
main()
os.remove("entrada.lp")