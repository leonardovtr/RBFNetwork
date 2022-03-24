import numpy as np
import pickle as pkl

def fbr(num_da_funcao,
        X,
        C,
        sigma):

   
    # k: constante dependente de sigma
    k = 1 / (2 * sigma ** 2)

    if num_da_funcao == 1: # M: Multiquadrática
        calculo = np.sqrt((np.linalg.norm(X - C)) ** 2 +
                          (1 / k) ** 2)
    elif num_da_funcao == 2: # MR: Multiquadrática Recíproca
        calculo = 1 / (np.sqrt((np.linalg.norm(X - C)) ** 2 +
                          (1 / k) ** 2))
    elif num_da_funcao == 3: # MRI: Multiquadrática Recíproca Inversa
        calculo = 1 / (1 / k) - \
                  1 / np.sqrt((np.linalg.norm(X - C)) ** 2 +
                          (1 / k) ** 2)
    elif num_da_funcao == 4: # G: Gaussiana
        calculo = np.exp(- k * (np.linalg.norm(X - C)) ** 2)
    elif num_da_funcao == 5: # SH: Secante Hiperbólica
        calculo = 2 / \
                  (np.exp(k * (np.linalg.norm(X - C)) ** 2) +
                   np.exp(- k * (np.linalg.norm(X - C)) ** 2))
    elif num_da_funcao == 6: # CH: Cosseno Hiperbólico
        calculo = (np.exp((1 / (2 * sigma ** 2))
                         * (np.linalg.norm(X - C)) ** 2) +
                   np.exp((-1 / (2 * sigma ** 2))
                          * (np.linalg.norm(X - C)) ** 2)) / 2
    elif num_da_funcao == 7: # SPF: Splines de Placas Finas
        # lim_inf = Limite inferior
        lim_inf = 0
        # lim_sup = Limite superior
        lim_sup = 10
        global b

        np.random.seed(42)

        b = np.random.randint(lim_inf, lim_sup)
        calculo = np.linalg.norm(X - C) ** (2 * b) * \
                  np.log(np.linalg.norm(X - C))

    return calculo

