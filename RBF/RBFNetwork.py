import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from funcoes import fbr
from sklearn.metrics import r2_score


def polos(x,y,n_neuronios, ativacao, epochs):


    np.random.seed(epochs)
        
    C = np.zeros((n_neuronios,x.shape[1]))
    for i in range(n_neuronios):
        C[i] = np.random.uniform(np.min(x), np.max(x), x.shape[1])
        
    #C[0] = 0 Um polo centrado na origem em alguns casos garante uma curva mais suave

        
    dist = np.zeros((n_neuronios, n_neuronios))
        
    for i in range(0,n_neuronios,1):
        for j in range(0,n_neuronios,1):
            dist[i,j] = np.linalg.norm(C[i] - C[j])
                
                
    dist_max = np.max(dist)
    sigma = dist_max / np.sqrt(2*n_neuronios)
        
    G = np.zeros((x.shape[0], n_neuronios))
        
    for n in range(0, x.shape[0],1):
        for i in range(0, n_neuronios,1):
            G[n,i] = fbr.fbr(ativacao,x[n],C[i], sigma)
                
                
    vetor_unitario = np.ones((x.shape[0],1))
        
    G = np.hstack((G, vetor_unitario))
        
    matriz_de_moore_penrose = np.linalg.pinv(G)
        
    sinapses = np.matmul(matriz_de_moore_penrose, y)
            
    y_pred = np.zeros((x.shape[0],1))
        
    for i in range(0,x.shape[0],1):
        y_pred[i] = sum(sinapses*G[i])
            
    resultado = r2_score(y, y_pred).round(2)
        
    return resultado


def melhor_polo(x,y,n_neuronios,ativacao, n_polos):
    r = []
    seeds = []
    for n_polos in range(0,n_polos):
        r.append(polos(x,y,n_neuronios,ativacao,n_polos))
        seeds.append(n_polos)
        df = pd.DataFrame(r, columns = ['r2'])
        df['semente'] = seeds  
        a = df[df['r2'] == df['r2'].max()]['semente']
    
    return np.max(a)




def treinamento(x,y,n_neuronios, ativacao, n_polos):

    semente = melhor_polo(x,y,n_neuronios,ativacao,n_polos)
    
    np.random.seed(semente)
        
    C = np.zeros((n_neuronios,x.shape[1]))
    for i in range(n_neuronios):
        C[i] = np.random.uniform(np.min(x), np.max(x), x.shape[1])
        
    #C[0] = 0

        
    dist = np.zeros((n_neuronios, n_neuronios))
        
    for i in range(0,n_neuronios,1):
        for j in range(0,n_neuronios,1):
            dist[i,j] = np.linalg.norm(C[i] - C[j])
                
                
    dist_max = np.max(dist)
    sigma = dist_max / np.sqrt(2*n_neuronios)
        
    G = np.zeros((x.shape[0], n_neuronios))
        
    for n in range(0, x.shape[0],1):
        for i in range(0, n_neuronios,1):
            G[n,i] = fbr.fbr(ativacao,x[n],C[i], sigma)
                
                
    vetor_unitario = np.ones((x.shape[0],1))
        
    G = np.hstack((G, vetor_unitario))
        
    matriz_de_moore_penrose = np.linalg.pinv(G)
        
    sinapses = np.matmul(matriz_de_moore_penrose, y)
            
    y_pred = np.zeros((x.shape[0],1))
        
    for i in range(0,x.shape[0],1):
        y_pred[i] = sum(sinapses*G[i])
     
    
    return sinapses, C, y_pred.flatten(),sigma


def teste(x,n_neuronios, C, sinapses,ativacao):
    
    dist = np.zeros((n_neuronios, n_neuronios))
        
    for i in range(0,n_neuronios,1):
        for j in range(0,n_neuronios,1):
            dist[i,j] = np.linalg.norm(C[i] - C[j])
                
                
    dist_max = np.max(dist)
    sigma = dist_max / np.sqrt(2*n_neuronios)
        
    G = np.zeros((x.shape[0], n_neuronios))
        
    for n in range(0, x.shape[0],1):
        for i in range(0, n_neuronios,1):
            G[n,i] = fbr.fbr(ativacao,x[n],C[i], sigma)
            
    
    vetor_unitario = np.ones((x.shape[0],1))
        
    G = np.hstack((G, vetor_unitario))
    
    y_pred = np.zeros((x.shape[0],1))
        
    for i in range(0,x.shape[0],1):
        y_pred[i] = sum(sinapses*G[i])
    
    
    return y_pred.flatten()

