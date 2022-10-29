import random


def max_cut(k,G, pesos):
    """
    k: entero
        el peso del corte
    G: Diccionario 
        Gráfica en forma de lista de nodos
    """
    # Paso no determinista
    n = len(G.keys())
    V1 = set()
    for k in G.keys():
        nodo =  k if random.uniform(0,1) >=  1/n else set()
        if isinstance(nodo,int):
            V1 = V1 | { nodo }
    
    #Paso determinista
    #verifica que el peso del corte sea >= k
    j = 0
    for v in V1:
        vecinos  = G[v]
        for u in vecinos:
            #u está en V2
            if u not in V1:
                j+= pesos[(v,u)]

    if j >= k:
        print("V1:",V1)
        print("j:", j )

    return j >= k 

MAX_STEPS = 1000
def decide(k,G,pesos):
    for i in range(MAX_STEPS):
        if max_cut(k,G,pesos ):
            print("Verdadero")
            return True
    print("No se pudo determinar.")

if __name__ == "__main__":
    G1 = {
        1:[2,4],
        2:[1,4,3],
        3:[2],
        4:[1,2],
    }

    pesos = {
        (1,2):1,
        (2,1):1,
        (2,4):1,
        (4,2):1,
        (1,4):2,
        (4,1):2,
        (2,3):2,
        (3,2):2
    }

    k = 2
    print(f"Grafica G1, k: {k} ")
    decide(k,G1,pesos)



    G2 = {
        1:[2,3,4],
        2:[1,],
        3:[1,4],
        4:[1,3],
    }

    pesos2 = {
        (1,2):1,
        (1,3):1,
        (1,4):1,
        (2,1):1,
        (3,1):1,
        (4,1):1,
        (3,4):5,
        (4,3):5
    }

    k2 = 4
    print(f"Grafica G2, k: {k2} ")
    decide(k2,G2,pesos2)

    k2 = 8
    print(f"Grafica G2, k: {k2} ")
    decide(k2,G2,pesos2)

    


