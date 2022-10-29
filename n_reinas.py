import random
from typing import List
import numpy as np

class Algos:
    def __init__(self,n:int,  n_soluciones:int = 1) -> None:
        """
        n: 
        tamaño del tablero 
        n_soluciones: 
            número de soluciones que produce.
        """
        self.padre_x = [] 
        self.padre_y = []

        #probabilidad de que un individuo mute.
        self.prob_mutacion = 0.1

        #tamaño de la población
        self.tamaño_pob  = n*n 

        self.poblacion = None 

        #número de reinas
        self.n_reinas = n 
        self.n_soluciones = n_soluciones
        self.soluciones = None 

        #esta cantidad se logra cuando no existen conflictos entre las n reinas
        if self.n_reinas <2:
            self.max_fitness = 0 
        if n== 2:
            self.max_fitness= 2
        else:
            self.max_fitness = self.n_reinas*(self.n_reinas - 1) /2

        self.indices = list(range(self.n_reinas))
    def max_ataques(self,n:int):
        '''
        Calcula el máximo número de pares de reinas 
        que se pueden atacar entre si.
        Se calcula de la combinaciones de 2 en n.
        '''
        if n < 2:
            return 0
        if n == 2:
            return 1
        return (n - 1) * n / 2

         
    def costo(self,ind:List):
        """
        ind: un tablero de ajedrez

        función de costo u objetivo 
        Regresa el numero de pared de reinas que se atacan 

        """
        C = 0
        izq = {}
        der = {}

        for j in ind:
            #renglon de la reina en columna j 
            j = j-1 # 0, n_reinas-1
            i = ind[j]-1 # 0, n_reinas-1
            ind_izq = j - i 
            ind_der = j + i

            if ind_izq not in izq:
                izq[ind_izq] = 1 
            else:
                izq[ind_izq]+= 1

            if ind_der not in der:
                der[ind_der] = 1 
            else:
                der[ind_der]+= 1

        for j in izq:
            C += self.max_ataques(izq[j])
        
        for k in der:
            C += self.max_ataques(der[k])

        return C


    def fitness(self,individuo:List):   
        """
        individuo: 
            vector (lista) con las posiciones de las 8 reinas.
        Calcula el número de conflictos entre las reinas 
        Mientras mas cerca de max_fitness mejor.
        Si es max_fitness entonces no hay conflicto entre las reinas del tablero.
        """
       
        colisones_h = sum([individuo.count(reina)-1 for reina in individuo])/2

        dia_izq = [0] * 2*self.n_reinas
        dia_der = [0] * 2*self.n_reinas
     
   
        for i in range(self.n_reinas):
            dia_izq[i + individuo[i] - 1] += 1
            dia_der[self.n_reinas - i + individuo[i] - 2] += 1

        colisiones_dia = 0 

        for i in range(2*self.n_reinas-1):
            c = 0
            if dia_izq[i] > 1:
                c += dia_izq[i]-1
            if dia_der[i] > 1:
                c += dia_der[i]-1

            colisiones_dia += c / (self.n_reinas-abs(i-self.n_reinas+1))

        return int(self.max_fitness - (colisones_h + colisiones_dia))
 

    def probabilidad(self,ind:List) -> float:
        """
        ind: 
            configuración del tablero con n reinas.

        regresa la probabilidad de que el ind 
        sea seleccionado según su aptitud
        """
        return self.fitness(ind)/self.max_fitness

    def selecciona(self,poblacion:List, probabilidades:List):
        """
        Selecciona 2 individuos de la población
        con respecto de la distribucion de probabilidad.
        Regresa dos individuos de la población.
        """

        i,j = np.random.choice(self.indices,2,probabilidades)
        ind_x,ind_y = poblacion[i], poblacion[j]

        return ind_x,ind_y

    def reproduce(self, padre_x:List, padre_y:List):
        """
        Produce un nuevo individuo.
        combinando  dos padres x & y usando el 
        crossover. Por ejemplo 
        n_reinas = 10, crossover = 5 
            la mitad de genes de un padre y la mitad del otro.

        """
        #crossover aleatorio
        crossover = random.randint(0, self.n_reinas -1)

        #selecciona genes de ambos individuos al azar
        #de 0 a crossover  los del padre x serán seleccionados, después se seleccionan los del  padre y
       
        #nuevo individuo
        ind = padre_x[0:crossover] + padre_y[crossover:self.n_reinas]

        #assert len(ind) == self.n_reinas,f"Produjo un hijo vacio. {ind}"

        return ind 

    def produce(self):
        """
        Produce una nueva población
        combinando  individuos de la población actual

        Regresa los valores de aptitud (lista) y el indice de la solución (si no hay es None)
        valores_fit,resultado

        """
        poblacion = []

        valores_fit = []

        resultado = None 
        #calcula la probabilidad de un individuo 
        #para ser elegido según su aptitud (fitness)
        probs =  [self.probabilidad(ind) for ind in self.poblacion]
        Z = sum(probs)#constante de normalización 
        probs = [x/Z for x in probs]
     
        for i in range(self.tamaño_pob):
            padre_x,padre_y = self.selecciona(self.poblacion, probs)
            
            hijo = self.reproduce(padre_x,padre_y)

            if random.random() < self.prob_mutacion:
                hijo = self.muta(hijo)
            
            poblacion.extend([hijo])
            fit = self.fitness(hijo)
            valores_fit.append(fit)

            if abs(fit-self.max_fitness) < 0.1:
                print("solucion: ",hijo,fit, self.max_fitness)

            if fit  == self.max_fitness:
                print("encontro solucion")
                print(f"fitness {self.fitness(hijo)}, max_fitness: {self.max_fitness}")
                resultado = i
                break 

        self.poblacion = poblacion
        
        return valores_fit,resultado

    def heuristica(self,individuo:List):
        """
        Calcula el número de conflictos entre reinas en 
        esta instancia del juego de la n reinas.

        Regresa una lista de enteros, donde cada entrada i es el número d conflictos de cada reina con otras en el tablero.
        """
        valor = []
        for i in range(self.n_reinas):
            j = i - 1
            valor.append(0)
            while j >= 0:
                if individuo[i] == individuo[j] or (abs(individuo[i] - individuo[j]) == abs(i - j)):
                    valor[i] += 1
                j -= 1
            j = i + 1
            while j < self.n_reinas:
                if individuo[i] == individuo[j] or (abs(individuo[i] - individuo[j]) == abs(i - j)):
                    valor[i] += 1
                j += 1
        return valor

    def muta(self,ind:List):
        """
        ind: Lista  (vecotr) de enteros
        Muta los genes de un individuo .
        Regresa un  individuo con su cromosoma modificado.
        """
        #crea una copia del individuo
        nuevo_individuo = ind[:]

        #cambia un gen del individuo de manera aleatoria
        nuevo_individuo[random.randint(0,self.n_reinas-1)] = random.randint(1, self.n_reinas)

        return nuevo_individuo
        
    def imprime_soluciones(self):
        """
        Imprime los tableros que son solución al problema de las
        ocho reinas
        """
        assert self.soluciones is not None, "No hay soluciones disponibles. Ejecuta el método self.resuelve o self.resuelve_sa."
    
        print("*******************************SOLUCIONES*******************************")

        print("TABLERO:")
        for k,sol in enumerate(self.soluciones):
            print(f"Solución {k + 1}: " +  str(sol))
            for i in range(self.n_reinas):
                x = sol[i] - 1
                for j in range(self.n_reinas):
                    if j == x:
                        print('[Q]', end='')
                    else:
                        print('[ ]', end='')
                print()
            
            print("\n\n")
    def individuo_aleatorio(self):
        """
        Regresa un nuevo individuo
        Un vector de enteros.
        """ 
        ind =  [random.randint(1,self.n_reinas) for _ in range(self.n_reinas)]

        return ind 

    def resuelve(self):
        print("Resolviendo...")
        #soluciones
        sols  = []

        while len(sols) < self.n_soluciones:
            poblacion = []
            generacion =  1
            #Genera configuraciones del tablero
            #con n reinas de manera aleatoria
            for i in range(self.tamaño_pob):
                poblacion.append(
                    self.individuo_aleatorio())
            self.poblacion = poblacion
            #Calcula la función fitness (aptitud)
            #para cada individuo
            valores = [self.fitness(ind) for ind in poblacion ]

            #Mientras la función de aptitud sea diferente de max_fitness para todos los individuos
            #continua la reproducción de la   poblacion.
            while not self.max_fitness in valores:
                valores, indice  = self.produce()
                print(f"max fitness {max(valores)} in gen: {generacion} ", self.max_fitness)
                if indice is not None:
                    if self.poblacion[indice] not in sols:
                        sols.append(self.poblacion[indice])
        
                generacion+= 1 

        self.soluciones = sols
        self.imprime_soluciones()

    def resuelve_sa(self, temperatura: int = 4000, descuento:float= 0.99):
        """
        temperatura
            numero de pasos 
        descuento
            factor de descuento 
        """
        
        ind = list(range(1,self.n_reinas+1))
        #mezcla in-place
        random.shuffle(ind)

        costo_ind = self.costo(ind)
        self.soluciones = []

        while temperatura > 0:
            temperatura = temperatura *descuento
            estado = ind[:] 

            while True:
                indice0, indice1 = random.randrange(0, self.n_reinas - 1),random.randrange(0, self.n_reinas - 1)
                if indice0 != indice1:
                    break 
                
            #intercambia los 2 elementos
            #de manera simultanea
            estado[indice0], estado[indice1] = estado[indice1],estado[indice0] 

            #diferencia en el costo  del nuevo estado con el anterior
            diff   = self.costo(estado) - costo_ind
            #print("temperatura: ",temperatura)
            #print("costo estado: ",self.costo(estado) )

            #Acepta un estado que es peor (mayor costo) con probabilidad
            # e^(-d/pasos) para evitar minimos locales
            if diff < 0 or random.uniform(0,1) < np.exp(-diff/temperatura):
                ind = estado[:]
                costo_ind = self.costo(ind)
            
            #encontramos una solución
            #pues el costo es mínimo (no hay reinas que se ataquen)
            if costo_ind == 0:
                self.soluciones.append(ind)
                break 
        self.imprime_soluciones()
        if len(self.soluciones) == 0:
            print(f"No se encontro solución, trata de aumentar la temperatura: {temperatura}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_reinas", required = True)
    parser.add_argument("-m", "--metodo", required = True)

    args = parser.parse_args()

    print(f'Número de reinas: {args.n_reinas}')
    print()
    
    n = args.n_reinas
    metodo = args.metodo
    A = Algos(n = int(n) ,n_soluciones = 1)

    assert  metodo in  ["ag","rs"],f"Metodo no válido: {metodo}: usa ag para algoritmo genético o rs para recocido simulado."

    if metodo == "ag":
        A.resuelve()

    if metodo == "rs":
        A.resuelve_sa()
    

    #B =  Algos(n = 4 ,n_soluciones = 1)
    #print(A.costo([2,4,1,3])) #0
    #print(B.costo([2,2,4,4]))




        