

# INSTALAR KERAS ANTES

# Instalamos python y un mini-entorno sobre el que trabajar con keras.
# install_miniconda()
# si no funciona, hacerlo desde aquí: 
# https://docs.conda.io/en/latest/miniconda.html

# install.packages("keras")
library(keras)
# install_keras() # tarda

# install.packages("tensorflow")
library(tensorflow)
# install_tensorflow()




# Ejercicio 3.2.1. Activación de la única neurona de una RN con la FdA ReLU y 
# actualización de sus pesos.

rm(list=ls(all=T)) # clears workspace

# Datos de entrada
X <- 2

# valor objetivo
y <- 4

# Inicialización de pesos y sesgos
W <- 3
b <- 1

learning_rate <- 0.1


# Cargamos la FdA ReLU para el forward

relu <- function(x) {
  pmax(0, x) # de (-1, 2, -3, 4) devolvería (0, 2, 0, 4), mientras que max(0, x) devolvería 4
}

# 1. Activación de la capa oculta con la FdA ReLU (forward pass)

# 1.1 Cálculo de la entrada ponderada (pre-activación)
Z <- X * W + b; Z  # 2*3 + 1 = 7

# 1.2 Aplicación de la función de activación -> salida de la capa oculta (A)     
A <- relu(Z); A 


# 2. Actualización de pesos y sesgos (backward pass)

# Cargamos la derivada de la FdP
relu_derivada <- function(x) {
  ifelse(x > 0, 1, 0)
}


# 2.1 Cálculo del error a partir de la derivada de la FdP con respecto a A 
error <- (y - A) # es el residuo

# 2.2 Cálculo del gradiente local (dZ)
# se obtiene a partir de la derivada de la función de activación con respecto a su entrada 
dZ <- error * relu_derivada(Z); dZ 

# 2.3 Cálculo del gradiente de los pesos (dW)
dW <- X * dZ; dW 

# 2.4 Cálculo del gradiente de sesgo (b)
db <- dZ; db # Gradiente del sesgo

# 2.5 Actualización de los pesos 
new_W <- W + learning_rate * dW; new_W 

# 2.6 Actualización de los pesos W      
new_b <- b + learning_rate * db; new_b # Nuevo sesgo

# originales
# W <- 3
# b <- 1

# esto se repite muchas veces

#----------------------------------------------------





# Ejercicio 3.2.2. Programar paso a paso una RN feedforward con una sola capa (oculta)
# y una sola neurona,sin utilizar ningún paquete.  


rm(list=ls(all=T)) # clears workspace
set.seed(2)  # Set random seed

# Datos de entrenamiento (2 predictoras y 4 muestras)
X <- matrix(c(0, 0, 1, 1, 0, 1, 1, 0), ncol = 2, byrow = T); X # Entradas

# Salidas esperadas
Y <- c(0, 1, 1, 1); Y # 4 salidas esperadas para las 4 muestras

# Inicialización de los pesos y sesgos
# Un solo conjunto de cada porque hay solo 1 neurona
# 2 pesos por neurona porque hay dos variables
W <- runif(2, -1, 1); W # 2 pesos aleatorios en el rango [-1, 1]
b <- runif(1, -1, 1); b # 1 sesgo aleatorio en el rango [-1, 1]

# Creamos la función de activación ReLU
relu <- function(x) {
  pmax(0, x) # compara cada elemento del vector con 0 y devuelve el valor de entrada si es positivo y 0 si es negativo
}

# Cargamos la derivada de la función de activación ReLU
relu_derivada <- function(x) {
  ifelse(x > 0, 1, 0) # devuelve 1 si es positivo (f'(x)=1) y 0 si es negativo
}

epochs <- 1000 # Nº de iteraciones
learning_rate <- 0.1 # Tasa de aprendizaje

# Entrenamiento de la red neuronal

for (epoch in 1:epochs) {
  
  # 1.  Activación de la capa oculta (forward pass)
  
    #   1.1 Cálculo de la entrada ponderada (pre-activación)
      Z <- X %*% W + b # %*% -> producto matricial
      
    #   1.2 Aplicación de la función de activación 
      A <- relu(Z) # Salida de la capa oculta
  
  # 2. Cálculo del error
    error <- (Y - A) # se aplica en la retropropagación
  
  # 3. Actualización de pesos y sesgos (retropropagación)
    
    # 3.1 Cálculo del gradiente local
      dZ <- error * relu_derivada(Z) 

    # 3.2 Cálculo del gradiente de los pesos
      dW <- t(X) %*% dZ # (%*% -> producto matricial)
      
# La transposición de X es necesaria para que las dimensiones de los 
# matrices (tensores si fueran de más dimensiones) sean compatibles para la multiplicación. De dicha multiplicación 
# resulta el gradiente correcto para cada peso en la red.

  
    # 3.3 Actualización de los pesos
      W <- W + learning_rate * dW
  
    # 3.4 Cálculo del gradiente de sesgo  
      db <- sum(dZ) # Se suma todos los elementos de la matriz dZ
      
    # 3.5 Actualización del sesgo
      b <- b + learning_rate * db
      
} #(final del for; su contenido se ha repetido 1000 veces)


# 4. Predicción con la RN ya entrenada
pred <- ifelse(A > 0.5, 1, 0); pred  # A se corresponde con la salida de la última iteración

Y # real
