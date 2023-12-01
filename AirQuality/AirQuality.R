
# Ejercicio 3.4.1: Entrenar un modelo de Red Neuronal Recurrente (RNN) 
# utilizando capas LSTM para predecir el nivel de ozono a partir de 
# ciertas características ambientales e indicadores de contaminación. 
# El objetivo es comprender cómo ciertas condiciones ambientales pueden 
# influir en la concentración de ozono y prever su nivel en el futuro 
# basándonos en datos históricos.  

library(keras)

rm(list=ls(all=T)) 
set.seed(3) 

setwd("D:/Science/Curso/AirQuality")
AirQuality <- read.table(file="AirQuality.csv",header=T,sep=",") 

data <- AirQuality

names(data)


#             ¡Diferencia fundamental!

# División de datos en conjuntos de entrenamiento y prueba (80% - 20%)
# A diferencia de una división aleatoria, esta división es secuencial 
# para simular una predicción de "futuro" sobre las muestras test.

t <- floor(0.8 * nrow(data)) # floor en vez de sample (  )
traindata <- data[1:t, ] # en vez de [t, ]
testdata  <- data[(t + 1):nrow(data), ] #  en vez de [-t, ]

# Selección de características predictoras
predictors <- c("T","RH","AH","CO","NOx","NO2")
target <- c("O3") 

# Separación de características y objetivo
traindatax <- traindata[, predictors]
traindatay <- as.matrix(traindata[, target])
testdatax <- testdata[, predictors]
testdatay <- as.matrix(testdata[, target])

# Estandarización de características respecto a datos de entrenamiento
meanx <- apply(traindatax, 2, mean) 
sdx <- apply(traindatax, 2, sd)
traindatax <- scale(traindatax, center = meanx, scale = sdx)
testdatax <- scale(testdatax, center = meanx, scale = sdx)


# ¡Diferente! Estandarización de las etiquuetas respecto a datos de entrenamiento
# No es exclusivo de las RNN, pero sí recomendable en modelos complejos que 
# manejan secuencias de datos, en los que la escala afecta al rendimiento.

meany <- apply(traindatay, 2, mean) 
sdy <- apply(traindatay, 2, sd)
traindatay <- scale(traindatay, center = meany, scale = sdy)
testdatay <- scale(testdatay, center = meany, scale = sdy)


# ¡FUNDAMENTAL!

dim (traindatax)
# 7485    6

# En las feedforward cada muestra es una entrada independiente y los 
# datos pueden (y deben) estar en forma de matriz 2D (como hasta ahora).
# En las LSTM las filas no son independientes, pues suponen una secuencia 
# (p. ej. temporal) y por eso los datos deben tener una estructura 3D 
# la transformación de 2D (muestras por características) a 3D
# (muestras, pasos de tiempo, características) se hace con la función array

traindatax <- array(traindatax, dim = c(nrow(traindatax), 1, ncol(traindatax)))
testdatax <- array(testdatax, dim = c(nrow(testdatax), 1, ncol(testdatax)))

dim (traindatax)
# 7485    1    6

# Se crea un tensor 3D, de forma  (7485,1,6) donde cada dimensión representa:
# muestras, pasos de tiempo, características, respectivamente
# Cada muestra, un tensor 2D, de 1X6, se considera ahora como una secuencia 
# de un solo paso en el tiempo X 6 características. 


# Función para construir el modelo LSTM
build_model <- function() {
  
  model <- keras_model_sequential()
  # 1ª LSTM, con input_shape. return_sequences determina si se debe retornar la 
  # secuencia completa de salidas a lo largo de todos los pasos de tiempo
  model <- layer_lstm(model, units = 100, return_sequences = TRUE, 
                      input_shape = tail(dim(traindatax), 2))# input_shape 
  # es la forma del tensor de entrada, sin incluir el nº de muestras

  # Añadir Dropout para reducir el sobreajuste
  model <- layer_dropout(model, rate = 0.2) 
  # Segunda capa LSTM con 50 unidades
  model <- layer_lstm(model, units = 50, return_sequences = FALSE) 
  # Capa densa intermedia para aprendizaje no secuencial
  model <- layer_dense(model, units = 50, activation = 'relu') 
  # Añadir Dropout después de la capa densa también puede ser beneficioso
  model <- layer_dropout(model, rate = 0.2) 
  # Capa de salida para predicción final, sin FdA para salida continua
  model <- layer_dense(model, units = length(target))
  # Optimizador a utilizar en la compilación
  optimizer = optimizer_adam(learning_rate = 0.001)
  # Compilación del modelo
  model <- compile(model, optimizer = optimizer, loss = "mse", metrics = "mae") 
  
  return(model)
}


# Crear un objeto ModelCheckpoint
checkpoint_callback <- callback_model_checkpoint(
  filepath = "best_model_AQ.h5",  # Ubicación donde guardar el mejor modelo
  monitor = "val_loss",        # Métrica a monitorizar
  save_best_only = TRUE,       # Guar¿correcto?da solo el modelo que tiene el menor valor de "val_loss"
  verbose = 1,                 # Muestra detalles durante el entrenamiento
)


# creamos el modelo
model <- build_model()

epochs = 50

# Entrenamiento del modelo utilizando un conjunto de validación del 20% y los callbacks definidos

history <- fit(model, 
               traindatax, 
               traindatay, 
               epochs = epochs, 
               initial_epoch = 0,  # Iniciar desde la época 1
               batch_size = 32, 
               validation_split = 0.2, 
               callbacks = list(checkpoint_callback))


# keras::save_model_hdf5(model, "model_AQ") # para salvar el modelo entrenado
# saveRDS(history, "history_AQ.rds") # para salvar history

# para cargarlos
# model <- keras::load_model_hdf5("model_AQ")
# history <- readRDS("history_AQ.rds")

names(history$metrics)

# Representamos mse (la FdP) sobre los datos de prueba y validación
trainmse <- history$metrics$loss # nos quedamos con mse
valmse <- history$metrics$val_loss # nos quedamos con mse

# ------------------ todo de golpe -----------
windows()
plot(c(1:epochs), trainmse[1:epochs], xlab = "epoch", type = 'l', 
     ylim = range(c(trainmse, valmse)), col = "blue", ylab = "mse", 
     main = "MSE")
lines(c(1:epochs), valmse[1:epochs], col = "red", type = 'l')
legend("topright", legend = c("Entrenamiento", "Validación"), 
       col = c("blue", "red"), lty = 1, cex = 0.8)
# --------------------------------------------
which.min(valmse)
# 15

# Representamos también mae ((la métrica) sobre los datos de prueba y validación
trainmae <- history$metrics$mae # nos quedamos con mae
valmae <- history$metrics$val_mae # nos quedamos con mae

# ------------------ todo de golpe -----------
windows()
plot(c(1:epochs), trainmae[1:epochs], xlab = "epoch", type = 'l', 
     ylim = range(c(trainmae, valmae)), col = "blue", ylab = "mae", 
     main = "MAE")
lines(c(1:epochs), valmae[1:epochs], col = "red", type = 'l')
legend("topright", legend = c("Entrenamiento", "Validación"), 
       col = c("blue", "red"), lty = 1, cex = 0.8)
# --------------------------------------------
which.min(valmae)
# 15

# **********************
#  en caso de que el gráfico sea confuso se puede promediar intervalos
ndiv<-3 # iteraciones por intervalo
segmae <- split(valmae, ceiling(seq_along(valmae)/ndiv))
meanmaes <- sapply(segmae, mean)
epoch <- seq(1, epochs, by = ndiv)
windows();plot(epoch, meanmaes, type = "b", col = "red", pch = 19,cex = 0.5,
               lty = 1,lwd = 1, xlab = "Epoch", ylab = "Mean MAEs",
               cex.lab = 1.2,  cex.main = 1.5,  cex.axis = 1.1, main = "MAE")         

# # si valores iniciales muy altos no dejan apreciar la curva
# n<-3 # nº de intervalos que se dejan de representar
# # Creamos el vector del eje x
# epoch <- seq(n*ndiv+1, epochs, by = ndiv)
# # Ahora podemos representar las medias
# windows();plot(epoch, meanmaes[-(1:n)], type = "b", col = "red", pch = 19,
#                cex = 0.5,lty = 1, lwd = 1, xlab = "Epoch", ylab = "Mean MAEs",
#                cex.lab = 1.2,  cex.main = 1.5,  cex.axis = 1.1, main = "MAE")         
# **********************  



# Cargar el modelo con el mejor desempeño
best_model <- keras::load_model_hdf5("best_model.h5")

# Realizar las predicciones con el mejor modelo
preds <- predict(best_model, testdatax)

# Calcular R2
rsq = as.numeric(cor(preds, testdatay))^2; round(rsq,2)

# 0.76

rmse = sqrt(mean((preds - testdatay)^2)); round(rmse,2)

# 0.69

# Visualización de predicciones vs valores reales
windows();plot(testdatay, preds, main = "Real vs Predicho", 
               xlab = "Valores reales", ylab = "Predicciones",
               xlim = range(c(testdatay, preds)), 
               ylim = range(c(testdatay, preds)))
abline(a = 0, b = 1)





