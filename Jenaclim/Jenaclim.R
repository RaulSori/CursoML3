

# Ejercicio 3.4.2: Entrena un modelo de RNN utilizando capas LSTM 
# bidireccionales para predecir la temperatura en el siguiente punto 
# de tiempo, a partir de mediciones de temperatura de las 24 horas 
# anteriores (medidas cada 10 minutos).


 library(keras)

 rm(list=ls(all=T)) 
 set.seed(2)

 setwd("D:/Science/Curso/Jenaclim")
 jenaclim <- read.table(file="jenaclim.csv", header=T, sep=",") 
 names(jenaclim)
 head(jenaclim) # 1 hora, de 00h a 1.00 h
 
 # Nº total de mediciones en el conjunto de datos
 nmeds <- nrow(jenaclim); nmeds
 
 # Estimación del nº de días, basado en mediciones tomados cada 10 minutos
 días<-round(nmeds/(24*6)); días
 
 # Estimación del nº de años
 años<-días/365; años

 # Extraer columna de temperatura y convertir en matriz
 temperature <- as.matrix(jenaclim$temp)

 
 # Cálculo de look_back (nº de pasos de tiempo anteriores a utilizar 
 # para predecir el próximo valor.
 
 # El enunciado dice: "a partir de mediciones de temperatura de las 24 
 # horas anteriores", por lo que hay que determinar el nº de mediciones 
 # que representan un día completo (mediciones cada 10 minutos)
  
# Como las mediciones son cada 10 minutos (6/hora), 
# en 1 día se toman 6*24 = 144
 look_back <- 144 
# esto quiere decir que las temperaturas a lo largo de dia se predicen a partir
# de las temperaturas que hubo a esa misma hora, el dia anterior.
 
# Cálculo del nº de mediciones/año
 NRpA<-365*24*6; NRpA
 
# Determinar puntos de inicio para dividir los datos en segmentos
 IdA <- nmeds - 2 * NRpA  # Inicio de los 2 últimos años
 IuA <- nmeds - NRpA  # Inicio del último año

 # Segmentar los datos en conjuntos de entrenamiento, validación y prueba
 traindata <- temperature[1:(IdA - 1)] # 1 a 6º año
 valdata <- temperature[IdA:(IuA - 1)] # 7º año
 testdata <- temperature[IuA:nmeds]     # 8º año

 # Normalización de los datos
 mean <- mean(traindata)
 std <- sd(traindata)
 traindata <- (traindata - mean) / std
 valdata <- (valdata - mean) / std
 testdata <- (testdata - mean) / std

 
# Función para crear un conjunto de datos adaptado a una RNN
 
# Define una función llamada create_dataset que toma dos argumentos: 
# - data (que serán traindata, valdata y testdata).
# - look_back, definido previamente (144).
create_dataset <- function(data, look_back) { 
  
  # Inicializa dos listas vacías, X para almacenar las secuencias de 
  # entrada e Y para almacenar los valores objetivo correspondientes.
  X <- list()
  Y <- list()
  
  for (i in 1:(length(data) - look_back)) {
    timesteps <- data[i:(i + look_back - 1)]
    X[[length(X) + 1]] <- timesteps # Extrae una secuencia de 144 mediciones 
  # de temperatura comenzando desde la i-ésima medición
    Y[[length(Y) + 1]] <- data[i + look_back]} # Extrae la temperatura 
  # inmediatamente después de la secuencia anterior (paso i + 144)
   
  # solo para clase 
  # i=1; i:(i + look_back - 1) # El 1er elemento de X contiene las primeras 144 mediciones de data
  # i + look_back # El primer elemento de Y contiene la temperatura en el paso de tiempo 145
  
  # Convertir las listas a tensores 3D y 2D (vector) respectivamente
  X_matrix <- array(unlist(X), dim = c(length(X), look_back, 1))
  Y_vector <- unlist(Y)
  # Cada tensor 3D corresponde a una secuencia de datos de entrada, 
  # y cada elemento en Y_vector al valor de Tª que debe predecir 
  
  list(X_matrix, Y_vector)
}

# Aplicación de la función para crear los conjuntos de datos de entrenamiento
train_set <- create_dataset(traindata, look_back)
X_train <- train_set[[1]]
y_train <- train_set[[2]]

val_set <- create_dataset(valdata, look_back)
X_val <- val_set[[1]]
y_val <- val_set[[2]]

test_set <- create_dataset(testdata, look_back)
X_test <- test_set[[1]]
y_test <- test_set[[2]]


# Función para construir el modelo RNN con capas LSTM bidireccionales
build_model <- function() {
  
  model <- keras_model_sequential() 
  
  model <- bidirectional(model,layer_lstm(units = 100, return_sequences = TRUE, 
                               input_shape = c(look_back, 1)), merge_mode = "concat")
  # merge_mode = "concat" toma las salidas de las direcciones hacia adelante y hacia 
  # atrás y las pega una tras otra, proporcionando al modelo una visión más completa.
  # Hay otras opciones.
  
  model <- bidirectional(model,layer_lstm(units = 50), merge_mode = "concat")
  model <- layer_dense(model,units = 50, activation = "relu")
  model <- layer_dense(model,units = 1)
  
  optimizer = optimizer_adam(learning_rate = 0.001)
  
  # Compilamos el modelo
  compile(model, loss = "mse",optimizer = optimizer, metrics = c("mae"))
  
  return(model)
}

# Crear un objeto ModelCheckpoint
checkpoint_callback <- callback_model_checkpoint(
  filepath = "best_model_JClim.h5",      # Ubicación donde guardar el modelo
  monitor = "val_loss",            # Métrica a monitorizar (puede ser "val_accuracy", "val_mae", etc.)
  save_best_only = TRUE,           # Guarda solo el modelo que tiene el menor valor de "val_loss"
  verbose = 1)

model <- build_model()

epochs = 20

# Train the model and capture the history
history <- fit(model, x=X_train, y=y_train, epochs = epochs, 
               batch_size = 128, validation_data = list(X_val, y_val),
               callbacks = list(checkpoint_callback))

# 
# Tardó más de24 horas y se entrenó sin  checkpoint_callback, por lo que 
# sobreajusta y su capacidad predictiva es muy baja

# keras::save_model_hdf5(model, "model_JClim")
# saveRDS(history, "history_JClim.rds")
# model <- keras::load_model_hdf5("model_JClim")
 history <- readRDS("history_JClim.rds")

# Representamos mse (la FdP) sobre los datos de prueba y validación
trainmse <- history$metrics$loss # nos quedamos con mse
valmse <- history$metrics$val_loss # nos quedamos con mse

n=1 # Inicio del rango de épocas para el gráfico
# ------------------ todo de golpe -----------
windows()
plot(c(n:epochs), trainmse[n:epochs], xlab = "epoch", type = 'l', 
     ylim = range(c(trainmse[n:epochs], valmse[n:epochs])), col = "blue", ylab = "mse", 
     main = "MSE")
lines(c(n:epochs), valmse[n:epochs], col = "red", type = 'l')
legend("topright", legend = c("Entrenamiento", "Validación"), 
       col = c("blue", "red"), lty = 1, cex = 0.8)

# Encuentra el punto de menor MSE en la validación
min_epoch <- which.min(valmse[n:epochs]) + n - 1; min_epoch
# 14

# Añade una línea vertical en el punto de menor MSE
abline(v = min_epoch, col = "darkgreen", lty = 3, lwd = 1)  
# --------------------------------------------


# Representamos también mae ((la métrica) sobre los datos de prueba y validación
trainmae <- history$metrics$mae # nos quedamos con mae
valmae <- history$metrics$val_mae # nos quedamos con mae

n=1 # Inicio del rango de épocas para el gráfico
# ------------------ todo de golpe -----------
windows()
plot(c(n:epochs), trainmae[n:epochs], xlab = "epoch", type = 'l', 
     ylim = range(c(trainmae[n:epochs], valmae[n:epochs])), col = "blue", ylab = "mae", 
     main = "MAE")
lines(c(n:epochs), valmae[n:epochs], col = "red", type = 'l')
legend("topright", legend = c("Entrenamiento", "Validación"), 
       col = c("blue", "red"), lty = 1, cex = 0.8)

# Encuentra el punto de menor MSE en la validación
min_epoch <- which.min(valmae[n:epochs]) + n - 1; min_epoch
# 14

# Añade una línea vertical en el punto de menor MSE
abline(v = min_epoch, col = "darkgreen", lty = 3, lwd = 1)  
# --------------------------------------------


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
best_model <- keras::load_model_hdf5("best_model_JClim.h5")

# Cargar el modelo con el mejor desempeño (que en este caso no lo es)
pred <- predict(best_model, X_test)

# Tarda, cargar abajo

# saveRDS(pred, "pred.rds")
pred <- readRDS("pred_JClim.rds")

# Revertir la estandarización para obtener predicciones en la escala original
pred <- (pred * std) + mean

# Cálculo de R² para evaluar la capacidad predictiva del modelo
R2 = as.numeric(cor(pred, y_test))^2; round(R2, 2)
# 0.2

# Cálculo del RMSE para evaluar el error de predicción del modelo
rmse <- sqrt(mean((pred - y_test)^2)); round(rmse)
# 17

