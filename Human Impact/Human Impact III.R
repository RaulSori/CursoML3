



#------------    DpL -  Clasificación binaria   -----------------



# Ejercicio 3.3.3. Entrenar un modelo de RNF para predecir si una comunidad estará 
# depauperada a partir de una serie de factores de impacto  humano.  
# Optimizar el nº de épocas con checkpoint_callback. 

library(keras)
library(irr)          # para obtener kappa


rm(list=ls(all=T)) # clears workspace
set.seed(3)  # Set random seed

setwd("D:/Science/Curso/Human Impact")
HImp <- read.table(file="HImpdata.csv",header=T,sep=",") 
HImp <- HImp[complete.cases(HImp), ]  # to delete cases with NA?s 

# La variable objetivo no se convierte en factor, como en el resto 
# de métodos, sino que se deja como integer

data <- subset(HImp, select = -c(lon, lat))

# Preparación de los datos

# división en train & val data
m <- sample.int(n=nrow(data), size=floor(.8*nrow(data)),replace = F)
traindata <- data[m,]
testdata <- data[-m,]

# Cargamos la función bsample
source("D:/Science/Curso/bsample.R") 

table(traindata$Dep)
mean<-mean(table(traindata$Dep)); mean
traindata <- bsample(traindata,'Dep',mean) 
table(traindata$Dep)

# Separa los predictores de la var objetivo y los transforma
traindatax <- as.matrix(subset(traindata, select=-Dep))# los predictores como matriz
traindatay <- as.numeric(as.character(traindata$Dep))
testdatax <- as.matrix(subset(testdata, select=-Dep))
testdatay <- as.numeric(as.character(testdata$Dep))

# Normalizar los datos

mean <- apply(traindatax, 2, mean) # apply es para aplicar mean a todas las predictoras (columnas) 
sd <- apply(traindatax, 2, sd)
traindatax <- scale(traindatax, center = mean, scale = sd)
testdatax <- scale(testdatax, center = mean, scale = sd)


# Construir el modelo de clasificación

build_model <- function() {
  model <- keras_model_sequential()
  
# 2 capas ocultas con 64 neuronas cada una y una capa de salida.
# La FdA de ambas capas ocultas es ReLU.

# En la 1ª se indican las dimensiones del tensor de entrada
  model <- layer_dense(model, units = 64, activation = "relu",
                       input_shape = dim(traindatax)[2])
  model <- layer_dense(model, units = 64, activation = "relu")
  
# La de salida tiene activación sigmoid porque es clasificación binaria.
  model <- layer_dense(model, units = 1, activation = "sigmoid")

# Una TdA de 0.001.
# loss = "binary_crossentropy" porque es una clasificación binaria.
# Además de la métrica accuracy, vamos a calcular kappa, pero en Keras no tenemos 
# una función directa para kappa, por lo que utilizaremos accuracy para 
# monitorizar y luego calcularemos kappa de manera separada.  
  model <- compile(model, optimizer = optimizer_adam(learning_rate = 0.001), 
                   loss = "binary_crossentropy", metrics = c("accuracy")) 
  
  return(model)
}

# ¿por qué no usar loss =  "accuracy"?
# "accuracy" es una métrica discreta, no es diferenciable y no podría usarse como FdP

# Para que guarde automáticamente el mejor modelo (save_best_only = T),
# de acuerdo con la métrica indicada en monitor, val_loss en este caso.
checkpoint_callback <- callback_model_checkpoint(
  filepath = "best_model_HI.h5",   # Ubicación donde guardar el modelo
  monitor = "val_loss",            # Métrica a monitorizar (puede ser "val_accuracy", "val_mae", etc.)
  save_best_only = TRUE,           # Guarda solo el modelo que tiene el menor valor de "val_loss"
  verbose = 1)

epochs = 1200

model <- build_model() # se crea el modelo

# Entrenamiento del modelo
history <- fit(model, traindatax, traindatay,epochs = epochs, batch_size = 16,
               validation_split = 0.2,callbacks = list(checkpoint_callback))

# accuracy ha de subir, en vez de bajar como mae

# keras::save_model_hdf5(model, "model_HI")
# saveRDS(history, "history_HI.rds")

# model <- keras::load_model_hdf5("model_HI")
history <- readRDS("history_HI.rds")

# Representamos crossentropy sobre los datos de prueba y validación
# es lo que utiliza checkpoint para determinar el mejor modelo (min(valcrossentropy))
names(history$metrics) 
traincrossentropy <- history$metrics$loss 
valcrossentropy <- history$metrics$val_loss 

n=1 # Inicio del rango de épocas para el gráfico
# ------------------ todo de golpe -----------
windows()
plot(c(n:epochs), traincrossentropy[n:epochs], xlab = "epoch", type = 'l', 
     ylim = range(c(traincrossentropy[n:epochs], valcrossentropy[n:epochs])), col = "blue", ylab = "crossentropy", 
     main = "Binary crossentropy")
lines(c(n:epochs), valcrossentropy[n:epochs], col = "red", type = 'l')
legend("topright", legend = c("Entrenamiento", "Validación"), 
       col = c("blue", "red"), lty = 2, cex = 0.8)


# Encuentra el punto de menor MSE en la validación
min_epoch <- which.min(valcrossentropy[n:epochs]) + n - 1; min_epoch
# 877

# Añade una línea vertical en el punto de menor MSE
abline(v = min_epoch, col = "darkgreen", lty = 3, lwd = 1)  

# --------------------------------------------


# Representamos también Accuracy sobre los datos de prueba y validación

trainaccuracy <- history$metrics$accuracy 
valaccuracy <- history$metrics$val_accuracy 

n=1 # Inicio del rango de épocas para el gráfico
# ------------------ todo de golpe -----------
windows()
plot(c(n:epochs), trainaccuracy[n:epochs], xlab = "epoch", type = 'l', 
     ylim = range(c(trainaccuracy[n:epochs], valaccuracy[n:epochs])), col = "blue", ylab = "accuracy", 
     main = "Accuracy")
lines(c(n:epochs), valaccuracy[n:epochs], col = "red", type = 'l')
legend("bottomright", legend = c("Entrenamiento", "Validación"), 
       col = c("blue", "red"), lty = 2, cex = 0.7)

# Encuentra el punto de mayor Accuracy en la validación
max_epoch <- which.max(valaccuracy[n:epochs]) + n - 1; max_epoch
# 967

# Añade una línea vertical en el punto de mayor Accuracy
abline(v = max_epoch, col = "darkgreen", lty = 3, lwd = 1)
# --------------------------------------------

# ¿por qué no usar loss =  "accuracy"?
# "accuracy" es una métrica discreta, no es diferenciable y no podría usarse como FdP

# Segmentación y Promedio de la Precisión en Intervalos
ndiv <- 20  # Iteraciones por intervalo
segaccuracy <- split(valaccuracy, ceiling(seq_along(valaccuracy)/ndiv))
meanaccuracies <- sapply(segaccuracy, mean)
epoch <- seq(1, epochs, by = ndiv)

# Gráfico de la Precisión Promedio
windows()
plot(epoch, meanaccuracies, type = "b", col = "red", pch = 19, cex = 0.5,
     lty = 1, lwd = 1, xlab = "Epoch", ylab = "Mean Accuracies",
     cex.lab = 1.2, cex.main = 1.5, cex.axis = 1.1, main = "Accuracy")

# Si los valores iniciales son muy altos y no dejan apreciar la curva
n <- 3  # Número de intervalos que se dejan de representar
epoch <- seq(n*ndiv+1, epochs, by = ndiv)

# Gráfico de las Medias excluyendo los primeros intervalos
windows()
plot(epoch, meanaccuracies[-(1:n)], type = "b", col = "red", pch = 19,
     cex = 0.5, lty = 1, lwd = 1, xlab = "Epoch", ylab = "Mean Accuracies",
     cex.lab = 1.2, cex.main = 1.5, cex.axis = 1.1, main = "Accuracy")



# Cargar el modelo con el mejor desempeño
best_model <- keras::load_model_hdf5("best_model_HI.h5")

probs <- predict(best_model, testdatax)
preds <- ifelse(probs > 0.5, 1, 0)

kappa <- kappa2(data.frame(testdatay, preds))
kappa$value

# 0.32




# -----------------------------------------------------------------

# Ejercicio 3.3.11. Entrenar un modelo de RNF optimizado para predecir si una 
# comunidad estará depauperada a partir de una serie de factores de impacto  humano


rm(list=ls(all=T)) # clears workspace
set.seed(3)  # Set random seed

setwd("D:/Science/Curso/Human Impact")
HImp <- read.table(file="HImpdata.csv",header=T,sep=",") 
HImp <- HImp[complete.cases(HImp), ]  # to delete cases with NA?s (RF y XgBoost no los admiten)

# La variable objetivo no se convierte en factor, como en el resto 
# de métodos, sino que se deja como integer

data <- subset(HImp, select = -c(X, lon, lat))
data <- data[complete.cases(data), ]  # to delete cases with NA?s (RF y XgBoost no los admiten)


# Preparación de los datos

# Selecciona el 80% de tus datos totales para ser tu conjunto de datos de entrenamiento y validación
s <- sample.int(n=nrow(data), size=floor(.8*nrow(data)), replace = F)
data2 <- data[s,]

# Asigna el resto (20%) a tus datos de prueba
testdata <- data[-s,]

# De los datos de entrenamiento y validación, selecciona otro 80% para ser tu conjunto de entrenamiento
m <- sample.int(n=nrow(data2), size=floor(.8*nrow(data2)), replace = F)
traindata <- data2[m,]

# Asigna el resto (20%) a tus datos de validación. Hay que extraer valdata 
# previamente, en vez de utilizar validation_split = 0.2 en history porque 
# se va a aplicar bsample y ese 20% incluiría datos que también se estaban
# utilzando para el entrenamiento. Sería lo que se conoce como !fuga de 
# datos! o "data leakage" por incorrecto preprocesamiento de los datos
valdata <- data2[-m,]

# Cargamos la función bsample
source("D:/Science/Curso/bsample.R") 


table(traindata$Dep)
mean<-mean(table(traindata$Dep)); mean
traindata <- bsample(traindata,'Dep',2*mean) 
table(traindata$Dep)

# Preparación de los datos

# Separa los predictores de la var objetivo y los transforma
traindatax <- as.matrix(subset(traindata, select=-Dep))# los predictores como matriz
traindatay <- as.array(traindata$Dep)   # la variable objetivo como vector empezando por 0 y no por 1
valdatax <- as.matrix(subset(valdata, select=-Dep))
valdatay <- as.array(valdata$Dep)   
testdatax <- as.matrix(subset(testdata, select=-Dep))
testdatay <- as.array(testdata$Dep) 

# Normalizar los datos

mean <- apply(traindatax, 2, mean) # apply es para aplicar mean a todas las predictoras (columnas) 
sd <- apply(traindatax, 2, sd)
traindatax <- scale(traindatax, center = mean, scale = sd)
valdatax <- scale(valdatax, center = mean, scale = sd)
testdatax <- scale(testdatax, center = mean, scale = sd)


modelo <- function(neuronas=1, activacion="relu", 
                   tasa_aprendizaje=0.1, capas=1, 
                   l1 = 0, l2 = 0, dropout = 0,
                   kernel_initializer='glorot_uniform', 
                   beta_1=0.9, beta_2=0.999, epsilon=1e-7) {
  model <- keras_model_sequential()
  model <- layer_dense(model, units = neuronas, 
                       activation = activacion, 
                       input_shape = c(dim(traindatax)[2]),
                       kernel_regularizer = regularizer_l1_l2(l1 = l1, l2 = l2),
                       kernel_initializer = kernel_initializer)
  
  for(i in 2:capas){
    model <- layer_dense(model, units = neuronas, 
                         activation = activacion,
                         kernel_regularizer = regularizer_l1_l2(l1 = l1, l2 = l2),
                         kernel_initializer = kernel_initializer)
    if(dropout > 0){
      model <- layer_dropout(model, rate = dropout)
    }
  }
  
  model <- layer_dense(model, units = 1, activation = 'sigmoid')
  
  model <- compile(model,
                   loss = 'binary_crossentropy',
                   optimizer = optimizer_adam(learning_rate = tasa_aprendizaje, 
                                              beta_1 = beta_1, beta_2 = beta_2,
                                              epsilon = epsilon),
                   metrics = list('accuracy')
  )
  
  return(model)
}

neuronas <- c(16, 32) # Este no tiene un valor por defecto, ya que depende del problema específico
tasa_aprendizaje <- c(0.01,0.001, 0.0001) # Valor por defecto para Adam

activacion <- c('relu') # 'relu' es una opción común, pero no hay un valor por defecto estricto
capas <- c(2,3) # Este no tiene un valor por defecto, ya que depende del problema específico
batch_size <- c(16) # Valor por defecto en Keras para fit()
l1 <- c(0) # 0, no hay regularización L1 por defecto
l2 <- c(0) # 0, no hay regularización L2 por defecto
dropout <- c(0.0, 0.1) # 0, no hay dropout por defecto
kernel_initializer <- c('glorot_uniform') # Valor por defecto para capas densas
beta_1 <- c(0.9) 
beta_2 <- c(0.999) 
epsilon <- c(1e-7) 

# neuronas <- c(8, 16, 32, 64) # Números comunes de neuronas por capa
# activacion <- c('relu', 'tanh', 'sigmoid', 'elu') # Funciones de activación comunes
# tasa_aprendizaje <- c(0.1, 0.01, 0.001, 0.0001) # Tasas de aprendizaje comunes
# capas <- c(1, 2, 3, 4) # Número común de capas ocultas en un modelo
# batch_size <- c(16, 32, 64, 128) # Tamaños de lote comunes
# l1 <- c(0.0, 0.001, 0.01, 0.1) # Valores comunes para regularización L1
# l2 <- c(0.0, 0.001, 0.01, 0.1) # Valores comunes para regularización L2
# dropout <- c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5) # Tasas de dropout comunes
# kernel_initializer <- c('glorot_uniform', 'he_normal', 'he_uniform') # Inicializadores de peso comunes
# beta_1 <- c(0.9, 0.8, 0.7) # Valores comunes para beta_1
# beta_2 <- c(0.999, 0.9, 0.8) # Valores comunes para beta_2
# epsilon <- c(1e-7, 1e-8, 1e-9) # Valores comunes para epsilon



grid <- expand.grid(neuronas=neuronas, activacion=activacion, 
                    tasa_aprendizaje=tasa_aprendizaje, capas=capas,
                    l1 = l1, l2 = l2, dropout = dropout,
                    batch_size=batch_size,
                    kernel_initializer = kernel_initializer,
                    beta_1 = beta_1,
                    beta_2 = beta_2,
                    epsilon = epsilon,
                    stringsAsFactors = F)

nc<-nrow(grid); nc  # nº total de combinaciones

grid$optepochs = 0
grid$accuracy = 0 
grid$minBCE = 0
grid$kappa = 0

early_stop <- callback_early_stopping(monitor = 'val_loss', patience = 10)

# Implementación de un callback para la reducción de la tasa de aprendizaje en un plateau
reduce_learn_rate <- callback_reduce_learn_rate_on_plateau(monitor = "val_loss", 
                                           factor = 0.2, patience = 5)

for(i in 1:nc) {
  model <- modelo(neuronas = grid$neuronas[i],
                  activacion = grid$activacion[i],
                  tasa_aprendizaje = grid$tasa_aprendizaje[i],
                  capas = grid$capas[i],
                  kernel_initializer = grid$kernel_initializer[i],
                  beta_1 = grid$beta_1[i],
                  beta_2 = grid$beta_2[i],
                  epsilon = grid$epsilon[i])
  
  history <- suppressWarnings(fit(model, traindatax, traindatay, 
                                  epochs = 1000, batch_size = grid$batch_size[i], 
                                  # para evitar la fuga de datos
                                  # en vez de validation_split = 0.2
                                  validation_data = list(valdatax, valdatay), 
                                  verbose = 0, 
                                  callbacks = list(early_stop, reduce_learn_rate)))
  
  
  grid$optepochs[i] <- which.min(history$metrics$val_loss)
  grid$minBCE[i] <- min(history$metrics$val_loss)
  
  predictions <- round(as.vector(predict(model, testdatax)))
  # mean() de un vector booleano, da el promedio de TRUEs en el vector
  grid$accuracy[i] <- mean(predictions == testdatay)
  testdata <- cbind(testdatay, predictions)
  kappa <- kappa2(testdata[,c(1,2)], "equal")
  grid$kappa[i] <- round(kappa$value, 2)
  
  print(i) 
}


grid <- grid[order(grid$minBCE),]
head(grid)

bestmod <- modelo(neuronas = grid$neuronas[1],
                  activacion = grid$activacion[1],
                  tasa_aprendizaje = grid$tasa_aprendizaje[1],
                  capas = grid$capas[1],
                  beta_1 = grid$beta_1[1],
                  beta_2 = grid$beta_2[1])

history <- suppressWarnings(fit(bestmod, traindatax, traindatay, 
                                epochs = grid$optepochs[1], batch_size = grid$batch_size[1], 
                                # para evitar la fuga de datos
                                # en vez de validation_split = 0.2
                                validation_data = list(valdatax, valdatay), 
                                callbacks = list(early_stop, reduce_learn_rate)))

# keras::save_model_hdf5(bestmod, "bestmodel.h5")
# bestmodel <- keras::load_model_hdf5("bestmodel.h5")

min(history$metrics$val_loss)

evaluate(bestmod, testdatax, testdatay)


predictions <- round(as.vector(predict(model, testdatax))); head(predictions)
testdata <- cbind(testdatay, predictions)
kappa <- kappa2(testdata[,c(1,2)], "equal"); round(kappa$value, 2)

# 0.36

# Para entender cómo el modelo está aprendiendo y cómo sus 
# predicciones están mejorando con cada época 

epochs = 200

history <- suppressWarnings(fit(bestmod, traindatax, traindatay, 
                                epochs = epochs, batch_size = grid$batch_size[1], 
                                # para evitar la fuga de datos
                                # en vez de validation_split = 0.2
                                validation_data = list(valdatax, valdatay)))

names(history$metrics)
val_BCE<-history$metrics$val_loss # nos quedamos con BCE, la FdP
n_epochs <- min(length(val_BCE), epochs)
windows();plot(c(1:n_epochs), val_BCE[1:n_epochs], type = "l")
which.min(val_BCE)

ndiv<-3
segBCE <- split(val_BCE, ceiling(seq_along(val_BCE)/ndiv))
meanBCE <- sapply(segBCE, mean)
which.min(meanBCE)*ndiv

# Creamos el vector del eje x
epoch <- seq(1, epochs, by = ndiv)
windows();plot(epoch, meanBCE, type = "b") 
windows();plot(epoch[1:60/ndiv], meanBCE[1:60/ndiv], type = "b") 






