



# Ejercicio 3.3.4. Solos. Entrenar un modelo de RNF para diagnosticar la posible 
# malignidad de un tumor a partir de 10 características diferentes que pueden 
# tomarse a partir de imágenes digitalizadas de una aguja de biopsia del seno. 
# Optimizar el nº de épocas con checkpoint_callback.  

library(keras)
library(irr)          # para obtener kappa


rm(list=ls(all=T)) # clears workspace
set.seed(3)  # Set random seed

setwd("D:/Science/Curso/BreastCancer")
BreastCancer <- read.table(file="BreastCancer.csv",header=T,sep=",") 

BreastCancer$Class <- ifelse(BreastCancer$Class == "benign", 0, 1)
BreastCancer$Class <- factor(BreastCancer$Class)  

data<-BreastCancer




# model <- keras::load_model_hdf5("model_BC")
# history <- readRDS("history_BC.rds")






# -----------------------------------------------------


# Ejercicio 3.3.11 Entrenar un modelo de RNF con la que diagnosticar la malignidad 
# de un tumor en el pecho a partir de nueve caracteristicas (grosor de las 
# células, tamaño, forma, etc.) que se toman en las biopsias. 

library(keras)
library(irr)          # para obtener kappa

rm(list=ls(all=T)) # clears workspace
set.seed(3)  # Set random seed

setwd("D:/Science/Curso/BreastCancer")
BreastCancer <- read.table(file="BreastCancer.csv",header=T,sep=",") 
BreastCancer <- subset(BreastCancer, select=-X)  # quitar X

table(BreastCancer$Class)

BreastCancer$Class <- ifelse(BreastCancer$Class == "benign", 0, 1)

data<-BreastCancer

names(data)

data <- data[complete.cases(data), ]  # to delete cases with NA 


# Preparación de los datos

# Selecciona el 80% de tus datos para entrenamiento y validación
s <- sample.int(n=nrow(data), size=floor(.8*nrow(data)), replace = F)
data2 <- data[s,]

# Toma el resto (20%) como datos de prueba
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

# remuestreamos traindata para equilibrar presencias y ausencias
bsample <- function(data,cname,n) {
  d <- data[-c(1:nrow(data)),]
  u <- unique(data[,cname])
  for (uu in u) {
    w <- which(data[,cname] == uu)
    if (length(w) >= n) {
      s <- sample(w,n)
    } else {
      s <- sample(w,n,replace=T)
    }
    d <- rbind(d,data[s,])
  }
  d
}

table(traindata$Class)
mean<-mean(table(traindata$Class)); mean
traindata <- bsample(traindata,'Class',2*mean) 
table(traindata$Class)

# Preparación de los datos

# Separa los predictores de la var objetivo y los transforma
traindatax <- as.matrix(subset(traindata, select=-Class))# los predictores como matriz
traindatay <- as.array(traindata$Class)   # la variable objetivo como vector empezando por 0 y no por 1
valdatax <- as.matrix(subset(valdata, select=-Class))
valdatay <- as.array(valdata$Class)   
testdatax <- as.matrix(subset(testdata, select=-Class))
testdatay <- as.array(testdata$Class) 

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

neuronas <- c(16, 32) # Este no tiene un valor por defecto, ya que Classende del problema específico
tasa_aprendizaje <- c(0.01,0.001) # Valor por defecto para Adam

activacion <- c('relu') # 'relu' es una opción común, pero no hay un valor por defecto estricto
capas <- c(2,3) # Este no tiene un valor por defecto, ya que Classende del problema específico
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
reduce_lr <- callback_reduce_lr_on_plateau(monitor = "val_loss", 
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
                                  callbacks = list(early_stop, reduce_lr)))
  
  
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
                                callbacks = list(early_stop, reduce_lr)))

# keras::save_model_hdf5(bestmod, "bestmodel.h5")
# bestmodel <- keras::load_model_hdf5("bestmodel.h5")

min(history$metrics$val_loss)

evaluate(bestmod, testdatax, testdatay)


predictions <- round(as.vector(predict(model, testdatax))); head(predictions)
testdata <- cbind(testdatay, predictions)
kappa <- kappa2(testdata[,c(1,2)], "equal"); round(kappa$value, 2)

# 0.96

# Para entender cómo el modelo está aprendiendo y cómo sus 
# predicciones están mejorando con cada época 

grid$optepochs[1]

epochs = 50

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

ndiv<-1
segBCE <- split(val_BCE, ceiling(seq_along(val_BCE)/ndiv))
meanBCE <- sapply(segBCE, mean)
which.min(meanBCE)*ndiv

# Creamos el vector del eje x
epoch <- seq(1, epochs, by = ndiv)
windows();plot(epoch, meanBCE, type = "b") 
windows();plot(epoch[1:10/ndiv], meanBCE[1:10/ndiv], type = "b") 





















