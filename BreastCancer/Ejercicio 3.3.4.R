


#------------    DpL -  Clasificación binaria   -----------------




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
data <- data[complete.cases(data), ]  # to delete cases with NA 

names(data)


# Preparación de los datos

# División en train & val data antes del balanceo
m <- sample.int(n=nrow(data), size=floor(.8*nrow(data)),replace = F)
traindata <- data[m,]
testdata <- data[-m,]

# Cargamos la función bsample
source("D:/Science/Curso/bsample.R") 

table(traindata$Class)
mean<-mean(table(traindata$Class)); mean
traindata <- bsample(traindata,'Class',mean) 
table(traindata$Class)

# Separa los predictores de la var objetivo y los transforma

traindatax <- as.matrix(subset(traindata, select=-Class))# los predictores como matriz
traindatay <- as.numeric(as.character(traindata$Class))
testdatax <- as.matrix(subset(testdata, select=-Class))
testdatay <- as.numeric(as.character(testdata$Class))

# Normalizar los datos

mean <- apply(traindatax, 2, mean) # apply es para aplicar mean a todas las predictoras (columnas) 
sd <- apply(traindatax, 2, sd)
traindatax <- scale(traindatax, center = mean, scale = sd)
testdatax <- scale(testdatax, center = mean, scale = sd)

# Construir el modelo de clasificación
# 2 capas ocultas con 64 neuronas cada una y una capa de salida.
# La FdA de ambas capas ocultas es ReLU.
# La de salida tiene activación sigmoid porque es clasificación binaria.
# Una TdA de 0.001.
# loss = "binary_crossentropy" porque es una clasificación binaria.
# Además de la métrica accuracy, vamos a calcular kappa, pero en Keras no tenemos 
# una función directa para el kappa, por lo que utilizaremos accuracy para 
# monitorizar y luego calcularemos kappa de manera separada.

build_model <- function() {
  model <- keras_model_sequential()
  model <- layer_dense(model, units = 64, activation = "relu",
                       input_shape = dim(traindatax)[2])
  model <- layer_dense(model, units = 64, activation = "relu")
  model <- layer_dense(model, units = 1, activation = "sigmoid")
  model <- compile(model, optimizer = optimizer_adam(learning_rate = 0.001), 
                   loss = "binary_crossentropy", metrics = c("accuracy")) 
  return(model)
}

# ¿por qué no usar loss =  "accuracy"?
# "accuracy" es una métrica discreta, no es diferenciable y no podría usarse como FdP


# Crear un objeto ModelCheckpoint
checkpoint <- callback_model_checkpoint(
  filepath = "best_model_BC.h5",      # Ubicación donde guardar el modelo
  monitor = "val_loss",            # Métrica a monitorizar (puede ser "val_accuracy", "val_mae", etc.)
  save_best_only = TRUE,           # Guarda solo el modelo que tiene el menor valor de "val_loss"
  verbose = 1)

epochs = 300

model <- build_model() # se crea el modelo


# Entrenamiento del modelo
history <- fit(model, traindatax, traindatay,epochs = epochs, 
               batch_size = 16,  validation_split = 0.2,
               callbacks = list(checkpoint))

# accuracy ha de subir, en vez de bajar como mae

# keras::save_model_hdf5(model, "model_BC")
# saveRDS(history, "history_BC.rds")

# model <- keras::load_model_hdf5("model_BC")
history <- readRDS("history_BC.rds")


# Representamos crossentropy (la FdP) sobre los datos de prueba y validación
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
legend("right", legend = c("Entrenamiento", "Validación"), 
       col = c("blue", "red"), lty = 2, cex = 0.8)

# Encuentra el punto de menor MSE en la validación
min_epoch <- which.min(valcrossentropy[n:epochs]) + n - 1; min_epoch
# 19

# Añade una línea vertical en el punto de menor MSE
abline(v = min_epoch, col = "darkgreen", lty = 3, lwd = 1)  

# --------------------------------------------


# Representamos también Accuracy ((la métrica) sobre los datos de prueba y validación

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
# 23

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
best_model <- keras::load_model_hdf5("best_model_BC.h5")

probs <- predict(best_model, testdatax)
preds <- ifelse(probs > 0.5, 1, 0)

kappa <- kappa2(data.frame(testdatay, preds))
kappa$value


# 1






