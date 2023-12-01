


#------------    DpL -  clasificación multiclase   -----------------



# Ejercicio 3.3.5. Entrenar un modelo de FFNN para predecir el comportamiento
# migratorio de las aves a partir de su morfología (clasificación multiclase). 
# a) Optimizar el nº de épocas con checkpoint_callback; 
# b) predecir el comportamiento migratorio de Accipiter brachyurus, A. brevipes 
#    y A. Butler (testspcs).   

library(keras)
library(ggplot2)
library(irr)

rm(list=ls(all=T)) # clears workspace
set.seed(3)

setwd("D:/Science/Curso/AVONETMgr")
AVONET <- read.table(file="AVONET.csv",header=T,sep=",") # es sin R xq es full data

# Establecer la columna 'Species' de AVONET como los nombres de las filas del dataframe
rownames(AVONET) <- AVONET$Species

# separamos las especies que utilizaremos como muestras de estudio
Newspcs<-subset(AVONET, AVONET$Species=="Accipiter brachyurus"|AVONET$Species=="Accipiter brevipes"|AVONET$Species=="Accipiter butleri")
AVONET <- subset(AVONET,!AVONET$Species %in% Newspcs$Species)

# nos quedamos con las variables que nos interesan
AVONET<- AVONET[c("Migrt","BkL","BNL","BkD","BkW","TsL","WnL","SWL","KpD","TlL","HWI")]
# sin Migrt
Newspcs<- Newspcs[c("BkL","BNL","BkD","BkW","TsL","WnL","SWL","KpD","TlL","HWI")]

# Mezcla las muestras por si tuvieran algún tipo de orden
AVONET <- AVONET[sample(nrow(AVONET)), ]

# Eliminar filas con valores NA en AVONET
AVONET <- AVONET[complete.cases(AVONET), ]  

# Mezclar las filas de AVONET, por si estuvieran ordenadas
AVONET <- AVONET[sample(nrow(AVONET)), ]

table(AVONET$Migrt)

data <- AVONET

# Creating a map of string labels to numeric
mapping <- c("Migr" = 0, "PMig" = 1, "Sed" = 2)

# Applying the mapping to create a new numeric variable
data$Migrt <- as.numeric(as.character(mapping[data$Migrt]))

num_classes <- length(unique(data$Migrt))  # O cualquier número que represente tus clases

# Selecciona el 80% de los datos para el entrenamiento
m <- sample.int(n=nrow(data), size=floor(.8*nrow(data)), replace = F)
traindata <- data[m,]
testdata <- data[-m,]

# Cargamos la función bsample
source("D:/Science/Curso/bsample.R") 


table(traindata$Migrt)
mean<-mean(table(traindata$Migrt)); mean
traindata <- bsample(traindata,'Migrt',mean) 
table(traindata$Migrt)

# Preparación de los datos

# Separa los predictores de la var objetivo y los transforma
traindatax <- as.matrix(subset(traindata, select=-Migrt))# los predictores como matriz
traindatay <- as.numeric(traindata$Migrt)   # la variable objetivo como vector empezando por 0 y no por 1
testdatax <- as.matrix(subset(testdata, select=-Migrt))# los predictores como matriz
testdatay <- as.array(testdata$Migrt)   # la variable objetivo como vector empezando por 0 y no por 1

# Normalizar los datos

mean <- apply(traindatax, 2, mean) # apply es para aplicar mean a todas las predictoras (el 2 es para las columnas) 
sd <- apply(traindatax, 2, sd)
traindatax <- scale(traindatax, center = mean, scale = sd)
testdatax <- scale(testdatax, center = mean, scale = sd)
Newspcs <- scale(Newspcs, center = mean, scale = sd)


# Construimos la FFNN con 2 capas intermedias
# Como es clasificación multiclase, indicamos también num_classes

build_model <- function(input_shape, num_classes) {
  model <- keras_model_sequential() # se genera el modelo vacío

  # se le agregan 2 capas  intermedias densas (totalmente conectadas)
  # con 64 neuronas (en vez de 16, como en IMDB, porque son 46 clases) 
  # se puede perder información relevante
  # y la FdA relu
  model <- layer_dense(model, units = 64, activation = "relu",
                       input_shape = input_shape)
  model <- layer_dense(model, units = 64, activation = "relu")
  # la capa de salida con FdA softmax (en vez de sigmoid), que genera una  
  # distribución de probabilidad sobre las 3 clases, que suman 1.
  # Es donde se especifica el nº de clases
  model <- layer_dense(model, units = num_classes, activation = "softmax")
  # optimizer rmsprop, buena opción para prácticamente cualquier problema
  # FdP entropía cruzada (pero no binaria), que mide la distancia entre 
  # dos distribuciones de probabilidad
  compile(model, optimizer = "rmsprop", loss = "categorical_crossentropy", 
          metrics = "accuracy")
  
  return(model)
}

input_shape <- dim(traindatax)[2] # sin el nº de muestras

# One-hot encoding
# "to_categorical" es una función de Keras que convierte un vector de números 
# enteros en una matriz binaria de clase categórica (one-hot encoding). 
traindatay <- to_categorical(traindatay, num_classes = num_classes) 
testdatay <- to_categorical(testdatay, num_classes = num_classes)

# Crear un objeto ModelCheckpoint
checkpoint <- callback_model_checkpoint(
  filepath = "best_model_AMg.h5",      # Ubicación donde guardar el modelo
  monitor = "val_loss",            # Métrica a monitorizar (puede ser "val_accuracy", "val_mae", etc.)
  save_best_only = TRUE,           # Guarda solo el modelo que tiene el menor valor de "val_loss"
  verbose = 1)

epochs = 500

model <- build_model(input_shape, num_classes)

# Entrenamiento del modelo
history<-fit(model, traindatax, traindatay, epochs = epochs, batch_size = 16, 
             validation_split = 0.2, callbacks = list(checkpoint))


# keras::save_model_hdf5(model, "model_AMg")
# saveRDS(history, "history_AMg.rds")

# model <- keras::load_model_hdf5("model_AMg")
history <- readRDS("history_AMg.rds")


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
# 72

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
# 449

# Añade una línea vertical en el punto de mayor Accuracy
abline(v = max_epoch, col = "darkgreen", lty = 3, lwd = 1)
# --------------------------------------------


# **********************
#  en caso de que el gráfico sea confuso se puede promediar intervalos
ndiv<-20 # iteraciones por intervalo
segaccuracy <- split(valaccuracy, ceiling(seq_along(valaccuracy)/ndiv))
meanaccuracy <- sapply(segaccuracy, mean)
epoch <- seq(1, epochs, by = ndiv)
bestepochs<-which.max(meanaccuracy)*ndiv; bestepochs# mejor modelo
windows()
plot(epoch, meanaccuracy, type = "b", col = "red", pch = 19,cex = 0.5,
     lty = 1, lwd = 1, xlab = "Epoch", ylab = "Mean Accuracy",
     cex.lab = 1.2,  cex.main = 1.5,  cex.axis = 1.1, main = "Accuracy")



# ¿por qué no usar loss =  "accuracy"?
# "accuracy" es una métrica discreta, no es diferenciable y no podría usarse como FdP

# Pero sí se puede utilizar para determinar el nº de epocas óptimo, 
# cambiando en checkpoit monitor = "val_loss" por monitor = "val_metric"
# También se puede entrenar el modelo (history) con epoca = which.max(valaccuracy)
# y ver si kappa aumenta 


# Cargar el modelo con el mejor desempeño
best_model <- keras::load_model_hdf5("best_model_AMg.h5")

probs <- predict(best_model, testdatax); head(probs)
preds <- apply(probs, 1, which.max) - 1; str(preds)
testdatay <- as.array(testdata$Migrt); str(testdatay)
tabrp<-table(testdatay, preds); tabrp # tabla de 46 x 46
tabprop<-round(100*(prop.table(tabrp,1)),0); tabprop # row percentages 
kappa <- kappa2(matrix(c(testdatay, preds), ncol = 2), "equal");round(kappa$value, 2)

# 0.37

mapping <- c("Migr" = 0, "PMig" = 1, "Sed" = 2)
inverse_mapping <- c("0" = "Migr", "1" = "PMig", "2" = "Sed")

preds <- inverse_mapping[as.character(preds)]
table(preds)

testdatay <- inverse_mapping[as.character(testdatay)]
table(testdatay)

tabrp<-table(testdatay, preds); tabrp # tabla de 46 x 46
tabprop<-round(100*(prop.table(tabrp,1)),0); tabprop # row percentages 


# b) predecir el comportamiento migratorio de Accipiter brachyurus, A. brevipes 
#    y A. Butler (testspcs).   

# Predecir las probabilidades de clase para las nuevas especies
Newspcsprobs <- predict(best_model, Newspcs)

# Obtener las clases predichas seleccionando la clase con la mayor probabilidad
Newspcspreds <- apply(Newspcsprobs, 1, which.max) - 1  # restamos 1 porque R es base-1 y tus etiquetas empiezan en 0

# Convertir las predicciones numéricas a las etiquetas de clase originales
Newspcspreds <- inverse_mapping[as.character(Newspcspreds)]

# Mostrar las predicciones
print(Newspcspreds)



