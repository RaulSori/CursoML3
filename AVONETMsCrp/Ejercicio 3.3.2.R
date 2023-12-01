



#------      DpL   Regresión   ----------


# Ejercicio 3.3.2. SOLOS Estimar con RNF la masa corporal de las aves a partir de su 
# morfología, a) Optimizar el nº de épocas checkpoint_callback; 
# b) estimar la masa corporal de Accipiter brachyurus, Cincloramphus cruralis y 
#    Tigriornis leucolopha (datos reales pero no de una investigación real). 

library(ggplot2)
library(keras)

rm(list=ls(all=T)) # clears workspace
set.seed(2)

setwd("D:/Science/Curso/AVONETMsCrp")
AVONET <- read.table(file="AVONET0.csv",header=T,sep=",") # es sin R xq es full data

row.names(AVONET)<-AVONET[,1] # le pone el nombre de la 1ª columna


# Separa las newspcs
newspcs <- AVONET[AVONET$Species == "Accipiter brachyurus" | AVONET$Species == "Tigriornis leucolopha" | AVONET$Species == "Cincloramphus cruralis", ]
# las elimina de AVONET
AVONET <- AVONET[!(AVONET$Species == "Accipiter brachyurus" | AVONET$Species == "Tigriornis leucolopha" | AVONET$Species == "Cincloramphus cruralis"), ]

names(AVONET)
# nos quedamos con las variables que nos interesan

AVONET<- AVONET[c("Mass","TsL","WnL","SWL","KpD","TlL","HWI")]
AVONET <- AVONET[complete.cases(AVONET), ]  # to delete cases with NA?s

data<-AVONET

# dividir en train & test data
# no se preselecciona valdata porque se hace con validation_split = 0.2 en history
# hay que hacerlo cuando los datos están estructurados para que sean independientes 
# de los de entrenamiento

m <- sample.int(n=nrow(data), size=floor(.7*nrow(data)),replace = F)
traindata <- data[m,]
testdata <- data[-m,]


# Preparación de los datos
# las variables predictoras solas (sin Mass -> variable objetivo) en una matriz
traindatax<-as.matrix(subset(traindata, select=-Mass)) 
traindatay<-c(traindata$Mass)# la variable objetivo como vector, aparte  

testdatax<-as.matrix(subset(testdata, select=-Mass)) 
testdatay<-c(testdata$Mass) 

# Normalizar los datos

# Datos son muy heterogéneos pueden dificultar el aprendizaje.
# Normalizar es recomendable. Se puede utilizar la función scale()
# No estandariza, solo hace la media 0 y la desviación típica 1.
# Todo (train, test y newdata) se normaliza con la media y sd de traindata

mean <- apply(traindatax, 2, mean) 
sd <- apply(traindatax, 2, sd)
traindatax <- scale(traindatax, center = mean, scale = sd)
testdatax <- scale(testdatax, center = mean, scale = sd)


# Construimos un modelo muy simple, de tan solo dos capas intermedias
# con 64 neuronas cada una.

# Antes se crea una función (no es imprescindible) para poder 
# renovar el modelo con facilidad en cada nueva prueba

build_model <- function() { 
  
  # Se crea un objeto "modelo secuencial" que alojará una pila lineal 
  # de capas en el orden que uno quiera
  model <- keras_model_sequential()
 
  # Se crean 2 capas ocultas densas (totalmente conectadas porque son FFNN)  
  # con 64 neuronas cada una, cuya FdA es ReLU. 
  
  # En la 1ª se indican las dimensiones del tensor de entrada, dim(traindatax)
  # (6988 muestras X 5 CCS), pero se excluye el nº de muestras (6988)
  model <- layer_dense(model, units = 64, activation = "relu",
                       input_shape = dim(traindatax)[2]) 

  model <- layer_dense(model, units = 64, activation = "relu")
  
  # Se crea una capa de salida que no tiene FdA porque es regresión
  # units = 1 porque esperamos un solo valor
  model <- layer_dense(model, units = 1)
  
  # Tipo de optimizador Adam y TdA (learning_rate)    
  optimizer = optimizer_adam(learning_rate = 0.001)
  
  # La función compile() sirve para configurar el proceso de aprendizaje del modelo.
  # loss = "mse" => La FdP que el algoritmo trata de minimizar.
  # metrics = "mae" => una o más metricas para monitorizar el rendimiento 
  # Se obtiene la evolución de loss y metrics sobre las muestras de entrenamiento 
  # (loss y metric) y sobre las muestras de validación (val_loss y val_metrica)
  model <- compile(model, optimizer = optimizer,loss = "mse", metrics = "mae") 
 
  return(model)
}

model <- build_model() # se crea el modelo

# Para que guarde automáticamente el mejor modelo (save_best_only = T),
# de acuerdo con la métrica indicada en monitor, val_loss en este caso.
checkpoint <- callback_model_checkpoint(
  filepath = "best_model_MsCrp.h5",# Ubicación donde guardar el modelo
  monitor = "val_loss",# Métrica para indentificar el mejor modelo. Puede ser también val_metrica
  save_best_only = T,# Guarda solo el modelo que tiene el menor valor de monitor
  verbose = 1)

epochs = 1000

# Entrenamos del modelo. history guarda el registro del proceso de entrenamiento
history <- fit(model, traindatax, traindatay, epochs = epochs, batch_size = 16, 
               validation_split = 0.2, callbacks = list(checkpoint))


# keras::save_model_hdf5(model, "model_MsCrp")
# saveRDS(history, "history_MsCrp.rds")

# model <- keras::load_model_hdf5("model_MsCrp")
history <- readRDS("history_MsCrp.rds")

names(history) # objetos que contiene
names(history$metrics)

# Representamos mse (la FdP) sobre los datos de prueba y validación
trainmse <- history$metrics$loss # nos quedamos con mse
valmse <- history$metrics$val_loss # nos quedamos con mse

n=100 # Inicio del rango de épocas para el gráfico
# ------------------ todo de golpe -----------
windows()
plot(c(n:epochs), trainmse[n:epochs], xlab = "epoch", type = 'l', 
     ylim = range(c(trainmse[n:epochs], valmse[n:epochs])), col = "blue", ylab = "mse", 
     main = "MSE")
lines(c(n:epochs), valmse[n:epochs], col = "red", type = 'l')
legend("right", legend = c("Entrenamiento", "Validación"), 
       col = c("blue", "red"), lty = 1, cex = 0.8)

# Encuentra el punto de menor MSE en la validación
min_epoch <- which.min(valmse[n:epochs]) + n - 1; min_epoch
# 473

# Añade una línea vertical en el punto de menor MSE
abline(v = min_epoch, col = "darkgreen", lty = 3, lwd = 1)  
# --------------------------------------------


# Representamos también mae ((la métrica) sobre los datos de prueba y validación
trainmae <- history$metrics$mae # nos quedamos con mae
valmae <- history$metrics$val_mae # nos quedamos con mae

n=50 # Inicio del rango de épocas para el gráfico
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
# 759

# Añade una línea vertical en el punto de menor MSE
abline(v = min_epoch, col = "darkgreen", lty = 3, lwd = 1)  
# --------------------------------------------


# **********************
#  en caso de que el gráfico sea confuso se puede promediar intervalos
ndiv<-40 # iteraciones por intervalo
segmae <- split(valmae, ceiling(seq_along(valmae)/ndiv))
meanmaes <- sapply(segmae, mean)
epoch <- seq(1, epochs, by = ndiv)
windows();plot(epoch, meanmaes, type = "b", col = "red", pch = 19,cex = 0.5,
               lty = 1,lwd = 1, xlab = "Epoch", ylab = "Mean MAEs",
               cex.lab = 1.2,  cex.main = 1.5,  cex.axis = 1.1, main = "MAE")         

# si valores iniciales muy altos no dejan apreciar la curva
n<-10 # nº de intervalos que se dejan de representar (n X ndiv)
# Creamos el vector del eje x
epoch <- seq(n*ndiv+1, epochs, by = ndiv)
# Ahora podemos representar las medias
windows();plot(epoch, meanmaes[-(1:n)], type = "b", col = "red", pch = 19,
               cex = 0.5,lty = 1, lwd = 1, xlab = "Epoch", ylab = "Mean MAEs",
               cex.lab = 1.2,  cex.main = 1.5,  cex.axis = 1.1, main = "MAE")         
# **********************  



# Cargar el modelo con el mejor desempeño
best_model <- keras::load_model_hdf5("best_model_MsCrp.h5")


# Cargar el modelo con el mejor desempeño

# Realizar las predicciones con el mejor modelo
preds <- predict(best_model_MsCrp, testdatax)

# Calcular R2
rsq = as.numeric(cor(preds, testdatay))^2; round(rsq,2)


# 0.8

rmse = sqrt(mean((preds - testdatay)^2)); rmse

# 310

# Visualización de predicciones vs valores reales
windows();plot(testdatay, preds, main = "Real vs Predicho", 
               xlab = "Valores reales", ylab = "Predicciones",
               xlim = range(c(testdatay, preds)), 
               ylim = range(c(testdatay, preds)))
abline(a = 0, b = 1)




# b) Inferir la masa corporal de nuevas especies (newspcs) con DpL


# ¡¡¡ sin la variable objetivo!!!
newspcs<- newspcs[c("TsL","WnL","SWL","KpD","TlL","HWI")]

# Preparación de los datos 
# los predictores solos (sin Mass - variable objetivo) en una matriz 
newspcs<-as.matrix(newspcs) 

# Normalizar los datos; se usa la media y std de los datos de entrenamiento
newspcs <- scale(newspcs, center = mean, scale = sd)

pred <- round(predict(best_model, newspcs))
pred = as.data.frame(pred) 
rownames(pred) = rownames(newspcs) 
pred

# Accipiter brachyurus    414
# Cincloramphus cruralis  61
# Tigriornis leucolopha   735

