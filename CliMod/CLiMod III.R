
#------------------   DpL   -----------------------------------

# Ejercicio 3.3.6. Entrenar un modelo de RNF para predecir la riqueza de 
# especies invertívoras (IFd) a partir del clima, con datos de validación 
# independientes (datos estructurados espacialmente).  


library(keras)
library(maps)
library(ggplot2)

rm(list=ls(all=T)) # clears workspace
set.seed(3)

setwd("D:/Science/Curso/CliMod")
dataCOOR <- read.table(file="dataCOOR.csv",header=T,sep=",") # es sin R xq es full data

data <- dataCOOR
names(data)


# Separamos datos de entrenamiento, prueba y validación
# Los de valdata porque son datos estructurados espacialmente y han de ser
# independientes de los de entrenamiento

# Seleccionar los datos dentro de la franja para el conjunto de validación
testdata <- subset(data, lon >= 10 & lon <= 30)

# Seleccionar todo lo que no esté en valdata ni en testdata para el conjunto de entrenamiento
traindata <- subset(data, !(lon >= 10 & lon <= 30) & lon >= -17)

#                        ¡CLAVE! (datos estructrados)

# Seleccionar los datos para el conjunto de pruebas
valdata <- subset(data, lon < -17)  # América + Groenlandia


world <- map_data('world')

# Añadir una nueva columna para identificar el conjunto de datos
traindata$set <- 'Traindata'
testdata$set <- 'Testdata'
valdata$set <- 'Valdata' #    ¡¡¡  CLAVE  !!!

# Combinar todos los datos en un marco de datos
all_data <- rbind(traindata, testdata, valdata)

# Representamos los datos de prueba en un mapa
windows(); ggplot(all_data, aes(x=lon, y=lat)) +
  geom_path(data=world, aes(x=long, y=lat, group=group)) +
  geom_point(aes(color=IFd), size=0.01) +
  scale_color_gradient(limits=c(0,300)) +
  facet_grid(. ~ set) + # Separar las gráficas por conjunto de datos
  guides(colour = guide_legend(override.aes = list(size=5))) +
  labs(title = "Data Distribution by Set",
       x = "Longitude",
       y = "Latitude",
       colour = "IFd")


traindatay <- traindata$IFd
# quitamos lo que no son predictoras
traindatax <- subset(traindata, select = -c (IFd, set, lon, lat))

testdatay <- testdata$IFd
testdatax <- subset(testdata, select =  -c (IFd, set, lon, lat))

valdatay <- valdata$IFd
valdatax <- subset(valdata, select =  -c (IFd, set, lon, lat))

# normalización o estandarización respecto a los datos de 
# entrenamiento (no repetir sin repetir lo anterior)
meanx <- apply(traindatax, 2, mean) 
sdx <- apply(traindatax, 2, sd)
traindatax <- scale(traindatax, center = meanx, scale = sdx)
valdatax <- scale(valdatax, center = meanx, scale = sdx)
testdatax <- scale(testdatax, center = meanx, scale = sdx)


# Construir el modelo de clasificación

build_model <- function(input_shape) {
  model <- keras_model_sequential()
  
  # Se crean 2 capas ocultas densas (totalmente conectadas porque son FFNN)  
  # con 64 neuronas cada una, cuya FdA es ReLU. 
  
  # En la 1ª se indican las dimensiones del tensor de entrada, dim(traindatax)
  model <- layer_dense(model, units = 64, activation = "relu",
                       input_shape = dim(traindatax)[2])
  model <- layer_dense(model, units = 64, activation = "relu")
  
  # Se crea una capa de salida que no tiene FdA porque es regresión
  # units = 1 porque esperamos un solo valor
  model <- layer_dense(model, units = 1)
  
  # Tipo de optimizador Adam y TdA (learning_rate) = 0.001, en este caso    
  optimizer = optimizer_adam(learning_rate = 0.001)
  
  # La función compile() sirve para configurar el proceso de aprendizaje del modelo.
  # loss = "mse" => La FdP que el algoritmo trata de minimizar.
  # metrics = "mae" => una o más metricas para monitorizar el rendimiento 
  # Se obtiene la evolución de loss y metrics sobre las muestras de entrenamiento 
  # (loss y metric) y sobre las muestras de validación (val_loss y val_metrica)
  model <- compile(model, optimizer = optimizer, 
                   loss = "mse", metrics = "mae") 

  return(model)
}

model <- build_model() # se crea el modelo

# Para que guarde automáticamente el mejor modelo (save_best_only = T),
# de acuerdo con la métrica indicada en monitor, val_loss en este caso.
checkpoint <- callback_model_checkpoint(
  filepath = "best_model_NSpc.h5",# Ubicación donde guardar el modelo
  monitor = "val_loss",# Métrica para indentificar el mejor modelo. Puede ser también val_metrica
  save_best_only = T,# Guarda solo el modelo que tiene el menor valor de monitor
  verbose = 1)

epochs = 50

#                        ¡CLAVE! (datos estructrados)

# validation_data = list(valdatax, valdatay) en vez de validation_split = 0.2


history <- fit(model, traindatax, traindatay, epochs = epochs, batch_size = 16, 
               validation_data = list(valdatax, valdatay), 
               callbacks = list(checkpoint))


# keras::save_model_hdf5(model, "model_CMd") # para salvar el modelo entrenado
# saveRDS(history, "history_CMd.rds") # para salvar history

# para cargarlos
# model <- keras::load_model_hdf5("model_CMd")
history <- readRDS("history_CMd.rds")

names(history$metrics)

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
legend("right", legend = c("Entrenamiento", "Validación"), 
       col = c("blue", "red"), lty = 1, cex = 0.8)

# Encuentra el punto de menor MSE en la validación
min_epoch <- which.min(valmse[n:epochs]) + n - 1; min_epoch
# 15

# Añade una línea vertical en el punto de menor MSE
abline(v = min_epoch, col = "darkgreen", lty = 3, lwd = 1)  
# --------------------------------------------


# Representamos también mae ((la métrica) sobre los datos de prueba y validación
trainmae <- history$metrics$mae # nos quedamos con mae
valmae <- history$metrics$val_mae # nos quedamos con mae

# ------------------ todo de golpe -----------
windows()
plot(c(1:epochs), trainmae[1:epochs], xlab = "epoch", type = 'l', 
     ylim = range(c(trainmae, valmae)), col = "blue", ylab = "mae", 
     main = "MAE")
lines(c(1:epochs), valmae[1:epochs], col = "red", type = 'l')
legend("right", legend = c("Entrenamiento", "Validación"), 
       col = c("blue", "red"), lty = 1, cex = 0.8)
# --------------------------------------------
which.min(valmae)
# 3

# **********************
#  en caso de que el gráfico sea confuso se puede promediar intervalos
ndiv<-5 # iteraciones por intervalo
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
best_model <- keras::load_model_hdf5("best_model_CMd.h5")

# Realizar las predicciones con el mejor modelo
preds <- predict(best_model, testdatax)

# Calcular R2
rsq = as.numeric(cor(preds, testdatay))^2; round(rsq,2)

# 0.81

rmse = sqrt(mean((preds - testdatay)^2)); rmse

# 9.53

# Visualización de predicciones vs valores reales
windows();plot(testdatay, preds, main = "Real vs Predicho", 
               xlab = "Valores reales", ylab = "Predicciones",
               xlim = range(c(testdatay, preds)), 
               ylim = range(c(testdatay, preds)))
abline(a = 0, b = 1)


# --------------------------------------------























# OLD Ejercicio 2.7.10. 
# a) Entrenar un algoritmo de RF y dos de XgBoost (sin optimizar y optimizado)
#    para predecir la riqueza de especies invertívoras a partir del clima, 
#    utilizando para ello un 10% de las muestras de DataCOOR, y comparar 
#    su capacidad predictiva sobre otro 10% de las muestras
# b) obtener la variable más importante de acuerdo con cada uno de los 
#    3 ?ndices: Gain, Cover y frequency
# c) analizar con PDPs sus respectivos efectos sobre la riqueza de especies 
#   con el paquete DALEX. 


library(randomForest)
library(gbm)
library(xgboost)
library(ipred) # for bagging
library(dismo)
library(iml)
library(vip)
library(ggthemes)
library(keras)
library(tensorflow)

rm(list=ls(all=T)) # clears workspace
set.seed(7)  # Set 

setwd("D:/Science/Curso/CliMod")
DataCOOR <- read.table(file="DataCOOR.csv",header=T,sep=",") # es sin R xq es full data
DataCOOR <- subset(DataCOOR, select=-X) 
DataCOOR <- subset(DataCOOR, select=-lon) 
DataCOOR <- subset(DataCOOR, select=-lat) 
DataCOOR <- DataCOOR[complete.cases(DataCOOR), ]  # to delete cases with NA?s


# Generamos dos bases de datos distintas con un 10% de las muestras

s <- sample.int(n=nrow(DataCOOR), size=floor(.2*nrow(DataCOOR)),replace = F)
data0 <- DataCOOR[s,]
m <- sample.int(n=nrow(data0), size=floor(.5*nrow(data0)),replace = F)
data1 <- data0[m,]
data2 <- data0[-m,]

summary(data1$IFd)
summary(data2$IFd)

# a) Entrenar un algoritmo de RF y dos de XgBoost (sin optimizar y optimizado)
#    para predecir la riqueza de especies invertívoras a partir del clima, 
#    utilizando para ello un 10% de las muestras de DataCOOR, y comparar 
#    su capacidad predictiva sobre otro 10% de las muestras


# RF 



# Xgboost 

#  Preparación de los datos


# sin optimizar




# Optimizando los hiperpar?metros 

# Para explorar el hiperespacio de hiperpar?metros hacemos una 1? b?squeda en rejilla

grid <- expand.grid(eta = c(.001, .01, .1), # distintas tasas de aprendizaje
                    max_depth = c(1, 3, 7),    # árboles sencillos o complejos
                    min_child_weight = c(1, 3, 5), # nº m?n de muestras requeridas en cada nodo terminal
                    subsample = c(.7, .8), # % de muestras de entrenamiento a usar para cada árbol
                    colsample_bytree = c(.8, 1),# porcentaje de predictores de los que tomar muestras para cada árbol (como RF)
                    alpha = c(.01, .1),
                    lambda = c(.1, 1, 10),
                    gamma = c(0, 10),
                    
                    rsq=0,          # para anotar los resultados
                    optntrees = 0,   # para anotar los resultados
                    minRMSE = 0 )   # para anotar los resultados

nc<-nrow(grid); nc  # n?mero total de combinaciones

#  ??? 1296 COMBINACIONES !!!

#    Probar con for(i in 1:10) en vez de for(i in 1:nc) para ver que funciona
#    y cargar grid abajo.





grid <- read.table(file="grid.csv",header=T,sep=",")
grid <- subset(grid, select=-X) 



# Identificar los hiperpar?metros ?ptimos y utilizarlos para entrenar un algoritmo 
# definitivo.

grid<-grid[order(grid$rsq,decreasing=T),]# ordena los casos en base a la var. Cls
head(grid)

# eta max_depth min_child_weight subsample colsample_bytree alpha lambda gamma    rsq       optntrees  minRMSE
# 0.01         7           3       0.7            0.8       0.10    1.0    10    0.9055654  1000       18.16745


# Aplicarlo al otro 10% con los hipermar?metros ya optimizados 



# Como con RF, aplicar a nuevas muestras tan solo para ver c?mo se hace,
# puesto que la capacidad predictiva es exactamente la misma que por VC




# con RF optimizado, 0.90
# con xgboost sin optimizar, 0.88
# con xgboost optimizado, 0.91


# b) obtener la variable más importante de acuerdo con cada uno de los 
#    3 ?ndices: Gain, Cover y frequency





# c) analizar con PDPs sus respectivos efectos sobre la riqueza de especies 
#   con el paquete DALEX. 

#   ?IMPORTANTE!    Reiniciar con cntrl+shift+F10 - mantiene todos los objetos

library(ggplot2)
library(randomForest) 
library(xgboost)
library(DALEX)






















