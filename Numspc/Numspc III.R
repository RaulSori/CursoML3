



#------      DpL   Regresión   ----------



  # Ejercicio 3.3.1. Estimar con RNF la riqueza de especies de aves invernantes 
  # a partir de predictores ambientales, 
  # a) Optimizar el nº de épocas con checkpoint_callback; 
  # b) Inferir la riqueza de especies esperable en una nueva localidad (NvLoc).   
  
  
  library(ggplot2)
  library(keras)
  
  rm(list=ls(all=T)) # clears workspace
  set.seed(3) 
  
  setwd("D:/Science/Curso/Numspc")
  Numspc <- read.table(file="Numspc.csv",header=T,sep=",") 
  names(Numspc)
  
  data<-Numspc
  
 # dividir en train & test data
 # no se preselecciona valdata porque se hace con validation_split = 0.2 en history
 # hay que hacerlo cuando los datos están estructurados para que sean independientes 
 # de los de entrenamiento
  
  m <- sample.int(n=nrow(data), size=floor(.8*nrow(data)),replace = F)
  traindata <- data[m,]
  testdata <- data[-m,]
    
  # Preparación de los datos
  # las variables predictoras solas (sin nspp -> variable objetivo) en una matriz
  traindatax<-as.matrix(subset(traindata, select=-nspp)) 
  traindatay<-c(traindata$nspp)# la variable objetivo como vector, aparte 
  
  # y lo mismo para las muestras test
  testdatax<-as.matrix(subset(testdata, select=-nspp)) 
  testdatay<-c(testdata$nspp) 
  
 # Normalizar los datos
  
 # Datos heterogéneos pueden dificultar el aprendizaje.
 # Normalizar es recomendable. Se puede utilizar la función scale()
 # scale no estandariza, solo hace la media 0 y la desviación típica 1.
 # Todo (train, test y newdata) se normaliza con la media y sd de traindata

  mean <- apply(traindatax, 2, mean) # obtiene la media de todas las muestras para cada columna
  sd <- apply(traindatax, 2, sd) # = la desv tipica
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
# (799 muestras X 5 CCS), pero se excluye el nº de muestras (799)
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
  filepath = "best_model_NSpc.h5",# Ubicación donde guardar el modelo
  monitor = "val_loss",# Métrica para indentificar el mejor modelo. Puede ser también val_metrica
  save_best_only = T,# Guarda solo el modelo que tiene el menor valor de monitor
  verbose = 1)
 
  epochs = 500
  
  # Entrenamos del modelo. history guarda el registro del proceso de entrenamiento
  history <- fit(model, traindatax, traindatay, epochs = epochs, batch_size = 16, 
                validation_split = 0.2, callbacks = list(checkpoint))
  
  # keras::save_model_hdf5(model, "model_NSpc") # para salvar el modelo entrenado
  # saveRDS(history, "history_NSpc.rds") # para salvar history
  
  # para cargarlos
  # model <- keras::load_model_hdf5("model_NSpc")
  history <- readRDS("history_NSpc.rds")
  
  names(history) # objetos que contiene
  names(history$metrics)

  # Representamos mse (la FdP) sobre los datos de prueba y validación
  trainmse <- history$metrics$loss # nos quedamos con mse
  valmse <- history$metrics$val_loss # nos quedamos con mse

  n=30 # Inicio del rango de épocas para el gráfico
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
# 208
  
  # Añade una línea vertical en el punto de menor MSE
  abline(v = min_epoch, col = "darkgreen", lty = 3, lwd = 1)  
  # --------------------------------------------
  
  
  # Representamos también mae ((la métrica) sobre los datos de prueba y validación
  trainmae <- history$metrics$mae # nos quedamos con mae
  valmae <- history$metrics$val_mae # nos quedamos con mae

  n=30 # Inicio del rango de épocas para el gráfico
  # ------------------ todo de golpe -----------
  windows()
  plot(c(n:epochs), trainmae[n:epochs], xlab = "epoch", type = 'l', 
       ylim = range(c(trainmae[n:epochs], valmae[n:epochs])), col = "blue", ylab = "mae", 
       main = "MAE")
  lines(c(n:epochs), valmae[n:epochs], col = "red", type = 'l')
  legend("right", legend = c("Entrenamiento", "Validación"), 
         col = c("blue", "red"), lty = 1, cex = 0.8)
  
  # Encuentra el punto de menor MSE en la validación
  min_epoch <- which.min(valmae[n:epochs]) + n - 1; min_epoch
  # 208
  
  # Añade una línea vertical en el punto de menor MSE
  abline(v = min_epoch, col = "darkgreen", lty = 3, lwd = 1)  
  # --------------------------------------------
 
# **********************
#  en caso de que el gráfico sea confuso se puede promediar intervalos
ndiv<-20 # iteraciones por intervalo
segmae <- split(valmae, ceiling(seq_along(valmae)/ndiv))
meanmaes <- sapply(segmae, mean)
epoch <- seq(1, epochs, by = ndiv)
windows();plot(epoch, meanmaes, type = "b", col = "red", pch = 19,cex = 0.5,
               lty = 1,lwd = 1, xlab = "Epoch", ylab = "Mean MAEs",
               cex.lab = 1.2,  cex.main = 1.5,  cex.axis = 1.1, main = "MAE")         

# si valores iniciales muy altos no dejan apreciar la curva
n<-3 # nº de intervalos que se dejan de representar (n X ndiv)
# Creamos el vector del eje x
epoch <- seq(n*ndiv+1, epochs, by = ndiv)
# Ahora podemos representar las medias
windows();plot(epoch, meanmaes[-(1:n)], type = "b", col = "red", pch = 19,
               cex = 0.5,lty = 1, lwd = 1, xlab = "Epoch", ylab = "Mean MAEs",
               cex.lab = 1.2,  cex.main = 1.5,  cex.axis = 1.1, main = "MAE")         
# **********************  



# Cargar el modelo con el mejor desempeño
best_model <- keras::load_model_hdf5("best_model_NSpc.h5")

# Realizar las predicciones con el mejor modelo
preds <- predict(best_model, testdatax)

# Calcular R2
rsq = as.numeric(cor(preds, testdatay))^2; round(rsq,2)

# 0.3

rmse = sqrt(mean((preds - testdatay)^2)); rmse

# 9.53

# Visualización de predicciones vs valores reales
windows();plot(testdatay, preds, main = "Real vs Predicho", 
               xlab = "Valores reales", ylab = "Predicciones",
               xlim = range(c(testdatay, preds)), 
               ylim = range(c(testdatay, preds)))
abline(a = 0, b = 1)

  
                 
  # b) Inferir la riqueza de especies en una nueva localidad (NvLoc) con DpL
  
 # ¡¡¡ sin la variable objetivo!!!
  NvLoc <- read.table(file="NvLoc.csv",header=T,sep=",") # cargamos una nueva localidad
  rownames(NvLoc) = "NvLoc"; NvLoc
  
    
  # Preparación de los datos 
  # los predictores solos (sin nspp - variable objetivo) en una matriz 
  NvLoc<-as.matrix(NvLoc) 
  
  # Normalizar los datos; se usa la media y std de los datos de entrenamiento
  NvLoc <- scale(NvLoc, center = mean, scale = sd)
  
  pred <- round(predict(best_model, NvLoc))
  pred = as.data.frame(pred) 
  rownames(pred) = rownames(NvLoc) 
  pred
  
  # se esperan 45 especies
  

  
  
  
  
  
  
  
  
  
# Ejercicio nº?. Análisis e interpretación (PDP e interacciones) de los 
# algoritmos de RF (a) y XgBoost (b) entrenados para estimar la riqueza de especies de aves 
# invernantes (paquete PDP). 


rm(list=ls(all=TRUE)) # clears workspace

set.seed(5)  # Set 
setwd("D:/Science/Curso/Numspc")
data <- read.table(file="Numspc.csv",header=T,sep=",") 

RFfit<- randomForest(nspp ~ ., data=data)

RFfit

windows(); plot(RFfit)


# Dependencia parcial (paquete PDP)


library(pdp)

windows();partial(RFfit, pred.var = "altmed", plot = T, plot.engine = "ggplot2", rug=T) # lo hace directamente con ggplot

# PDP suavizado

p1 <- partial(RFfit, pred.var = "altmed") # usamos los resultados del PDP para representarlos como queremos

windows();ggplot(p1, aes(x = altmed, y = yhat))+ 
  geom_point() + 
  geom_smooth() + 
  theme_bw() + 
  labs(x = "Altitud media", y = "nº de especies")

pdtmin <- plotPartial(pdptmin)

pdpdmar <- partial(RFfit, pred.var = "distmar")
windows();ggplot(pdpdmar, aes(x = distmar, y = yhat))+ 
  geom_point() + 
  geom_smooth() + 
  theme_bw() + 
  labs(x = "", y = "Abundancia")

pdpralt <- partial(RFfit, pred.var = "rangoalt")
windows();ggplot(pdpralt, aes(x = rangoalt, y = yhat))+ 
  geom_point() + 
  geom_smooth() + 
  theme_bw() + 
  labs(x = "", y = "Abundancia")

pdpamed <- partial(RFfit, pred.var = "altmed")
windows();ggplot(pdpamed, aes(x = altmed, y = yhat))+ 
  geom_point() + 
  geom_smooth() + 
  theme_bw() + 
  labs(x = "", y = "Abundancia")

pdpprec <- partial(RFfit, pred.var = "precip")
windows();ggplot(pdpprec, aes(x = precip, y = yhat))+ 
  geom_point() + 
  geom_smooth() + 
  theme_bw() + 
  labs(x = "", y = "Abundancia")

pdptmed <- partial(RFfit, pred.var = "tempmedia")
windows();ggplot(pdptmed, aes(x = tempmedia, y = yhat))+ 
  geom_point() + 
  geom_smooth() + 
  theme_bw() + 
  labs(x = "", y = "Abundancia")


# Interacción entre los predictores


# pdp2<-partial(RFfit, pred.var = c("tempmin", "distmar"))  # ¡¡¡  TOMA MUCHO TIEMPO !!!
# save(pdp2, file = "pdp2.rda") 

load(file = "pdp2.rda")


windows();plotPartial(pdp2)

rwb <- colorRampPalette(c("red", "white", "blue"))
windows();plotPartial(pdp2, contour = T, col.regions = rwb)

windows();plotPartial(pdp2, levelplot = F, zlab = "Abund", drape = T, 
                      colorkey = F, screen = list(z = 0, x = 0))# con plotpartial (3D)

windows();plotPartial(pdp2, levelplot = F, zlab = "Abund", drape = T, 
                      colorkey = F, screen = list(z = 150, x = -60))# con plotpartial (3D)


# tambión puede mostrar individual conditional expectation (ICE) 

windows();partial(RFfit, pred.var = "tempmin",plot = T, chull=T, 
                  plot.engine = "ggplot2", rug=T, ice = T, progress = "text")


detach("package:pdp", unload=T) # puede interferir con iml

































