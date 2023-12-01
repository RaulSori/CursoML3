# Ejercicio 2.3.8.SOLOS
# Estimar con GBM para regresión la masa corporal de las 
# aves a partir de su morfología (referencia 2.3.5), 
# a) sin optimizar, 
# b) optimizado con una rejilla de búsqueda, 
# c) testar su capacidad predictiva con una 5-folds CV, 
# d) estimar la masa corporal de Accipiter brachyurus, Cincloramphus cruralis y Tigriornis leucolopha (datos reales pero no de una investigación real).


library(ggplot2)
library(irr)          # para obtener kappa
library(gbm) 

rm(list=ls(all=T)) # clears workspace
set.seed(12)

setwd("D:/Science/Curso/AVONETMsCrp")
AVONET <- read.table(file="AVONET0.csv",header=T,sep=",") # es sin R xq es full data
summary(round(AVONET$Mass))

# Mezcla las muestras por si tuvieran algún tipo de orden
AVONET <- AVONET[sample(nrow(AVONET)), ]
rownames(AVONET) <- AVONET$Species

# Quitamos a los emues y pingüinos, no voladores
AVONET <- AVONET[!rownames(AVONET) %in% c("Rhea americana", "Rhea pennata", "Aptenodytes patagonicus"), ]

# separamos las especies que utilizaremos como muestras de estudio
Newspcs <- AVONET[rownames(AVONET) %in% c("Accipiter brachyurus", "Tigriornis leucolopha", "Cincloramphus cruralis"), ]
Newspcs<- Newspcs[c("Mass","TsL","WnL","SWL","KpD","TlL","HWI")]

AVONET <- AVONET[!rownames(AVONET) %in% c("Accipiter brachyurus", "Tigriornis leucolopha", "Cincloramphus cruralis"), ]

# nos quedamos con las variables que nos interesan
AVONET<- AVONET[c("Mass","TsL","WnL","SWL","KpD","TlL","HWI")]

data <- AVONET

# Entrenamos con un 10% de las muestras y dejamos el resto para testar
set.seed(5)
m <- sample.int(n=nrow(data), size=floor(.1*nrow(data)),replace = F)
traindata <- data[m,]
testdata <- data[-m,]

summary(round(traindata$Mass)) # nos aseguramos de que recoja todo el rango (2-23900)
# puesto que el modelo final se entrenará con todas


# a) Sin optimización




# b) Optimización con una grid aleatoria



load(file = "gridgbm.rda")


load("besthppsGBM.RData")


# Testar su capacidad predictiva con testdata

# Entrena el modelo final con los mejores hiperparámetros.

# Calcula R2

# Calcula el RMSE

# Gráfica de dispersión




# c) Optimización bayesiana.


# d) Estimar la masa corporal de Accipiter brachyurus, Cincloramphus 
#    cruralis y Tigriornis leucolopha.

# Elegimos BestmodelGBM, que dió mejores resultados

preds <- as.data.frame(round(predict(BestmodelGBM, newdata = Newspcs, n.trees = besthppsGBM$opttrees)))
colnames(preds)<-"Pred"
rownames(preds)<-rownames(Newspcs) # le ponemos los nombres a las variables
preds

#                        Pred
# Tigriornis leucolopha  1308
# Accipiter brachyurus    256
# Cincloramphus cruralis   51


# Ver PP

#--------------------------------------------------





#--------------------------------------------------


#     XgBoost  Regresión


# Ejercicio 2.5.2.a) Entrenar un algoritmo de XgBoost para estimar la masa 
#    corporal de las aves, a partir de la morfología de su ala, 
# b) estimar la masa corporal de Accipiter brachyurus, A. brevipes y 
#    A. Butler (datos reales pero  no de una investigación real).  


library(ggplot2)
library(irr)
library(XgBoost) 


#     XgBoost  Regresión


# Ejercicio 2.5.2.a) Entrenar un algoritmo de XgBoost para estimar la masa 
#    corporal de las aves voladoras, a partir de la morfología de su ala, 
# b) estimar la masa corporal de Accipiter brachyurus, A. brevipes y 
#    A. Butler (datos reales pero  no de una investigación real).  


library(ggplot2)
library(irr)
library(XgBoost) 

rm(list=ls(all=T)) # clears workspace

setwd("D:/Science/Curso/AVONETMsCrp")
AVONET <- read.table(file="AVONET0.csv",header=T,sep=",") # es sin R xq es full data
summary(round(AVONET$Mass))

# Mezcla las muestras por si tuvieran algún tipo de orden
AVONET <- AVONET[sample(nrow(AVONET)), ]
rownames(AVONET) <- AVONET$Species

# separamos las especies que utilizaremos como muestras de estudio
Newspcs<-subset(AVONET, AVONET$Species=="Accipiter brachyurus"|AVONET$Species=="Tigriornis leucolopha"|AVONET$Species=="Cincloramphus cruralis")
AVONET <- subset(AVONET,!AVONET$Species %in% Newspcs$Species)
# nos quedamos con las variables que nos interesan
Newspcs<- Newspcs[c("TsL","WnL","SWL","KpD","TlL","HWI")]
AVONET<- AVONET[c("Mass","TsL","WnL","SWL","KpD","TlL","HWI")]

AVONET <- AVONET[complete.cases(AVONET), ]  # to delete cases with NAs

data <- AVONET

set.seed(5)
m <- sample.int(n=nrow(data), size=floor(.8*nrow(data)),replace = F)
traindata <- data[m,]
testdata <- data[-m,]


#       Preparación de los datos para XgBoost




# a) Entrenar un algoritmo de XgBoost para estimar la masa 
#    corporal de las aves, a partir de la morfología de su ala 




# rsq = 0.83   Auténticas predicciones (por VC), 
# aunque sobre traindata. Puede haber FdV porque se optimizó 
# sobre ellos el nº óptimo de iteraciones.




# Optimizando los hiperparámetros la predicción puede mejorar

# probar que el loop funciona, cargar los guardado con load
# y comparar los resultados




# R2 alcanzó 0.86



load("besthpps1.RData")


load("besthpps2.RData")







# b) Estimar la masa corporal de Accipiter brachyurus, A. brevipes y A. Butler. 




#                        Newspcspred
# Cincloramphus cruralis       54.56
# Accipiter brachyurus        342.34
# Tigriornis leucolopha      1318.67

#---------------------------------------------------------------------------



#     XGBOOST, REGRESIÓN IMPORTANCIA y PDPs


# Ejercicio 2.8.7 Solos (Con solo el 10% de las muestras)

# Con el Paquete xgboost
# a) Entrenar un algoritmo xgBoost para estimar la masa corporal de las aves 
#    (regresión) a partir de la morfología de su ala. 
# b) Determinar la importancia de las variables. 
# c) Determinar la masa corporal de 3 nuevas especies: Accipiter brachyurus, 
#    Cincloramphus cruralis y Tigriornis leucolopha.


# Con el paquete DALEX
# d) Determinar la importancia de las variables
# e) Obtener el perfil desglosado (Break Down profile) para la 1ª muestra de 
#    entrenamiento y las 3 nuevas especies.
# f) Obtener la Importancia Basada en el Valor de Shapley (SHAP values) 
#    para la 1ª muestra de entrenamiento y NvLoc.
# g) Determinar el efecto de la variable más importante con ayuda de un PDP 
#    individual.
# h) Determinar el efecto de todas las variables, simultáneamente, con sus 
#    PDPs ya estandarizados. 


library(ggplot2)
library(xgboost)
library(DALEX)

rm(list=ls(all=T)) # clears workspace

setwd("D:/Science/Curso/AVONETMsCrp")
AVONET <- read.table(file="AVONET0.csv",header=T,sep=",") # es sin R xq es full data
summary(round(AVONET$Mass))

# Mezcla las muestras por si tuvieran algún tipo de orden
AVONET <- AVONET[sample(nrow(AVONET)), ]
rownames(AVONET) <- AVONET$Species

# separamos las especies que utilizaremos como muestras de estudio
Newspcs<-subset(AVONET, AVONET$Species=="Accipiter brachyurus"|AVONET$Species=="Tigriornis leucolopha"|AVONET$Species=="Cincloramphus cruralis")
AVONET <- subset(AVONET,!AVONET$Species %in% Newspcs$Species)
# nos quedamos con las variables que nos interesan
Newspcs<- Newspcs[c("Mass","TsL","WnL","SWL","KpD","TlL","HWI")]
AVONET<- AVONET[c("Mass","TsL","WnL","SWL","KpD","TlL","HWI")]

data <- AVONET

# Quedarnos con solo el 25%
set.seed(5)
data <- AVONET[sample(nrow(AVONET), nrow(AVONET) * 0.25), ]



# Con el Paquete xgboost

# a) Entrenar un algoritmo xgBoost para estimar la masa corporal de las aves 
#    (regresión) a partir de la morfología de su ala (ya se hizo en el 2.5.2). 



# b) Determinar la importancia de las variables en el modelo XgBoost



# c) Determinar la masa corporal de 3 nuevas especies: Accipiter brachyurus, 
#    Cincloramphus cruralis y Tigriornis leucolopha (ya se hizo en el 2.5.2).




# Con el paquete DALEX

# Transformamos el modelo XgBoost en explain (al que llamamos explainer) con el que trabaja DALEX


# d) Determinar la importancia de las variables



# e) Obtener el perfil desglosado para las 3 nuevas especies 
#    (la 1ª, 2ª y 3ª de Newspcs). 

Newspcs1 <- Newspcsx[1, , drop = F] # drop = F impide que R convierta el df en un vector
name1 <- rownames(Newspcs1);name1
# continuar


# f) Obtener la Importancia Basada en el Valor de Shapley (SHAP values) 
#    para las 3 nuevas especies (la 1ª, 2ª y 3ª de Newspcs). 

Newspcsx_df <- as.data.frame(Newspcsx) # Convertir datax a data.frame
pred <- predict(fitxgb, as.matrix(Newspcsx_df[1:3, ]))
print(round(pred))


# Para la 1ª muestra de Newspcs
# Calcula valores SHAP para la 1ª Newspcs (se le añade "predcontrib = T", respecto a pred)


# Para la 2ª Newspcs


# Para la 3ª Newspcs




# g) Determinar el efecto de la variable más importante con PDP 




# h) Determinamos el efecto de todas las variables simultáneamente, 
#    estandarizando los gráficos









# ---------------------------------------------------------


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




# a) Estimar la masa corporal de las aves con DpL, a partir de su morfología 




# b) Inferir la masa corporal de nuevas especies (newspcs) con DpL

# sin la variable objetivo
newspcs<- newspcs[c("TsL","WnL","SWL","KpD","TlL","HWI")]








