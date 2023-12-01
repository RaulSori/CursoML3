

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