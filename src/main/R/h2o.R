## H2O integration
# dependency packages
pkgs <- c("methods","statmod","stats","graphics",
          "RCurl","jsonlite","tools","utils", "devtools" )
for (pkg in pkgs) {
  if (!(pkg %in% rownames(installed.packages()))) install.packages(pkg)
}

# install H2O from localfile
#install.packages("/home/iwannab1/tools/h2o-3.10.0.10/R/h2o_3.10.0.10.tar.gz", repos=NULL, type="source")
install.packages("h2o", type = "source", repos = "http://h2o-release.s3.amazonaws.com/h2o/rel-turing/6/R")

# install sparling 
library(devtools)
devtools::install_github("h2oai/sparkling-water", subdir = "/r/rsparkling")

library(sparklyr)
library(rsparkling)
library(dplyr)

config <- spark_config()
config$spark.executor.cores <- 4
config$spark.executor.memory <- "4G"
options(rsparkling.sparklingwater.version = "1.6.7")
sc <- spark_connect("local[*]", version = "1.6.2", spark_home="/home/iwannab1/tools/spark-1.6.2-bin-hadoop2.6", config=config)

mtcars_tbl <- copy_to(sc, mtcars, "mtcars", overwrite = TRUE)

# convert h2o frame
iris_hf <- as_h2o_frame(sc, iris_tbl)

y = "Species"
x = setdiff(names(iris_tbl), y)
iris_hf[,y] <- as.factor(iris_tbl[,y])



