library(sparklyr)
#spark_install(version = "2.1.0")

spark = spark_connect(master = "local[*]", spark_home="C:/spark-2.3.1-bin-hadoop2.7", app_name = "sparklr")

library(dplyr)
library(ggplot2)

df = read.csv("C:/machinelearning/IdeaProject/Kdata/cancer.csv", header = F)
head(df)
str(df)

sparkdf <- copy_to(spark, df, "cancer")
src_tbls(spark)
head(sparkdf)

sparkdf = spark_read_csv(spark, "cancer", "C:/machinelearning/IdeaProject/Kdata/cancer.csv", header=F, infer_schema = T)
head(sparkdf)
sparkdf = sparkdf %>% na.omit()

cancer  = sparkdf %>% sdf_partition(train=0.7, test=0.3, seed=1234)
cancer

rf = cancer$train %>%  ml_random_forest(V11 ~ ., type = "classification")
prediction = ml_predict(rf, cancer$test) %>% collect
head(prediction)

table(prediction$V11, prediction$prediction)
