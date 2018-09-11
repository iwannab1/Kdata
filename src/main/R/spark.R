setwd('C:/machinelearning/IdeaProject/Kdata')
Sys.setenv(JAVA_HOME = 'C:/Progra~1/Java/jdk1.8.0_121')

if (nchar(Sys.getenv("SPARK_HOME")) < 1) {
  Sys.setenv(SPARK_HOME = "C:/spark-2.3.1-bin-hadoop2.7")
}
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "2g"))

schema = structType(structField("scn", "double"),
                    structField("ct", "double"),
                    structField("ucs", "double"),
                    structField("ucs2", "double"),
                    structField("ma", "double"),
                    structField("secz", "double"),
                    structField("bn", "double"),
                    structField("bc", "double"),
                    structField("nn", "double"),
                    structField("mit", "double"),
                    structField("class", "double"))

df = read.df("C:/machinelearning/IdeaProject/Kdata/cancer.csv", "csv", header=F, schema=schema) 
selectedCol = colnames(df)[!colnames(df) %in% c("scn")]
df = select(df, selectedCol)
head(df)
printSchema(df)
str(df)

df2 = dropna(df)

indexer = function(df){
  out = df
  out$classIndex = sapply(df$class, function(x){ ifelse((x==2), 0, 1) })
  out
}


df3 <- dapply(df2, indexer, schema)
head(df3)


df_list = randomSplit(df3, c(7,3), 2)

train = df_list[[1]]
test = df_list[[2]]

model = spark.svmLinear(train, classIndex ~ . - class)
summary(model)
prediction = predict(model, test)
head(prediction)


prediction$result  = ifelse((prediction$classIndex == prediction$prediction),"TRUE", "FALSE")
correct = NROW(prediction[prediction$result == "TRUE",])
accuracy = correct/nrow(test)
cat(accuracy*100, "%")


## dataframe

library(ggplot2)

qplot(df$ct,
      geom="histogram", 
      binwidth = 0.5)

Rdf = collect(df)
Rdf

qplot(Rdf$ct,
      geom="histogram", 
      binwidth = 0.5)
