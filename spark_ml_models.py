import pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .master("local[20]") \
    .appName("COM6012 Spark Intro") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

from pyspark.sql import Row,Column
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.feature import Binarizer
import pandas as pd
import time

newCol = ['label','epton pT', 'lepton eta', 'lepton phi', 'missing energy magnitude', \
'missing energy phi', 'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', \
'jet 2 eta', 'jet 2 phi', 'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', \
'jet 3 b-tag', 'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag', 'm_jj', \
'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']
oldCol = ['_c0','_c1','_c2','_c3','_c4','_c5','_c6','_c7','_c8','_c9','_c10',\
'_c11','_c12','_c13','_c14','_c15','_c16','_c17','_c18','_c19','_c20','_c21',\
'_c22','_c23','_c24','_c25','_c26','_c27','_c28']
#loading data
data = spark.read.csv("../Data/HIGGS.csv.gz", header=False)
data = data.repartition(20)

#change column names
for i in range(29):
    data = data.withColumn(oldCol[i], data[oldCol[i]].cast(DoubleType()))
    data = data.withColumnRenamed(oldCol[i],newCol[i])

#build features
assembler = VectorAssembler(inputCols = newCol[1:], outputCol = 'features')
raw_plus_vector = assembler.transform(data)
(trainingData, testData) = raw_plus_vector.randomSplit([0.7, 0.3], 66)
trainingData = trainingData.select([c for c in trainingData.columns if c in ['features','label']])
testData = testData.select([c for c in testData.columns if c in ['features','label']])
trainingData.cache()
testData.cache()

def performance(prediction):
    '''
    performance of model
    '''
    binarizer = Binarizer(threshold=0.5, inputCol="prediction", outputCol="binarized_feature")
    binarizedDataFrame = binarizer.transform(prediction)
    prediction_label = binarizedDataFrame.select(['binarized_feature','label'])
    metrics = BinaryClassificationMetrics(prediction_label.rdd)
    return metrics.areaUnderROC

def get_top_3(data, scores):
    col_info = data.schema['features'].metadata["ml_attr"]["attrs"]
    #transfer column info into list
    lis = []
    for i in col_info:
        lis = lis + col_info[i]
    #link column name with their feature importance scores and sort
    df = pd.DataFrame(lis)
    df['score'] = df['idx'].apply(lambda x: scores[x])
    top3 = df.sort_values('score', ascending = False).head(3)
    top3_names = [name for name in top3['name']]
    return top3_names

#doing DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=13, maxBins = 64, impurity='entropy')
time1 = time.time()
dtc_model = dt.fit(trainingData)
time2 = time.time()
dtc_time = time2 - time1
dtc_prediction = dtc_model.transform(testData)
evaluator = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(dtc_prediction)
dtc_3 = get_top_3(trainingData, dtc_model.featureImportances)

#doing DecisionTreeRegressor
dt = DecisionTreeRegressor(labelCol="label", featuresCol="features", maxDepth=12,maxBins=64,minInstancesPerNode = 2, minInfoGain = 0.0)
time1 = time.time()
dtr_model = dt.fit(trainingData)
time2 = time.time()
dtr_time = time2 - time1
dtr_prediction = dtr_model.transform(testData)
dtr_3 = get_top_3(trainingData, dtr_model.featureImportances)

#doing LogisticRegression
lr = LogisticRegression(maxIter=10, regParam=0.01, elasticNetParam = 0.0)
time1 = time.time()
logi_model = lr.fit(trainingData)
time2 = time.time()
logi_time = time2 - time1
logi_prediction = logi_model.transform(testData)

#get top 3 features
coef = logi_model.coefficientMatrix
coef_arr = coef.toArray()
col_coef = {}
for i in range(len(newCol)-1):
    col_coef[newCol[i+1]] = coef_arr[0][i]
rank = sorted(col_coef.items(), key=lambda kv: abs(kv[1]), reverse = True)
logi_3 = [item[0] for item in rank[:3]]

#evaluation
dtc_ROC = performance(dtc_prediction)
dtr_ROC = performance(dtr_prediction)
logi_ROC = performance(logi_prediction)

#output
print("Area under DecisionTreeClassifier ROC =", dtc_ROC, '\n')
print("Accuracy of DecisionTreeClassifier =", accuracy, '\n')
print("Area under DecisionTreeRegressor ROC =", dtr_ROC, '\n')
print("Area under LogisticRegression ROC =", logi_ROC, '\n')
print("the training time of DecisionTreeClassifier is:",dtc_time,'\n')
print("the training time of DecisionTreeRegressor is:",dtr_time,'\n')
print("the training time of LogisticRegression is:",logi_time,'\n')
#output for question1.3
print("the top 3 features of DecisionTreeClassifier is:",dtc_3,"\n")
print("the top 3 features of DecisionTreeRegressor is:",dtr_3,"\n")
print("the top 3 features of LogisticRegression is:",logi_3,"\n")


spark.stop()
