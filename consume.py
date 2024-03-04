import json
import os
import random
import time

import pyspark as ps
from kafka import KafkaConsumer, KafkaProducer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

os.environ["PYSPARK_PYTHON"] = "C:/Users/Viktor/AppData/Local/Programs/Python/Python311/python.exe"

spark = ps.sql.SparkSession.builder \
    .master("local[4]") \
    .appName("diabetes classification offline") \
    .getOrCreate()

sc = spark.sparkContext

diabetes_df = spark.read.csv('diabetes_binary_health_indicators_BRFSS2015.csv',
                             header=True,
                             quote='"',
                             sep=",",
                             inferSchema=True)
diabetes_df.printSchema()

offline_df, online_df = diabetes_df.randomSplit([0.8, 0.2], seed=1335)

assembler = VectorAssembler(inputCols=[ftr for ftr in diabetes_df.columns if ftr != 'Diabetes_binary'],
                            outputCol='features')
offline_df = assembler.transform(offline_df)
indexer = StringIndexer(inputCol='Diabetes_binary', outputCol='label')
offline_df = indexer.fit(offline_df).transform(offline_df)

training_df, test_df = offline_df.randomSplit([0.8, 0.2], seed=654)

lr = LogisticRegression(maxIter=10, labelCol='label')

# 2 * 2 * 3 kombinacii na parametri
paramGrid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .addGrid(lr.fitIntercept, [False, True]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

f1_evaluator = MulticlassClassificationEvaluator(metricName='f1', labelCol='label')

# sekoja kombinacija se trenira na 70% od trening mnozestvoto;
# za validacija se koristat ostanatite 30%;
# ne se koristi k-fold (sekogas se istite 2 mnozestva za trening i validacija)
tvs = TrainValidationSplit(estimator=lr,
                           estimatorParamMaps=paramGrid,
                           evaluator=f1_evaluator,
                           trainRatio=0.7)

# model == najdobriot model
model = tvs.fit(training_df)

pred = model.transform(test_df)

print(f'F1: {f1_evaluator.evaluate(pred)}')

roc_ev = BinaryClassificationEvaluator(labelCol='label')
print(f'ROC AUC: {roc_ev.evaluate(pred)}')

# online faza

consumer = KafkaConsumer(
    'health_data',
    bootstrap_servers='localhost:9092',
    group_id='diabetes_group',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

producer = KafkaProducer(bootstrap_servers='localhost:9092', security_protocol="PLAINTEXT")

for message in consumer:
    data = message.value
    json_data = json.loads(json.dumps(data))
    json_rdd = spark.sparkContext.parallelize([json_data])
    df = spark.read.json(json_rdd)
    df = assembler.transform(df)
    prediction = model.transform(df)
    prediction.show()
    row = json.loads(prediction.toPandas().to_json(orient='records'))
    producer.send(
        topic="health_data_predicted",
        value=json.dumps(row).encode("utf-8")
    )
    time.sleep(random.randint(500, 1000) / 1000.0)
