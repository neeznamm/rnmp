import json
import random
import time

import pyspark as ps
from kafka import KafkaProducer

spark = ps.sql.SparkSession.builder \
    .master("local[4]") \
    .appName("diabetes classification offline") \
    .getOrCreate()

sc = spark.sparkContext

online_df = spark.read.csv('online.csv',
                           header=True,
                           quote='"',
                           sep=",",
                           inferSchema=True)

producer = KafkaProducer(bootstrap_servers='localhost:9092', security_protocol="PLAINTEXT")

for row in json.loads(online_df.toPandas().drop(['Diabetes_binary'], axis=1).to_json(orient='records')):
    producer.send(
        topic="health_data",
        value=json.dumps(row).encode("utf-8")
    )
    time.sleep(random.randint(500, 2000) / 1000.0)
