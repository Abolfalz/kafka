from pyspark.ml.clustering import KMeansModel
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import unix_timestamp
from pyspark.ml.feature import StringIndexer
import matplotlib.pyplot as plt
from kafka import KafkaConsumer
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime

spark = SparkSession.builder.appName("LoadStream").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

model = KMeansModel.load("/home/am/T7/K10_means_Model")

bootstrap_servers = ['localhost:9092']
topicName = 'csv_data'

consumer = KafkaConsumer(topicName, bootstrap_servers = bootstrap_servers, auto_offset_reset = 'latest')

schema = StructType([   
    StructField("Lat", FloatType(),True),
    StructField("Lon", FloatType(),True),
    StructField("Date/Time_unix", IntegerType(),True),
    StructField("BaseIndex", IntegerType(), True)
])

for message in consumer:
    f = open("/home/am/T7/Result_Out_predict.txt", "a")
    message_str = message.value.decode('utf-8')
    data_list = message_str.split(', ')
    lat = float(data_list[1])
    lon = float(data_list[2])
    datetime_unix = int(data_list[4])
    bi = data_list[5]
    base_index = float(bi.split('.')[0])
    data = spark.createDataFrame([(datetime_unix ,lat ,lon ,base_index)], ["Date/Time_unix", "Lat", "Lon", "BaseIndex"])
    vectorAssembler = VectorAssembler(inputCols=["Date/Time_unix", "Lat", "Lon", "BaseIndex"], outputCol="features")
    data = vectorAssembler.transform(data).select('features')
    predictions = model.transform(data) 
    cluster = predictions.select("prediction").collect()[0][0] 
    print(data,"Predicted clusters:", cluster) 
    f.write('{} \t {} \n'.format(data,cluster))
    f.close()

consumer.close()