from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import unix_timestamp
from pyspark.ml.feature import StringIndexer
import matplotlib.pyplot as plt
from pyspark.sql.functions import count

spark = SparkSession.builder.appName("KMeansExample").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load data
data = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/home/am/T7/uber-raw-data-aug14.csv")

data = data.withColumn('Date/Time_unix', unix_timestamp('Date/Time', 'M/d/yyyy H:m:s'))
stringIndexer = StringIndexer(inputCol="Base", outputCol="BaseIndex")
data = stringIndexer.fit(data).transform(data)
print('\n\n\n\n')


train_data, test_data = data.randomSplit([0.8, 0.2])

test_data.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save("/home/am/T7/stream_test_set.csv")


# Prepare train_data for k-means
vectorAssembler = VectorAssembler(inputCols=["Date/Time_unix", "Lat", "Lon", "BaseIndex"], outputCol="features")
train_data = vectorAssembler.transform(train_data).select('features')

vectorAssembler = VectorAssembler(inputCols=["Date/Time_unix", "Lat", "Lon", "BaseIndex"], outputCol="features")
test_data = vectorAssembler.transform(test_data).select('features')

print(train_data.count(), test_data.count())


silh = []
for i in range(2,11):
    kmeans = KMeans().setK(i).setSeed(1)
    model = kmeans.fit(train_data)
    predictions = model.transform(train_data)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    silh.append(silhouette)
    print("Silhouette - k" + str(i) + ":\t"+ str(silhouette))

sse = []
for k in range(2, 11):
    kmeans = KMeans().setK(k).setSeed(1)
    model = kmeans.fit(train_data)
    se= model.summary.trainingCost
    print("Elbow Method - k" + str(k) + ":\t"+ str(se))
    sse.append(model.summary.trainingCost)


fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot(range(2, 11), sse)
axs[0].set_title('Elbow Method')
axs[0].set_xlabel('Number of clusters')
axs[0].set_ylabel('SSE')

axs[1].plot(range(2, 11), silh)
axs[1].set_title('Silhouette')
axs[1].set_xlabel('Number of clusters')
axs[1].set_ylabel('Silhouette')

plt.show()


# Train k-means model
kmeans = KMeans().setK(3).setSeed(1)
model = kmeans.fit(train_data)

predictions = model.transform(train_data)
cluster_count = predictions.groupBy("prediction").agg(count("*").alias("count"))
cluster_count.show()

predictions = model.transform(test_data)
cluster_count = predictions.groupBy("prediction").agg(count("*").alias("count"))
cluster_count.show()

model.save("/home/am/T7/K10_means_Model")

spark.stop()