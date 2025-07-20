from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
import logging
import transform
import os
import time

spark = SparkSession.builder \
    .appName("Streaming Prediction") \
    .getOrCreate()

log4j = spark.sparkContext._jvm.org.apache.log4j
log4j.Logger.getLogger("org").setLevel(log4j.Level.ERROR)

ssc = StreamingContext(spark.sparkContext, 5)

demo_path = "/data/processed/demo.csv"
model_path = "/data/processed/model"
output_path = "/data/streaming_results.csv"

static_df = spark.read.option("header", "true").option("inferSchema", "true").csv(demo_path)
batch_size = 10 
batches = [static_df.limit(batch_size * (i + 1)).exceptAll(static_df.limit(batch_size * i)) for i in range((static_df.count() + batch_size - 1) // batch_size)]

def send_batch(batch, host="localhost", port=9999):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        for row in batch.collect():
            data = ",".join(str(row[c]) for c in row.__fields__ if c != "Price") + "\n"
            s.sendall(data.encode())
        s.close()

lines = ssc.socketTextStream("localhost", 9999)

def process_batch(time, rdd):
    if not rdd.isEmpty():
        df = spark.createDataFrame(rdd.map(lambda x: x.split(",")), schema=["SquareFeet", "Bedrooms", "Bathrooms", "Neighborhood", "YearBuilt"])
        
        df_transformed = transform.transform_data(df)
        
        model = PipelineModel.load(model_path)
        predictions = model.transform(df_transformed).select("SquareFeet", "Bedrooms", "Bathrooms", "Neighborhood", "YearBuilt", "prediction")
        
        combined_df = df.join(predictions, ["SquareFeet", "Bedrooms", "Bathrooms", "Neighborhood", "YearBuilt"], "inner")
        combined_df.write.mode("append").option("header", "true").csv(output_path)
        
        print(f"Batch at {time}:")
        df.show()
        print(f"Predictions at {time}:")
        predictions.show()

lines.foreachRDD(process_batch)

def send_batches():
    while True:
        for batch in batches:
            send_batch(batch)
            time.sleep(6)  

ssc.start()
import threading
threading.Thread(target=send_batches, daemon=True).start()

ssc.awaitTermination()


