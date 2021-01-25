import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Dat, TimestampType

config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config.get('aws_key','AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY']=config.get('aws_key','AWS_SECRET_ACCESS_KEY')

def create_spark_session():
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    # get filepath to song data file
    song_data = input_data + 'song_data/*/*/*/*.json'
    
    songSchema = R([
                    Fld("artist_id",Str()),
                    Fld("artist_latitude",Dbl()),
                    Fld("artist_location",Str()),
                    Fld("artist_longitude",Dbl()),
                    Fld("artist_name",Str()),
                    Fld("duration",Dbl()),
                    Fld("num_songs",Int()),
                    Fld("title",Str()),
                    Fld("year",Int()),
                ])

    # read song data file
    df = spark.read.json(song_data, schema=songSchema)

    # extract columns to create songs table
    songs_table =  df.select(["title","artist_id","year","duration"]).dropDuplicates().withColumn("song_id", monotonically_increasing_id())
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode("overwrite").parquet(output_data + 'songs')


    # extract columns to create artists table
    artists_table = df.selectExpr(["artist_id","artist_name as name", "artist_location as location","artist_latitude as lattitude", "artist_longitude as longitude"]).dropDuplicates()
    
    # write artists table to parquet files
    artists_table.write.mode("overwrite").parquet(output_data + 'artists')


def process_log_data(spark, input_data, output_data):
    # get filepath to log data file
    log_data = input_data + 'log_data/*/*/*.json'

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter("page == 'NextSong'")

    # extract columns for users table    
    user_table = df.selectExpr(["useId as user_id","firstName as first_name","lastName as last_name","gender","level"])
    
    # write users table to parquet files
    users_table.write.mode("overwrite").parquet(output_data + 'users')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: x / 1000, TimestampType())
    df = df.withColumn("start_time",get_timestamp('ts'))
    
    # create datetime column from original timestamp column
    get_datetime = udf(datetime.fromtimestamp(x),TimestampType())
    df = df.withColumn("datetime",get_datetime('ts'))
    
    # extract columns to create time table
    time_table = df.select("start_time").withColumn("hour",hour("start_time")).withColumn("day",dayofmonth("start_time"))\
                .withColumn("week",weekofyear("start_time")).withColumn("month",month("start_time")).withColumn("year",year("start_time"))\
                .withColumn("weekday",dayofweek("start_time"))
    
    # write time table to parquet files partitioned by year and month
    time_table.write.mode("overwrite").partitionBy("year", "month").parquet(output_data + "time")

    # read in song data to use for songplays table
    song_df = spark.read.parquet(output_data + 'songs/*/*/*')
    # read in artists data to use for songplays table
    artists_df = spark.read.parquet(output_data + 'artists')
    
    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = df.join(song_df, song_df.title == df.song).join(artists_df, artists_df.name == df.artist)
    songplays_table = songplays_table.selectExpr(["start_time", "userId as user_id", "level","song_id", "artist_id",\
                                                "sessionId as session_id","location","userAgent as user_agent"])\
                                                .withColumn("year", year("start_time")).withColumn("month", month("start_time"))
    
    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode("overwrite").partitionBy("year", "month").parquet(output_data, 'songplays')


def main():
    spark = create_spark_session()
    input_data = "s3a://udacity-dend/"
    output_data = "s3a://data-lake-buckets-jy/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
