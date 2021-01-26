# DataLake_ETL
Builds an ETL pipeline for a data lake hosted on S3.  

The data that will be used:   
> Song data: s3://udacity-dend/song_data  

The first dataset is a subset of real data from the Million Song Dataset. Each file is in JSON format and contains metadata about a song and the artist of that song. 

> Log data: s3://udacity-dend/log_data   

The second dataset consists of log files in JSON format generated by this event simulator based on the songs in the dataset above. These simulate app activity logs from an imaginary music streaming app based on configuration settings.

----------------------   

#### dl.cfg
Before running etl.py, edit this file and input your AWS access key id and AWS secert access key.  

#### etl.py
Creates a spark session that will extra data from the song data and log data AWS s3 buckets. After extraction the data will be transformed and loaded to another AWS s3 bucket.
--------------------  

### Instructions
* Replace the AWS IAM credentials found in dl.cfg
* Modify the output path in etl.py
* Run etl.py 
