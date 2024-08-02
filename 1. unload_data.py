from pyspark.sql import SparkSession, types
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Unload Reddit Data").getOrCreate()
assert spark.version >= '3.2'
spark.sparkContext.setLogLevel('WARN')

submissions_schema = types.StructType([
    types.StructField('archived', types.BooleanType()),
    types.StructField('author', types.StringType()),
    # types.StructField('author_flair_css_class', types.StringType()),
    types.StructField('author_flair_text', types.StringType()),
    types.StructField('created', types.LongType()),
    types.StructField('created_utc', types.StringType()),
    # types.StructField('distinguished', types.StringType()),
    # types.StructField('domain', types.StringType()),
    types.StructField('downs', types.LongType()),
    types.StructField('edited', types.BooleanType()),
    types.StructField('from', types.StringType()),
    types.StructField('from_id', types.StringType()),
    types.StructField('from_kind', types.StringType()),
    # types.StructField('gilded', types.LongType()),
    types.StructField('hide_score', types.BooleanType()),
    types.StructField('id', types.StringType()),
    types.StructField('is_self', types.BooleanType()),
    # types.StructField('link_flair_css_class', types.StringType()),
    types.StructField('link_flair_text', types.StringType()),
    # types.StructField('media', types.StringType()),
    types.StructField('name', types.StringType()),
    types.StructField('num_comments', types.LongType()),
    types.StructField('over_18', types.BooleanType()),
    types.StructField('permalink', types.StringType()),
    types.StructField('quarantine', types.BooleanType()),
    # types.StructField('retrieved_on', types.LongType()),
    types.StructField('saved', types.BooleanType()),
    types.StructField('score', types.LongType()),
    # types.StructField('secure_media', types.StringType()),
    types.StructField('selftext', types.StringType()),
    types.StructField('stickied', types.BooleanType()),
    types.StructField('subreddit', types.StringType()),
    types.StructField('subreddit_id', types.StringType()),
    # types.StructField('thumbnail', types.StringType()),
    types.StructField('title', types.StringType()),
    types.StructField('ups', types.LongType()),
    types.StructField('url', types.StringType()),
    types.StructField('year', types.IntegerType()),
    types.StructField('month', types.IntegerType()),
])



def main():
    def load_all_json_gz_to_dataframe_spark(directory):
        # Read all JSON.gz files into a Spark DataFrame
        df = spark.read.json(f"{directory}")
        # Filter the DataFrame
        filtered_df = df.filter(
            (col('selftext') != '[removed]') &
            (col('selftext') != '[deleted]') &
            (col('num_comments').cast('int') > 10) &
            ((col('link_flair_text') == 'Asshole') | (col('link_flair_text') == 'Not the A-hole'))
        )

        return filtered_df

    directory_path = 'reddit-subset/submissions' # change this to the reddit subset you desire, need to be done once for 2022 and 2023 
    df = load_all_json_gz_to_dataframe_spark(directory_path)
    # df.write.json('output/filtered_not_balanced_2022')
    pd_df = df.toPandas() # this is okay as the data set is only 100k rows 
    pd_df.to_json('output/filtered_not_balanced.json.gz' , compression='gzip') # make output file name correspond with the input subset 
if __name__ == '__main__':
    main()