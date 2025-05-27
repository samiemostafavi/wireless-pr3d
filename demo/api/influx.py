from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from loguru import logger
from datetime import datetime, timedelta
import pandas as pd

MAX_PAST_DAYS = 7

class InfluxClient:
    def __init__(self, influx_db_address, token, bucket, org, point_name, fields = None, time_key = "send.timestamp"):
        self.point_name = point_name
        self.org = org
        self.time_key = time_key
        self.fields = fields
        self.influx_db_address = influx_db_address
        self.token = token


        # Connect to InfluxDB server
        self.client = InfluxDBClient(
          url=influx_db_address,
          token=token,
          org=org
        )
        ready_dict = self.client.ready()
        logger.info(f"InfluxDB client:\n{ready_dict}")
        
        self.create_bucket(bucket)
        self.bucket = bucket
        
        logger.info(f"InfluxDB initialized with:")
        logger.info(f"  URL: {self.influx_db_address}")
        logger.info(f"  Org: {self.org}")
        logger.info(f"  Bucket: {self.bucket}")
        logger.info(f"  Point Name: {self.point_name}")
        logger.info(f"  Time Key: {self.time_key}")
        logger.info(f"  Fields: {self.fields if self.fields else 'All DataFrame columns'}")
        
        # configure write api
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def create_bucket(self, bucket):
        buckets_api = self.client.buckets_api()

        bucket_names = [b.name for b in buckets_api.find_buckets().buckets]
        if bucket not in bucket_names:
            buckets_api.create_bucket(bucket_name=bucket, org=self.org, retention_rules=[])

    #def push_dataframe(self, df, point_name):
    #    point = Point(point_name)
    #    for index, row in df.iterrows():
    #        for f in df.keys():
    #            point.field(f, row[f])
    #    point.time(datetime.utcnow(), WritePrecision.NS)
    #    self.write_api.write(self.bucket, self.org, point)

    def push_dataframe(self, df, point_name):
        points = []

        # All points share this timestamp
        shared_time = datetime.utcnow()

        for i, row in df.iterrows():
            point = Point(point_name)

            # Add a unique tag for each row
            point = point.tag("point_id", str(i))

            # Add all fields (cast to float for safety)
            for f in df.columns:
                point = point.field(f, float(row[f]))

            # Use the same timestamp for all rows
            point = point.time(shared_time, WritePrecision.NS)

            points.append(point)

        self.write_api.write(self.bucket, self.org, points)


    # this new function takes care of the case when we have different sizes of columns
    def run_query(self, query):
        logger.debug(f"Sending query to influxDB: {query}")
        query_result = self.client.query_api().query(org=self.org, query=query)
        
        all_dfs = []

        for table in query_result:
            field = table.records[0].get_field()
            records = [(r.get_time(), r.get_value()) for r in table.records]
            temp_df = pd.DataFrame(records, columns=["_time", field])
            all_dfs.append(temp_df)

        # Merge on _time using outer join
        if not all_dfs:
            return pd.DataFrame()

        df = all_dfs[0]
        for other_df in all_dfs[1:]:
            df = pd.merge(df, other_df, on="_time", how="outer")

        # Optional: sort by time and reset index
        df = df.sort_values("_time").reset_index(drop=True)

        return df

    def get_recent_samples_dur(self, duration : timedelta, field=None):
        if field is not None:
            query = f'''
                from(bucket: "{self.bucket}")
                    |> range(start: -duration(v:{int(duration.total_seconds()*10.0**9)}))
                    |> filter(fn: (r) => r._field == \"{field}\")
                    |> filter(fn: (r) => r._measurement == \"{self.point_name}\")
            '''
        else:
            query = f'''
                from(bucket: "{self.bucket}")
                    |> range(start: -duration(v:{int(duration.total_seconds()*10.0**9)}))
                    |> filter(fn: (r) => r._measurement == \"{self.point_name}\")
            '''
        return self.run_query(query)

    def get_latest_samples_num(self, number : int):
        query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: -{MAX_PAST_DAYS}d)
                |> filter(fn: (r) => r._measurement == \"{self.point_name}\")
                |> tail(n: {number})
        '''
        return self.run_query(query)
    
    def get_latest_samples_dur(self, duration : timedelta):
        latest_sample_df = self.get_latest_samples_num(1)
        latest_sample = latest_sample_df.iloc[0].to_dict()
        end_time = latest_sample['time'] # in seconds
        end_time_dt = datetime.fromtimestamp(end_time)
        start_time = end_time - duration.total_seconds() # in seconds
        start_time_dt = datetime.fromtimestamp(start_time)
        query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: -{MAX_PAST_DAYS}d)
                |> filter(fn: (r) => r._measurement == \"{self.point_name}\")
                |> range(start: time(v:{int(start_time_dt.timestamp()*10**9):d}), stop: time(v:{int(end_time_dt.timestamp()*10**9):d}))
        '''
        res = self.run_query(query)

    def __del__(self):
        self.client.close()
