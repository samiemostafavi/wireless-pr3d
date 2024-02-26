from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from loguru import logger
from datetime import datetime, timedelta
import pandas as pd

MAX_PAST_DAYS = 7

class InfluxClient:
    def __init__(self, influx_db_address, token, bucket, org, point_name, fields = None, time_key = "send.timestamp"):
        self.point_name = point_name
        self.bucket = bucket
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
        
        # configure write api
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)

    def push_dataframe(self, df, point_name):
        for index, row in df.iterrows():
            point = Point(point_name)
            for f in df.keys():
                point.field(f, row[f])

            point.time(datetime.utcnow(), WritePrecision.NS)
            self.write_api.write(self.bucket, self.org, point)

    def run_query(self, query):
        logger.debug(f"Sending query to influxDB: {query}")
        query_result = self.client.query_api().query(org=self.org, query=query)
        df = pd.DataFrame()
        for table in query_result:
            # table.records is the columns
            column_name = table.records[0].get_field()
            column_values = []
            for record in table.records:
                column_values.append(record.get_value())
            df[column_name] = column_values

        # add time
        if query_result:
            column_name = "time"
            column_values = []
            for record in query_result[0].records:
                ts = datetime.timestamp(record.get_time())
                column_values.append(ts)
            df[column_name] = column_values

        logger.debug(f"Received query results:\n{df}")
        return df

    def get_recent_samples_dur(self, duration : timedelta):
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
