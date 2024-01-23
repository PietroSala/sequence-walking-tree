import pandas as pd
from sklearn.cluster import MeanShift
import numpy as np

def create_buckets(df, bucket_values, group_name, value_name, prefix):
    df['bucket'] = pd.cut(df[group_name], bins=[-float('inf')] + bucket_values + [float('inf')],
                           labels=['{}_{}'.format(prefix,i) for i in range(1, len(bucket_values) + 2)])
    result_df = df.groupby('bucket', observed=False)[value_name].apply(list).reset_index(name='values_in_bucket')
    return result_df

def bucket_to_interval(lst): #we suppose that the list has a single feature of the values in the bucket
    ms = MeanShift(bandwidth=None, bin_seeding=True)
    ms.fit(np.array(lst).reshape(-1, 1)) 
    labels = ms.labels_
    
    intervals = []
    current_label = labels[0]
    start_index = 0
    
    # Iterates through labels to identify continuous intervals
    for i, label in enumerate(labels[1:], start=1):
        if label != current_label:
            intervals.append([start_index, i - 1])
            start_index = i
            current_label = label
    
    # Adds the last interval
    intervals.append([start_index, len(labels) - 1])
    
    # Converts intervals to start and end values
    start_end_values = []
    for interval in intervals:
        start_value = lst[interval[0]]
        end_value = lst[interval[1]]
        start_end_values.append([start_value, end_value])
    
    return start_end_values

def create_intervals(created_buckets):
    cb = created_buckets[created_buckets.values_in_bucket.apply(lambda x: len(x)>0)].copy()
    cb["intervals_in_bucket"] = cb.values_in_bucket.apply(lambda x: bucket_to_interval(x)) 
    r = pd.concat([pd.DataFrame([{"bucket": row[1].bucket, "interval": i} for i in row[1].intervals_in_bucket])  for row in cb.iterrows()])
    r.reset_index(inplace=True, drop=True)
    return r