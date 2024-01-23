import pandas as pd
from slicer import *
from time_series_to_intervals import *
from tqdm import tqdm


def dataset_history(
        df_list,
        main_feature,
        time_feature,
        main_value,
):
    start_time = None
    end_time = None

    for df_path in df_list:
        df = pd.read_csv(df_path, parse_dates=[time_feature])
        df_filtered = df[df[main_feature] == main_value]

        min_time = df_filtered[time_feature].min()
        max_time = df_filtered[time_feature].max()

        # Converts to UTC to avoid time zone error
        min_time_utc = min_time.tz_localize(None).tz_localize('UTC')
        max_time_utc = max_time.tz_localize(None).tz_localize('UTC')

        if start_time is None or min_time_utc < start_time:
            start_time = min_time_utc
        if end_time is None or max_time_utc > end_time:
            end_time = max_time_utc

        return[start_time, end_time]    

def samples_from_df_ts(
        dataframe, hs, he, 
        main_feature,  # "participant"
        time_feature,  # "vt" -> "seconds" before calling create_buckets
        group_feature,  # "calories"
        main_value,   # "p01"
        slicer_params,  # { "stride":3,  "ow":3, "ww":6, "pw":14, "granularity":'hours'}
        bucket_params):
    
    df = dataframe[dataframe[main_feature] == main_value].reset_index().copy()
    slices = slicer(hs,he, **slicer_params)
    st_ow_dfs = [df[(slices.iloc[i].st <= df[time_feature]) & (df[time_feature] <= slices.iloc[i].ow)].copy() for i in range(len(slices))]
    ww_pw_dfs = [df[(slices.iloc[i].ww <= df[time_feature]) & (df[time_feature] <= slices.iloc[i].pw)].copy() for i in range(len(slices))]

    for i, ww_pw_df in enumerate(tqdm(ww_pw_dfs)):
        ww_pw_df["i"] = i
    r = []    
    for i, st_ow_df in enumerate(tqdm(st_ow_dfs)):
        transaction_start = slices.iloc[i].st
        st_ow_df["seconds"]  = (st_ow_df[time_feature] - transaction_start).dt.total_seconds() #.apply(lambda x: x.total_seconds())
        bucket_values = bucket_params["bucket_values"]
        bi = create_intervals(create_buckets(st_ow_df,bucket_values, group_feature, "seconds", group_feature))
        bi["i"] = i
        bi["vt"] = bi["interval"].apply(lambda x: x[0])
        bi["value"] = bi["interval"].apply(lambda x: x[1] - x[0])
        bi["label"] = bi["bucket"]
        bi = bi[["i", "vt", "label", "value"]].copy()
        r.append(bi)
    return r, ww_pw_dfs

def samples_from_df_ev(dataframe, #"exercise.csv as example"
        hs, # history start
        he, # history end
        main_feature,  # "participant"
        time_feature,  # "vt" -> then adding "seconds" columns before calling create_buckets
        group_feature,  # "Activity"
        value_feature, # "value"--new-- and lacks bucket_params
        main_value,   # "p01"
        slicer_params): # { "stride":3,  "ow":3, "ww":6, "pw":14, "granularity":'hours'})  
    
    df = dataframe[dataframe[main_feature] == main_value].reset_index().copy()
    slices = slicer(hs, he, **slicer_params)
    
    st_ow_dfs = [df[(slices.iloc[i].st <= df[time_feature]) & (df[time_feature] <= slices.iloc[i].ow)].copy() for i in range(len(slices))]
    ww_pw_dfs = [df[(slices.iloc[i].ww <= df[time_feature]) & (df[time_feature] <= slices.iloc[i].pw)].copy() for i in range(len(slices))]
    
    for i, ww_pw_df in enumerate(tqdm(ww_pw_dfs)):
        ww_pw_df["i"] = i
    
    r = []    
    for i, st_ow_df in enumerate(tqdm(st_ow_dfs)):
        transaction_start = slices.iloc[i].st
        if not st_ow_df.empty:
            #st_ow_df["vt"] = (st_ow_df[time_feature] - min(st_ow_df[time_feature])).apply(lambda x: x.total_seconds())
            st_ow_df["vt"] = (st_ow_df[time_feature] - transaction_start).dt.total_seconds()


        bi = st_ow_df[[time_feature, group_feature, value_feature]].copy()
        bi["i"] = i
        bi["vt"] = bi["vt"] #or can be time_feature !
        bi["label"] = bi[group_feature]
        bi["value"] = bi[value_feature]
        bi = bi[["i", "vt", "label", "value"]].copy()
        r.append(bi)
    
    return r, ww_pw_dfs            

def samples_from_dataset(main_value, main_feature, time_feature, timeseries_data, event_data, slicer_params):
    
    all_odfs, all_pdfs = [], {}

    # Step a: Compute histories (hs, he) on all the .csv files

    history = dataset_history(timeseries_data["paths"]+event_data["paths"], main_feature, time_feature, main_value)

    for path, group_feature, bucket_values in zip(timeseries_data["paths"], timeseries_data["group_features"], timeseries_data["bucket_values"]):

            # Step b & c: Load the file in a DataFrame and fix time_feature's timezone
        ts_df = pd.read_csv(path, parse_dates=[time_feature])
        ts_df[time_feature] = ts_df[time_feature].apply(lambda x: x.tz_localize(None).tz_localize('UTC'))

            # Step d.a timeseries: Generate dataframes with samples_from_df_ts function
        t_odfs, t_pdfs = samples_from_df_ts(
                ts_df, history[0], history[1], main_feature,
                time_feature, group_feature,
                main_value, slicer_params, {"bucket_values": bucket_values}
            )
            #Iterates over its argument and adding each element to the list and extending the list,O(k)(used instead of append,O(1))
        all_odfs.extend(t_odfs)
        all_pdfs[group_feature] = t_pdfs

    for path, group_feature, value_feature in zip(event_data["paths"], event_data["group_features"], event_data["value_feature"]):
            # Step a: Compute histories (hs, he) on all the .csv files

            # Step b & c: Load the file in a DataFrame and fix time_feature's timezone
        ev_df = pd.read_csv(path, parse_dates=[time_feature])
        ev_df[time_feature] = ev_df[time_feature].apply(lambda x: x.tz_localize(None).tz_localize('UTC'))

            # Step d.b event: Generate dataframes with samples_from_df_ev function
        v_odfs, v_pdfs = samples_from_df_ev(
                ev_df, history[0], history[1], main_feature,
                time_feature, group_feature,
                value_feature, main_value, slicer_params
            )

        all_odfs.extend(v_odfs)
        all_pdfs[group_feature] = v_pdfs

    # Step e: Concatenate all the returned "odfs" and "odfs_ev" in a ordered DataFrame
    final_odf_df = pd.concat(all_odfs).sort_values(by=["i", "vt"]).reset_index(drop=True)
    return final_odf_df, all_pdfs