from sklearn.cluster import DBSCAN
import pickle
import re


def insure_long_lat(coordinate_list):
    """
    relevant only for places in center of earth (in terms of east and west)
    """
    if not coordinate_list:
        return 0
    if len(coordinate_list) == 1:
        return 0
    if abs(coordinate_list[1]) > abs(coordinate_list[0]):
        return [coordinate_list[1], coordinate_list[0]]
    else:
        return coordinate_list


def create_long_lat(df):
    df["geolocation"].fillna(0, inplace=True)
    df["long_lat"] = None
    empty_idx = list(df[df["geolocation"] == 0].index)
    empty_idx += list(df[df["geolocation"] == '0'].index)
    df["long_lat"].loc[empty_idx] = 0
    true_idx = set(df.index) - set(empty_idx)
    df["long_lat"].loc[true_idx] = df["geolocation"].loc[true_idx].apply \
        (lambda x: [float(s) for s in re.findall(r'-?\d+\.?\d*', x)][-2:])
    df["long_lat"] = df["long_lat"].apply(lambda x: insure_long_lat(x))
    print("created 'lon_lat' col")


def anomaly_geolocation(df):
    """

    :param df: Dataframe of the attractions
    :return: list of the outliers uuids and list of long,lat of the prime geolocation
    """
    uuid_list = []
    create_long_lat(df)
    df['correct_geo'] = df['long_lat'].apply(lambda x: insure_long_lat(x))
    uuid_list += list(df[df['geolocation'] == '0']['uuid'])
    uuid_list += list(df[df['geolocation'] == 0]['uuid'])
    len_uuid_list = len(uuid_list)
    df = df[df['geolocation'] != '0']
    df = df[df["geolocation"] != 0]
    geos = list(df['correct_geo'])
    dbscan = DBSCAN(eps=0.6, min_samples=3)
    dbscan.fit(geos)
    df['cluster_id'] = list(dbscan.labels_)
    df_1 = df.groupby('cluster_id')['long_lat'].apply(list).reset_index(name='new')
    a = list(df['cluster_id'].unique())
    try:
        a.remove(-1)
    except:
        pass
    largest_cluster = 0
    largest_count = 0
    for i in a:
        count = len(df[df['cluster_id'] == i])
        if count > largest_count:
            largest_cluster = i
            largest_count = count
    uuid_list += list(df[df['cluster_id'] != largest_cluster]['uuid'])
    central_geolocation = [float(sum(col)) / len(col) for col in
                           zip(*list(df[df['cluster_id'] == largest_cluster]['correct_geo']))]

    return [uuid_list, central_geolocation]

