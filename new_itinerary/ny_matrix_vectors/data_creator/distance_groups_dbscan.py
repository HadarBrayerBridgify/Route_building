import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import re

def create_long_lat(df):
    df["geolocation"].fillna(0, inplace=True)
    df["long_lat"] = None
    empty_idx = df[df["geolocation"] == 0].index
    df["long_lat"].loc[empty_idx] = 0
    true_idx = set(df.index) - set(empty_idx)
    df["long_lat"].loc[true_idx] = df["geolocation"].loc[true_idx].apply \
        (lambda x: [float(s) for s in re.findall(r'-?\d+\.?\d*', x)][-2:])

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


def anomaly_geolocation(df, max_number_of_centers=1, main_cluster_percent=90, secondary_cluster_percent=20):
    # handle edge case of second_largest_cluster_size############################################################

    try:
        df.set_index('uuid', drop=True, inplace=True)
    except:
        pass

    uuid_Nones = []

    create_long_lat(df)
    df['correct_geo'] = df['long_lat'].apply(lambda x: insure_long_lat(x))

    uuid_Nones += list(df[df['geolocation'] == '0'].index)
    uuid_Nones += list(df[df['geolocation'] == 0].index)

    attractions_amount = len(df)
    print(attractions_amount)

    df = df[df['geolocation'] != '0']
    df = df[df['geolocation'] != 0]

    geos = list(df['correct_geo'])

    did_break = 0

    if max_number_of_centers == 2:

        for eps in np.linspace(0.05, 1, num=20):
            if did_break:
                break

            for min_samples in np.linspace(1, 10, num=10):

                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan.fit(geos)

                df['cluster_id'] = list(dbscan.labels_)

                a = list(df['cluster_id'].unique())

                try:
                    a.remove(-1)
                except:
                    pass

                largest_cluster_id = 0
                largest_cluster_size = 0

                second_largest_cluster_id = 0
                second_largest_cluster_size = 0

                for i in a:
                    count = len(df[df['cluster_id'] == i])
                    if count > largest_cluster_size:
                        second_largest_cluster_id = largest_cluster_id
                        second_largest_cluster_size = largest_cluster_size

                        largest_cluster_id = i
                        largest_cluster_size = count

                    elif count > second_largest_cluster_size:
                        second_largest_cluster_id = i
                        second_largest_cluster_size = count

                if len(a) < 5 and largest_cluster_size + second_largest_cluster_size > main_cluster_percent / 100 * attractions_amount \
                        and second_largest_cluster_size > secondary_cluster_percent / 100 * attractions_amount:
                    did_break = 1
                    break
        # print(eps)
        # print(min_samples)
        # print(len(a))
        # print((largest_cluster_size + second_largest_cluster_size)/attractions_amount)
        # print(second_largest_cluster_size)
        print(largest_cluster_id)
        print(second_largest_cluster_id)

        # 2 dicts of 2 main regions:
        if did_break:

            main_region_dict = {}
            secondary_region_dict = {}

            main_region_dict['nones'] = uuid_Nones
            main_region_dict['epsilon'] = eps
            main_region_dict['min_samples'] = min_samples
            main_region_dict['main'] = list(df[df['cluster_id'] == largest_cluster_id].index)

            secondary_region_dict['nones'] = uuid_Nones
            secondary_region_dict['epsilon'] = eps
            secondary_region_dict['min_samples'] = min_samples
            secondary_region_dict['main'] = list(df[df['cluster_id'] == second_largest_cluster_id].index)

            a.remove(largest_cluster_id)
            try:
                a.remove(second_largest_cluster_id)
            except:
                pass

            main_region_centeral_geolocation = [float(sum(col)) / len(col) for col in
                                                zip(*list(df[df['cluster_id'] == largest_cluster_id]['correct_geo']))]
            second_region_centeral_geolocation = [float(sum(col)) / len(col) for col in zip(*list(
                df[df['cluster_id'] == second_largest_cluster_id]['correct_geo']))]

            df_help = pd.DataFrame(columns=['old_cluster_index', 'cluster_distance_from_main_region_center',
                                            'cluster_distance_from_second_region_center'])
            for cluster_id in a:
                new_row = {'old_cluster_index': cluster_id, 'cluster_distance_from_main_region_center': None,
                           'cluster_distance_from_second_region_center': None}
                df_help = df_help.append(new_row, ignore_index=True)
                cluster_center = [float(sum(col)) / len(col) for col in
                                  zip(*list(df[df['cluster_id'] == cluster_id]['correct_geo']))]
                df_help['cluster_distance_from_main_region_center'].iloc[-1] = ((cluster_center[0] -
                                                                                 main_region_centeral_geolocation[
                                                                                     0]) ** 2 \
                                                                                + (cluster_center[0] -
                                                                                   main_region_centeral_geolocation[
                                                                                       0]) ** 2) ** 0.5
                df_help['cluster_distance_from_second_region_center'].iloc[-1] = ((cluster_center[0] -
                                                                                   second_region_centeral_geolocation[
                                                                                       0]) ** 2 \
                                                                                  + (cluster_center[0] -
                                                                                     second_region_centeral_geolocation[
                                                                                         0]) ** 2) ** 0.5

            df_help['is_closer_to_main_region'] = df_help['cluster_distance_from_main_region_center'] <= df_help[
                'cluster_distance_from_second_region_center']

            df_help_main_region = df_help.loc[df_help['is_closer_to_main_region'] == True]
            df_help_secondary_region = df_help.loc[df_help['is_closer_to_main_region'] == False]

            df_help_main_region.sort_values(by='cluster_distance_from_main_region_center', inplace=True)
            for idx in range(len(df_help_main_region)):
                main_region_dict[str(idx + 1)] = list(
                    df[df['cluster_id'] == df_help_main_region['old_cluster_index'].iloc[idx]].index)

            df_help_secondary_region.sort_values(by='cluster_distance_from_main_region_center', inplace=True)
            for idx in range(len(df_help_secondary_region)):
                secondary_region_dict[str(idx + 1)] = list(
                    df[df['cluster_id'] == df_help_secondary_region['old_cluster_index'].iloc[idx]].index)

            anomalies_uuid_list = list(df[df['cluster_id'] == -1].index)
            anomalies_main_region_uuid_list = []
            anomalies_secondary_region_uuid_list = []

            for uuid in anomalies_uuid_list:

                uuid_geolocation = df.loc[uuid]['correct_geo']
                distance_from_main_region = ((uuid_geolocation[0] - main_region_centeral_geolocation[0]) ** 2 \
                                             + (uuid_geolocation[0] - main_region_centeral_geolocation[0]) ** 2) ** 0.5
                distance_from_secondary_region = ((uuid_geolocation[0] - second_region_centeral_geolocation[0]) ** 2 \
                                                  + (uuid_geolocation[0] - second_region_centeral_geolocation[
                            0]) ** 2) ** 0.5

                if distance_from_main_region <= distance_from_secondary_region:
                    anomalies_main_region_uuid_list += [uuid]
                else:
                    anomalies_secondary_region_uuid_list += [uuid]

            main_region_dict['anomalies'] = anomalies_main_region_uuid_list
            secondary_region_dict['anomalies'] = anomalies_secondary_region_uuid_list

            # df_1 = df.groupby('cluster_id')['long_lat'].apply(list).reset_index(name='new')

            dicts_list = [main_region_dict, secondary_region_dict]
            centeral_geolocations_list = [main_region_centeral_geolocation, second_region_centeral_geolocation]

            # dict keys: 'main','1','2'...,'anomalies','nones','centeral_geolocation', 'epsilon', 'min_samples'
            return dicts_list, centeral_geolocations_list

        # 1 dict of 1 main region
        else:

            main_region_dict = {}
            main_region_dict['nones'] = uuid_Nones
            main_region_dict['epsilon'] = eps
            main_region_dict['min_samples'] = min_samples
            main_region_dict['anomalies'] = list(df[df['cluster_id'] == -1].index)
            main_region_dict['main'] = list(df[df['cluster_id'] == largest_cluster_id].index)

            a.remove(largest_cluster_id)

            main_region_centeral_geolocation = [float(sum(col)) / len(col) for col in
                                                zip(*list(df[df['cluster_id'] == largest_cluster_id]['correct_geo']))]

            df_help = pd.DataFrame(columns=['old_cluster_index', 'cluster_distance_from_region_center'])
            for cluster_id in a:
                new_row = {'old_cluster_index': cluster_id, 'cluster_distance_from_region_center': None}
                df_help = df_help.append(new_row, ignore_index=True)
                cluster_center = [float(sum(col)) / len(col) for col in
                                  zip(*list(df[df['cluster_id'] == cluster_id]['correct_geo']))]
                df_help['cluster_distance_from_region_center'].iloc[-1] = ((cluster_center[0] -
                                                                            main_region_centeral_geolocation[0]) ** 2 \
                                                                           + (cluster_center[0] -
                                                                              main_region_centeral_geolocation[
                                                                                  0]) ** 2) ** 0.5

            df_help.sort_values(by='cluster_distance_from_region_center', inplace=True)
            for idx in range(len(df_help)):
                main_region_dict[str(idx + 1)] = list(
                    df[df['cluster_id'] == df_help['old_cluster_index'].iloc[idx]].index)

            # df_1 = df.groupby('cluster_id')['long_lat'].apply(list).reset_index(name='new')

            dicts_list = [main_region_dict]
            centeral_geolocations_list = [main_region_centeral_geolocation]

            # dict keys: 'main','1','2'...,'anomalies','nones','centeral_geolocation', 'epsilon', 'min_samples'
            return dicts_list, centeral_geolocations_list

    elif max_number_of_centers == 1:

        for eps in np.linspace(0.05, 1, num=20):
            if did_break:
                break

            for min_samples in np.linspace(1, 10, num=10):

                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                dbscan.fit(geos)

                df['cluster_id'] = list(dbscan.labels_)

                a = list(df['cluster_id'].unique())

                try:
                    a.remove(-1)
                except:
                    pass

                largest_cluster_id = 0
                largest_cluster_size = 0

                second_largest_cluster_id = 0
                second_largest_cluster_size = 0

                for i in a:
                    count = len(df[df['cluster_id'] == i])
                    if count > largest_cluster_size:
                        second_largest_cluster_id = largest_cluster_id
                        second_largest_cluster_size = largest_cluster_size

                        largest_cluster_id = i
                        largest_cluster_size = count

                    elif count > second_largest_cluster_size:
                        second_largest_cluster_id = i
                        second_largest_cluster_size = count

                if len(a) < 5 and largest_cluster_size > main_cluster_percent / 100 * attractions_amount:
                    did_break = 1
                    break

        main_region_dict = {}
        main_region_dict['nones'] = uuid_Nones
        main_region_dict['epsilon'] = eps
        main_region_dict['min_samples'] = min_samples
        main_region_dict['anomalies'] = list(df[df['cluster_id'] == -1].index)
        main_region_dict['main'] = list(df[df['cluster_id'] == largest_cluster_id].index)

        a.remove(largest_cluster_id)

        main_region_centeral_geolocation = [float(sum(col)) / len(col) for col in
                                            zip(*list(df[df['cluster_id'] == largest_cluster_id]['correct_geo']))]

        df_help = pd.DataFrame(columns=['old_cluster_index', 'cluster_distance_from_region_center'])
        print(a)
        for cluster_id in a:
            new_row = {'old_cluster_index': cluster_id, 'cluster_distance_from_region_center': None}
            df_help = df_help.append(new_row, ignore_index=True)
            cluster_center = [float(sum(col)) / len(col) for col in
                              zip(*list(df[df['cluster_id'] == cluster_id]['correct_geo']))]
            df_help['cluster_distance_from_region_center'].iloc[-1] = ((cluster_center[0] -
                                                                        main_region_centeral_geolocation[0]) ** 2 \
                                                                       + (cluster_center[0] -
                                                                          main_region_centeral_geolocation[
                                                                              0]) ** 2) ** 0.5

        df_help.sort_values(by='cluster_distance_from_region_center', inplace=True)

        for idx in range(len(df_help)):
            main_region_dict[str(idx + 1)] = list(df[df['cluster_id'] == df_help['old_cluster_index'].iloc[idx]].index)

        # df_1 = df.groupby('cluster_id')['long_lat'].apply(list).reset_index(name='new')

        dicts_list = [main_region_dict]
        centeral_geolocations_list = [main_region_centeral_geolocation]

        # dict keys: 'main','1','2'...,'anomalies','nones','centeral_geolocation', 'epsilon', 'min_samples'
        return dicts_list, centeral_geolocations_list

    else:
        print('max_number_of_centers should be either 1 or 2')
        return None, None
