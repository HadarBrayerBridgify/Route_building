import re
import gmplot
import random
import string
import json
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from itertools import combinations
from typing import Any, Dict, List
from datetime import datetime, date, timedelta
import time
import restaurants as rest
from typing import Dict, List, Tuple

# import restaurants as rest


# new data preprocessing
NUM_ATTRACTIONS = 8


def update_columns_names(df):
    # return df.rename({"title": "name", "description": "about", "categories_list": "tags", 'inventory_supplier': "soure"}, axis='columns')
    return df.rename({'about': 'description', 'name': 'title', 'soure': 'inventory_supplier', 'tags': 'categories_list',
                      'location_point': 'geolocation'}, axis='columns')


def unavailable_to_empty_str(df: DataFrame, col_list: List[str]) -> None:
    """
    change 'unavailable' to empty string in the specified columns

    Args:
      df: raw DataFrame of attractions
      col_list: list of text columns

    Returns:
      None

    """

    for col in col_list:
        df[col] = df[col].apply(lambda x: np.nan if x == "unavailable" else x)
        df[col] = df[col].fillna("")


def tags_format(df):
    """
  creating a new column of 'prediction' which is the tag value as a list of tags
  (for data not from google that skip the 'tagging' process
  """
    df["prediction"] = df["categories_list"].apply(
        lambda x: list(set([j.strip().title() for j in re.sub('[\[\'"{}\]]', '', x).strip().split(",")])) if type(
            x) != list else x)


def data_preprocessing(file_path):
    # load the data
    raw_df = pd.read_csv(file_path, encoding='UTF-8')
    raw_df = update_columns_names(raw_df)
    raw_df.drop_duplicates(subset=['title', 'description', 'inventory_supplier'],
                           inplace=True)  # I need to delete this line!
    df = raw_df.reset_index(drop=True)
    unavailable_to_empty_str(df, ["title", "description"])
    df["text"] = df["title"] + '. ' + df["description"]

    if "prediction" not in df.columns:
        tags_format(df)

    return df


class Anchors:

    def __init__(self, anchors):  # (anchors= {uuid: idx_in_itinerary})   * idx_in_itinerary start at index 1 (not 0)
        self.anchors = dict(sorted(anchors.items(), key=lambda x: x[1]))
        self.keys_list = list(self.anchors.keys())
        self.position_list = list(self.anchors.values())
        self.num_anchors = len(anchors)

    def num_attrac_between(self, position1, position2):
        return abs(position2 - position1) - 1

    def num_attrac_before(self):
        return min(self.position_list) - 1

    def num_attrac_after(self, tot_num_attrac):
        return tot_num_attrac - max(self.position_list)


##### route with restaurants
class RouteBulider:

    def __init__(self, df, chosen_tags, df_similarity_norm, df_distances_norm, tot_num_attractions,
                 weight_dict, user_dates, availability_df, anchors=False):
        self.df = df
        self.problematic_uuids = self.find_nan_attractions_uuid()
        # self.tags_vec = tags_vec
        self.chosen_tags = chosen_tags
        self.popularity_vec = self.create_popularity_vec()
        self.df_similarity_norm = df_similarity_norm
        self.df_distances_norm = df_distances_norm
        self.tot_attractions_duration = tot_num_attractions
        self.weight_dict = weight_dict
        self.user_dates = user_dates
        self.availability_df = availability_df
        self.availability_user_dates_dict = self.availability_user_dates()
        self.availability_per_date = None
        self.anchors = anchors

        self.current_anchors = dict()
        self.chosen_idx = list()  # np.nan(self.tot_num_attractions)
        self.similarity_vec = 0
        self.distance_vec = None
        self.tags_vec = None
        self.duration_vec = self.create_duration_vec()
        self.duration_vec_norm = None
        self.paid_attrac_vec = None
        self.availability_vec = None
        if len(self.chosen_idx) > 0:
            self.last_idx = self.chosen_idx[-1]
        self.sum_duration = 0
        self.duration_paid_attrac = 0
        self.final_route_df = pd.DataFrame()

        self.df_similarity_norm.columns = self.df_similarity_norm.index
        self.all_days_attractions_idx = list()
        self.all_days_restaurants_idx = list()


    def find_nan_attractions_uuid(self) -> List[str]:
        """
        Extract the uuids without 'title' and 'description'

        Return:
            list of uuids
        """
        nan_title = self.df["uuid"][self.df["title"].isnull()].values
        nan_description = self.df["uuid"][self.df["description"].isnull()].values
        nan_index = list(set(nan_title) & set(nan_description))
        return nan_index


    def first_attraction(self):
        """
        return the first attraction according to 2*popularity and chosen tags
        """
        vec_result = self.tags_vec + 2 * self.popularity_vec * self.availability_vec
        #vec_result = vec_result.drop(index=self.all_days_attractions_idx)
        #self.chosen_idx = vec_result.sort_values().index[0]
        self.chosen_idx = [(self.tags_vec + 2 * self.popularity_vec * self.availability_vec).sort_values().index[0]]


    def vec_scaling(self, vec):
        chosen_median = 0.5
        difference = chosen_median // vec.median()
        if difference:
            return vec * difference
        else:
            return vec


    def attractions_score_vec(self, idx_to_drop):
        """
        return a vector with a score for every attraction according to the last chosen attraction/s
        """
        vectors_results = (
                                  (self.weight_dict["popular"] * self.vec_scaling(self.popularity_vec)) +
                                  (self.weight_dict["distance"] * self.vec_scaling(self.distance_vec)) +
                                  (self.weight_dict["similarity"] * self.vec_scaling(self.similarity_vec)) +
                                  (self.weight_dict["tags"] * self.vec_scaling(self.tags_vec) * 0.25)
                          ) * self.duration_vec_norm * self.paid_attrac_vec * self.availability_vec

        #print("vector results with zeros:", self.vec_scaling(self.popularity_vec).sort_values())
        print("popular vec:\n", self.vec_scaling(self.popularity_vec).sort_values()[:10], self.popularity_vec.shape)
        print("distance vec:\n", self.vec_scaling(self.distance_vec).sort_values()[:10], self.distance_vec.shape)
        print("similarity vec:\n", self.vec_scaling(self.similarity_vec).sort_values()[:10], self.similarity_vec.shape)
        print("tags vec:\n", self.vec_scaling(self.tags_vec).unique())




        # drop duplicates of the chosen uuids
        duplicates_idx = list()
        for idx in idx_to_drop:
            duplicates = list(self.df_similarity_norm.loc[idx][self.df_similarity_norm.loc[idx] > 0.7].index)
            if len(duplicates) > 1:
                duplicates.remove(idx)
                duplicates_idx += duplicates
        idx_to_drop += duplicates_idx

        # remove problematic attractions #################################################################
        relevant_uuids = list(set(vectors_results.index) - set(self.problematic_uuids))
        vectors_results = vectors_results.loc[relevant_uuids]
        # drop chosen uuids and their duplicates
        vectors_results.drop(index=idx_to_drop, inplace=True)
        #vectors_results.drop(index=self.all_days_attractions_idx, inplace=True)
        vectors_results = vectors_results[vectors_results.values != 0]

        return vectors_results

    def next_best(self, score_vector):
        """
        input: Vector with the score of each attraction according to the last chosen attraction
        output: the next best attraction/s
        """
        print("vector results\n",score_vector.sort_values()[:10])
        if score_vector.sort_values().index[0] in self.all_days_attractions_idx:
            print("chosen attraction!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return score_vector.sort_values().index[0]

    def df_tags(self):
        """
        Creating DataFrame for all tags (each tag will have a different column)
        """

        tags_dict = {tag: [] for tag in self.chosen_tags}
        for tag_name in tags_dict.keys():
            for tags in self.df["prediction"]:
                if tag_name in tags:
                    tags_dict[tag_name].append(1)
                else:
                    tags_dict[tag_name].append(0)
        tags_df = pd.DataFrame(tags_dict)
        tags_df.index = self.df["uuid"]
        return tags_df

    def create_popularity_vec(self):
        df = self.df.set_index("uuid")
        df["popularity_norm"] = None
        for supplier in df["inventory_supplier"].unique():
            uuids = df[df["inventory_supplier"] == supplier].index
            max_reviews = df["number_of_reviews"].loc[uuids].sort_values(ascending=False).values[0]
            if max_reviews == 0:
                max_reviews = 1

            df["popularity_norm"].loc[uuids] = df["number_of_reviews"].loc[uuids].apply(lambda x: x / max_reviews)
        df["popularity_norm"] = df["popularity_norm"].fillna(0)

        attrac_with_reviews = df["popularity_norm"][df["popularity_norm"] != 0].sort_values()
        attrac_without_reviews = df["popularity_norm"][df["popularity_norm"] == 0]
        num_attrac_with_reviews = len(attrac_with_reviews)
        norm_values = np.arange(0, 1 + (1/(num_attrac_with_reviews - 1)), 1/(num_attrac_with_reviews - 1))
        df["popularity_norm"].loc[attrac_with_reviews.index] = norm_values
        df["popularity_norm"].loc[attrac_without_reviews.index] = 0.8
        df["popularity_norm"] = 1 - df["popularity_norm"]
        return df["popularity_norm"]

    def update_tags_vec(self, chosen_idx):

        # creating a dataframe for the chosen tags
        tags_df = self.df_tags()

        # reduce the values of the tags that already been chosen
        if len(chosen_idx) > 0:

            # checking how many times each 'chosen tag' has been chosen
            selected_tags_count = tags_df.loc[chosen_idx].sum()

            # Consider only those above 0
            selected_tags_count = selected_tags_count[selected_tags_count > 0]

            for i in range(selected_tags_count.shape[0]):
                tag_name = selected_tags_count.index[i]
                tag_value = selected_tags_count.values[i]
                tags_df[tag_name] = tags_df[tag_name] * (1 / (2 * tag_value))

        # sum the tags to a vector
        tags_sum = tags_df.sum(axis=1)

        # normalized the vector to be between 0-1. transforming the results with '1-norm' in order for the best attraction in terms of tags to be the lowest. Adding 0.01 in order not to reset the results
        self.tags_vec = 1 - self.norm_df(tags_sum) + 0.01
        self.tags_vec.index = self.df["uuid"]

    def norm_df(self, df):
        return (df - df.min()) / (df.max() - df.min())

    def current_similarity_vec(self, chosen_idx):
        """
        Return the similarity vector associated with the last attraction along with the other selected attractions
        """
        current_similarity_vec = self.df_similarity_norm.loc[chosen_idx[-1]]
        current_similarity_vec.index = self.df["uuid"]
        self.similarity_vec = current_similarity_vec + (1 / 3) * self.similarity_vec

    def update_distance_vec(self, chosen_idx):
        self.distance_vec = self.df_distances_norm.loc[chosen_idx[-1]]
        self.distance_vec.index = self.df["uuid"]

    def create_duration_vec(self):
        def str_to_hours(hour_string):
            try:
                t = datetime.strptime(hour_string, "%H:%M:%S")
                delta = timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
                hour = delta.seconds / 3600
                # if the duration is weird, probably it's transportation
                remainder = str(hour)
                if len(remainder) > 4:
                    return 0
                else:
                    return hour
            except:
                return float(hour_string[:2])

        self.df["hour"] = self.df["duration"].apply(lambda x: str_to_hours(x))
        duration_vec = self.df[["uuid", "hour"]].set_index("uuid", drop=True)
        return duration_vec

    def create_availability_vec(self, hour: int) -> Series:
        return self.availability_per_date[hour]

    def create_duration_vec_norm(self, chosen_idx):
        time_left = (self.tot_attractions_duration - self.sum_duration) + 0.5
        return self.duration_vec["hour"].apply(lambda x: 1 if x <= time_left and x != 0 else 0)

    def create_paid_attractions_vec(self):
        """
        we want paid attractions to be till half of the daily route duration
        return: Series, vector for paid attractions duration. 0: if not relevant, 1:if free(google), 0.5: if paid and relevant
        """
        df = self.df.set_index("uuid", drop=True)
        # we want paid attractions to be till half of the daily route duration
        time_left_for_paid_attractions = self.tot_attractions_duration / 2 - self.duration_paid_attrac
        df["if_google"] = df["inventory_supplier"].apply(lambda x: 1 if x == 'GoogleMaps' else 0.2)
        df["paid_duration"] = df["hour"].apply(lambda x: 1 if x <= time_left_for_paid_attractions else 0)
        df["if_paid_results"] = df["if_google"] + df["paid_duration"]
        df["paid_attraction_vec_final"] = df["if_paid_results"].apply(lambda x: 1 if ((round(x) != 0) & (x % 1 == 0)) else (0.8 if round(x) > 0 else 0))
        return df["paid_attraction_vec_final"]

    def update_paid_attrac_duration(self, chosen_uuid):
        df = self.df.copy()
        df.set_index("uuid", drop=True, inplace=True)
        if df.loc[chosen_uuid]["inventory_supplier"].values != 'GoogleMaps':
            print(df[["title", "hour"]].loc[chosen_uuid])
            self.duration_paid_attrac += df.loc[chosen_uuid]["hour"].values[0]
            print("paid attractions durations:", self.duration_paid_attrac)

    def update_vectors(self, chosen_uuids, hour):
        """
        update the vectors according to chosen_idx (according to the attractions that were chosen)
        """
        hour = round(hour)
        # update the tags_vector
        self.update_tags_vec(chosen_uuids)
        # find similarity_vec according to current attraction
        self.current_similarity_vec(chosen_uuids)
        # similarity_vec = current_similarity_vec(chosen_idx, df_similarity_norm)
        self.similarity_vec = self.norm_df(self.similarity_vec)
        # extract distance vector
        self.update_distance_vec(chosen_uuids)

        # duration vector norm
        self.duration_vec_norm = self.create_duration_vec_norm(chosen_uuids)
        self.paid_attrac_vec = self.create_paid_attractions_vec()
        self.availability_vec = self.availability_per_date[hour]
        print("Successfully updated vectors!")

    def update_sum_duration(self, chosen_uuid) -> None:
        print("*******sum duration before adding new attraction******:", self.sum_duration)
        df = self.df.copy()
        df.set_index("uuid", drop=True, inplace=True)
        print("added time:", df.loc[chosen_uuid]["hour"].values[0])
        self.sum_duration += df.loc[chosen_uuid]["hour"].values[0]
        print("*******sum duration after adding new attraction******:", self.sum_duration)

    def availability_user_dates(self) -> Dict[timedelta, DataFrame]:
        """
        Create a dictionary with the user dates as keys and availability dataframe for each date as a value

        Return:
            a dictionary with the user dates as keys and availability dataframe for each date as a value
        """
        def convert_time(time_string):
            date_var = time.strptime(time_string, "%H:%M:%S")
            return date_var.tm_hour

        def convert_date(date_str):
            return datetime.strptime(date_str, "%Y-%m-%d")

        # create a new column with the hour as an int
        self.availability_df["hour"] = self.availability_df["time"].apply(lambda x: convert_time(x))
        # create a new column of formatted date
        self.availability_df["formatted_date"] = self.availability_df["date"].apply(lambda x: convert_date(x))

        # from_date = self.user_dates[0]
        # to_date = self.user_dates[1]
        #
        # user_dates = [from_date + timedelta(days=i) for i in range((to_date - from_date).days + 1)]

        # list of all the id attractions
        list_attractions_id = self.availability_df["attraction_id"].unique()

        # empty dict of the user dates and their availability matrices
        availability_dict = {date: None for date in self.user_dates}
        hours = sorted(self.availability_df["hour"].unique())

        for date in self.user_dates:

            # define a matrix with a column of the attractions id
            availability_per_date = pd.DataFrame(list_attractions_id, columns=["attraction_id"])

            for hour in hours:
                # extract the selected date and hour from the original availability dataframe
                hour_availability = self.availability_df[["date", "attraction_id", "hour"]][
                    (self.availability_df['formatted_date'] == date) & (self.availability_df["hour"] == hour)].drop_duplicates()
                hour_availability[hour] = 1
                # merge the specific hour to availability_per_date
                availability_per_date = pd.merge(availability_per_date, hour_availability, how='left')
                availability_per_date.fillna(0, inplace=True)
                availability_per_date.drop(columns=["date", "hour"], inplace=True)
                #availability_per_date.rename(columns={0: 24}, inplace=True)

            # insert the matrix to the dictionary as a value for its date
            availability_dict[date] = availability_per_date
        return availability_dict

    def availability_df_per_date(self, date: str) -> DataFrame:
        specific_date_availability = self.availability_user_dates_dict[date]
        specific_date_availability = specific_date_availability.merge(self.df["uuid"], how="outer",
                                                                      left_on="attraction_id", right_on='uuid')
        specific_date_availability.drop(columns="attraction_id", inplace=True)
        specific_date_availability.fillna(1, inplace=True)
        specific_date_availability.set_index("uuid", drop=True, inplace=True)
        return specific_date_availability



    def select_attractions_idx(self, num_attractions, chosen_idx,
                               uuids_to_drop):  # idx_to_drop was added mainly for the anchors

        # for i in range(num_attractions):
        time_left = self.tot_attractions_duration - self.sum_duration
        while time_left >= 1:
            hour = round(self.final_route_df["end"].values[-1] + 0.5)
            self.update_vectors(chosen_idx, hour)

            vector_results = self.attractions_score_vec(uuids_to_drop)
            attraction_uuid = self.next_best(vector_results)
            # append the next index to the indices list
            chosen_idx.append(attraction_uuid)
            print("chosen_idx:", chosen_idx)
            uuids_to_drop.append(attraction_uuid)
            self.update_sum_duration([attraction_uuid])
            self.update_paid_attrac_duration([attraction_uuid])
            time_left = self.tot_attractions_duration - self.sum_duration
            self.final_route_df = self.uuid_to_df([attraction_uuid], self.final_route_df)
        return chosen_idx


    def route_without_anchors(self):
        """
        return idx list of the chosen attractions
        """
        # for now, define a daily route
        date = self.user_dates[0]
        self.availability_per_date = self.availability_df_per_date(date)
        self.update_tags_vec([])
        self.availability_vec = self.availability_per_date[9]
        self.first_attraction()
        self.final_route_df = self.uuid_to_df(self.chosen_idx, self.final_route_df)
        # drop the already selected attraction from the list of attractions (self.chosen_idx)
        idx_to_drop = self.chosen_idx.copy()
        self.update_sum_duration(self.chosen_idx)
        self.update_paid_attrac_duration(self.chosen_idx)
        return self.select_attractions_idx(self.sum_duration, self.chosen_idx, idx_to_drop)



    def full_route_without_anchors(self, rest_instance=False):
        route_dict = dict()
        for i in range(len(self.user_dates)):
            self.availability_per_date = self.availability_df_per_date(self.user_dates[i])
            if i != 0:
                # reset all the vectors
                self.final_route_df = pd.DataFrame()
                self.chosen_idx = list()
                self.duration_vec = self.create_duration_vec()
                self.duration_vec_norm = None
                self.paid_attrac_vec = None
                self.availability_vec = None
                if len(self.chosen_idx) > 0:
                    self.last_idx = self.chosen_idx[-1]
                self.sum_duration = 0
                self.duration_paid_attrac = 0

            if type(rest_instance) != bool:
                daily_route = self.route_with_restaurants(rest_instance)
                route_dict[i+1] = daily_route
                self.all_days_restaurants_idx += list(set(self.final_route_df["uuid"].values) & set(rest_instance.df_rest.index))
                self.all_days_attractions_idx += list(set(self.final_route_df["uuid"].values) & set(self.df["uuid"].values))
            else:
                daily_route = self.route_without_anchors()
                route_dict[self.user_dates[i]] = daily_route
                self.all_days_attractions_idx += list(set(self.final_route_df["uuid"].values) & set(self.df["uuid"].values))

        return route_dict



    def attraction_between_anchors(self, vector1, vector2):
        """
        input: 2 vectors with the attraction score according to each anchor
        output: the best attraction idx between the anchors
        """
        print("vector_score:", (vector1 + vector2))
        return (vector1 + vector2).sort_values().index[0]

    def select_middle_attrac(self, anchors, idx_to_drop=[]):
        """
         #input anchors= {position: idx, position: idx}): {0:[10], 2:[40,30]}, the anchore always should be the last item in the list
         input anchors= {position: idx, position: idx}): {0:[uuid], 2:[uuid,uuid]}, the anchore always should be the last item in the list
         output: attraction idx between the anchors: (between 10 and 30)
        """
        ## select the attraction in the middle
        # reset the chosen_idx and update the vectors according to the first anchore and its chosen attractions
        idx_to_drop += anchors[0] + anchors[2]
        chosen_idx = anchors[0]
        self.similarity_vec = 0
        self.update_vectors(anchors[0])
        new_anchore1_vec = self.attractions_score_vec(idx_to_drop)

        # reset the chosen_idx and update the vectors according to the second anchore and its chosen attractions
        chosen_idx = anchors[2]
        self.similarity_vec = 0
        self.update_vectors(anchors[2])
        new_anchore2_vec = self.attractions_score_vec(idx_to_drop)

        # find the middle attraction according to both anchores
        middle_attraction_idx = self.attraction_between_anchors(new_anchore1_vec, new_anchore2_vec)
        # print("middle attraction", middle_attraction_idx)

        return middle_attraction_idx

    def even_attractions_between_anchors(self, idx_vec):  # np.array([0,10,20,30,40,50])
        """
        find the best 2 middle attractions according to both anchors (find the best "20, 30")
        """
        for i in range(5):
            print(f"itteration {i}")
            middle1_position = len(idx_vec) // 2 - 1  # 2
            middle2_position = len(idx_vec) // 2  # 3
            middle1 = idx_vec[middle1_position]  # 20
            middle2 = idx_vec[middle2_position]  # 30

            # find middle 1
            print("middle1:", middle1)
            middle1_new = self.select_middle_attrac(
                {0: idx_vec[:middle1_position], 2: idx_vec[middle2_position::][::-1]})
            print("middle1_new:", middle1_new)
            # update the new middle 1 in the list
            idx_vec[middle1_position] = middle1_new

            # find middle 2
            print("middle2:", middle2)
            middle2_new = self.select_middle_attrac(
                {0: idx_vec[:middle2_position], 2: idx_vec[middle2_position + 1::][::-1]})
            print("middle2:", middle2_new)
            # update the new middle 2 in the list
            idx_vec[middle2_position] = middle2_new

            if middle1 == middle1_new and middle2 == middle2_new:
                print("found the best middle attractions!")
                return idx_vec
        return idx_vec

    def select_attractions_between_anchors(self):
        """
        input: a dictionary with 2 items (anchors)
        output: a list with the anchors and the selected attractions between them
        """

        # for testing only!!
        # anchors = {self.anchors.keys_list[0]: self.anchors.position_list[0] , self.anchors.keys_list[0+1]: self.anchors.position_list[0+1] }
        # self.current_anchors = Anchors(anchors)
        # print(self.current_anchors.anchors)

        # number of attractions to select according to each anchore
        num_attractions_per_anchore = self.current_anchors.num_attrac_between(self.current_anchors.position_list[0],
                                                                              self.current_anchors.position_list[
                                                                                  1]) // 2
        print("num_attractions_per_anchore:", num_attractions_per_anchore)

        chosen_idx = list()
        # create a list with the anchors and their chosen attractions idx
        idx_to_drop = self.chosen_idx.copy() + self.anchors.keys_list

        # first anchore:
        # insert the first anchore to chosen_idx
        print(self.current_anchors.keys_list[0])
        chosen_idx.append(self.current_anchors.keys_list[0])
        print(chosen_idx)

        # selecting attraction/s according to the first anchore
        chosen_idx = self.select_attractions_idx(num_attractions_per_anchore, chosen_idx, idx_to_drop)
        first_anchore_idx = chosen_idx.copy()
        print("chosen_idx:", chosen_idx)  # [10, 20]
        print("idx to drop:", idx_to_drop)

        # add the second anchore to the list
        idx_to_drop.append(self.current_anchors.keys_list[1])  # [10, 20, 50]

        # add chosen_idx the second anchore
        chosen_idx.append(self.current_anchors.keys_list[1])

        # reset self.similarity_vec (that we will not have the history value of the first anchore)
        # self.similarity_vec = 0

        # adding attractions according to the second anchore, drop the attractions that were already chosen by the first anchore
        attractions_anchore2 = self.select_attractions_idx(num_attractions_per_anchore, chosen_idx,
                                                           idx_to_drop)  # [10, 20, 50, 40]
        # chosen_idx += attractions_anchore2
        second_anchore_idx = chosen_idx[num_attractions_per_anchore + 1:]  # [50, 40]
        print("second_anchore_idx:", second_anchore_idx)
        print("chosen_idx:", chosen_idx)  # [10, 20, 50, 40]

        ### select the attraction in the middle
        # middle_attraction_idx = self.select_middle_attrac({0:first_anchore_idx, 2:second_anchore_idx})

        # join the indices to one list and reverse the second list so that the anchore will be the last item
        anchors_idx = first_anchore_idx + second_anchore_idx[::-1]  # [10, 20, 40, 50]
        print("anchors idx", anchors_idx)

        # no attractions between anchors
        num_attractions_between = self.current_anchors.num_attrac_between(self.current_anchors.position_list[0],
                                                                          self.current_anchors.position_list[1])
        if num_attractions_between == 0:
            chosen_idx = anchors_idx

        # even number of attractions between anchors
        elif num_attractions_between % 2 == 0:
            chosen_idx = self.even_attractions_between_anchors(anchors_idx)

        # odd number of attractions between anchors
        else:
            # idx_to_drop = chosen_idx.copy() + self.anchors.keys_list
            print("idx_to_drop:", idx_to_drop)
            ### select the attraction in the middle
            middle_attraction_idx = self.select_middle_attrac({0: first_anchore_idx, 2: second_anchore_idx},
                                                              idx_to_drop)

            # add the middle attraction to the idx list
            anchors_idx.insert(num_attractions_per_anchore + 1, middle_attraction_idx)
            chosen_idx = anchors_idx

        print("chosen attractions by both anchors", chosen_idx)
        return chosen_idx

    def attractions_before_anchore(self, chosen_idx):
        """
        select attractions idx before anchore
        """
        num_attrac_to_select = self.anchors.num_attrac_before()
        # Drop off the attractions that have already been selected
        idx_to_drop = chosen_idx
        # reverse chosen_idx in order to choose the "before attractions"
        chosen_idx = chosen_idx[::-1]
        self.select_attractions_idx(num_attrac_to_select, chosen_idx, idx_to_drop)
        # reverse again chosen_idx
        chosen_idx = chosen_idx[::-1]
        return chosen_idx

    def attractions_after_anchors(self, chosen_idx):
        """
        select attractions idx after anchore
        """
        num_attrac_to_select = self.anchors.num_attrac_after(self.tot_attractions_duration)
        # Drop off the attractions that have already been selected
        idx_to_drop = chosen_idx.copy()
        self.select_attractions_idx(num_attrac_to_select, chosen_idx, idx_to_drop)
        return chosen_idx

    def route_with_anchors(self):
        """
        select attractions idx for route with anchors
        """
        # Verify that the anchors are within range of the number of attractions for the route
        for anchore, position in self.anchors.anchors.items():
            assert position <= self.tot_attractions_duration, "The location of the anchor exceeds the amount of attractions"
            assert position != 0, "The location of the anchor exceeds the amount of attractions"

        if self.anchors.num_anchors > 1:
            print("anchore!")
            self.chosen_idx = [self.anchors.keys_list[0]]

            for i in range(len(self.anchors.keys_list) - 1):
                # if i != self.anchors.keys_list[-1]:
                anchors = {self.anchors.keys_list[i]: self.anchors.position_list[i],
                           self.anchors.keys_list[i + 1]: self.anchors.position_list[i + 1]}
                print("current anchore:", anchors)
                self.current_anchors = Anchors(anchors)
                self.chosen_idx += self.select_attractions_between_anchors()[1:]

            # add attractions before anchors
            self.chosen_idx = self.attractions_before_anchore(self.chosen_idx)
        else:
            self.chosen_idx = [self.anchors.keys_list[0]]
            self.chosen_idx = self.attractions_before_anchore(self.chosen_idx)

        # add attractions after anchors
        self.attractions_after_anchors(self.chosen_idx)
        print("final idx:", self.chosen_idx)


    def idx_to_dataframe(self):
        """
        receive a list of the chosen idx and retrieve the attractions in a dataframe
        """
        df_uuid = self.df.copy()
        df_uuid.set_index("uuid", drop=True)
        selected_attractions = df_uuid.loc[self.chosen_idx].drop(columns=["index", "Unnamed: 0"])
        return selected_attractions

    def uuid_to_df(self, uuid, selected_attractions_df, start_hour=False):
        """
        extract the row of the chosen uuid from attractions df
        :param uuid:
        :return:
        """
        df_uuid = self.df.copy()
        df_uuid.set_index("uuid", drop=True, inplace=True)
        selected_attraction = df_uuid[["title", "hour", "inventory_supplier"]].loc[uuid]
        selected_attraction["start"] = None
        selected_attraction["end"] = None
        selected_attraction.reset_index(inplace=True)
        selected_attractions_df = pd.concat([selected_attractions_df, selected_attraction], ignore_index=True)
        if start_hour:
            selected_attractions_df["start"].iloc[-1] = start_hour
            selected_attractions_df["end"].iloc[-1] = start_hour + selected_attractions_df.iloc[-1]["hour"]

        else:
            if len(selected_attractions_df) == 1:
                selected_attractions_df["start"] = 9.5
                selected_attractions_df["end"] = 9.5 + selected_attractions_df.iloc[-1]["hour"]
            else:
                selected_attractions_df["start"].iloc[-1] = selected_attractions_df.iloc[-2]["end"] + 0.5
                selected_attractions_df["end"].iloc[-1] = selected_attractions_df["start"].iloc[-1] + \
                                                          selected_attractions_df["hour"].iloc[-1]

                # if instead of attraction we need to have lunch
                if 13 < selected_attractions_df["end"].iloc[-1] < 16:
                    restaurant_start = selected_attractions_df["end"].iloc[-1]
                    selected_attractions_df = selected_attractions_df.append(
                        [{"uuid": 0, "title": 'Lunch time', "hour": 1, "start": restaurant_start, "end": restaurant_start + 1}],
                        ignore_index=True)

        return selected_attractions_df

    def build_route(self):
        """
        return a dataframe of the selected route
        """
        if self.anchors:
            self.route_with_anchors()
            #self.test_route()
            print("final attractions:", self.chosen_idx)
            return self.final_route_df[self.final_route_df["uuid"] != 0]
        else:
            self.route_without_anchors()
            #self.test_route()
            print("final attractions:", self.chosen_idx)
            return self.final_route_df[self.final_route_df["uuid"] != 0]












    def route_with_restaurants(self, rest_instance):

        # create a dataframe of the selected attractions
        selected_attractions = self.build_route()

        # create a copy of the selected attractions indices
        route_uuid_list = self.chosen_idx.copy()
        rest_uuid_list = list()  # chosen uuids to drop

        rest_kind_dict = {'breakfast': 0, 'lunch': 1, 'dinner': 2}

        ## choose the first restaurant at the begining of the route
        rest_uuid = rest_instance.best_rest_uuid(
            rest_instance.rest_vec(route_uuid_list[0], rest_kind_dict['breakfast'], rest_uuid_list))
        rest_uuid_list.append(rest_uuid)
        # add the chosen restaurant to the route dataframe
        rest_title = rest_instance.df_rest.loc[rest_uuid]["title"]
        self.final_route_df = pd.DataFrame(
            np.insert(self.final_route_df.values, 0, values=[rest_uuid, rest_title, 1, np.nan, 8, 9], axis=0))
        self.final_route_df.columns = ["uuid", "title", "hour", "inventory_supplier", "start", "end"]

        # choose lunch
        lunch_idx = self.final_route_df[self.final_route_df["uuid"] == 0].index[0]
        attrac_before = self.final_route_df.iloc[lunch_idx - 1]["uuid"]
        attrac_after = self.final_route_df.iloc[lunch_idx + 1]["uuid"]
        rest_uuid = rest_instance.best_rest_uuid(
            rest_instance.rest_between_attractions(attrac_before, attrac_after,
                                                   rest_uuid_list, rest_kind_dict['lunch']))
        rest_uuid_list.append(rest_uuid)
        # add the chosen restaurant to the route dataframe
        rest_title = rest_instance.df_rest.loc[rest_uuid]["title"]
        self.final_route_df["title"].iloc[lunch_idx] = rest_title
        self.final_route_df["uuid"].iloc[lunch_idx] = rest_uuid

        ## choose dinner
        rest_uuid = rest_instance.best_rest_uuid(
            rest_instance.rest_vec(route_uuid_list[-1], rest_kind_dict['dinner'], rest_uuid_list))
        rest_uuid_list.append(rest_uuid)

        # add the chosen restaurant to the route dataframe
        rest_title = rest_instance.df_rest.loc[rest_uuid]["title"]
        dinner_start = self.final_route_df["end"].iloc[-1] + 0.5
        self.final_route_df = self.final_route_df.append(
            {"title": rest_title, "uuid": rest_uuid, "hour": 1, "start": dinner_start, "end": dinner_start + 1.5},
            ignore_index=True)

        return self.final_route_df

    def create_map_file(self, selected_attractions, api_key, file_name, rest_df):

        #extract 'geolocation' from attractions df and from restaurant df

        rest_df.reset_index(inplace=True)
        if "index" in rest_df.columns:
            rest_df.rename(columns={"index": "uuid"}, inplace=True)
        selected_attractions_rest_geo = selected_attractions.merge(rest_df[["uuid", "geolocation"]], how="inner")
        selected_attractions = selected_attractions.merge(self.df[["uuid", "geolocation"]], how="left")
        rest_uuids = selected_attractions_rest_geo["uuid"].values
        selected_attractions.set_index("uuid", inplace=True)
        selected_attractions["geolocation"].loc[rest_uuids] = selected_attractions_rest_geo["geolocation"].values

        #create 'lon' and 'lat' columns
        selected_attractions = selected_attractions[selected_attractions.index != 0]
        selected_attractions["lon"] = selected_attractions["geolocation"].apply(
            lambda x: [float(s) for s in re.findall(r'-?\d+\.?\d*', x)][-2:][0])
        selected_attractions["lat"] = selected_attractions["geolocation"].apply(
            lambda x: [float(s) for s in re.findall(r'-?\d+\.?\d*', x)][-2:][1])

        # optional colors
        #colors = ["b", "g", "r", "y", "m", "w", "c"]
        colors = ["y" for i in range(selected_attractions.shape[0])]

        latitude_list = selected_attractions["lat"]
        longitude_list = selected_attractions["lon"]

        gmap = gmplot.GoogleMapPlotter(latitude_list.mean(), longitude_list.mean(), 11)
        gmap.scatter(latitude_list, longitude_list, color=random.sample(colors, selected_attractions.shape[0]),
                     s=60,
                     ew=2,
                     marker=True,
                     symbol='+',
                     label=[l for l in string.ascii_uppercase[:selected_attractions.shape[0]]])

        # polygon method Draw a polygon with
        # the help of coordinates
        gmap.polygon(latitude_list, longitude_list, color='cornflowerblue')
        gmap.apikey = api_key
        gmap.draw(file_name)

    def test_route(self):
        # test if we have duplications
        assert len(self.chosen_idx) == len(set(self.chosen_idx)), "found duplicates!"

        # test the len of the route
        # assert len(
        #     self.chosen_idx) == self.tot_attractions_duration, "The length of the route is not equal to the one defined!"

        # test the position of each anchore in the selected route
        if self.anchors:
            for anchore in self.anchors.keys_list:
                assert self.anchors.anchors[anchore] - 1 == self.chosen_idx.index(
                    anchore), "Anchors not in the right position!"

        print("The tests passed successfully!")


def main():
    # New york attractions
    # df = data_preprocessing("new_york_attractions.csv")
    df = pd.read_csv("new_york_attractions.csv")
    availability_df = pd.read_csv("availability_newyork.csv")

    df_distances_norm = pd.read_csv("ny_distance_norm.csv")
    df_distances_norm.set_index("uuid", drop=True, inplace=True)

    df_similarity_norm = pd.read_csv("ny_similarity_norm.csv")
    df_similarity_norm.set_index("uuid", drop=True, inplace=True)


    # New york restaurants

    RESTAURANTS_TAGS_LIST = rest.RESTAURANTS_TAGS_LIST
    rest_tags_weights = pd.read_csv(
        r"C:\Users\user\PycharmProjects\bridgify\route_builder\new_itinerary\restaurants_data\tags_weights.csv")

    rest_df = pd.read_csv(
        r"C:\Users\user\PycharmProjects\bridgify\route_builder\new_itinerary\restaurants_data\ny_rest_tagged.csv")
    rest_df.set_index("uuid", drop=True, inplace=True)

    rest_distances_norm = pd.read_csv(
        r"C:\Users\user\PycharmProjects\bridgify\route_builder\new_itinerary\restaurants_data\ny_attrac_rest_distances_matrix.csv")

    rest_distances_norm.set_index("uuid", drop=True, inplace=True)
    rest_instance = rest.Restaurants(rest_df, rest_distances_norm, rest_tags_weights, RESTAURANTS_TAGS_LIST, [])

    chosen_tags = ["Architecture", "Culinary Experiences", "Shopping", "Art", "Urban Parks", "Museums"]
    #chosen_tags = ["Museums", "Urban Parks", "Shows/Performance"]

    ATTRACTIONS_DURATION = 7
    # anchors = Anchors({"565e696f-4a4c-414b-afb4-70c97f094214": 3,
    #                    "96fe7ece-ef7d-4c1f-b7fa-ef6b1d5b12a0": 2,
    #                    "16149ccd-7b45-453f-981b-114cfb24f2de": 7})  # (idx: location in the route. starts at 1, not 0)

    # select formula weights
    weight_dict = {"popular": 2, "distance": 4, "similarity": 1, "tags": 1}

    # user d
    from_date = datetime.strptime("2022-10-14", "%Y-%m-%d")
    to_date = datetime.strptime("2022-10-18", "%Y-%m-%d")
    #user_dates = (datetime.strptime("2022-10-14", "%Y-%m-%d"), datetime.strptime("2022-10-18", "%Y-%m-%d"))
    user_dates = [from_date + timedelta(days=i) for i in range((to_date - from_date).days + 1)]
    # new_route = RouteBulider(df_reduced, chosen_tags, df_popularity_vec_or, df_similarity_norm_or, df_distances_norm_or, num_attractions, weight_dict, anchors)
    new_route = RouteBulider(df, chosen_tags, df_similarity_norm, df_distances_norm, ATTRACTIONS_DURATION,
                             weight_dict, user_dates, availability_df)

    #selected_route = new_route.build_route()
    selected_route = new_route.route_with_restaurants(rest_instance)
    #all_days_route = new_route.full_route_without_anchors(rest_instance)
    api_key = 'AIzaSyCoqQ2Vu2yD99PqVlB6A6_8CyKHKSyJDyM'
    new_route.create_map_file(new_route.final_route_df, api_key, "NY_route.html", rest_df)
    #return all_days_route
    return selected_route


if __name__ == "__main__":
    route = main()
    #print(route[["title", "uuid"]])
    print(route)

