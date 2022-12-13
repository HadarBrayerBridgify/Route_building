import pytest
import json
import pandas as pd
import multiple_days_route_anchors as new_ip


# setup
with open("config.json") as config_file:
    config = json.load(config_file)
route_object = new_ip.main(config)

# setup- distance
config_distance = config.copy()
config_distance["PARAMETERS_WEIGHTS"] = {"popular": 0, "distance": 1, "tags": 0, "price_level": 2}
route_object_distance = new_ip.main(config_distance)

# setup- tags
config_tags = config.copy()
config_tags["PARAMETERS_WEIGHTS"] = {"popular": 0, "distance": 0, "tags": 1, "price_level": 2}
route_object_tags = new_ip.main(config_tags)

# setup- popularity
config_popularity = config.copy()
config_popularity["PARAMETERS_WEIGHTS"] = {"popular": 1, "distance": 0, "tags": 0, "price_level": 2}
route_object_popularity = new_ip.main(config_popularity)

def test_popularity_vec_len():
    assert len(route_object.popularity_vec) == len(route_object.df)


def test_popularity_vec_range():
    assert route_object.popularity_vec.max() == 1
    assert route_object.popularity_vec.min() == 0


def test_popularity_vec_nan():
    assert route_object.popularity_vec.isna().sum() == 0


def test_popularity_vec_one_one_value():
    assert len(route_object.popularity_vec[route_object.popularity_vec == 0]) == 1


def test_popularity_vec_not_binary():
    assert len(route_object.popularity_vec.value_counts()) > 2


def test_popularity_vec_nan_number_reviews():
    nan_uuids = list(route_object.df["uuid"][route_object.df["number_of_reviews"].isna()].values)
    nan_uuids += list(route_object.df["uuid"][route_object.df["number_of_reviews"] == 0].values)
    popularity_one = route_object.popularity_vec[route_object.popularity_vec == 1].index
    assert len(set(nan_uuids) - set(popularity_one)) == 0


def test_popularity_vec_max_number_reviews():
    sort_reviews = route_object.df["number_of_reviews"].sort_values(ascending=False)
    if sort_reviews.values[0] > 50:
        max_uuid = route_object.df.loc[sort_reviews.index[0]]["uuid"]
        best_popularity_uuids = route_object.popularity_vec.sort_values()[:10].index
        assert max_uuid in best_popularity_uuids


def test_tags_vec_range():
    tags_weight = route_object_tags.weight_dict["tags"]
    if tags_weight:
        a = route_object_tags.tags_vec
        a_min = route_object_tags.tags_vec.min()
        assert route_object_tags.tags_vec.min() == 1 - 0.1 * tags_weight
    assert route_object_tags.tags_vec.max() == 1


def test_tags_vec_len():
    assert len(route_object_tags.tags_vec) == len(route_object_tags.df)


def test_tags_vec_unique_val_initial():
    assert route_object_tags.tags_vec.nunique() == 2


def test_tags_vec_unique_val_after_route():
    full_route = route_object_tags.create_full_route()
    assert route_object_tags.tags_vec.nunique() == 3


# distance vec

def test_df_distances_norm_range():
    if route_object.distance_outliers:
        assert route_object.df_distances_norm.max().max() == 1.1
    assert route_object.df_distances_norm.min().min() == 0


def test_df_distances_norm_outliers_values():
    for outlier in route_object.distance_outliers:
        assert round(route_object.df_distances_norm.loc[outlier].sum()) == round(1.1 * route_object.df_distances_norm.shape[1])
        assert round(route_object.df_distances_norm[outlier].sum()) == round(1.1 * route_object.df_distances_norm.shape[0])


def test_df_distances_norm_correlative_to_distance():
    i = 0
    uuid = route_object.df_distances.index[i]

    while True:
        if uuid in route_object.distance_outliers:
            i += 1
            uuid = route_object.df_distances.index[i]
        else:
            break
    assert route_object.df_distances_norm.loc[uuid, uuid] == 0
    highest_distance_uuid = route_object.df_distances[uuid].sort_values(ascending=False).index[0]
    highest_val = route_object.df_distances_norm.loc[uuid][route_object.df_distances_norm.loc[uuid] != 1.1].sort_values(ascending=False).values[0]
    assert route_object.df_distances_norm.loc[uuid, highest_distance_uuid] == 1.1 or \
           route_object.df_distances_norm.loc[uuid, highest_distance_uuid] == highest_val



def test_first_attraction_distance():
    """
     Validates if the selected attraction is the closest to the prime location
    """
    route_object_distance.availability_per_date = route_object_distance.availability_df_per_date(route_object_distance.user_dates[0])
    route_object_distance.availability_vec = route_object_distance.availability_per_date[9]
    availability_distance_df = pd.concat([route_object_distance.availability_vec, route_object_distance.first_attraction_distance_vec], axis=1)
    availability_distance_df.columns = ["availability", "distance"]
    availability_distance_df = availability_distance_df[availability_distance_df["availability"] == 1]
    selected_uuid = availability_distance_df.sort_values(by="distance").index[0]
    assert route_object_distance.route_without_anchors()[0] == selected_uuid


def test_first_attraction_tags():
    """
    Validates if the selected attraction has at least one of the user's tags
    """
    first_attraction_uuid = route_object_tags.route_without_anchors()[0]
    attraction_tags = route_object_tags.df[route_object_tags.df["uuid"] == first_attraction_uuid]["categories_list"]
    a = set(route_object_tags.chosen_tags)
    b = set(attraction_tags) & set(route_object_tags.chosen_tags)
    assert len(set(attraction_tags) & set(route_object_tags.chosen_tags)) > 0


def test_first_attraction_popularity():
    """
     Validates if the selected attraction is the most popular out of the available attractions
    """
    availability_pop_df = pd.concat([route_object_popularity.availability_vec, route_object_popularity.popularity_vec], axis=1)
    availability_pop_df.columns = ["availability", "popularity"]
    availability_pop_df = availability_pop_df[availability_pop_df["availability"] == 1]
    selected_uuid = availability_pop_df.sort_values(by="popularity").index[0]
    assert route_object_popularity.route_without_anchors()[0] == selected_uuid


