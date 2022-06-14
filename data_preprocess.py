import os
import re
import sys
import logging
import numpy as np
import pandas as pd
import argparse
from configparser import ConfigParser


def update_columns_names(df):
    # return df.rename({"title": "name", "description": "about", "categories_list": "tags", 'inventory_supplier': "soure"}, axis='columns')
    return df.rename({'about': 'description', 'name': 'title', 'source': 'inventory_supplier', 'tags': 'categories_list', 'location_point': 'geolocation'}, axis='columns')


def parse_args(logger):
    """
    This function initialize the parser and the input parameters
    """

    my_parser = argparse.ArgumentParser(description=config_object['params']['description'])
    my_parser.add_argument('--path', '-p',
                           required=True,
                           type=str,
                           help="config_object['params']['path_help']")

    my_parser.add_argument('--save', '-s',
                           required=False,
                           type=str, default=None,
                           help=config_object['params']['save_help'])

    args = my_parser.parse_args()
    logger.info('Parsed arguments')
    return args


def val_input(args, logger):
    """
    This function validated that the input file exists and that the output path's folder exists
    """

    if not os.path.isfile(args.path):
        logger.debug('the input file doesn\'t exists')
        return False

    if args.save:
        if '/' in args.save:
            folder = "/".join(args.save.split('/')[:-1])
            if not os.path.exists(folder):
                logger.debug('the output folder doesn\'t exists')
                return False
        else:
            folder = "/".join(args.path.split('/')[:-1])
            args.save = folder + '/' + args.save

    else:
        current_time = datetime.datetime.now()
        save_path = f'processed_data_{current_time.year}-{current_time.month}-{current_time.day}-{current_time.hour}-' \
                    f'{current_time.minute}.xlsx'
        args.save = save_path
        logger.info('the save path was set to default')
    logger.info(f'args={args}')
    logger.info('input was validated')
    return True


def init_logger():
    """
    This function initialize the logger and returns its handle
    """

    log_formatter = logging.Formatter('%(levelname)s-%(asctime)s-FUNC:%(funcName)s-LINE:%(lineno)d-%(message)s')
    logger = logging.getLogger('log')
    logger.setLevel('DEBUG')
    file_handler = logging.FileHandler('pr_log.txt')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    return logger


def unavailable_to_nan(df, col_list, logger):
    """ change the 'unavailable' to nan, replace nan with empty string for joining the text columns"""

    for col in col_list:
        try:
            df[col] = df[col].apply(lambda x: np.nan if x == 'unavailable' else x)
            df[col] = df[col].fillna("")
        except KeyError as er:
            logger.debug(f'{col} column is missing from the DataFrame!')
            print(er)
            sys.exit(1)


def remove_duplicates_and_nan(df, logger):
    """ Remove rows which are exactly the same """

    logger.info(f"Shape before removing duplicates and Nans: {df.shape}")
    print("Shape before removing duplicates and Nans:", df.shape)
    try:
        # I exclude 'address' from 'drop_duplicates' because in many rows the address is inaccurate or missing so the
        # duplicates will be expressed especially according to 'title' and 'description'
        df.drop_duplicates(subset=['title', 'description'], inplace=True)
        df.dropna(subset=["text"], inplace=True)
        df.reset_index(inplace=True)

    except KeyError as er:
        logger.debug("One or more columns from the list ['title','description'] are missing from the "
                     "DataFrame!")
        print(er)
        sys.exit(1)

    logger.info(f"Shape after removing duplicates: {df.shape}")
    print("Shape after removing duplicates:", df.shape)
    return df


def tags_format(df):
    """
    creating a new column of 'prediction' which is the tag value as a list of tags
    (for data not from google that skip the 'tagging' proccesse
    """
    df["prediction"] = df["categories_list"].apply \
        (lambda x : list(set([j.strip().title() for j in re.sub('[()\[\'"{}\]]', '', x).strip().split(",")])) if type
            (x) != list else x)


def data_processing(file_path):
    logger = init_logger()
    logger.info('STARTED RUNNING')

    # load the data
    raw_df = pd.read_csv(file_path, encoding='UTF-8')
    # if 'about' in raw_df.columns:
    raw_df = update_columns_names(raw_df)

    # 'unavailable' to NAN
    unavailable_to_nan(raw_df, ["title", "description"], logger)

    # Remove rows which are exactly the same
    raw_df["text"] = raw_df["title"] + ' ' + raw_df["description"]
    df_reduced = remove_duplicates_and_nan(raw_df, logger)

    if "prediction" not in df_reduced.columns:
        tags_format(df_reduced)

    return df_reduced

