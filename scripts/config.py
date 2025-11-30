import os.path as op
import pandas as pd
import numpy as np
import time
from functools import wraps

def calculate_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Function {func.__name__} took {runtime:.6f} seconds to run.")
        return result
    return wrapper

this_file_abs_path = op.abspath(__file__)
active_device = 'nut_cloud_pc' # modify manually
is_in_server = active_device == 'sustech_stat_ds_public_server'
# cmd_running = False # set True only when is_in_server = False
# filedir = '../files' if cmd_running else './files'

class Path(object):
     # suitable for running at mutiple sync PC
    active_device_data_dir = {'sustech_stat_ds_public_server': '../../../data/files/', 
                              'nut_cloud_pc': op.dirname(this_file_abs_path), 
                              'other_pc': 'modify_this_dir'}
    base_data_dir = active_device_data_dir[active_device]
    embedding_model_dir = op.join(base_data_dir, 'Embeddings')

class DatabaseParams:
    # the following default parameters are randomly set for example
    def __init__(self, server_ip: str = '195.20.15.167', server_port: int = 5508, database_name: str = 'databasename', 
                 username: str = 'username', password: str = 'password', charset:str = 'utf8mb4'):
        self.server_ip = server_ip
        self.server_port = server_port
        self.database_name = database_name
        self.username = username
        self.password = password
        self.charset = charset

class ModelParams():
    embedding_dim = 24

class OpenAlex(Path):
    def __init__(self):
        super().__init__()
        self.concept_lists = self.get_concept_lists()

    @classmethod
    def get_concept_lists(cls):
        concept_lists_path = op.join(cls.train_data_dir, 'concept_names_train.npy') # level 0 to level 5
        print(time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()))
        ts = time.time()
        concept_lists = np.load(concept_lists_path, allow_pickle=True)
        te = time.time()
        print(time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()), te-ts)
        return concept_lists

if __name__=='__main__':
    path = Path()
    embedding_model_path = OpenAlex.embedding_model_dir
    print('Testing')


