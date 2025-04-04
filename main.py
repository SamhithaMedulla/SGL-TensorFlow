import os
import sys
import random
import numpy as np
import tensorflow as tf
import importlib
from data.dataset import Dataset
from util import Configurator, tool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
    is_windows = sys.platform.startswith('win')
    if is_windows:
        root_folder = 'C:/Users/samhi/SGL-TensorFlow/'
    else:
        root_folder = '/home/wujc/PythonProjects/SGL/'
    # Load the configuration from conf/SGL.properties
    config_path = os.path.join(root_folder, "conf", "SGL.properties")
    conf = Configurator(config_path, default_section="hyperparameters")
    
    seed = conf["seed"]
    print('seed=', seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    gpu_id = str(conf["gpu_id"])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    recommender = conf["recommender"]

    dataset = Dataset(conf)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = conf["gpu_mem"]
    
    with tf.compat.v1.Session(config=config) as sess:
        if importlib.util.find_spec("model.general_recommender." + recommender) is not None:
            my_module = importlib.import_module("model.general_recommender." + recommender)
        elif importlib.util.find_spec("model.social_recommender." + recommender) is not None:
            my_module = importlib.import_module("model.social_recommender." + recommender)
        else:
            my_module = importlib.import_module("model.sequential_recommender." + recommender)
        
        MyClass = getattr(my_module, recommender)
        model = MyClass(sess, dataset, conf)

        model.build_graph()
        sess.run(tf.compat.v1.global_variables_initializer())
        model.train_model()
