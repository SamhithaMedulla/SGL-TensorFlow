import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import os
import sys
import random
import argparse
import numpy as np
import importlib
from data.dataset import Dataset
from util import Configurator, tool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --------- Add CLI Argument Parser ---------
parser = argparse.ArgumentParser()
parser.add_argument('--fast_debug', action='store_true', help='Run in fast debug mode with reduced settings')
args = parser.parse_args()
# -------------------------------------------

if __name__ == "__main__":
    is_windows = sys.platform.startswith('win')
    if is_windows:
        root_folder = 'C:/Users/samhi/Documents/SGL-TensorFlow/'
    else:
        root_folder = '/home/wujc/PythonProjects/SGL/'
        
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--fast_debug")]
    conf = Configurator(root_folder + "NeuRec.properties", default_section="hyperparameters")


    # --------- Apply fast_debug config overrides ---------
    if args.fast_debug:
        print("⚡ Fast debug mode activated!")
        conf.add_config("epochs", 10)
        conf.add_config("start_testing_epoch", 5)
        conf.add_config("verbose", 10)
        conf.add_config("rec.evaluate.neg", 100)
        conf.add_config("embed_size", 32)
        conf.add_config("ssl_mode", "user_side")
        conf.add_config("pretrain", 0)  # ✅ ← IMPORTANT: disable pretraining

    # ------------------------------------------------------

    seed = conf["seed"]
    print('seed=', seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
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
