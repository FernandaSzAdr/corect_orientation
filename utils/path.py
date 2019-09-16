import numpy as np
import os
import pandas as pd
import pickle as pkl
import shutil

from utils.work_images import zip_path


def create_path(path_results):
    if not os.path.exists(path_results):
        os.makedirs(path_results)


def remove_path(path_results):
    shutil.rmtree(path_results)


def dump_numpy(output):
    with open('output/corrected_orientation_np.pkl', 'wb') as files:
        pkl.dump(output, files)


def dump_results(imgs, predicts, path_results, rotate_img):
    dump_numpy(np.array(rotate_img))

    df = pd.DataFrame({'fn': imgs, 'label': predicts})
    df.to_csv('output/test.preds.csv', index=False)

    imgs = [path_results + '/' + img for img in imgs]
    zip_path('output/test_result.zip', imgs)
