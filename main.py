import logging
import os
import pandas as pd
import shutil

from utils.model import Model
from utils.work_images import rotate, zip_path


logging.getLogger().setLevel(logging.ERROR)


def have_model():
    return os.path.exists('utils/model.h5')


def create_path(path_results):
    if not os.path.exists(path_results):
        os.makedirs(path_results)


def remove_path(path_results):
    shutil.rmtree(path_results)


if __name__ == '__main__':
    path_results = 'results'
    path_test = 'database/test'
    _model = Model()

    if not have_model():
        _model.create_model()
        _model.compile_model()
        _model.train()
        _model.save_model()
    else:
        _model.load_model()

    create_path(path_results)
    imgs = []
    predicts = []

    labels = {
        '0': 'rotated_left',
        '1': 'rotated_right',
        '2': 'upright',
        '3': 'upside_down',
    }

    for pred, img in zip(_model.predictions, os.listdir(path_test)):
        pred = str(pred)

        imgs.append(img)
        predicts.append(labels.get(pred))

        rotate(path_test, img, pred, path_results)

    df = pd.DataFrame({'fn': imgs, 'label': predicts})
    df.to_csv('test.preds.csv', index=False)

    imgs = [path_results + '/' + img for img in imgs]
    zip_path('test_result.zip', imgs)

    remove_path(path_results)
