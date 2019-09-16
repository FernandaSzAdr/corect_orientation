import logging
import os

from utils.model import model
from utils.path import create_path, remove_path, dump_results
from utils.work_images import rotate


logging.basicConfig(format='%(levelname)s:%(message)s',
                    level=logging.INFO)


if __name__ == '__main__':
    path_results = 'test'
    path_test = 'database/test'

    """
        Return the model trained. If don't have already a model trained, will
        going to train the model.
    """
    _model = model(shape=(32, 32, 3), num_classes=4, is_plot=True,
                   path_database='database/{}', epochs=8)

    """
        Path 'test/' is create exclusively to save model corrected images.
        After zip the directory 'test/' it is removed.
    """
    create_path(path_results)

    imgs = []
    rotate_img = []
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

        rotate_img.append(rotate(path=path_test, image=img, orientation=pred,
                                 save_path=path_results))

    """
        Dump in 'output/': numpy array with corrected orientation faces,
        csv with the labels resulting from the model and the zip files with
        images after corrected orientation process.
    """
    dump_results(imgs, predicts, path_results, rotate_img)
    logging.info('The result is dump in "output/"!')

    remove_path(path_results)
