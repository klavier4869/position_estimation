import os
import os.path
import numpy as np
import pickle


dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/pdsolv.pkl"


def load_pdsolv(normalize=True, flatten=True):
    """ pd-solvの位置推定データを読み込み
    Parameters
    ----------
    normalize : 画像のピクセル値を0.0~1.0に正規化する
    flatten : 画像を一次元配列に平にするかどうか

    Returns
    -------
    (訓練画像, 訓練位置情報), (検証画像, 検証位置情報), (テスト画像, テスト位置情報)
    """

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'valid_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if flatten:
         for key in ('train_img', 'valid_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 3, 1200)

    return (dataset['train_img'], dataset['train_pos']), \
            (dataset['valid_img'], dataset['valid_pos']), \
            (dataset['test_img'], dataset['test_pos'])
