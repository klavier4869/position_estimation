import os
import os.path
import numpy as np
import cv2
import random
import pickle


def fetch_game_img():
    filename = str(random.randint(0,9))
    img = cv2.imread('trim/' + filename + '.png', -1)
    return img


def composite_img(dest, src):
    w_src, h_src = src.shape[:2]
    w_dest, h_dest = dest.shape[:2]
    # init position of composite
    op_eria = { 'width' : w_dest - w_src,
                'height': h_dest - h_src }
    w_offset = random.randint(0, op_eria['width'])
    h_offset = random.randint(0, op_eria['height'])
    # composite image
    mask = src[:,:,3]
    mask = (mask / 255.0).reshape(w_src, h_src, 1)
    src = src[:,:,:3]
    trim_dest = dest[w_offset:w_offset+w_src, h_offset:h_offset+h_src]
    dest[w_offset:w_offset+w_src, h_offset:h_offset+h_src] = trim_dest * (1 - mask)
    dest[w_offset:w_offset+w_src, h_offset:h_offset+h_src] = trim_dest + (src * mask)
    img_info = [w_offset, h_offset, w_src, h_src]

    return dest, img_info


def save_data(imgs, imgs_info):
    dataset_dir = os.path.dirname(os.path.abspath(__file__))
    save_file   = dataset_dir + "/pd-solv.pkl"
    dataset     = {}
    imgs_len    = len(imgs)
    # splid data
    train_img, valid_img, test_img = np.split(
        imgs, [int(imgs_len*0.6), int(imgs_len*0.8)])
    train_pos, valid_pos, test_pos = np.split(
        imgs_info, [int(imgs_len*0.6), int(imgs_len*0.8)])
    # set data
    dataset['train_img'] = train_img
    dataset['train_pos'] = train_pos
    dataset['valid_img'] = valid_img
    dataset['valid_pos'] = valid_pos
    dataset['test_img']  = test_img
    dataset['test_pos']  = test_pos
    # save pickle file
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)


if __name__ == '__main__':
    img_count = 0
    cap = cv2.VideoCapture('background.mp4') # 800 frame
    imgs_info = []
    imgs = []

    print('Creating imgs...')
    while(cap.isOpened() | img_count < 5000):
        ret, frame = cap.read()
        # repeat
        if(ret == False):
            cap.set(1, 0.0)
            continue
        # composite img
        resize_img = cv2.resize(frame, (400, 300))
        game_img = fetch_game_img()
        comp_img, img_info = composite_img(resize_img, game_img)
        # add img & info
        imgs_info.append(img_info)
        comp_img = cv2.cvtColor(comp_img, cv2.COLOR_BGR2RGB)
        imgs.append(comp_img)
        img_count = img_count + 1
    # save train data
    print('Creating pickle...')
    imgs = np.array(imgs)
    imgs_info = np.array(imgs_info)
    save_data(imgs, imgs_info)
    print('Done')

    cap.release()
    cv2.destroyAllWindows()
