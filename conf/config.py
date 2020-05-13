# coding=utf-8
# author=yphacker

import os

conf_path = os.path.dirname(os.path.abspath(__file__))
work_path = os.path.dirname(os.path.dirname(conf_path))
data_path = os.path.join(work_path, "data", "af2020cv-2020-05-09-v5-dev")
model_path = os.path.join(work_path, "model")
submission_path = os.path.join(work_path, "submission")
for path in [model_path, submission_path]:
    if not os.path.isdir(path):
        os.makedirs(path)

image_train_path = os.path.join(data_path, 'data')
image_test_path = os.path.join(data_path, 'data')
train_path = os.path.join(data_path, 'training.csv')
test_path = os.path.join(data_path, 'test.csv')

num_classes = 20
img_size = 224
batch_size = 32
epochs_num = 8
train_print_step = 20
patience_epoch = 4
