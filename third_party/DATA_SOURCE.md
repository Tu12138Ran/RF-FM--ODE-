# Fashion-MNIST attribution

Dataset: Fashion-MNIST, copyright 2017 Zalando SE, MIT license.
Source: https://github.com/zalandoresearch/fashion-mnist
Paper: https://arxiv.org/abs/1708.07747

Dataset files are not committed. `flowlab/io_utils.py` automatically downloads
60,000 training images (28×28, grayscale), verifies MD5
`8d4fb7e6c68d591d4c3dfef9ec88bf0d`, and normalizes them to [-1,1].
Labels are unused. The initial mirror is
https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
See `Fashion-MNIST-LICENSE` for the upstream license.
