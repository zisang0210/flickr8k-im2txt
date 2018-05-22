#!/usr/bin/python
import tensorflow as tf

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data
import os

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('faster_rcnn_ckpt', './vgg16_no_fc.npy',
                       'The file containing a pretrained faster rcnn model')

tf.flags.DEFINE_string('faster_rcnn_frozen', './vgg16_no_fc.npy',
                       'The file containing a frozened faster rcnn model')

tf.flags.DEFINE_boolean('joint_train', False,
                        'Turn on to train both RNN and attention mechanism. \
                         Otherwise, only attention mechanism is trained')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')

tf.flags.DEFINE_string("attention", "bias",
                        "fc1, fc2, bias, bias2, bias_fc1, bias_fc2, rnn")

def main(argv):
    os.system("pwd")
    os.system("ls -al")
    os.system("ls /")
    os.system("ls /output")
    os.system("ls /prev-output")
    os.system("ls /tinysrc")
    os.system("python tinysrc/download_flickr8k.py")
    # config = Config()
    # config.phase = FLAGS.phase
    # config.train_cnn = FLAGS.train_cnn
    # config.joint_train = FLAGS.joint_train
    # config.beam_size = FLAGS.beam_size
    # config.attention_mechanism = FLAGS.attention
    # config.faster_rcnn_frozen = FLAGS.faster_rcnn_frozen

    # with tf.Session() as sess:
    #     if FLAGS.phase == 'train':
    #         # training phase
    #         data = prepare_train_data(config)
    #         model = CaptionGenerator(config)
    #         sess.run(tf.global_variables_initializer())
    #         if FLAGS.load:
    #             model.load(sess, FLAGS.model_file)
    #         if FLAGS.load_cnn:
    #             model.load_faster_rcnn_feature_extractor(sess, FLAGS.faster_rcnn_ckpt)
    #         tf.get_default_graph().finalize()
    #         model.train(sess, data)

    #     elif FLAGS.phase == 'eval':
    #         # evaluation phase
    #         coco, data, vocabulary = prepare_eval_data(config)
    #         model = CaptionGenerator(config)
    #         model.load(sess, FLAGS.model_file)
    #         tf.get_default_graph().finalize()
    #         model.eval(sess, coco, data, vocabulary)

    #     else:
    #         # testing phase
    #         data, vocabulary = prepare_test_data(config)
    #         model = CaptionGenerator(config)
    #         model.load(sess, FLAGS.model_file)
    #         tf.get_default_graph().finalize()
    #         model.test(sess, data, vocabulary)

    os.system("rm -rf /output/Flickr8k_Dataset/")
    os.system("rm -rf /output/Flickr8k_text/")

if __name__ == '__main__':
    tf.app.run()
