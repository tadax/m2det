import numpy as np
import tensorflow as tf
import logging
import os
import argparse

from m2det import M2Det
from utils.data import Data
from utils.loss import calc_loss

def main(args):
    logger = logging.getLogger()
    hdlr = logging.FileHandler(args.log_path)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(threadName)-10s] %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    
    databox = Data(image_dir=args.image_dir, 
                   label_dir=args.label_dir,
                   num_classes=args.num_classes,
                   input_size=args.input_size, 
                   shapes=args.shapes, 
                   assignment_threshold=args.assignment_threshold)
    databox.start()
    dataset_size = databox.size
    logger.info('Dataset size: {}'.format(dataset_size))

    '''
    y_true_size = 4 + num_classes + 1 + 1 = num_classes + 6
    * 4 => bbox coordinates (x1, y1, x2, y2);
    * num_classes + 1 => including a background class;
    * 1 => denotes if the prior box was matched to some gt boxes or not;
    '''
    y_true_size = args.num_classes + 6
    inputs = tf.placeholder(tf.float32, [None, args.input_size, args.input_size, 3])
    y_true = tf.placeholder(tf.float32, [None, args.num_boxes, y_true_size])
    is_training = tf.constant(True)
    net = M2Det(inputs, is_training, args.num_classes, args.sfam)
    y_pred = net.prediction
    total_loss = calc_loss(y_true, y_pred, box_loss_weight=args.scale)

    weights = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if 'kernel' in v.name]
    decay = tf.reduce_sum(tf.stack([tf.nn.l2_loss(w) for w in weights])) * 1e-3
    total_loss += decay

    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_var = tf.trainable_variables()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        opt = tf.train.MomentumOptimizer(learning_rate=args.learning_rate, momentum=0.9)
        grads = tf.gradients(total_loss, train_var)
        train_op = opt.apply_gradients(zip(grads, train_var), global_step=global_step)

    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    if tf.train.get_checkpoint_state(args.model_dir):
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(args.model_dir, 'variables'))
        logger.info('Resuming training')

    while True:
        x_batch, t_batch = databox.get(args.batch_size)
        _, loss_value = sess.run([train_op, total_loss], feed_dict={inputs: x_batch, y_true: t_batch})
        step_value = sess.run(global_step)
        logger.info('step: {}, loss: {}'.format(step_value, loss_value))
        if (step_value) % 1000 == 0:
            saver = tf.train.Saver()
            dst = os.path.join(args.model_dir, 'variables')
            saver.save(sess, dst, write_meta_graph=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--label_dir', required=True)
    parser.add_argument('--model_dir', default='weights/')
    parser.add_argument('--log_path', default='weights/out.log')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--scale', type=float, default=20.0)
    parser.add_argument('--num_classes', type=int, default=80)
    parser.add_argument('--input_size', type=int, default=320)
    parser.add_argument('--assignment_threshold', type=float, default=0.5)
    parser.add_argument('--num_boxes', type=int, default=19215) # (40x40+20x20+10x10+5x5+3x3+1x1)x9=19215
    parser.add_argument('--shapes', type=int, nargs='+', default=[40, 20, 10, 5, 3, 1]) # for 320x320
    parser.add_argument('--sfam', action='store_true', default=False)
    parser.add_argument('--gpu', type=str, default='0')
    os.environ['CUDA_VISIBLE_DEVICES'] = parser.parse_args().gpu
    main(parser.parse_args())
