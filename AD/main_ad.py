from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
#from scipy.misc import imresize
#import keras.backend as K
import numpy as np
import time, os, h5py
import functools, itertools, six
from six.moves import xrange

from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops

## Import Customized Functions
import network
from flags import FLAGS, TRAIN_RANGE, VAL_RANGE, option

#import tflearn_dev as tflearn
import tflearn
from data_flow import input_fn, IMG_SIZE

from utils import multig, prune, op_utils, train_ops
import keras.backend as K


def weighted_jaccard_loss_v2(logits, y_trues):
    dice = 0
    y_preds = tf.nn.softmax(logits)
    weights = [1, 1]
    for i in range(2):
        y_pred = y_preds[:,:,:,i+1]
        y_true = y_trues[:,:,:,i+1]
        union = y_pred+y_true - y_pred*y_true
        dice_list = -1*tf.reduce_sum(y_pred*y_true,(1,2))/(tf.reduce_sum(union,(1,2))+1)
        dice = dice+tf.reduce_mean(dice_list)*weights[i]
    return dice

def weighted_jaccard_loss(y_preds, y_trues):
    dice = 0
    weights = [1, 1]
    for i in range(2):
        y_pred = y_preds[:,:,:,i+1]
        y_true = y_trues[:,:,:,i+1]
        union = y_pred+y_true - y_pred*y_true
        dice_list = -1*tf.reduce_sum(y_pred*y_true,(1,2))/(tf.reduce_sum(union,(1,2))+1)
        dice = dice+tf.reduce_mean(dice_list)
    return dice


def get_trainable_variables(checkpoint_file, layer=None):
    reader = tf.train.NewCheckpointReader(checkpoint_file)
    saved_shapes = reader.get_variable_to_shape_map()

    checkp_var = [var for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes]

    checkp_name2var = dict(zip(map(lambda x:x.name.split(':')[0], checkp_var), checkp_var))
    all_name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))

    if layer==None:
        for name, var in all_name2var.items():
            if name in checkp_name2var:
                tf.add_to_collection('restore_vars', all_name2var[name])
            else:
                print(name)
                tf.add_to_collection('my_new_vars', all_name2var[name])
    else:
        for name, var in all_name2var.items():
            if name in checkp_name2var and 'Block8' not in name:
                tf.add_to_collection('restore_vars', all_name2var[name])
                if 'Block8' in name:
                    tf.add_to_collection('my_new_vars', all_name2var[name])
            else:
                print(name)
                tf.add_to_collection('my_new_vars', all_name2var[name])

    my_trainable_vars = tf.get_collection('my_new_vars')
    restore_vars = tf.get_collection('restore_vars')
    return my_trainable_vars, restore_vars



def dice(labels, logits, am_training):
    class_weights = tf.constant([[1.0]])
    y_pred = tf.nn.softmax(logits)
    if am_training:
        y_pred = y_pred[:,:,:,1:]
    else:
        maxvalue = tf.reduce_max(y_pred, axis=-1)
        y_pred = tf.equal(y_pred, tf.stack([maxvalue,maxvalue,maxvalue], axis=-1))
        y_pred = tf.cast(y_pred[:,:,:,1:], tf.float32)

    y_true = labels[:,:,:,1:]

    #union = y_pred+y_true - y_pred*y_true
    union = y_pred+y_true
    inter = y_pred*y_true
    under = union-inter
    dice_list = -1*(tf.reduce_sum(y_pred*y_true*class_weights,axis=(1,2,3))+1e-7)/(tf.reduce_sum(under,axis=(1,2,3))+2e-7)
    return dice_list


class Train():
    def __init__(self, run_id, config):
        self.img_size = IMG_SIZE
        self.run_id = run_id
        self.dict_widx = None
        self.tf_config = config

        # From FLAGS
        self.train_range = TRAIN_RANGE
        self.vali_range = VAL_RANGE
        self.wd = FLAGS.weight_decay
        self.wd_scale = FLAGS.weight_scale
        self.set_id = FLAGS.set_id
        self.am_training = True
    
    def _dice(self, logits):

        y_pred = tf.nn.softmax(logits)
        y_true = self.batch_labels[0]
        # add a number at end (20) to avoid dividing by 0
        y_pred = y_pred[:,:,:,1:2]
        #y_pred = y_pred[:,:,:,1:]
        y_true = y_true[:,:,:,1:2]
        #y_true = y_true[:,:,:,1:]
        union = y_pred+y_true
        union = y_pred+y_true - y_pred*y_true
        dice_list = K.sum(y_pred*y_true,(1,2,3))/(K.sum(union,(1,2,3))+1)
        dice = K.mean(dice_list)
        #dice = weighted_dice_loss(y_preds, y_trues)
        return dice

    def _dice_loss(self, logits, label):

        y_pred = tf.nn.softmax(logits)
        y_true = label
        #dice = weighted_cross_entropy(y_pred, y_true)
        #dice = weighted_dice_loss(y_pred, y_true)
        dice = weighted_jaccard_loss(y_pred, y_true)
        return dice


    def _build_graph(self):
        global_step = tf.contrib.framework.get_or_create_global_step()

        # Calculate logits using training data and vali data seperately
        #logits, logits_2 = getattr(model, self.model)(self.batch_data, self.wd)
        with tf.variable_scope(FLAGS.net_name):

            logits = getattr(network, FLAGS.net_name)(inputs=self.batch_data[0], 
                prob_fc=self.prob_fc, prob_conv=self.prob_conv, 
                wd=self.wd, wd_scale=self.wd_scale, 
                training_phase=self.am_training)


        print(logits.shape)

        label = self.batch_labels[0][:,:,:,1:3]
        #label = tf.greater(label, tf.ones(label.get_shape().as_list())*0.5)
        label = tf.cast(label, tf.float32)
        #label = tf.cast(tf.argmax(self.label_pl[:,:,:,0:2],3), tf.float32)
        #label = tf.expand_dims(label, -1)

        al_true = getattr(network, 'adversatial_2')(self.batch_data[0], label,reuse=False)

        y_pred = tf.nn.softmax(logits[0])
        maxvalue = tf.reduce_max(y_pred, axis=-1)
        y_pred = tf.equal(y_pred, tf.stack([maxvalue,maxvalue,maxvalue], axis=-1))
        pred = tf.cast(y_pred[:,:,:,1:], tf.float32)

        #print(pred)
        al_false = getattr(network, 'adversatial_2')(self.batch_data[0], pred,reuse=True)
        
        
        #size = al_true.get_shape().as_list()
        label_1 = tf.ones([FLAGS.batch_size,1])
        label_0 = tf.zeros([FLAGS.batch_size, 1])
        label_for_true = tf.concat([label_1, label_0], axis=1)
        label_for_false = tf.concat([label_0, label_1], axis=1)

        w_adversatial = 0.5*(tf.nn.softmax_cross_entropy_with_logits(labels=label_for_true, logits=al_true) + \
                        tf.nn.softmax_cross_entropy_with_logits(labels=label_for_false, logits=al_false))
        w_for_s = tf.nn.softmax_cross_entropy_with_logits(labels=label_for_false, logits=al_true)
        #w_adversatial = (tf.reduce_mean(-1*tf.map_fn(tf.log, al_true[:,0]+1e-3)) + \
        #               tf.reduce_mean(-1*tf.map_fn(tf.log, al_false[:,1]+1e-3)))

        self.al_true = al_true[0]
        self.al_false = al_false[1]


        with tf.name_scope("traing_dice"):
            self.accuracy = self._dice(logits[0])
            self.loss = self._dice_loss(logits[0], self.batch_labels[0])
            #self.loss = self._dice_loss(logits[0]) + self._dice_loss(logits[1]) + self._dice_loss(logits[2]) + self._dice_loss(logits[3]) +\
            #                        self._dice_loss(logits[4]) + self._dice_loss(logits[5])


        with tf.name_scope("total_loss"):
            weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            scale_loss = tf.get_collection('losses')

            w1 = 0; w2 = 0
            if len(weight_loss)>0:
                w1 = tf.add_n(weight_loss)
            
            if len(scale_loss)>0:
                w2 = tf.add_n(scale_loss)

            self.total_loss = self.loss + w1 + w2 - 1 * w_adversatial

            #self.total_loss = self.loss + w1 + w2 + w_for_s
            self.ad_loss = w_adversatial


        all_name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.trainable_variables()), tf.trainable_variables()))

        for name, var in all_name2var.items():
            if 'AD' in name:
                tf.add_to_collection('ad_vars', all_name2var[name])
            else:
                tf.add_to_collection('regular_vars', all_name2var[name])

        ad_vars = tf.get_collection('ad_vars')
        regular_vars = tf.get_collection('regular_vars')


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


        with tf.control_dependencies(update_ops):

            self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                           600, 0.9, staircase=True) # 10000

            learning_rate_ad = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                           600, 0.9, staircase=True)

            opt = tf.train.AdamOptimizer(self.learning_rate)
            opt_ad = tf.train.AdamOptimizer(learning_rate_ad)
            
            #grads_and_vars = opt.compute_gradients(self.total_loss, var_list=regular_vars)
            grads_and_vars_ad = opt_ad.compute_gradients(w_adversatial+w1, var_list=ad_vars)


            #apply_gradient_op = opt.apply_gradients(grads_and_vars, global_step=global_step)
            apply_gradient_op_ad = opt_ad.apply_gradients(grads_and_vars_ad, global_step=global_step)


            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

        tf.summary.histogram('al_true', tf.argmax(al_true[0,:]))
        tf.summary.histogram('al_false', tf.argmax(al_false[0,:]))
        tf.summary.image(
                        'pred',
                        pred[:,:,:,0:1],
                        max_outputs=6,
                        collections=None,
                        family=None
                        )

        tf.summary.image(
                        'label',
                        label[:,:,:,0:1],
                        max_outputs=6,
                        collections=None,
                        family=None
                        )

        #self.train_op = tf.group(apply_gradient_op, apply_gradient_op_ad) #
        self.train_op = tf.group(apply_gradient_op_ad)
        self.summary_op = tf.summary.merge_all()
        return ad_vars

    def train(self, **kwargs):
        #with tf.Graph().as_default():
        ops.reset_default_graph()
        sess = tf.Session(config=self.tf_config)

        with sess.as_default():
            # Data Reading objects
            tflearn.is_training(True, session=sess)

            self.am_training = tf.placeholder(dtype=bool, shape=())
            self.prob_fc = tf.placeholder_with_default(0.5, shape=())
            self.prob_conv = tf.placeholder_with_default(0.5, shape=())


            data_fn = functools.partial(input_fn, data_dir=os.path.join(FLAGS.data_dir, FLAGS.set_id), 
                num_shards=FLAGS.num_gpus, batch_size=FLAGS.batch_size, use_distortion_for_training=True)

            self.batch_data, self.batch_labels, _, _ = tf.cond(self.am_training, 
                lambda: data_fn(data_range=self.train_range, subset='train'),
                lambda: data_fn(data_range=self.vali_range, subset='test'))

            #self.batch_data = self.batch_data[0]
            #self.batch_labels = self.batch_labels[0]
            if FLAGS.status=='scratch':
                self.dict_widx = None
                self._build_graph()
                self.saver = tf.train.Saver(tf.global_variables())
                # Build an initialization operation to run below
                init = tf.global_variables_initializer()
                sess.run(init)

            elif FLAGS.status=='tune':
                self.dict_widx = kwargs['dict_widx']
                pruned_model = kwargs['pruned_model_path']

                #tflearn.config.init_training_mode()
                self._build_graph()             
                #tflearn.config.init_training_mode()
                init = tf.global_variables_initializer()
                sess.run(init)

                self.saver = tf.train.Saver(tf.global_variables())
                self.saver.restore(sess, pruned_model)
                print('Pruned model restored from ', pruned_model)

            elif FLAGS.status=='transfer':  
                regular_vars  = self._build_graph()             
                #tflearn.config.init_training_mode()
                init = tf.global_variables_initializer()
                sess.run(init)
                saver = tf.train.Saver(regular_vars)
                saver.restore(sess, FLAGS.checkpoint_path)
                print('Model restored from ', FLAGS.checkpoint_path)
                self.saver = tf.train.Saver(tf.global_variables())
                


            #coord = tf.train.Coordinator()
            #threads = tf.train.start_queue_runners(coord=coord)

            # This summary writer object helps write summaries on tensorboard
            summary_writer = tf.summary.FileWriter(FLAGS.log_dir+self.run_id)
            summary_writer.add_graph(sess.graph)

            train_error_list = []
            val_error_list = []

            print('Start training...')
            print('----------------------------------')


            train_steps_per_epoch = FLAGS.num_train_images//FLAGS.batch_size
            report_freq = train_steps_per_epoch

            train_steps = FLAGS.train_epoch * train_steps_per_epoch

            durations = []
            train_loss_list = []
            train_total_loss_list = []
            train_ad_loss_list = []
            train_accuracy_list = []
            best_epoch = 0
            best_accuracy = 0
            best_loss = 1

            nparams = op_utils.calculate_number_of_parameters(ad_vars)
            print(nparams)

            for step in range(train_steps):

                #print('{} step starts'.format(step))
                start_time = time.time()
                tflearn.is_training(True, session=sess)

                data, labels, _, summary_str, loss_value, total_loss, ad_loss, accuracy = sess.run(
                    [self.batch_data, self.batch_labels, self.train_op, self.summary_op, self.loss, self.total_loss, self.ad_loss, self.accuracy], 
                    feed_dict={self.am_training: True, self.prob_fc: FLAGS.keep_prob_fc, self.prob_conv: FLAGS.keep_prob_conv})

                #np.save('./test.npy',labels[0])
                tflearn.is_training(False, session=sess)
                duration = time.time() - start_time
                durations.append(duration)
                train_loss_list.append(loss_value)
                train_total_loss_list.append(total_loss)
                train_ad_loss_list.append(ad_loss)
                train_accuracy_list.append(accuracy)

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                
                #if step%(report_freq*10)==0:
                #   print(self.learning_rate)
                if step%report_freq == 0:
                    start_time = time.time()

                    summary_writer.add_summary(summary_str, step)

                    sec_per_report = np.sum(np.array(durations))
                    train_loss = np.mean(np.array(train_loss_list))
                    train_total_loss = np.mean(np.array(train_total_loss_list))
                    train_ad_loss =  np.mean(np.array(train_ad_loss_list))
                    train_accuracy_value = np.mean(np.array(train_accuracy_list))

                    train_loss_list = []
                    train_total_loss_list = []
                    train_accuracy_list = []
                    durations = []
                    train_ad_loss_list = []

                    train_summ = tf.Summary()
                    train_summ.value.add(tag="train_loss", simple_value=train_loss.astype(np.float))
                    train_summ.value.add(tag="train_ad_loss", simple_value=train_ad_loss.astype(np.float))
                    train_summ.value.add(tag="train_total_loss", simple_value=train_total_loss.astype(np.float))
                    train_summ.value.add(tag="train_accuracy", simple_value=train_accuracy_value.astype(np.float))

                    summary_writer.add_summary(train_summ, step)
                                                      
                    vali_loss_value, vali_accuracy_value, vali_ad_loss_value = self._full_validation(sess)
                    
                    if step%(report_freq*5)==0:
                        epoch = step/report_freq
                        model_dir = os.path.join(FLAGS.log_dir, self.run_id, 'model')
                        if not os.path.isdir(model_dir):
                            os.mkdir(model_dir)
                        checkpoint_path = os.path.join(model_dir, 'epoch_{}_acc_{:.3f}'.format(epoch, vali_accuracy_value))

                        self.saver.save(sess, checkpoint_path, global_step=step)

                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag="vali_loss", simple_value=vali_loss_value.astype(np.float))
                    vali_summ.value.add(tag="vali_accuracy", simple_value=vali_accuracy_value.astype(np.float))


                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                    vali_duration = time.time() - start_time

                    format_str = ('Epoch %d, loss = %.4f, total_loss = %.4f, ad_loss = %.4f, acc = %.4f, vali_loss = %.4f, vali_ad_loss= %.4f, val_acc = %.4f (%.3f ' 'sec/report)')
                    print(format_str % (step//report_freq, train_loss, train_total_loss, train_ad_loss, train_accuracy_value, vali_loss_value, vali_ad_loss_value, vali_accuracy_value, sec_per_report+vali_duration))


    def _full_validation(self, sess):
        tflearn.is_training(True, session=sess)
        num_batches_vali = FLAGS.num_val_images // FLAGS.batch_size

        loss_list = []
        ad_loss_list = []
        accuracy_list = []

        for step_vali in range(num_batches_vali):
            _, _, loss, ad_loss, accuracy = sess.run([self.batch_data, self.batch_labels,self.loss, self.ad_loss, self.accuracy], 
                feed_dict={self.am_training:False, self.prob_fc: 1, self.prob_conv: 1})
                                            #feed_dict={self.am_training: False, self.prob_fc: FLAGS.keep_prob_fc, self.prob_conv: 1})
            
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            ad_loss_list.append(ad_loss)

        vali_loss_value = np.mean(np.array(loss_list))
        vali_accuracy_value = np.mean(np.array(accuracy_list))
        vali_ad_loss_value = np.mean(np.array(ad_loss_list))
        return vali_loss_value, vali_accuracy_value, vali_ad_loss_value


def test(tf_config, test_path, probs):
    ops.reset_default_graph()
    sess = tf.Session(config=tf_config)
    save_root = FLAGS.save_root_for_prediction
    with sess.as_default():
        tflearn.is_training(False, session=sess)
        batch_data, batch_labels, subject, index = input_fn(data_dir=os.path.join(FLAGS.data_dir, FLAGS.set_id), 
            num_shards=1, batch_size=FLAGS.test_batch_size, data_range=TEST_RANGE, subset='test')
        with tf.variable_scope(FLAGS.net_name):
            logits = getattr(network, FLAGS.net_name)(inputs=batch_data[0], 
                prob_fc=1, prob_conv=1, wd=0, wd_scale=0, 
                training_phase=False)

        preclass = tf.nn.softmax(logits)
        prediction = tf.argmax(preclass,-1)
        label =  tf.argmax(batch_labels[0],-1)
        if FLAGS.seg:
            accuracy = dice(labels=batch_labels[0], logits=logits, am_training=False)
        else:
            correct_prediction = tf.equal(prediction, label)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, test_path)
        print('Model restored from ', test_path)

        if probs:
            prediction_array = []
        else:
            prediction_array = np.array([])

        label_array = np.array([])

        num_batches = FLAGS.num_test_images//FLAGS.test_batch_size

        accuracy_list = []
        prediction_array = []
        label_array = []

        #var = [v for v in tf.global_variables() if v.name == 'prediction'][0]
        for step in range(num_batches):
            #print(step)
            if probs: # TODO
                s, batch_prediction_array, batch_accuracy, batch_label_array = sess.run(
                [batch_data, preclass, accuracy, label])
                prediction_array.append(batch_prediction_array)
                label_array = np.concatenate((label_array, batch_label_array))
                accuracy_list.append(batch_accuracy)        

            else:
                images, annotations, sub_ind, ind, batch_prediction_array, batch_accuracy = sess.run(
                    [batch_data, label, subject, index, prediction, accuracy])

                op_utils.draw_images(save_root, images[0], annotations, sub_ind[0], ind[0], batch_prediction_array, -1*batch_accuracy)

                if FLAGS.seg:
                    pass
                else:
                    prediction_array = np.concatenate((prediction_array, batch_prediction_array))
                    label_array = np.concatenate((label_array, batch_label_array))
                accuracy_list.append(batch_accuracy)

        if probs:
            prediction_array = np.concatenate(prediction_array, axis=0)

        """
        if FLAGS.seg:
            accuracy_list = np.array(accuracy_list)
            rem = np.where(accuracy_list==-1)
            mask = np.ones(len(accuracy_list), dtype=bool)
            mask[rem[0]] = False
            accuracy = np.mean(accuracy_list[mask])
        else:
        """
        accuracy = np.mean(np.array(accuracy_list, dtype=np.float32))
        #print(prediction_array)
        diceoverall = op_utils.calcuate_dice_per_subject(save_root=save_root, img_size=IMG_SIZE[1])
        print('{:.3f}, {}'.format(accuracy, np.round(diceoverall, 3)))
        return prediction_array, label_array, accuracy


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("gi", nargs='?', help="index of the gpu",
                        type=int)
    gi = parser.parse_args().gi
    if gi==None:
        os.environ["CUDA_VISIBLE_DEVICES"]= "2,3"
    else: 
        os.environ["CUDA_VISIBLE_DEVICES"]= str(gi)

    #os.environ["CUDA_VISIBLE_DEVICES"]= str(gi)
    tf_config=tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True 

    #set_id = 'eval'

    if option == 1:
        run_id = '{}_{}_lr_{}_wd_{}_{}'.format(FLAGS.net_name, FLAGS.run_name, FLAGS.learning_rate, FLAGS.weight_decay, time.strftime("%b_%d_%H_%M", time.localtime()))

        # First Training
        train = Train(run_id, tf_config)
        train.train()

    elif option == 2:
        test_path = '/home/spc/Dropbox/Filter_Prune/Logs/mergeon_fourier_deep_test_lr_0.0001_wd_0.0001_Jun_17_10_45/model/epoch_5.0_acc_-0.061-10000'
        test(tf_config, test_path, probs=False)

    elif option == 3:
        # Filter pruning
        trained_path = '/home/spc/Dropbox/Drowsiness/3D/log_0.0001/Sparse_fourier_new_7__lr_0.001_wd_0.0001_Feb_06_13_47/model/epoch_11.0_dice_-0.832-6600'

        run_name = 'Prune_1'
        run_id = '{}_{}_lr_{}_wd_{}_{}'.format(model_name, run_name, learning_rate, weight_decay, time.strftime("%b_%d_%H_%M", time.localtime()))
        layer_names = ['Conv3D', 'Conv3D_1', 'Conv3D_2', 'Conv3D_3'] 

        dict_widx, pruned_model_path = prune.apply_pruning_scale(layer_names, trained_path, run_id, tf_config)
        #dict_widx, pruned_model_path = apply_pruning_random(layer_names, [58,52,52,63], trained_path, run_id, tf_config, random=False)

        train = Train(model_name, run_id, tf_config, set_id, img_size, learning_rate, train_range, vali_range, weight_decay)
        train.train(dict_widx=dict_widx, pruned_model_path=pruned_model_path)

    elif option == 4:
        diceoverall = op_utils.calcuate_dice_per_subject(save_root=FLAGS.save_root_for_prediction, img_size=IMG_SIZE[1])
        #print(diceoverall)


if __name__ == '__main__':
    tf.app.run()


