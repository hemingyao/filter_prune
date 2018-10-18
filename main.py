"""
Author: Heming Yao
System: Linux

This is the main function for model training, test, and filter pruning
See flag.py for all relevent hyper-parameters 
See network.py for deep architectures
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time, os, h5py
import functools, itertools, six
from six.moves import xrange

from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops

## Import Customized Functions
import network
from flags import FLAGS, TRAIN_RANGE, VAL_RANGE, TEST_RANGE, option, IMG_SIZE
import tflearn_dev as tflearn
from data_flow import input_fn
from utils import multig, prune, op_utils, train_ops


def get_trainable_variables(checkpoint_file, layer=None):
    """This function is used for transfer learning
    to customize restore variables and new trainabel variables
    """

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
    y_pred = tf.nn.softmax(logits)
    if am_training:
        y_pred = y_pred[:,:,:,1]
    else:
        y_pred = tf.cast(tf.argmax(y_pred,-1), tf.float32)

    y_true = labels[:,:,:,1]

    #union = y_pred+y_true - y_pred*y_true
    union = y_pred+y_true
    dice_list = -2*(tf.reduce_sum(y_pred*y_true,axis=(1,2))+1e-7)/(tf.reduce_sum(union,axis=(1,2))+2e-7)
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
        

    def _tower_fn(self, index):
        """
        Build computation tower 
        Args:
          index: should be from 0 to n. n is the number of gpus.
        """
        logits = getattr(network, FLAGS.net_name)(inputs=self.batch_data[index], 
            prob_fc=self.prob_fc, prob_conv=self.prob_conv, 
            wd=self.wd, wd_scale=self.wd_scale, 
            training_phase=self.am_training)

        tower_pred = {
            'classes': tf.argmax(input=logits, axis=-1),
            'probabilities': tf.nn.softmax(logits)
            }

        labels = self.batch_labels[index]


        if FLAGS.loss_type == 'softmax_cross_entropy':

            temp_labels = tf.reshape(labels, [-1, labels.shape[-1].value])
            temp_logits = tf.reshape(logits, [-1,logits.shape[-1].value])
            #class_weights = tf.constant([[1.0, 1.0]])
            #tower_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            #    labels=class_weights *temp_labels, logits=temp_logits))

            tower_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=temp_labels, logits=temp_logits, name='cross_entropy'))

        elif FLAGS.loss_type == 'dice': 
            tower_loss = tf.reduce_mean(dice(labels, logits), am_training=True)

        # Add regularization loss and scale loss
        tower_total_loss = train_ops.add_all_losses(tower_loss)

        if FLAGS.status=='transfer':
            model_params, restore_vars = get_trainable_variables(FLAGS.checkpoint_path)
            model_params = model_params+restore_vars
        else:
            model_params = tf.trainable_variables()
            
        
        tower_grad = tf.gradients(tower_total_loss, model_params)

        return tower_loss, tower_total_loss, zip(tower_grad, model_params), tower_pred


    def _build_graph(self):
        """Resnet model body.
        Support single host, one or more GPU training. Parameter distribution can
        be either one of the following scheme.
        1. CPU is the parameter server and manages gradient updates.
        2. Parameters are distributed evenly across all GPUs, and the first GPU
           manages gradient updates.
        """
        global_step = tf.contrib.framework.get_or_create_global_step()
        tower_losses = []
        tower_total_losses = []
        tower_gradvars = []
        tower_preds = []
        num_gpus = FLAGS.num_gpus
        variable_strategy = 'GPU'

        if num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = num_gpus
            device_type = 'gpu'

        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if variable_strategy == 'CPU':
                device_setter = multig.local_device_setter(
                  worker_device=worker_device)
            elif variable_strategy == 'GPU':
                device_setter = multig.local_device_setter(
                  ps_device_type='gpu',
                  worker_device=worker_device,
                  ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                      num_gpus, tf.contrib.training.byte_size_load_fn))
            with tf.variable_scope(FLAGS.net_name, reuse=bool(i != 0)):

                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):
                        loss, total_loss, gradvars, preds = self._tower_fn(i)
                        tower_losses.append(loss)
                        tower_total_losses.append(total_loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                        # Only trigger batch_norm moving mean and variance update from
                        # the 1st tower. Ideally, we should grab the updates from all
                        # towers but these stats accumulate extremely fast so we can
                        # ignore the other stats from the other towers without
                        # significant detriment.
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                       name_scope)
        # Now compute global loss and gradients.
        gradlst = []
        varlst = []

        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in six.iteritems(all_grads):
                # Average gradients on the same device as the variables
                # to which they apply.
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                gradlst.append(avg_grad)
                varlst.append(var)

                if FLAGS.clip_gradients > 0.0:
                    gradlst, grad_norm = tf.clip_by_global_norm(gradlst, FLAGS.clip_gradients)
                gradvars = list(zip(gradlst, varlst))

        if FLAGS.status=='prune' and self.dict_widx:
            print('prune')
            gradvars = prune.apply_prune_on_grads(gradvars, self.dict_widx)

        # Device that runs the ops to apply global gradient updates.
        consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
        

        with tf.device(consolidation_device):
          # Suggested learning rate scheduling from

            self.loss = tf.reduce_mean(tower_losses, name='loss')
            self.total_loss = tf.reduce_mean(tower_total_losses, name='total_loss')

            _, apply_gradients_op, learning_rate = train_ops.train_operation(lr=FLAGS.learning_rate, global_step=global_step, 
                            decay_rate=FLAGS.decay_rate, decay_steps=FLAGS.decay_steps, optimizer=FLAGS.optimizer, clip_gradients=FLAGS.clip_gradients,
                            loss=self.total_loss, var_list=tf.trainable_variables(), grads_and_vars=gradvars)
            
            predictions = {
                'classes':
                    tf.concat([p['classes'] for p in tower_preds], axis=0),
                
                #'probabilities':
                #    tf.concat([p['probabilities'] for p in tower_preds], axis=0)
            }
            stacked_labels = tf.concat(self.batch_labels, axis=0)
            #stacked_labels = tf.argmax(input=stacked_labels, axis=1),
            

            if FLAGS.seg:
                y_pred = tf.cast(predictions['classes'], tf.float32)
                y_true = tf.cast(stacked_labels[:,:,:,1], tf.float32)
                union = y_pred + y_true
                dice_list = -2*tf.reduce_sum(y_pred*y_true,axis=(1,2))/(tf.reduce_sum(union,axis=(1,2))+0.001)
                metrics = {
                    'accuracy': tf.reduce_mean(dice_list)
                    }                          
            else:
                correct_prediction = tf.equal(predictions['classes'], tf.argmax(stacked_labels,-1))
                metrics = {
                    'accuracy': tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                        #tf.metrics.accuracy(stacked_labels, predictions['classes'])
                }

            # Create single grouped train op
            train_op = [apply_gradients_op]
            train_op.extend(update_ops)
            self.train_op = tf.group(*train_op)
            
            self.prediction = predictions['classes']
            self.accuracy = metrics['accuracy']

        """
        # This block may be uncommened in debug stage 
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        for grad, var in gradvars:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
        """

        for var in tf.get_collection('scale_vars'):
            tf.summary.histogram(var.op.name, var)
        # Concatenate Images
        tf.summary.image('images', self.batch_data[0], max_outputs=10)
        self.summary_op = tf.summary.merge_all()



    def train(self, **kwargs):
        """ Training body.
        if the filter prune is used, the input should be:
          dict_widx: the pruned weight matrix
          pruned_model_path: the path to the pruned model.
        """
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

            if FLAGS.status=='scratch':
                self.dict_widx = None
                self._build_graph()
                self.saver = tf.train.Saver(tf.global_variables())
                # Build an initialization operation to run below
                init = tf.global_variables_initializer()
                sess.run(init)

            elif FLAGS.status=='prune':
                self.dict_widx = kwargs['dict_widx']
                pruned_model = kwargs['pruned_model_path']

                self._build_graph()             
                init = tf.global_variables_initializer()
                sess.run(init)

                all_name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
                v_sel = []

                for name, var in all_name2var.items():
                    if 'Adam' not in name:
                        v_sel.append(all_name2var[name])
                self.saver = tf.train.Saver(v_sel)
                self.saver.restore(sess, pruned_model)
                print('Pruned model restored from ', pruned_model)

            elif FLAGS.status=='transfer':  
                self._build_graph()             
                init = tf.global_variables_initializer()
                sess.run(init)
                v1, v2 = get_trainable_variables(FLAGS.checkpoint_path)
                self.saver = tf.train.Saver(tf.global_variables())
                saver =  tf.train.Saver(v2)
                saver.restore(sess, FLAGS.checkpoint_path)
                print('Model restored.')


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
            train_accuracy_list = []
            best_epoch = 0
            best_accuracy = 0
            best_loss = 1

            nparams = op_utils.calculate_number_of_parameters(tf.trainable_variables())
            print(nparams)

            for step in range(train_steps):

                start_time = time.time()
                tflearn.is_training(True, session=sess)

                data, labels, _, summary_str, loss_value, total_loss, accuracy = sess.run(
                    [self.batch_data, self.batch_labels, self.train_op, self.summary_op, self.loss, self.total_loss, self.accuracy], 
                    feed_dict={self.am_training: True, self.prob_fc: FLAGS.keep_prob_fc, self.prob_conv: FLAGS.keep_prob_conv})

                tflearn.is_training(False, session=sess)
                duration = time.time() - start_time
                durations.append(duration)
                train_loss_list.append(loss_value)
                train_total_loss_list.append(total_loss)
                train_accuracy_list.append(accuracy)

                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                
                if step%report_freq == 0:
                    start_time = time.time()

                    summary_writer.add_summary(summary_str, step)

                    sec_per_report = np.sum(np.array(durations))
                    train_loss = np.mean(np.array(train_loss_list))
                    train_total_loss = np.mean(np.array(train_total_loss_list))
                    train_accuracy_value = np.mean(np.array(train_accuracy_list))

                    train_loss_list = []
                    train_total_loss_list = []
                    train_accuracy_list = []
                    durations = []

                    train_summ = tf.Summary()
                    train_summ.value.add(tag="train_loss", simple_value=train_loss.astype(np.float))
                    train_summ.value.add(tag="train_total_loss", simple_value=train_total_loss.astype(np.float))
                    train_summ.value.add(tag="train_accuracy", simple_value=train_accuracy_value.astype(np.float))

                    summary_writer.add_summary(train_summ, step)
                                                      
                    vali_loss_value, vali_accuracy_value = self._full_validation(sess)
                    
                    if step%(report_freq*FLAGS.save_epoch)==0:
                        epoch = step/(report_freq*FLAGS.save_epoch)
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

                    format_str = ('Epoch %d, loss = %.4f, total_loss = %.4f, acc = %.4f, vali_loss = %.4f, val_acc = %.4f (%.3f ' 'sec/report)')
                    print(format_str % (step//report_freq, train_loss, train_total_loss, train_accuracy_value, vali_loss_value, vali_accuracy_value, sec_per_report+vali_duration))


    def _full_validation(self, sess):
        """ Validation.
        After each epoch training, the loss value and accuracy will be 
        calculated for the validation dataset.
        The output can be used to implement early stopping.
        """
        tflearn.is_training(False, session=sess)
        num_batches_vali = FLAGS.num_val_images // FLAGS.batch_size

        loss_list = []
        accuracy_list = []

        for step_vali in range(num_batches_vali):
            _, _, loss, accuracy = sess.run([self.batch_data, self.batch_labels,self.loss, self.accuracy], 
                feed_dict={self.am_training:False, self.prob_fc: 1, self.prob_conv: 1})
                                            #feed_dict={self.am_training: False, self.prob_fc: FLAGS.keep_prob_fc, self.prob_conv: 1})
            #accuracy = 0
            loss_list.append(loss)
            accuracy_list.append(accuracy)

        vali_loss_value = np.mean(np.array(loss_list))
        vali_accuracy_value = np.mean(np.array(accuracy_list))

        return vali_loss_value, vali_accuracy_value


def test(tf_config, test_path, probs):
    """ Test function. To evaluate the performance of trained models
    tf_config: tensorflow and gpu configurations
    test_path: url. the path of trained model for test
    probs: boolean. True: calculate the probability. False: not calcualte.
    """
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

                accuracy_list.append(batch_accuracy)

        if probs:
            prediction_array = np.concatenate(prediction_array, axis=0)


        accuracy = np.mean(np.array(accuracy_list, dtype=np.float32))
        print('{:.3f}'.format(accuracy))
        return prediction_array, label_array, accuracy


def main(argv=None):
    """ The main function
      gi: specify which gpus installed on your computer the program will use
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("gi", nargs='?', help="index of the gpu",
                        type=int)
    gi = parser.parse_args().gi
    if gi==None:
        os.environ["CUDA_VISIBLE_DEVICES"]= "1,2,3"
    else: 
        os.environ["CUDA_VISIBLE_DEVICES"]= str(gi)

    tf_config=tf.ConfigProto()
    tf_config.gpu_options.allow_growth=True 


    if option == 1:
        run_id = '{}_{}_wd_{}_{}'.format(FLAGS.net_name, FLAGS.run_name, FLAGS.weight_scale, time.strftime("%b_%d_%H_%M", time.localtime()))

        # First Training
        train = Train(run_id, tf_config)
        train.train()

    elif option == 2:
        trained_path = '/home/spc/Dropbox/Filter_Prune/Logs/baseline_rescale_Prune_1_lr_0.01_wd_0.001_Jul_07_09_51/model/epoch_2.0_acc_0.900-1940'
        #test_path = '/home/spc/Dropbox/Filter_Prune/Logs/baseline_rescale_more_scale_wd_0.0001_Jul_06_11_10/epoch_21.0_acc_0.906-10185'
        test(tf_config, test_path, probs=False)

    elif option == 3:
        # Filter pruning
        trained_path = '/home/spc/Dropbox/Filter_Prune/Logs/prune_network/baseline_rescale_ScaleFC_wd_0.0001_Jul_06_14_52/model/epoch_49.0_acc_0.912-47530'

        run_name = 'Prune_1'
        run_id = '{}_{}_lr_{}_wd_{}_{}'.format(FLAGS.net_name, run_name, FLAGS.learning_rate, 
                 FLAGS.weight_decay, time.strftime("%b_%d_%H_%M", time.localtime()))
        layer_names = ['Conv1_1', 'Conv1_2', 'Conv2_1', 'Conv2_2', 'Conv3_1', 'Conv3_2', 
                       'Conv3_3', 'Conv4_1', 'Conv4_2', 'Conv4_3', 'Conv5_1', 'Conv5_2', 'Conv5_3']
        layer_names = [FLAGS.net_name+'/'+name  for name in layer_names]

        dict_widx, pruned_model_path = prune.apply_pruning_scale(layer_names, trained_path, run_id, tf_config)
        #
        #dict_widx, pruned_model_path = prune.apply_pruning_random(layer_names, a, trained_path, run_id, tf_config, random=False)
        #test(tf_config, pruned_model_path, probs=False)
        print(trained_path)
        train = Train(run_id, tf_config)
        train.train(dict_widx=dict_widx, pruned_model_path=pruned_model_path)

if __name__ == '__main__':
    tf.app.run()



