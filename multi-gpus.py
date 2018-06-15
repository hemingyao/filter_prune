import multig_utils 


def get_model_fn(num_gpus, variable_strategy, num_workers):
    """
    Returns a function that will build the resnet model."""
    # features
    # labels
    # mode
    # params

    """Resnet model body.
    Support single host, one or more GPU training. Parameter distribution can
    be either one of the following scheme.
    1. CPU is the parameter server and manages gradient updates.
    2. Parameters are distributed evenly across all GPUs, and the first GPU
       manages gradient updates.
    Args:
      features: a list of tensors, one for each tower
      labels: a list of tensors, one for each tower
      mode: ModeKeys.TRAIN or EVAL
      params: Hyperparameters suitable for tuning
    Returns:
      A EstimatorSpec object.
    """

    #def _resnet_model_fn(features, labels, mode, params):

    #is_training = self.is_training
    #weight_decay = params.weight_decay
    #momentum = params.momentum

    variable_strategy = 'GPU'
    num_gpus = FLAGS.num_gpus
    tower_features = features
    tower_labels = labels
    tower_losses = []
    tower_gradvars = []
    tower_preds = []
    data_format = 'channels_last'
    # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
    # on CPU. The exception is Intel MKL on CPU which is optimal with
    # channels_last.

    if num_gpus == 0:
        num_devices = 1
        device_type = 'cpu'
    else:
        num_devices = num_gpus
        device_type = 'gpu'
  
    for i in range(num_devices):
        worker_device = '/{}:{}'.format(device_type, i)
        if variable_strategy == 'CPU':
            device_setter = multig_utils.local_device_setter(
              worker_device=worker_device)
        elif variable_strategy == 'GPU':
            device_setter = multig_utils.local_device_setter(
              ps_device_type='gpu',
              worker_device=worker_device,
              ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                  num_gpus, tf.contrib.training.byte_size_load_fn))
        with tf.variable_scope(FLAGS.net_name, reuse=bool(i != 0)):
            with tf.name_scope('tower_%d' % i) as name_scope:
                with tf.device(device_setter):
                    loss, gradvars, preds = _tower_fn(
                        is_training, weight_decay, tower_features[i], tower_labels[i])
                    tower_losses.append(loss)
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
    gradvars = []
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
            gradvars.append((avg_grad, var))

    # Device that runs the ops to apply global gradient updates.
    consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
    with tf.device(consolidation_device):
      # Suggested learning rate scheduling from


        loss = tf.reduce_mean(tower_losses, name='loss')

        _, apply_gradients_op, learning_rate = train_operation(loss, var_list, global_step, decay_rate, decay_steps, lr=0.001, 
                          optimizer='Adam',dict_widx=None, clip_gradients=0)

        # log
        examples_sec_hook = multig_utils.ExamplesPerSecondHook(
            FLAGS.batch_size, every_n_steps=100)

        tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        train_hooks = [logging_hook, examples_sec_hook]

        if params.sync:
            optimizer = tf.train.SyncReplicasOptimizer(
                optimizer, replicas_to_aggregate=num_workers)
            sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
            train_hooks.append(sync_replicas_hook)

        # Create single grouped train op
        train_op = [apply_gradients_op]
        train_op.extend(update_ops)
        train_op = tf.group(*train_op)

        predictions = {
            'classes':
                tf.concat([p['classes'] for p in tower_preds], axis=0),
            'probabilities':
                tf.concat([p['probabilities'] for p in tower_preds], axis=0)
        }
        stacked_labels = tf.concat(labels, axis=0)
        metrics = {
            'accuracy':
                tf.metrics.accuracy(stacked_labels, predictions['classes'])
        }
    """
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=train_hooks,
        eval_metric_ops=metrics)
    """
  return _resnet_model_fn



def _tower_fn(is_training, weight_decay, feature, label, data_format,
              num_layers, batch_norm_decay, batch_norm_epsilon):
  """Build computation tower (Resnet).
  Args:
    is_training: true if is training graph.
    weight_decay: weight regularization strength, a float.
    feature: a Tensor.
    label: a Tensor.
    data_format: channels_last (NHWC) or channels_first (NCHW).
    num_layers: number of layers, an int.
    batch_norm_decay: decay for batch normalization, a float.
    batch_norm_epsilon: epsilon for batch normalization, a float.
  Returns:
    A tuple with the loss for the tower, the gradients and parameters, and
    predictions.
  """
    logits = getattr(network, FLAGS.net_name)(inputs=self.batch_data, 
        prob_fc=self.prob_fc, prob_conv=self.prob_conv, 
        wd=self.wd, wd_scale=self.wd_scale, 
        training_phase=self.am_training)
    

    tower_pred = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits)
    }

    tower_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=self.batch_labels, logits=logits, name='cross_entropy'))

    model_params = tf.trainable_variables()
    tower_loss += weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in model_params])

    tower_grad = tf.gradients(tower_loss, model_params)

    return tower_loss, zip(tower_grad, model_params), tower_pred

