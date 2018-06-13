import tensorflow as tf
import tflearn_dev as tflearn


##################################################################################################################

def _conv2d(x, spatial_size, out, name, wd=0):
    '''
    shape = [filter_size, filter_size, number of channel, number of filter]
    '''
    shape = [spatial_size, spatial_size, x.shape[-1], out]

    with tf.variable_scope(name):
        W = tf.get_variable(name='W', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='b', shape=[shape[3]], initializer=tf.constant_initializer(0.01))
        out = tf.add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'), b)
        activation = tf.nn.relu(out)

        if wd > 0:
            weight_decay = tf.multiply(tf.norm(W, 2), wd, name='weight_loss')
            #weight_decay = tf.multiply(tf.nn.l2_loss(tf.multiply(W, scale)), 10, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return activation


def _max_pool_2x2(x, name):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


def _fully_connected(x, num_neuro, name):
    '''
    x: [batch_size, spatial_size, spatial_size, channel]
    '''
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(x, [-1, dim])

        W = tf.get_variable(name='W', shape=[dim,num_neuro], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='b', shape=[num_neuro], initializer=tf.constant_initializer(0.01))
        out = tf.add(tf.matmul(x, W), b)

        return out


##################################################################################################################

def _fully_connected_scale(x, num_neuro, name):
    '''
    x: [batch_size, spatial_size, spatial_size, channel]
    '''
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(x, [-1, dim])

        W = tf.get_variable(name='W', shape=[dim,num_neuro], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='b', shape=[num_neuro], initializer=tf.constant_initializer(2))
        out = tf.add(tf.matmul(x, W), b)

        return out

def _fully_connected_rescale(x, num_neuro, name):
    '''
    x: [batch_size, spatial_size, spatial_size, channel]
    '''
    with tf.variable_scope(name):
        shape = x.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(x, [-1, dim])

        W = tf.get_variable(name='W', shape=[dim,num_neuro], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='b', shape=[num_neuro], initializer=tf.constant_initializer(2))
        out = tf.add(tf.matmul(x, W), b)

        return out


def _conv2d_rescale(x, spatial_size, out_dim, name, wd=0):
    shape = [spatial_size, spatial_size, x.shape[-1], out_dim]
    with tf.variable_scope(name):

        W = tf.get_variable(name='W', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.01))

        with tf.variable_scope('scale'):
            W_flatten = tf.reshape(W, [-1, out_dim])
            mag = tf.norm(W_flatten, 1, axis=0, keep_dims=True)
            out = _fully_connected_rescale(mag, out_dim/32, 'fc1')
            out = tf.nn.relu(out)
            out = _fully_connected_rescale(out, out_dim, 'fc2')
            #out = tflearn.batch_normalization(outf.norm(mean,1)t, scope='bn')
            scale = tf.nn.sigmoid(out)
            tf.summary.histogram('scale', scale)

        b = tf.get_variable(name='b', shape=[out_dim], initializer=tf.constant_initializer(0.01))
        out = tf.add(tf.nn.conv2d(x, tf.multiply(W, scale), strides=[1, 1, 1, 1], padding='SAME'), b)

        #out = tf.add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'), b)
        #out = tflearn.batch_normalization(out, scope='bn')
        activation = tf.nn.relu(out)        

        if wd > 0:
            mean, var = tf.nn.moments(scale, axes=[1])
            #scale = scale - tf.maximum(0.8, tf.norm(mean,1))
            #scale = scale - tf.norm(mean,1)
            #scale_decay = -tf.norm(scale, 2)*out_dim*0.1 + tf.multiply(tf.norm(scale, 1), wd*0.005)
            scale_decay = -tf.norm(var,1)*out_dim + tf.multiply(tf.norm(scale, 1), wd*0.005)

            #_, scale_decay = tf.nn.moments(scale, axes=[1])
            #scale_decay = tf.log(1/(tf.norm(scale_decay, 1)+1))
            #scale_decay = -tf.norm(scale_decay, 1)*out_dim + tf.multiply(tf.norm(scale, 1), wd*0.001)

            #scale_decay = tf.multiply(tf.norm(scale, 1), wd*0.001, name='scale_loss')
            tf.add_to_collection('losses', scale_decay)
            weight_decay = tf.multiply(tf.nn.l2_loss(W), wd, name='weight_loss')
            #weight_decay = tf.multiply(tf.norm(tf.multiply(W, scale), 1), 0.1, name='weight_loss')
            #weight_decay = tf.multiply(tf.nn.l2_loss(tf.multiply(W, scale)), 1, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        return activation


def _conv2d_rescale_sigmoid(x, spatial_size, out_dim, name, wd=0):
    shape = [spatial_size, spatial_size, x.shape[-1], out_dim]
    with tf.variable_scope(name):
        scale = tf.get_variable(name='scale', shape=[out_dim], initializer=tf.constant_initializer(2))
        #scale = tf.nn.softmax(scale) 
        scale = tf.nn.sigmoid(scale)
        tf.summary.histogram('scale', scale)
        W = tf.get_variable(name='W', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.01))
        b = tf.get_variable(name='b', shape=[out_dim], initializer=tf.constant_initializer(0.01))
        out = tf.add(tf.nn.conv2d(x, tf.multiply(W, scale), strides=[1, 1, 1, 1], padding='SAME'), b)
        #out = tf.add(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'), b)
        
        activation = tf.nn.relu(out)
        #activation = tflearn.batch_normalization(activation, scope='bn')

        if wd > 0:
            #_, scale_decay = tf.nn.moments(scale, axes=[0])
            scale_decay = tf.multiply(tf.norm(scale, 1), wd, name='scale_loss')
            tf.add_to_collection('losses', scale_decay)
            #tf.add_to_collection('losses', 1/(scale_decay+1))
            #weight_decay = tf.multiply(tf.nn.l2_loss(tf.multiply(W, scale)), wd, name='weight_loss')
            weight_decay = tf.multiply(tf.nn.l2_loss(W), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

        return activation


##################################################################################################################
def rescale_fc_slim(inputs, wd):
    x = _conv2d_rescale(inputs, 3, 128, 'conv_1', wd=wd)
    #x = _conv2d_rescale(x, 3, 128, 'conv_2', wd=wd)
    x = _max_pool_2x2(x, 'pool_1')
    x = tflearn.batch_normalization(x, gm_trainable=False)

    x = _conv2d_rescale(x, 3, 256, 'conv_3', wd=wd)
    #x = _conv2d_rescale(x, 3, 256, 'conv_4', wd=wd)
    x = _max_pool_2x2(x, 'pool_2')
    x = tflearn.batch_normalization(x, gm_trainable=False)

    x = _conv2d_rescale(x, 3, 512, 'conv_5', wd=wd)
    #x = _conv2d_rescale(x, 3, 512, 'conv_6', wd=wd)
    x = _max_pool_2x2(x, 'pool_3')
    x = tflearn.batch_normalization(x, gm_trainable=False)


    x = _fully_connected(x, 512, name='fc_1')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    #x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    #x = tflearn.dropout(x, 0.5, name='dropout2')

    x = _fully_connected(x, 10, name='fc_2')
    #x = tf.nn.softmax(x, name='softmax')
    return x


def rescale_fc(inputs, wd):
    x = _conv2d_rescale(inputs, 3, 128, 'conv_1', wd=wd)
    x = _conv2d_rescale(x, 3, 128, 'conv_2', wd=wd)
    x = _max_pool_2x2(x, 'pool_1')
    x = tflearn.batch_normalization(x, gm_trainable=False)

    x = _conv2d_rescale(x, 3, 256, 'conv_3', wd=wd)
    x = _conv2d_rescale(x, 3, 256, 'conv_4', wd=wd)
    x = _max_pool_2x2(x, 'pool_2')
    x = tflearn.batch_normalization(x, gm_trainable=False)

    x = _conv2d_rescale(x, 3, 512, 'conv_5', wd=wd)
    x = _conv2d_rescale(x, 3, 512, 'conv_6', wd=wd)
    x = _max_pool_2x2(x, 'pool_3')
    x = tflearn.batch_normalization(x, gm_trainable=False)


    x = _fully_connected(x, 512, name='fc_1')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    #x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    #x = tflearn.dropout(x, 0.5, name='dropout2')

    x = _fully_connected(x, 10, name='fc_2')
    #x = tf.nn.softmax(x, name='softmax')
    return x



def rescale_fc_variable_of_interest():
    return ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'fc_1', 'fc_2']
    

def rescale_2D_sigmoid(inputs, wd):
    x = _conv2d_rescale_sigmoid(inputs, 3, 64, 'conv_1', wd=wd)
    x = _conv2d_rescale_sigmoid(x, 3, 64, 'conv_2', wd=wd)
    x = _max_pool_2x2(x, 'pool_1')
    x = tflearn.batch_normalization(x)

    x = _conv2d_rescale_sigmoid(x, 3, 128, 'conv_3', wd=wd)
    x = _conv2d_rescale_sigmoid(x, 3, 128, 'conv_4', wd=wd)
    x = _max_pool_2x2(x, 'pool_2')
    x = tflearn.batch_normalization(x)

    x = _conv2d_rescale_sigmoid(x, 3, 256, 'conv_5', wd=wd)
    x = _conv2d_rescale_sigmoid(x, 3, 256, 'conv_6', wd=wd)
    x = _max_pool_2x2(x, 'pool_3')
    x = tflearn.batch_normalization(x)


    x = _fully_connected(x, 512, name='fc_1')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    #x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    #x = tflearn.dropout(x, 0.5, name='dropout2')

    x = _fully_connected(x, 10, name='fc_2')
    #x = tf.nn.softmax(x, name='softmax')
    return x
    

def tflearn_model(inputs):
    network = _conv2d(inputs, 3,32, 'conv_1')
    network = _max_pool_2x2(network, 'pool_1')

    network = _conv2d(network, 3,64, 'conv_2')
    network = _conv2d(network, 3,64, 'conv_3')

    nnetwork = _max_pool_2x2(network, 'pool_2')
    network = _fully_connected(network, 512, 'fc_1')
    network = tflearn.dropout(network, 0.5, name='drop_1')
    network = _fully_connected(network, 10, 'fc_2')
    network = tf.nn.softmax(network, name='softmax')

    return network


def vgg16_all(input, wd):
    x = _conv2d(input, 3, 128, 'conv_1', wd=wd)
    x = _conv2d(x, 3, 128, 'conv_2', wd=wd)
    x = _max_pool_2x2(x, 'pool_1')
    x = tflearn.batch_normalization(x, gm_trainable=True)

    x = _conv2d(x, 3, 256, 'conv_3', wd=wd)
    x = _conv2d(x, 3, 256, 'conv_4', wd=wd)
    x = _max_pool_2x2(x, 'pool_2')
    x = tflearn.batch_normalization(x, gm_trainable=True)

    x = _conv2d(x, 3, 512, 'conv_5', wd=wd)
    x = _conv2d(x, 3, 512, 'conv_6', wd=wd)
    x = _max_pool_2x2(x, 'pool_3')
    x = tflearn.batch_normalization(x, gm_trainable=True)


    x = _fully_connected(x, 512, name='fc_1')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    #x = tflearn.fully_connected(x, 4096, activation='relu', scope='fc7')
    #x = tflearn.dropout(x, 0.5, name='dropout2')

    x = _fully_connected(x, 10, name='fc_2')
    #x = tf.nn.softmax(x, name='softmax')
    return x
