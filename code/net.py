import tensorflow as tf, os, sys
import data 

def conv_block(layer, fsize, training, name, pool=True):
    """
    Method to perform basic CNN convolution block pattern
    
      [ CONV --> BN --> RELU ] x2 --> POOL (optional)

    :params
      
      (tf.Tensor) layer : input layer
      (int) fsize : output filter size
      (tf.Tensor) training : boolean value regarding train/valid cohort
      (str) name : name of block 
      (bool) pool : if True, pooling is performed

    :return

      (tf.Tensor) layer : output layer 

    """
    with tf.variable_scope(name):

        for i in range(1, 3):

            layer = tf.layers.conv2d(layer, filters=fsize, kernel_size=(3, 3), padding='same',
                kernel_regularizer=l2_reg(1e-1), name='conv-%i' % i)
            layer = tf.layers.batch_normalization(layer, training=training, name='norm-%s' % i)
            layer = tf.nn.relu(layer, name='relu-%i' % i)

        if pool:
            pool = tf.layers.max_pooling2d(layer, pool_size=(2, 2), strides=(2, 2), name='pool-%i' % i)

        return layer, pool

def convt_block(layer, concat, fsize, name):
    """
    Method to perform basic CNN convolutional-transpose block pattern

      CONVT (applied to `layer`) --> CONCAT (with `concat`) 

    :params
      
      (tf.Tensor) layer : input layer 
      (tf.Tensor) concat : tensor to be concatenated
      (str) name : name of block 

    :return

      (tf.Tensor) layer : output layer

    """
    with tf.variable_scope(name):

        layer = tf.layers.conv2d_transpose(layer, filters=fsize, kernel_size=2, strides=2, 
            kernel_regularizer=l2_reg(1e-1),  name='convt')
        layer = tf.concat([layer, concat], axis=-1, name='concat')

        return layer

def l2_reg(scale):

    return tf.contrib.layers.l2_regularizer(scale)

def create_classifier(X, training):
    """
    Method to implement simple classifier

    :params

      (tf.Tensor) X : input tensor
      (tf.Tensor) training : boolean value regarding train/valid cohort

    :return

      (tf.Tensor) layer : output layer

    """
    block1, pool1 = conv_block(X, 8, training, name='block01')
    block2, pool2 = conv_block(pool1, 16, training, name='block02')
    block3, pool3 = conv_block(pool2, 32, training, name='block03')
    block4, pool4 = conv_block(pool3, 64, training, name='block04')
    block5, pool5 = conv_block(pool4, 96, training, name='block05')
    block6, pool6 = conv_block(pool5, 128, training, name='block06')
    
    pool6 = tf.reshape(pool6, shape=[-1, 1, 1, 1152]) 
    pred = tf.layers.conv2d(pool6, 2, (1, 1), name='pred', padding='same')
    pred = tf.contrib.layers.flatten(pred)

    return pred

def loss_sce(y_pred, y_true):
    """
    Method to implement simple softmax cross-entropy loss

    """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    return tf.reduce_mean(loss)

def error_topk(y_pred, y_true, k=1):
    """
    Method to calculate top-k error

    """
    error = tf.nn.in_top_k(predictions=y_pred, targets=y_true, k=k)

    return tf.reduce_mean(tf.cast(error, dtype=tf.float32))

def create_unet(X, training):
    """
    Method to implement U-net

    :params

      (tf.Tensor) X : input tensor
      (tf.Tensor) training : boolean value regarding train/valid cohort

    :return

      (tf.Tensor) layer : output layer

    Original Paper: https://arxiv.org/abs/1505.04597

    """
    # --- Contracting arm
    block1, pool1 = conv_block(X, 8, training, name='block01')
    block2, pool2 = conv_block(pool1, 16, training, name='block02')
    block3, pool3 = conv_block(pool2, 32, training, name='block03')
    block4, pool4 = conv_block(pool3, 64, training, name='block04')
    block5, pool5 = conv_block(pool4, 96, training, name='block05', pool=False)

    # --- Expanding arm
    block6 = convt_block(block5, block4, 64, name='block06')
    block7, _ = conv_block(block6, 64, training, name='block07', pool=False)

    block8 = convt_block(block7, block3, 32, name='block08')
    block9, _ = conv_block(block8, 32, training, name='block09', pool=False)

    block10 = convt_block(block9, block2, 16, name='block10')
    block11, _ = conv_block(block10, 16, training, name='block11', pool=False)

    block12 = convt_block(block11, block1, 8, name='block12')
    block13, _ = conv_block(block12, 8, training, name='block13', pool=False)

    # --- Collapse to number of classes
    pred = tf.layers.conv2d(block13, 5, (1, 1), name='final', activation=tf.nn.softmax, padding='same')

    return pred 

def loss_dice(y_pred, y_true):
    """
    Method to approximate Dice score loss function

      Dice (formal) = 2 x (y_pred UNION y_true) 
                      -------------------------
                       | y_pred | + | y_true | 

      Dice (approx) = 2 x (y_pred * y_true) + d 
                      -------------------------
                     | y_pred | + | y_true | + d 

      where d is small delta == 1e-7 added both to numerator/denominator to
      prevent division by zero.

    :params

        (tf.Tensor) y_pred : predictions 
        (tf.Tensor) y_true : ground-truth 

    :return

        (dict) scores : {
          'final': final weighted Dice score,
          0: score for class 1,
          1: score for class 2, ...
        }

    """
    # --- Loop over the channels
    channels = y_pred.shape.as_list()[-1]
    losses = {}
    for ch in range(channels):
        num = 2 * tf.reduce_sum(y_pred[..., ch] * y_true[..., ch])
        den = tf.reduce_sum(y_pred[..., ch]) + tf.reduce_sum(y_true[..., ch])
        losses[ch] = (num + 1e-7) / (den + 1e-7) 

    # --- Calculate weighted dice score loss function
    weight = lambda ch : (tf.reduce_sum(y_true[..., ch]) + 1e-7) / (tf.reduce_sum(y_true) + 1e-7)
    losses['dice'] = tf.reduce_sum([losses[ch] * weight(ch) for ch in range(channels)])

    return losses 

def init_metrics(losses):
    """
    Method to initialize a metrics dictionary based on keys in losses

    """
    metrics = {'train': {}, 'valid': {}}
    for mode, d in metrics.items():
        metrics[mode] = dict([(k, 0) for k in losses])

    return metrics

def update_ema(metrics, metric, mode, iteration):
    """
    Method to update the metrics dict with exponential moving average of metric.

    :params

      (dict) errors : dictionary with errors
      (float) error : current error value
      (str) mode : 'train' or 'valid'
      (int) iteration : update iteration (to determine EMA vs average)

    """
    decay = 0.99 if mode == 'train' else 0.9
    d = decay if iteration > 10 else 0.5

    for key, value in metric.items():
        metrics[mode][key] = metrics[mode][key] * d + value * (1 - d)

    return metrics 

def print_status(metrics, step, metric_names=[], end='\r'):
    """
    Method to print iteration and metrics for train/valid

    :params

      (dict) metrics
      (int) step
      (list) metric_names : list of keys within metrics to print 

    """
    printf = {'train': '', 'valid': ''}
    values = {'train': [], 'valid': []} 

    for name in metric_names:
        for mode in ['train', 'valid']:
            printf[mode] += '- %s : %s ' % (name, '%0.4f')
            values[mode].append(metrics[mode][name])

    printf = '%s | TRAIN %s | VALID %s' % ('%07i', printf['train'], printf['valid'])
    values = [step] + values['train'] + values['valid']

    print(printf % tuple(values), end=end)
    sys.stdout.flush()

def init_session(sess, output_dir):
    """
    Method to initialize generic Tensorflow objects

    :params
      
      (tf.Session) sess
      (str) output_dir

    """
    writer_train = tf.summary.FileWriter('%s/logs/train' % output_dir, sess.graph)
    writer_valid = tf.summary.FileWriter('%s/logs/valid' % output_dir)

    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    # --- Restore checkpoints if available
    latest_check_point = tf.train.latest_checkpoint('%s/checkpoint' % output_dir)
    if latest_check_point is not None:
        saver.restore(sess, latest_check_point)
    else:
        os.makedirs('%s/checkpoint' % output_dir, exist_ok=True)

    return sess, saver, writer_train, writer_valid

def init_batch(batch_size, one_hot=True, root=None):
    """
    Method to return batched data

    """
    if root is not None:
        data.set_root(root)

    def generator_train():
        while True:
            dat, lbl, msk = data.load(mode='train', return_mask=True)
            yield (dat[0], lbl[0, ..., 0], msk[0]) 

    def generator_valid():
        while True:
            dat, lbl, msk = data.load(mode='valid', return_mask=True)
            yield (dat[0], lbl[0, ..., 0], msk[0]) 

    batch = {}
    for mode, generator in zip(['train', 'valid'], [generator_train, generator_valid]): 

        ds = tf.data.Dataset.from_generator(generator, 
            output_types=(tf.float32, tf.uint8, tf.float32),
            output_shapes=([240, 240, 4], [240, 240], [240, 240, 1]))
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size * 5)
        its = ds.make_one_shot_iterator()
        it = its.get_next()
        y = tf.one_hot(it[1], depth=5) if one_hot else it[1]

        batch[mode] = {'X': it[0], 'y': y, 'mask': it[2]}

    return batch
