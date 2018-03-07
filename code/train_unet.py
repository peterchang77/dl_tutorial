import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import net

# --- Set global variables
iterations = 10 
batch_size = 16
learning_rate = 1e-3

# --- Prepare output folder
output_dir = '../exp_unet' 
os.makedirs(output_dir, exist_ok=True)

# --- Initialize data inputs
ops = {}
tf.reset_default_graph()
batch = net.init_batch(batch_size)

# --- Create placeholders into graph
X = tf.placeholder(tf.float32, shape=[None, 240, 240, 4], name='X')
y = tf.placeholder(tf.float32, shape=[None, 240, 240, 5], name='y')
mode = tf.placeholder(tf.bool, name='mode')

# --- Create unet and loss
print('Creating U-net graph')
pred = net.create_unet(X, training=mode)
ops['dice'] = net.loss_dice(pred, y)

# --- Create operation for single optimizer iteration
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):

    print('Creating optimizer')
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    ops['train'] = optimizer.minimize(-ops['dice'], global_step=global_step)

# --- Save key placeholders/operations for future reference
tf.add_to_collection("inputs", X)
tf.add_to_collection("inputs", mode)
tf.add_to_collection("outputs", pred)

# --- Add data to TensorBoard
tf.summary.histogram('softmax-scores', pred)
tf.summary.scalar('dice', ops['dice'])
ops['summary'] = tf.summary.merge_all()

with tf.Session() as sess:

    # --- Run graph
    sess, saver, writer_train, writer_valid = net.init_session(sess, output_dir)

    try:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        errors = {'train': 0, 'valid': 0}

        for i in range(iterations):

            # --- Run 10 batches of training 
            for j in range(10):

                X_batch, y_batch = sess.run([batch['train']['X'], batch['train']['y']])
                _, error, summary, step  = sess.run(
                    [ops['train'], ops['dice'], ops['summary'], global_step],
                    feed_dict={
                        X: X_batch, 
                        y: y_batch, 
                        mode: True})

                writer_train.add_summary(summary, step)
                errors = net.update_ema(errors, error, mode='train', iteration=i)
                net.print_status(errors, step)

            # --- Run 1 batch of validation 
            for j in range(1):

                X_batch, y_batch = sess.run([batch['valid']['X'], batch['valid']['y']])
                error, summary = sess.run(
                    [ops['dice'], ops['summary']],
                    feed_dict={
                        X: X_batch, 
                        y: y_batch, 
                        mode: False})

                writer_valid.add_summary(summary, step / 10)
                errors = net.update_ema(errors, error, mode='valid', iteration=i)
                net.print_status(errors, step)

        saver.save(sess, '%s/checkpoint/model.ckpy' % output_dir)

    finally:
        coord.request_stop()
        coord.join(threads)
        saver.save(sess, '%s/checkpoint/model.ckpy' % output_dir)
