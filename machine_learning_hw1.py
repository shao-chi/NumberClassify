import tensorflow as tf
import numpy as np
import math

def read_and_decode(filename_queue, batch_size, train=True):
    # create TFRecordReader
    reader = tf.TFRecordReader()

    # read TFRecords 的ata
    _, serialized_example = reader.read(filename_queue)


    # read one data Example
    features = tf.parse_single_example(serialized_example,
            features={
                'image_string': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.string)
            })

    # 將序列化的圖片轉為 uint8 的 tensor
    image = tf.decode_raw(features['image_string'], tf.uint8)

    # 將 label 的資料轉為 uint8 的 tensor
    label = tf.decode_raw(features['label'],  tf.uint8)

    # 將圖片調整成正確的尺寸
    image = tf.reshape(image, [128, 128, 1])
    label = tf.reshape(label, [10])

    # 打散資料順序
    if train:        
        batch_size = batch_size
        min_after_dequeue = 1000
        capacity = 10000 + 4 * batch_size
        image, label = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity, num_threads=4, min_after_dequeue=min_after_dequeue)
    return image, label

def train(data_dir):
    # train your model with images from da
    # the following code is just a placeholder
    batch_size = 32
    # 建立檔名佇列
    filename_queue = tf.train.string_input_producer([data_dir], num_epochs=None)

    # 讀取並解析 TFRecords 的資料
    train_images, train_labels = read_and_decode(filename_queue, batch_size=batch_size)

    v_filename_queue = tf.train.string_input_producer(['./validation/validation.tfrecords'], num_epochs=None)
    valid_images, valid_labels = read_and_decode(filename_queue, batch_size=batch_size)

    def weight_variable(shape, name):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial, name = name)

    def bias_variable(shape, name):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial, name = name)

    def conv2d(x, w):
        return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
        exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
        bnepsilon = 1e-5
        if convolutional:
            mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
        else:
            mean, variance = tf.nn.moments(Ylogits, [0])
        update_moving_everages = exp_moving_avg.apply([mean, variance])
        m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
        v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
        Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
        return Ybn, update_moving_everages

    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None, 128, 128], name = 'train_data') # 48*48
        ys = tf.placeholder(tf.float32, [None, 10], name = 'train_label')
        lr = tf.placeholder(tf.float32) #For learning rate
        # test flag for batch normalization
        tst = tf.placeholder(tf.bool) 
        iter = tf.placeholder(tf.int32)
        keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    x_image = tf.reshape(xs, [-1, 128, 128, 1])

    # conv1 layer
    with tf.name_scope('layer_1'):
        with tf.name_scope('weights'):
            w_conv1_1 = weight_variable([3, 3, 1, 64], 'w_conv1')
        tf.summary.histogram('/weights', w_conv1_1)
        with tf.name_scope('bias'):
            b_conv1_1 = bias_variable([64], 'b_conv1')
        tf.summary.histogram('/bias', b_conv1_1)
        with tf.name_scope('outputs'):
            c_conv1_1 = conv2d(x_image, w_conv1_1) + b_conv1_1
            bn_1, update_ema1 = batchnorm(c_conv1_1, tst, iter, b_conv1_1, convolutional=True)
            h_conv1_1 = tf.nn.relu(bn_1)
            h_pool1 = max_pool_2x2(h_conv1_1)
        tf.summary.histogram('/outputs', h_pool1)

    # conv2 layer
    with tf.name_scope('layer_2'):
        with tf.name_scope('weights'):
            w_conv2_1 = weight_variable([3, 3, 64, 128], 'w_conv2')
        tf.summary.histogram('/weights', w_conv2_1)
        with tf.name_scope('bias'):
            b_conv2_1 = bias_variable([128], 'b_conv2')
        tf.summary.histogram('/bias', b_conv2_1)
        with tf.name_scope('outputs'):
            c_conv2_1 = conv2d(h_pool1, w_conv2_1) + b_conv2_1
            bn_2, update_ema2 = batchnorm(c_conv2_1, tst, iter, b_conv2_1, convolutional=True)
            h_conv2_1 = tf.nn.relu(bn_2)
            h_pool2 = max_pool_2x2(h_conv2_1)
        tf.summary.histogram('/outputs', h_pool2)

    # conv3 layer
    with tf.name_scope('layer_3'):
        with tf.name_scope('weights'):
            w_conv3_1 = weight_variable([3, 3, 128, 256], 'w_conv3')
        tf.summary.histogram('/weights', w_conv3_1)
        with tf.name_scope('bias'):
            b_conv3_1 = bias_variable([256], 'b_conv3')
        tf.summary.histogram('/bias', b_conv3_1)
        with tf.name_scope('outputs'):
            c_conv3_1 = conv2d(h_pool2, w_conv3_1) + b_conv3_1
            bn_3_1, update_ema3_1 = batchnorm(c_conv3_1, tst, iter, b_conv3_1, convolutional=True)
            h_conv3_1 = tf.nn.relu(bn_3_1)
            h_pool3 = max_pool_2x2(h_conv3_1)
        tf.summary.histogram('/outputs', h_pool3)

    #func1 layer
    with tf.name_scope('layer_func1'):
        with tf.name_scope('weights'):
            w_f1 = weight_variable([32*32*256, 1024], 'w_f1')  #new5 [256]
        tf.summary.histogram('/weights', w_f1)
        with tf.name_scope('bias'):
            b_f1 = bias_variable([1024], 'b_f1')  #new5 [256]
        tf.summary.histogram('/bias', b_f1)
        with tf.name_scope('outputs'):
            h_pool3_flat = tf.reshape(h_pool3, [-1, 32*32*256])
            h_m1 = tf.matmul(h_pool3_flat, w_f1) + b_f1
            bn_fl, update_ema_f1 = batchnorm(h_m1, tst, iter, b_f1)
            h_f1 = tf.nn.relu(bn_fl)
            h_f1_drop = tf.nn.dropout(h_f1, keep_prob)
        tf.summary.histogram('/outputs', h_f1_drop)

    #func2 layer
    with tf.name_scope('layer_func2'):
        with tf.name_scope('weights'):
            w_f2 = weight_variable([1024, 10], 'w_f2')  #new5 [256]
        tf.summary.histogram('/weights', w_f2)
        with tf.name_scope('bias'):
            b_f2 = bias_variable([10], 'b_f2')
        tf.summary.histogram('/bias', b_f2)
        with tf.name_scope('outputs'):
            h_m2 = tf.matmul(h_f1_drop, w_f2) + b_f2
            bn_fl, update_ema_f2 = batchnorm(h_m2, tst, iter, b_f2)
            prediction = tf.nn.softmax(bn_fl, name = 'prediction')
            tf.add_to_collection('outputs', prediction)
        tf.summary.histogram('/outputs', prediction)
        
    update_ema = tf.group(update_ema1, update_ema2, update_ema3_1, update_ema_f1, update_ema_f2)

    # loss
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=ys, logits=tf.log(tf.clip_by_value(prediction, 1e-10, 1.0))))
        tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('optimizer'):
        train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step = global_step)
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))  
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
        tf.summary.scalar('accuracy', accuracy)
    

    
    # 初始化變數
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session()  as sess:
        # 初始化
        sess.run(init_op)

        coord = tf.train.Coordinator() #創建協调器，管理線程
        threads = tf.train.start_queue_runners(coord=coord)

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter("logs_hw1_1/train", sess.graph)
        test_writer = tf.summary.FileWriter("logs_hw1_1/test")  
        saver = tf.train.Saver(max_to_keep=5)
        saver_max_acc = 0 

        for i in range(1001):
            # learning rate decay
            max_learning_rate = 0.02
            min_learning_rate = 0.0001
            decay_speed = 1600
            learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-i/decay_speed)
            batch_xs, batch_ys = sess.run([train_images, train_labels])
            batch_xs = batch_xs/255.
            feed_dict = {xs: batch_xs, ys: batch_ys, keep_prob: 0.7, lr: learning_rate, tst: False}
            sess.run(update_ema, {xs: batch_xs, ys: batch_ys, tst: False, iter: i, keep_prob: 1.0})

            if i % 100 == 0:
                ab, _, c , summary= sess.run([accuracy, train_step, cross_entropy, merged], feed_dict = feed_dict)
                train_writer.add_summary(summary, i)
                x_valid, y_valid = sess.run([valid_images, valid_labels])
                a, l , result= sess.run([accuracy, cross_entropy, merged], feed_dict = {xs: x_valid, ys: y_valid, tst: False, iter: i, keep_prob: 1})
                test_writer.add_summary(result, i)
                print('..........test_acc ', a, '  cross entropy:', l,  '------batch_acc:', ab, '  b_cross:', c)

                if a > saver_max_acc:
                    print('%d times valid accuracy: %.4f%%  loss: %.4f' % (i/100, a*100, l))
                    print('----------- batch accuracy: %.4f%%  loss: %.4f' % (ab*100, c))
                    saver.save(sess, '../model/model_hw1_1/model_hw1_1.ckpt', global_step = global_step)
                    print('saved\n')
                    saver_max_acc = a
            elif i % 200 == 99:
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step], feed_dict = feed_dict, options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict= feed_dict)
                train_writer.add_summary(summary, i)

        train_writer.close()  
        test_writer.close() 
        coord.request_stop()
        coord.join(threads)

def test(data_dir):
    # make your model give prediction for images from data_dir
    # the following code is just a placeholder
    return [1,3,4,5], [1,4,3,5]