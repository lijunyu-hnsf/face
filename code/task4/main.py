import numpy as np
import tensorflow as tf

Dtr=np.load("E:/renlianshibie/Data/data_tr.npy")
Ltr=np.load("E:/renlianshibie/Data/label_tr.npy")
Dte=np.load("E:/renlianshibie/Data/data_te.npy")
Lte=np.load("E:/renlianshibie/Data/label_te.npy")
lst=np.load("E:/renlianshibie/model/lst.npy")
lb=np.load("E:/renlianshibie/model/lb.npy")

ytr=[]

for i in Ltr:
    ind=list(lst).index(i)
    ytr.append(lb[ind])

yte=[]
for i in Lte:
    ind = list(lst).index(i)
    yte.append(lb[ind])
#xtr=[]
#for i in Dtr:
#    xtr.append(np.float32(np.reshape(i,6400)))

x=tf.placeholder(tf.float32, [None,160,160])
y_=tf.placeholder(tf.float32, [None,10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def max_pool_4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,160,160,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_4(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([20*20* 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 20*20*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

saver = tf.train.Saver() #定义saver

def next_suiji(sz):#随机抽sz个
    sui=np.random.randint(4800,size=sz)
    train_x = []
    train_y = []
    for i in sui:
        train_x.append(Dtr[i])
        train_y.append(ytr[i])
    return [train_x,train_y]

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
config.gpu_options.allow_growth = True
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch = next_suiji(100)
        if i % 100 == 0:
             train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
             print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    #train_step.run(feed_dict={x: Dtr, y_: ytr, keep_prob: 0.5})
    #train_accuracy = accuracy.eval(feed_dict={x: Dtr, y_: ytr, keep_prob: 1.0})
    saver.save(sess, 'E:/renlianshibie/model/model.ckpt')

    print('test accuracy %g' % accuracy.eval(feed_dict={x: Dte, y_: yte, keep_prob: 1.0}))