from scipy import misc
import tensorflow as tf
import detect_face
import cv2
import numpy as np

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
gpu_memory_fraction = 1.0

print('Creating networks and loading parameters')

config = tf.ConfigProto()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True
with tf.Graph().as_default():
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

capture = cv2.VideoCapture(0)


lst=np.load("E:/renlianshibie/model/lst.npy")
lb=np.load("E:/renlianshibie/model/lb.npy")

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

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'E:/renlianshibie/model/model.ckpt') #使用模型，参数和之前的代码保持一致
    prediction = tf.argmax(y_conv, 1)
    ii=0
    t=False
    name=""
    while (1):
        #
        ret, img = capture.read()
        #img = misc.imread("E:/renlianshibie/test/4.jpg")
        k=cv2.waitKey(1)
        if k==ord('q'):
            break
        b, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        tti=0
        for i in b:
            i = i.astype(int)
            crop=img[i[1]:i[3],i[0]:i[2],]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop=cv2.resize(crop,(160,160))
            crop = crop+15
            data=np.array(crop)/255
            predint = prediction.eval(feed_dict={x:[data], keep_prob: 1.0}, session=sess)
            name=lst[predint[0]]
            pre = tf.reshape(y_conv, [-1])
            if max(pre.eval(feed_dict={x: [data],keep_prob: 1.0}, session=sess)) / sum(pre.eval(feed_dict={x: [data],keep_prob: 1.0}, session=sess))>0.9:
                img=cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),(255,255,0),thickness=2)
                img=cv2.putText(img, name, (i[0], i[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                t=True
            else:
                img = cv2.rectangle(img, (i[0], i[1]), (i[2], i[3]), (255, 255, 0), thickness=2)
                img = cv2.putText(img, "UNKONW", (i[0], i[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('gray'+str(tti),crop)
            tti+=1
        cv2.imshow('frame', img)
        #ii+=1

    capture.release()
    cv2.destroyAllWindows()