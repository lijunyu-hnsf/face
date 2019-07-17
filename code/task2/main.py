from scipy import misc
import tensorflow as tf
import detect_face
import os
import cv2

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
gpu_memory_fraction = 1.0

print('Creating networks and loading parameters')

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config.gpu_options.allow_growth = True
with tf.Graph().as_default():
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

dbtype_list = os.listdir("E:/renlianshibie/facelmages")
for dbtype in dbtype_list:
    if os.path.isfile(os.path.join("E:/renlianshibie/facelmages",dbtype)):
        dbtype_list.remove(dbtype)

print(dbtype_list)
for na in ["niruxing"]:
    ii=0
    if not os.path.exists("E:/renlianshibie/faceImageGray/" + na):
        os.makedirs("E:/renlianshibie/faceImageGray/" + na)
    coll = os.walk("E:/renlianshibie/facelmages/"+na+"/")
    flist = []
    for path, d, filelist in coll:
        for filename in filelist:
            if filename.endswith('jpg'):
                flist.append(os.path.join(path, filename))

    for image_path in flist:
        img = misc.imread(image_path)
        b, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = b.shape[0]  # 人脸数目
        print('找到人脸数目为：{}'.format(nrof_faces))

        for i in b:
            i = i.astype(int)
            crop = img[i[1]:i[3],i[0]:i[2],]
            try:
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            except:
                u=1
            else:
                crop=cv2.resize(crop,(160,160))
                cv2.imwrite("E:/renlianshibie/faceImageGray/"+na+"/"+str(ii)+".jpg",crop)
                cv2.imshow("1",crop)
                cv2.waitKey(1)
                ii+=1

