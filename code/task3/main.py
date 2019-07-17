import numpy as np
import cv2
import os

dbtype_list = os.listdir("E:/renlianshibie/faceImageGray")
for dbtype in dbtype_list:
    if os.path.isfile(os.path.join("E:/renlianshibie/faceImageGray",dbtype)):
        dbtype_list.remove(dbtype)

Data=[]#总数据
Label=[]#总标签

for i in dbtype_list:
    coll = os.walk("E:/renlianshibie/faceImageGray/" + i + "/")
    flist=[]
    for path,d,filelist in coll:
        for filename in filelist:
            if filename.endswith('jpg'):
                flist.append(os.path.join(path,filename))
    for fdir in flist:
        #img=cv2.imread(fdir,cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(fdir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(160,160))
        Data.append(np.array(img)/255)
        Label.append(i)

b=set()
num=int(len(Data)*0.8)
Dtr=[]
Ltr=[]
Dte=[]
Lte=[]
i=0
while(i<num):
    randi=np.random.randint(len(Data))
    if randi not in b:
        Dtr.append(Data[randi])
        Ltr.append(Label[randi])
        b.add(randi)
        i+=1
for i in range(len(Data)):
    if i not in b:
        Dte.append(Data[i])
        Lte.append(Label[i])

Dtr=np.array(Dtr)
Ltr=np.array(Ltr)
Dte=np.array(Dte)
Lte=np.array(Lte)
np.save("E:/renlianshibie/Data/data_tr.npy",Dtr)
np.save("E:/renlianshibie/Data/label_tr.npy",Ltr)
np.save("E:/renlianshibie/Data/data_te.npy",Dte)
np.save("E:/renlianshibie/Data/label_te.npy",Lte)