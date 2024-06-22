#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Kod bloklarında fonksiyon tanımlanmasından öncesinde kullanıldığı için ilk çalıştırmada hata verecektir. 
#Lütfen iki kez çalıştırınız.
import os
import matplotlib.pyplot as plt 
import cv2 #opencv kütüphanesi
import numpy as np
import pandas as pd


# In[6]:


veri = os.listdir("veriseti")


# In[7]:


#İmg öznitelklerini çıkarmak için bu kod bloğunu kullanıldı.
for image_url in veri:
    img = cv2.imread("veriseti/"+image_url)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(500,500)) #resimler aynı boyutta olursa plakalar hangi boyutlarda görünür anlamak için yapıldı.
    plaka=plaka_konum_don(img) 
    x,y,w,h=plaka
    if(w>h):
        plaka_bgr=img[y:y+h,x:x+w].copy()
    else:                
        plaka_bgr=img[y:y+w,x:x+h].copy()
    img = cv2.cvtColor(plaka_bgr,cv2.COLOR_BGR2RGB) 
    plt.imshow(img)
    plt.show() 


# In[8]:


for image_url in veri:
    img = cv2.imread("veriseti/"+image_url)
    img = cv2.resize(img,(500,500)) #resimler aynı boyutta olursa plakalar hangi boyutlarda görünür anlamak için yapıldı.
    img_bgr = img
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Resimleri grileştiriyorildi.
    plt.imshow(img_gray, cmap="gray")
    plt.show()


# In[9]:


#Gürültü eleme işlemi ve kenarlık tespiti yapıldı
#Median bulanıklaştırma işlemi ile kenarlıkları netleştirildi.
for image_url in veri:
    img = cv2.imread("veriseti/"+image_url)
    img = cv2.resize(img,(500,500)) #Resimler aynı boyutta olursa plakalar hangi boyutlarda görünür anlamak için yapıldı.
    img_bgr = img
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Resimleri grileştirildi.
    ir_img = cv2.medianBlur(img_gray,5)
    ir_img = cv2.medianBlur(ir_img,5)
    #plt.imshow(img_gray, cmap="gray")
    #plt.show()
    
    medyan = np.median(ir_img)
    low = 0.67*medyan
    high = 1.33*medyan
    
    kenarlik = cv2.Canny(ir_img, low, high)
    plt.imshow(kenarlik, cmap="gray")
    plt.show()

    #Genişletme - erozyon işlemi yapıldı.
    kenarlik = cv2.dilate(kenarlik,np.ones((3,3),np.uint8), iterations=1)
    plt.imshow(kenarlik, cmap="gray")
    plt.show()
    


# In[10]:


#Contours bulma işlemi yapıldı.
#cv2.RETR_TREE hiyerarşik yapıda gelmesi için kullanıldı. 
#cv2.CHAİN_APPROX_SİMPLE tüm pikseller yerine sadece köşegenler alındı.
cnt = cv2.findContours(kenarlik,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnt = cnt[0]
cnt = sorted(cnt, key = cv2.contourArea,reverse=True)
H,W = 500,500
plaka = None

for c in cnt: 
    rect = cv2.minAreaRect(c) #dikdörtgen yapıda alındı.
    (x,y),(w,h),r=rect
    if(w>h and w>h*2) or (h>w and h>w*2): #oranın iki olması için bu şekilde tanımlandı.
        box= cv2.boxPoints(rect)
        box=np.int64(box) 
        
        minx= np.min(box[:,0])
        miny=np.min(box[:,1])
        maxx=np.max(box[:,0])
        maxy=np.max(box[:,1])
        
        muh_plaka = img_gray[miny:maxy, minx:maxx].copy()  #İmgelerin aslı bozulmaması için kopyası kullanıldı.
        muh_medyan= np.median(muh_plaka)
        
        kon1 = muh_medyan>85 and muh_medyan<200 #medyan değer kontrolü (yoğunluk kontrolü) yapıldı.
        kon2 = h<50 and w<150 #sınır kontrolü yapıldı.
        kon3 = w<50 and h<150 #sınır kontrolü yapıldı.
        
        print(f"muh_plaka medyan:{muh_medyan} genislik: {w} yükseklik: {h}")
        plt.figure()
        kon=False
        if(kon1 and (kon2 or kon3)):
            #plakadır.
            cv2.drawContours(img,[box],0,(0,255,0),2) 
            plaka = [minx,miny,w,h]
            
            plt.title("Plaka tespit edildi.")
            kon=True
        else:
            #plaka değildir
            cv2.drawContours(img,[box],0,(0,0,255),2)
            plt.title("Plaka tespit edilemedi.")
        
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.show()
        
        if(kon):
            break
#plaka bulunmuştur.   


# In[11]:


def plaka_konum_don(img):

    img_bgr = img
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    

    ir_img = cv2.medianBlur(img_gray,5) #5x5 kernel boyutu kullanıldı.
    ir_img = cv2.medianBlur(ir_img,5)  

    medyan = np.median(ir_img)

    low = 0.67*medyan
    high = 1.33*medyan

    #Canny algoritması kullanıldı. 
    kenarlik = cv2.Canny(ir_img,low,high)
    kenarlik = cv2.dilate(kenarlik,np.ones((3,3),np.uint8),iterations=1)
    
    
    #CHAIN_APPROX_SIMPLE -> Tüm pikseller yerine köşegenleri alındı.
    cnt = cv2.findContours(kenarlik,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0]
    cnt = sorted(cnt,key=cv2.contourArea,reverse=True)

    H,W = 500,500
    plaka = None

    for c in cnt:
        rect = cv2.minAreaRect(c) 
        (x,y),(w,h),r = rect
        if(w>h and w>h*2) or (h>w and h>w*2):
            box = cv2.boxPoints(rect) 
            box = np.int64(box)

            minx = np.min(box[:,0])
            miny = np.min(box[:,1])
            maxx = np.max(box[:,0])
            maxy = np.max(box[:,1])


            muh_plaka = img_gray[miny:maxy,minx:maxx].copy()
            muh_medyan = np.median(muh_plaka)

            
            kon1 = muh_medyan>84 and muh_medyan<200 
            kon2 = h<50 and w<150 
            kon3 = w<50 and h<150 

            print(f"muh_plaka medyan:{muh_medyan} genislik: {w} yukseklik:{h}")

            kon=False
            if(kon1 and (kon2 or kon3)):
                #plaka'dır
                plaka =[int(i) for i in [minx,miny,w,h]]#x,y,w,h
                kon=True
            else:
                #plaka değidir
                #cv2.drawContours(img,[box],0,(0,0,255),2)
                pass
            if(kon):
                return plaka
    return []


# In[12]:


plaka_konumu=plaka_konum_don(img)


# In[13]:


#Karakter ayrıştırma işlemi yapıldı.
veriler = os.listdir("veriseti")
isim=veriler[1]
img=cv2.imread("veriseti/"+isim)
img = cv2.resize(img,(500,500))
plaka = plaka_konum_don(img)
x,y,w,h = plaka
if(w>h):
        plaka_bgr=img[y:y+h,x:x+w].copy()
else:                
        plaka_bgr=img[y:y+w,x:x+h].copy()

#Piksellerin giderilmesi için daha geniş bir görüntü alındı.
H,W = plaka_bgr.shape[:2]
print("Orjinal boyut: ", W,H)
H,W=H*2,W*2 
print("Yeni boyut: ", W,H)
plaka_bgr = cv2.resize(plaka_bgr, (W,H))
plt.imshow(plaka_bgr)
plt.imshow

#Renkler kullanılmadığı için grileştirme işlemi yapıldı.
plaka_resim = cv2.cvtColor(plaka_bgr,cv2.COLOR_BGR2GRAY)
plt.title("Gri Format")
plt.imshow(plaka_resim,cmap="gray")
plt.imshow

#Threshold işlemi ve ortalama algoritması kullanıldı.
th_img = cv2.adaptiveThreshold(plaka_resim,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
plt.title("Eşiklenmiş Format")
plt.imshow(th_img,cmap="gray")
plt.imshow

#Open (açma) işlemi ile gürültüler yok edildi. 
kernel=np.ones((3,3),np.uint8)
th_img=cv2.morphologyEx(th_img, cv2.MORPH_OPEN, kernel, iterations = 1)
plt.title("Gürültü yok edildi")
plt.imshow(th_img,cmap="gray")
plt.imshow

#Contours işlemi yapıldı. 
cnt=cv2.findContours(th_img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt=cnt[0]
cnt=sorted(cnt,key=cv2.contourArea,reverse=True)[:15]

for i,c in enumerate(cnt):
    rect= cv2.minAreaRect(c)
    (x,y),(w,h),r=rect
    
    kon1= max([w,h]) <W/4
    kon2 = w*h>200
    
    if(kon1 and kon2):
        print("Karakter ->",x,y,w,h)
        
        box = cv2.boxPoints(rect)
        box= np.int64(box)
        
        minx = np.min(box[:,0])
        miny = np.min(box[:,1])
        maxx = np.max(box[:,0])
        maxy = np.max(box[:,1])
        
        odak=2
        minx = max(0,minx-odak)
        miny = max(0,miny-odak)
        maxx = min(W,maxx+odak)
        maxy = min(H,maxy+odak)
        
        kesim = plaka_bgr[miny:maxy,minx:maxx].copy()
        
        try:
            cv2.imwrite("karakterseti/{isim}_{i}.jpg",kesim)
        except:   
            pass
        yaz=plaka_bgr.copy()
        cv2.drawContours(yaz,[box],0,(0,255,0),1)
        plt.imshow(yaz)
        plt.show()


# In[14]:


import pickle 
import os
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 

path = "karakterseti/"
siniflar =  [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
tek_batch = 0

urls = []
sinifs = []

print("Veriler okunuyor.")

for sinif in siniflar:
    sinif_path=os.path.join(path,sinif)
    resimler = os.listdir(sinif_path)
    for resim in resimler:
        urls.append(path+sinif+"/"+resim)
        sinifs.append(sinif)
        tek_batch+=1
        
        
df = pd.DataFrame({"adres":urls,"sinif":sinifs})
        


def islem(img):
    yeni_boy=img.reshape((1600,5,5))
    orts=[]
    for parca in yeni_boy:
        ort=np.mean(parca)
        orts.append(ort)
    orts=np.array(orts)   
    orts = orts.reshape(1600,)
    return orts

#ön işleme işlemi yapıldı.
def on_isle(img):
    return img/255

target_size=(200,200)
batch_size= tek_batch


#Veri setlerini oluşturuldu ve eğitildi.
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
  preprocessing_function = on_isle)

train_set=train_gen.flow_from_dataframe(df,x_col="adres",
                                        y_col="sinif",
                                       target_size=target_size,
                                        color_mode="grayscale",
                                        shuffle=True,
                                        class_mode="sparse",
                                        batch_size=batch_size)

images, train_y = next(train_set)
train_x=np.array(list(map(islem,images))).astype("float32")
train_y=train_y.astype(int)

print("Random forest /Rassal orman eğitiliyor")

rfc = RandomForestClassifier(n_estimators=10,criterion="entropy")
rfc.fit(train_x,train_y)
print("Eğitildi")
pred= rfc.predict(train_x)
acc = accuracy_score(pred,train_y)
print("Başarılı:",acc)

dosya="rfc_model.rfc"
pickle.dump(rfc,open(dosya,"wb")) #byte olarak yazmamızı sağladı.


# In[15]:


#Model test edildi.
def islem(img):
    yeni_boy=img.reshape((1600,5,5))
    orts=[]
    for parca in yeni_boy:
        ort=np.mean(parca)
        orts.append(ort)
    orts=np.array(orts)   
    orts = orts.reshape(1600,)
    return orts

path = "karakterseti/"
siniflar =  [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
tek_batch = 0

urls = []
sinifs = []

print("Veriler okunuyor.")

for sinif in siniflar:
    sinif_path=os.path.join(path,sinif)
    resimler = os.listdir(sinif_path)
    for resim in resimler:
        urls.append(path+sinif+"/"+resim)
        sinifs.append(sinif)
        tek_batch+=1
        
        
df = pd.DataFrame({"adres":urls,"sinif":sinifs})

sinifs = { '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'A':10,
          'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'I':18,'J':19,'K':20,
          'L':21,'M':22,'N':23,'O':24,'P':25,'Q':26,'R':27,'S':28,'T':29,'U':30,
          'V':31,'W':32,'X':33,'Y':34,'Z':35,'arkaplan':36  
}

dosya = "rfc_model.rfc"
rfc = pickle.load(open(dosya,"rb"))
index=list(sinifs.values())
siniflar = list(sinifs.keys())
df = df.sample(frac=1)

for adres,sinif in df.values: 
    image = cv2.imread(adres,0)
    resim=cv2.resize(image,(200,200))
    resim = resim/255
    oznitelikler=islem(resim)
    
    sonuc = rfc.predict([oznitelikler])[0]
    print("sonuc",sonuc)
    
    ind = index.index(sonuc)
    sinif=siniflar[ind]
    plt.imshow(resim,cmap="gray")
    plt.title(f"fotoğraftaki karakter: {sinif}")
    plt.show()


# In[16]:


import pickle
dosya="rfc_model.rfc"
rfc= pickle.load(open(dosya,"rb"))

sinifs = { '0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'A':10,
          'B':11,'C':12,'D':13,'E':14,'F':15,'G':16,'H':17,'I':18,'J':19,'K':20,
          'L':21,'M':22,'N':23,'O':24,'P':25,'Q':26,'R':27,'S':28,'T':29,'U':30,
          'V':31,'W':32,'X':33,'Y':34,'Z':35,'arkaplan':36  
}

index = list(sinifs.values())
siniflar=list(sinifs.keys())

def islem(img):
    yeni_boy=img.reshape((1600,5,5))
    orts=[]
    for parca in yeni_boy:
        ort=np.mean(parca)
        orts.append(ort)
    orts=np.array(orts)   
    orts = orts.reshape(1600,)
    return orts

#Plaka için sıralama işlemi yapıldı.
def plakaAyristir(mevcutPlaka):
    mevcutPlaka= sorted(mevcutPlaka,key= lambda x:x[1]) 
    mevcutPlaka = np.array(mevcutPlaka)
    mevcutPlaka=mevcutPlaka[:,0]
    mevcutPlaka=mevcutPlaka.tolist()
    
    #Plaka her zaman sayı ile başlar. Plakanın başındaki rakamların tespitinin kontrol kısmı yapıldı.
    karakterAdim=0
    for i in range(len(mevcutPlaka)):
        try:
            int(mevcutPlaka[i])
            karakterAdim+=1
        except:
            if karakterAdim>0:
                if i-2>=0:
                    mecvutPlaka = mevcutPlaka[i-2] 
                break          
            mevcutPlaka.pop(i)  
    
    karakterAdim=0
    #Plaka her zaman sayı ile biter. Plakanın sonundaki rakamların tespitinin kontrol kısmı yapıldı.
    for i in range(len(mevcutPlaka)):
        kontrolIndex = -1+(-1*karakterAdim)
        try:
            int(mevcutPlaka[kontrolIndex])
            karakterAdim+=1
        except:
            if karakterAdim>0:
                karIndex = len(mevcutPlaka)-karakterAdim
                mevcutPlaka = mevcutPlaka[:karIndex+4]
                break    
            mevcutPlaka.pop(kontrolIndex)
    return mevcutPlaka    

def plakaTani(img,plaka):
    x,y,w,h = plaka
    if(w>h):
        plaka_bgr=img[y:y+h,x:x+w].copy()
    else:                
        plaka_bgr=img[y:y+w,x:x+h].copy()

    H,W = plaka_bgr.shape[:2]
    H,W=H*2,W*2 
   
    plaka_bgr = cv2.resize(plaka_bgr, (W,H))
    plaka_resim = cv2.cvtColor(plaka_bgr,cv2.COLOR_BGR2GRAY)
    plt.title("Gri Format")
    th_img = cv2.adaptiveThreshold(plaka_resim,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
    kernel=np.ones((3,3),np.uint8)
    th_img=cv2.morphologyEx(th_img, cv2.MORPH_OPEN, kernel, iterations = 1)
    cnt=cv2.findContours(th_img,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt=cnt[0]
    cnt=sorted(cnt,key=cv2.contourArea,reverse=True)[:15]
    yaz=plaka_bgr.copy()
    mevcutPlaka = []
   
    for i,c in enumerate(cnt):
        rect= cv2.minAreaRect(c)
        (x,y),(w,h),r=rect
    
        kon1= max([w,h]) <W/4
        kon2 = w*h>200
    
        if(kon1 and kon2):
            print("Karakter ->",x,y,w,h)
            box = cv2.boxPoints(rect)
            box= np.int64(box)
        
            minx = np.min(box[:,0])
            miny = np.min(box[:,1])
            maxx = np.max(box[:,0])
            maxy = np.max(box[:,1])
        
            odak=2
            minx = max(0,minx-odak)
            miny = max(0,miny-odak)
            maxx = min(W,maxx+odak)
            maxy = min(H,maxy+odak)
        
            kesim = plaka_bgr[miny:maxy,minx:maxx].copy()
            tani = cv2.cvtColor(kesim,cv2.COLOR_BGR2GRAY)
            tani = cv2.resize(tani,(200,200))
            tani = tani/255
            oznitelikler= islem(tani)
            karakter = rfc.predict([oznitelikler])
            ind = index.index(karakter)
            sinif=siniflar[ind]
            if sinif =="arkaplan":
                continue
            
            mevcutPlaka.append([sinif,minx])
            cv2
            cv2.putText(yaz,sinif,(minx-2,miny-2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,(0,255,0),1)
            cv2.drawContours(yaz,[box],0,(0,255,0),1)
    if len(mevcutPlaka)>0:
        mevcutPlaka = plakaAyristir(mevcutPlaka)
    return yaz, mevcutPlaka  


# In[19]:


def plakaAyristir(mevcutPlaka):
    karakterAdim = 0
    while karakterAdim < len(mevcutPlaka):
        try:
            int(''.join(map(str, mevcutPlaka[karakterAdim])))
            karakterAdim += 1
        except ValueError:
            mevcutPlaka.pop(karakterAdim)

    karakterAdim = 0
    while karakterAdim < len(mevcutPlaka):
        try:
            int(''.join(map(str, mevcutPlaka[karakterAdim])))
            if len(mevcutPlaka) > karakterAdim + 2:
                mevcutPlaka = mevcutPlaka[:karakterAdim + 2]
            break
        except ValueError:
            karakterAdim += 1
    return mevcutPlaka

veriler = os.listdir("veriseti")

for dosya in veriler:
    img = cv2.imread("veriseti/" + dosya)
    img = cv2.resize(img, (500, 500))

    plaka = plaka_konum_don(img)
    plakaImg, plakaKarakter = plakaTani(img, plaka)

    print("img name:", dosya)
    print("arabanın plakası:", plakaKarakter)

    plt.imshow(cv2.cvtColor(plakaImg, cv2.COLOR_BGR2RGB))
    plt.show()


# In[ ]:




