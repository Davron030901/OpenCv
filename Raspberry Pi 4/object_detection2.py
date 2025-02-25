import cv2
import numpy as np
from os import listdir

# Create image window
def display(name, image):
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()

# Preprocess images
def preprocess(img_dir, img_file):
    img = cv2.imread(img_dir + '/' + img_file, cv2.IMREAD_COLOR)
    # Rasmni standart o'lchamga keltirish
    img = cv2.resize(img, (64, 128))  # HOG uchun standart o'lcham
    img = img.astype(np.float64) - np.mean(img)
    img /= np.std(img)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return img

# HOG detector sozlamalari
win_size = (64, 128)  # Standart o'lcham
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

# HOG descriptorni yaratish
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)

try:
    # Training rasmlarini o'qish
    train_features = []
    train_labels = []
    img_dir = 'train_img'
    
    if not listdir(img_dir):
        raise Exception(f"'{img_dir}' papkasi bo'sh yoki mavjud emas")
        
    for img_file in listdir(img_dir):
        img = preprocess(img_dir, img_file)
        train_features.append(hog.compute(img))
        train_labels.append(int(img_file.split('_')[0]))

    # SVM ni yaratish va o'qitish
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.train(np.array(train_features), cv2.ml.ROW_SAMPLE, np.array(train_labels))
    svm.save('svm.xml')

    # Detektorni sozlash
    vec = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    vec = np.append(vec, -rho)
    hog.setSVMDetector(vec)

    # Test rasmlarini tekshirish
    img_dir = 'test_img'
    for img_file in listdir(img_dir):
        img = preprocess(img_dir, img_file)
        res = hog.detectMultiScale(img)
        
        if len(res[0]) > 0:
            for (x, y, w, h) in res[0]:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        display('Img', img)

except Exception as e:
    print(f"Xatolik yuz berdi: {e}")