from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt

from keras.preprocessing.image import load_img


img_dog = load_img('./DATA/dog_cat/dog.jpg', target_size=(224, 224))
img_cat = load_img('./DATA/dog_cat/cat.jpg', target_size=(224, 224))
img_suit = load_img('./DATA/dog_cat/suit.jpg', target_size=(224, 224))
img_onion = load_img('./DATA/dog_cat/onion.jpg', target_size=(224, 224))

plt.imshow(img_suit)
plt.imshow(img_onion)
plt.imshow(img_dog)
plt.imshow(img_cat)
# plt.show()

from keras.preprocessing.image import img_to_array

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_onion = img_to_array(img_onion)
arr_suit = img_to_array(img_suit)

print(arr_dog)
print(type(arr_dog))
print(arr_dog.shape)

# RGB -> BGR
from keras.applications.vgg16 import preprocess_input

# 데이터 전처리
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_suit = preprocess_input(arr_suit)
arr_onion = preprocess_input(arr_onion)

print(arr_dog)

# 이미지를 하나로 합친다.
import numpy as np 
arr_input = np.stack([arr_dog, arr_cat, arr_onion, arr_suit])
print(arr_input.shape)  # (4, 224, 224, 3)

# 모델 구성
model = VGG16()
probs = model.predict(arr_input)

print(probs)
'''
[[4.8219149e-09 1.6681724e-07 6.7962125e-10 ... 1.9896062e-08
  1.1708904e-06 3.0176499e-05]
 [2.9127793e-07 2.7100705e-07 8.9896370e-05 ... 7.6635370e-06
  1.3547397e-04 2.1124631e-03]
 [2.0978274e-07 2.9952767e-07 4.6800531e-07 ... 7.9884029e-07
  6.9449170e-06 7.9233039e-05]
 [3.8404458e-08 8.8637086e-07 1.1326568e-06 ... 3.6307114e-08
  1.2270691e-05 7.8003234e-07]]
'''
print('probs.shape: ', probs.shape) # probs.shape:  (4, 1000)

# 이미지 결과
from keras.applications.vgg16 import decode_predictions

results = decode_predictions(probs)

print('-------------------')
print(results[0])
print('-------------------')
print(results[1])
print('-------------------')
print(results[2])
print('-------------------')
print(results[3])
# decode_predictions 하면 이렇게 됌.
'''
 VGG16
[('n02112018', 'Pomeranian', 0.99160516), ('n02086079', 'Pekinese', 0.0023396492), ('n02113624', 'toy_poodle', 0.0010944789), ('n02085620', 'Chihuahua', 0.00065595587), ('n02112350', 'keeshond', 0.0006173576)]
-------------------
[('n02342885', 'hamster', 0.2402213), ('n02441942', 'weasel', 0.21489294), ('n02909870', 'bucket', 0.13900372), ('n02443484', 'black-footed_ferret', 0.099000424), ('n02124075', 'Egyptian_cat', 0.02320484)]
-------------------
[('n03404251', 'fur_coat', 0.53184867), ('n03877472', 'pajama', 0.1044571), ('n04584207', 'wig', 0.10424174), ('n04325704', 'stole', 0.03015583), ('n03980874', 'poncho', 0.030063394)]
-------------------
[('n03888257', 'parachute', 0.96147615), ('n03388043', 'fountain', 0.0048395456), ('n10565667', 'scuba_diver', 0.0029878255), ('n04552348', 'warplane', 0.002623788), ('n03903868', 'pedestal', 0.0018622425)]

VGG19
[('n02112018', 'Pomeranian', 0.94120306), ('n02086079', 'Pekinese', 0.015324775), ('n04399382', 'teddy', 0.010623759), ('n02113624', 'toy_poodle', 0.003585076), ('n07930864', 'cup', 0.002246619)]
-------------------
[('n02441942', 'weasel', 0.2807945), ('n02443484', 'black-footed_ferret', 0.225002), ('n02342885', 'hamster', 0.0663418), ('n02124075', 'Egyptian_cat', 0.052604407), ('n02443114', 'polecat', 0.038919427)]
-------------------
[('n03404251', 'fur_coat', 0.14603388), ('n03877472', 'pajama', 0.09541144), ('n07248320', 'book_jacket', 0.088964), ('n03617480', 'kimono', 0.044973023), ('n06359193', 'web_site', 0.041755933)]
-------------------
[('n03888257', 'parachute', 0.21474329), ('n03388043', 'fountain', 0.10137556), ('n04228054', 'ski', 0.069416404), ('n10565667', 'scuba_diver', 0.057921495), ('n03903868', 'pedestal', 0.030797953)]
'''