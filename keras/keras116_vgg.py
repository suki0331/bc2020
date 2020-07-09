from keras.applications import VGG16, VGG19, Xception, ResNet101, ResNet101V2, ResNet152
from keras.applications import ResNet152V2, ResNet50, ResNet50V2, InceptionV3, InceptionResNetV2, NASNetMobile
from keras.applications import MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201,NASNetLarge


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam


applications = [VGG19, Xception, ResNet101, ResNet101V2, ResNet152,ResNet152V2, ResNet50, 
                ResNet50V2, InceptionV3, InceptionResNetV2,MobileNet, MobileNetV2, 
                DenseNet121, DenseNet169, DenseNet201]

for i in applications:
    take_model = i()

model = NASNetLarge()
model = NASNetMobile()

'''
vgg16 = VGG16()     # (None, 224, 224, 3)

vgg16.summary()

model = Sequential()
model.add(vgg16)

# model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
'''