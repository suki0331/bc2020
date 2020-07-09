from keras.applications import VGG16,VGG19, Xception, ResNet101,ResNet101V2,ResNet152
from keras.applications import ResNet152V2,ResNet50,ResNet50V2,InceptionV3,InceptionResNetV2
from keras.applications import MobileNet,MobileNetV2,DenseNet121,DenseNet169,DenseNet201
from keras.applications import NASNetLarge, NASNetMobile
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, Activation,Flatten
from keras.optimizers import Adam

# applications = [VGG19, Xception, ResNet101, ResNet101V2, ResNet152,ResNet152V2, ResNet50, 
#                 ResNet50V2, InceptionV3, InceptionResNetV2,MobileNet, MobileNetV2, 
#                 DenseNet121, DenseNet169, DenseNet201]

# for i in applications:
#     take_model = i()

# vgg16 = VGG16()
nasnetlarge = NASNetLarge()
nasnetmobile =  NASNetMobile() 