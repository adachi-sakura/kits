from unet import *
from data import *
from visualize import *
from utils import *
import SimpleITK as sitk
from keras.utils import to_categorical

datagen=ImageDataGenerator(rotation_range=0.2,
                           width_shift_range=0.05,
                           height_shift_range=0.05,
                           shear_range=0.05,
                           zoom_range=0.05,
                           horizontal_flip=True,
                           fill_mode='nearest')

itk_img=sitk.ReadImage(r'../data/case_00000/imaging.nii.gz')
itk_seg=sitk.ReadImage(r'../data/case_00000/segmentation.nii.gz')

img=sitk.GetArrayFromImage(itk_img)
seg=sitk.GetArrayFromImage(itk_seg)
seg=seg.astype(np.int32)

img=np.transpose(img)
seg=np.transpose(seg)
img=img[:,:,:,np.newaxis]
seg=seg[:,:,:,np.newaxis]
seg=to_categorical(seg)

model=unet()
model_checkpoint=ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(datagen.flow(img,seg,batch_size=32),steps_per_epoch=len(seg)/32+1,epochs=5,callbacks=[model_checkpoint])