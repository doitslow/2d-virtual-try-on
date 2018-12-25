import h5py as h5
import numpy as np
from matplotlib import pyplot as plt

filename = 'G2.h5'

with h5.File(filename, 'r') as f:
    # train_images = f['ih'][:100]
    # print(train_images.shape)
    # train_images = train_images.transpose(0, 3 , 2, 1)
    # print(train_images.shape)
    # peak_on = train_images[5]
    # plt.imshow(peak_on)
    # plt.show()

    #print(peak_on[0,:,:])

    for i in f.keys():
        print(i)

    b_ = f['b_'][:100]
    b_ = b_.transpose(0, 3, 2, 1)
    print(b_.shape)
    #print(np.unique(b_))

    peak = b_[34, :, :, :]
    peak = peak.reshape(128, 128)
    peak[peak!=3] = 0
    peak[peak==3] = 255
    plt.imshow(peak, cmap = 'gray')
    plt.show()
    print(peak)

    # ih_mean = f['ih_mean'][:]
    # print(ih_mean.shape)
    # ih_mean = ih_mean.transpose(2, 1, 0)
    # plt.imshow(ih_mean)
    # plt.show()

# code to find out what crop do to the image

# image_before = ih[30]
# plt.figure(2)
# plt.imshow(image_before)
# plt.show()
#
# image_after = get_image(ih[30],
#                         input_height = 108,
#                         input_width = 108,
#                         resize_height = 64,
#                         resize_width = 64,
#                         crop = True,
#                         grayscale = False)
# plt.imshow(image_after)
# plt.show()
