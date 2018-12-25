from Utility import *


batchSize = 10

train_A = load_data('G2.h5')

train_batch = minibatchAB(train_A, batchSize)

epoch, A = next(train_batch)

print(epoch)

plt.imshow(A[1, :, :, 3], cmap = 'gray')
plt.show()
