import numpy as np
import matplotlib.pyplot as plt

from scipy import misc

image = misc.ascent()  # load image from scipy
t = image.shape
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(image)
plt.show()

print('Original image size:',image.shape)

# This filter detects edges nicely
# It creates a convolution that only passes through sharp edges and straight lines.
# If all the digits in the filter don't add up to 0 or 1, you should probably do a
# weight to get it to do so, for example, if your weights are 1,1,1 1,2,1 1,1,1
# They add up to 10, so you would set a weight of .1 if you want to normalize them
#filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]
filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]   # horizontal line
#filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]  # vertical line
weight  = 1
filtered_image = np.zeros_like(image)
for x in range(1,image.shape[0]-1):
  for y in range(1,image.shape[1]-1):
      output_pixel = 0.0
      output_pixel = output_pixel + (image[x - 1, y-1] * filter[0][0])
      output_pixel = output_pixel + (image[x, y-1] * filter[0][1])
      output_pixel = output_pixel + (image[x + 1, y-1] * filter[0][2])
      output_pixel = output_pixel + (image[x-1, y] * filter[1][0])
      output_pixel = output_pixel + (image[x, y] * filter[1][1])
      output_pixel = output_pixel + (image[x+1, y] * filter[1][2])
      output_pixel = output_pixel + (image[x-1, y+1] * filter[2][0])
      output_pixel = output_pixel + (image[x, y+1] * filter[2][1])
      output_pixel = output_pixel + (image[x+1, y+1] * filter[2][2])
      output_pixel = output_pixel * weight
      if output_pixel<0:
          output_pixel=0
      if output_pixel>255:
          output_pixel=255
      filtered_image[x, y] = output_pixel

print('Filtered image size:',filtered_image.shape)
plt.gray()
plt.grid(False)
plt.imshow(filtered_image)
plt.show()

# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
maxpool_image = np.zeros((int(image.shape[0] / 2), int(image.shape[1] / 2)))
for x in range(0, filtered_image.shape[0], 2):
    for y in range(0, filtered_image.shape[0], 2):
        pixels = []
        pixels.append(filtered_image[x, y])
        pixels.append(filtered_image[x + 1, y])
        pixels.append(filtered_image[x, y + 1])
        pixels.append(filtered_image[x + 1, y + 1])
        pixels.sort(reverse=True)
        maxpool_image[int(x / 2), int(y / 2)] = pixels[0]

print('Maxpool image size:',maxpool_image.shape)
plt.gray()
plt.grid(False)
plt.imshow(maxpool_image)
# plt.axis('off')
plt.show()
