from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale mode
img = Image.open('moon.bmp').convert('L')
p = np.array(img)

# Define the number of bins to use for quantization
num_bins = 128

# Compute the histogram of the image
hist = list(np.zeros(256,dtype=int))
for x in p:
    for y in x:
        hist[y]= hist[y] + 1

# Quantize the histogram by dividing it into bins
bin_size = 256 // num_bins
quantized_hist = [sum(hist[i*bin_size:(i+1)*bin_size]) for i in range(num_bins)]

# Compute the cumulative distribution function (CDF) of the quantized histogram
cdf = []
cdf_normalized = []
for i in range(len(quantized_hist)):
    cdf.append(sum(quantized_hist[:i+1]))
for x in cdf:
    cdf_normalized.append((x - min(cdf)) * 255 / (max(cdf) - min(cdf)))

# Compute the equalized histogram
equalized_hist = []
for x in cdf_normalized:
    equalized_hist.append(int(round(x)))

# Compute the lookup table for the equalization
lut = [equalized_hist[i // bin_size] for i in range(256)]

# Apply the equalization to the image
img_equalized = img.point(lut)
p1 = np.array(img_equalized)

hist_eq = list(np.zeros(256,dtype=int))
# bin_size=1
for xx in p1:
    for yy in xx:
        hist_eq[yy]= hist_eq[yy] + 1
# Display the original and equalized images side by side
# img.show()
img_equalized.save("mooneq128.png")