import imageio
import os
images = []
filenames=[fn for fn in os.listdir('.') if fn.endswith('.png')]
filenames.sort()
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('prediction.gif', images, duration=0.05)
