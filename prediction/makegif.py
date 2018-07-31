import matplotlib.pyplot as plt
import imageio,os
images = []
filenames=sorted((fn for fn in os.listdir('.') if fn.endswith('.png')))
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('prediction.gif', images,duration=0.2)
