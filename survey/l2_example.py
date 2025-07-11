import torch
import math
import matplotlib.pyplot as plt

img_1 = torch.full((224, 224), 0, dtype=torch.float32) # 
img_2 = torch.full((224, 224), 2, dtype=torch.float32)

img_3 = torch.full((224, 224), 0, dtype=torch.float32)

difference = (img_2 - img_1).sum().item() # holds complete delta
diff_p_pixel = math.sqrt(difference) / (400)

img_3[112-5:112+5, 112-5:112+5] += 44.8

plt.imshow(img_1)
plt.close()
plt.imshow(img_2.int())
plt.close()
plt.imshow(img_3.int())
plt.close()

l2_1 = (img_2 - img_1).pow(2).sum().sqrt().item()
l2_2 = (img_3 - img_1).pow(2).sum().sqrt().item()

if l2_1 == l2_2:
    print('same')
else:
    print('diff')


