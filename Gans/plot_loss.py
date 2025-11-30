import json
import matplotlib.pyplot as plt
losses="losses.json"
with open(losses,'r') as f:
    data=json.load(f)
loss_G=data['loss_G']
loss_D=data['loss_D']

plt.plot(loss_G,label="Generator Loss")
plt.plot(loss_D,label="Discriminator Loss")
plt.legend()
plt.show()