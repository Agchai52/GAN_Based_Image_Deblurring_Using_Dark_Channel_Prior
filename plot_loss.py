import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


loss_record = "loss_record.txt"

losses_dg = np.loadtxt(loss_record)
loss_d = losses_dg[:,0]
loss_g = losses_dg[:,1]
loss_p = losses_dg[:,2]

plt.figure()
plt.plot(losses_dg[0:-1:100, 0], 'r-', label='d_loss')
plt.xlabel("iteration*100")
plt.ylabel("Error")
plt.title("Discriminator Losse")
plt.savefig("plot_d_loss.jpg")

plt.figure()
plt.plot(losses_dg[0:-1:100, 1], 'g-', label='g_loss')
plt.xlabel("iteration*100")
plt.ylabel("Error")
plt.title("Generator Losses")
plt.savefig("plot_g_loss.jpg")

plt.figure()
plt.plot(losses_dg[0:-1:100, 2], 'b-', label='dc_loss')
plt.xlabel("iteration*100")
plt.ylabel("Error")
plt.title("DarkChannel Losses")
plt.savefig("plot_dc_loss.jpg")
plt.show()


