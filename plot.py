import matplotlib.pyplot as plt
import xarray as xr
import numpy as np


ds = xr.open_dataset("./extra_mooring.nc")
l_ar_x = []
l_ar_y = []
l_ar_z = []
times = [0, 30, 60, 90, 120]

for t in times:
	u_t = ds["u"].sel(time=t, method="nearest")
	ar = np.array(u_t).reshape(-1, 6)
	print(f"u for t={t}:")
	ar_ux = ar[:, 0]
	ar_uy = ar[:, 1]
	ar_uz = ar[:, 2]

	# only keep first 200 points (or fewer if array shorter)
	ar_ux = ar_ux[:800]
	ar_uy = ar_uy[:800]
	ar_uz = ar_uz[:800]

	l_ar_x.append(ar_ux)
	l_ar_y.append(ar_uy)
	l_ar_z.append(ar_uz)

fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

for i, t in enumerate(times):
	axes[0].plot(l_ar_x[i], label=f"t={t}")
	axes[1].plot(l_ar_y[i], label=f"t={t}")
	axes[2].plot(l_ar_z[i], label=f"t={t}")

axes[0].set_title("l_ar_x")
axes[1].set_title("l_ar_y")
axes[2].set_title("l_ar_z")

for ax in axes:
	ax.set_ylabel("Value")
	ax.legend()
	ax.grid(True)

axes[2].set_xlabel("Index")
plt.tight_layout()
plt.show()
plt.savefig('results_extra_mooring.png')


