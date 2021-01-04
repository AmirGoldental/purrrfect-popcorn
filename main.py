# %%
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import spectrogram
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from modest_image import ModestImage, imshow
import plotly.express as px

# %%
file_wav = read("popcorn.wav")
fs = file_wav[0]
recorded_signal = file_wav[1]
plt.plot(recorded_signal)
# cut area of interest
recorded_signal = recorded_signal[1700000:3400000, :]
plt.figure()
plt.plot(recorded_signal)
# zoom
plt.figure()
plt.plot(recorded_signal)
plt.xlim([250000, 255000])
# %% use pca to get a good signal
pca_singal = PCA().fit_transform(recorded_signal)
plt.figure()
plt.plot(pca_singal)
plt.xlim([250000, 255000])
signal = pca_singal[:, 1]
# %%
f, t, Sxx = spectrogram(signal, fs=fs)
fig = px.imshow(Sxx, zmax=100)
fig.update_xaxes(
    scaleanchor="x",
    scaleratio=0.1,
)
fig.show()
# %%
print("a")
