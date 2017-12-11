
# coding: utf-8

# In[3]:


import tensorflow as tf
import librosa
import os
from IPython.display import Audio, display
import numpy as np
import matplotlib.pyplot as plt


# ### Load style and content

# In[4]:


CONTENT_FILENAME = "inputs/imperial.mp3"
STYLE_FILENAME = "inputs/nwa.mp3"


# In[5]:


#display(Audio(CONTENT_FILENAME))
#display(Audio(STYLE_FILENAME))


# In[6]:


# Reads wav file and produces spectrum
# Fourier phases are ignored
N_FFT = 2048
def read_audio_spectum(filename):
    x, fs = librosa.load(filename)
    S = librosa.stft(x, N_FFT)
    S = np.log1p(np.abs(S[:,:430]))  
    return S, fs


def mel_spec(S):
    return librosa.feature.melspectrogram(S=S)


a_content, fs = read_audio_spectum(CONTENT_FILENAME)
a_style, fs = read_audio_spectum(STYLE_FILENAME)

mel_content = mel_spec(a_content)
mel_style = mel_spec(a_style)

N_SAMPLES = a_content.shape[1]
N_CHANNELS = a_content.shape[0]
a_style = a_style[:N_CHANNELS, :N_SAMPLES]


N_SAMPLES_MEL = mel_content.shape[1]
N_CHANNELS_MEL = mel_content.shape[0]
mel_style = mel_style[:N_CHANNELS_MEL, :N_SAMPLES_MEL]
# ### Visualize spectrograms for content and style tracks

# In[ ]:

"""
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Content')
plt.imshow(a_content[:400,:])
plt.subplot(1, 2, 2)
plt.title('Style')
plt.imshow(a_style[:400,:])
plt.show()
"""

# ### Compute content and style feats

N_FILTERS = 4096
N_FILTERS_MEL = 1025

FILTER_WIDTH = 11
MEL_FILTER_WIDTH = 50

a_content_tf = np.ascontiguousarray(a_content.T[None,None,:,:])
a_style_tf = np.ascontiguousarray(a_style.T[None,None,:,:])

mel_content_tf = np.ascontiguousarray(mel_content.T[None,None,:,:])
mel_style_tf = np.ascontiguousarray(mel_style.T[None,None,:,:])

# filter shape is "[filter_height, filter_width, in_channels, out_channels]"
std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
kernel = np.random.randn(1, FILTER_WIDTH, N_CHANNELS, N_FILTERS)*std

std_mel = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS_MEL + N_FILTERS_MEL) * 11))
kernel_mel = np.random.randn(1, MEL_FILTER_WIDTH, N_CHANNELS_MEL, N_FILTERS_MEL)*std


g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    # data shape is "[batch, in_height, in_width, in_channels]",
    x = tf.placeholder('float32', [1,1,N_SAMPLES,N_CHANNELS], name="x")

    # convert back to 2d array
    x_as_np = np.squeeze(x)
    mel_x = mel_spec(x)
    mel_x = np.ascontiguousarray(mel_x.T[None,None,:,:])

    # STFT Net
    kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
    conv = tf.nn.conv2d(
        x,
        kernel_tf,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")

    net = tf.nn.relu(conv)

    # now do mel net
    kernel_tf_mel = tf.constant(kernel_mel, name="kernel_mel", dtype='float32')
    conv_mel = tf.nn.conv2d(
        mel_x,
        kernel_tf_mel,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv_mel")

    mel_net = tf.nn.relu(conv_mel)



    content_features = net.eval(feed_dict={x: a_content_tf})
    style_features = net.eval(feed_dict={x: a_style_tf})
    
    features = np.reshape(style_features, (-1, N_FILTERS))
    style_gram = np.matmul(features.T, features) / N_SAMPLES


# ### Optimize

# In[ ]:


from sys import stderr

ALPHA= 1e-2
learning_rate= 1e-3
iterations = 100

result = None
with tf.Graph().as_default():

    # Build graph with variable input
#     x = tf.Variable(np.zeros([1,1,N_SAMPLES,N_CHANNELS], dtype=np.float32), name="x")
    x = tf.Variable(np.random.randn(1,1,N_SAMPLES,N_CHANNELS).astype(np.float32)*1e-3, name="x")

    kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
    conv = tf.nn.conv2d(
        x,
        kernel_tf,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    
    
    net = tf.nn.relu(conv)

    content_loss = ALPHA * 2 * tf.nn.l2_loss(
            net - content_features)

    style_loss = 0

    _, height, width, number = map(lambda i: i.value, net.get_shape())

    size = height * width * number
    feats = tf.reshape(net, (-1, number))
    gram = tf.matmul(tf.transpose(feats), feats)  / N_SAMPLES
    style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

     # Overall loss
    loss = content_loss + style_loss

    opt = tf.contrib.opt.ScipyOptimizerInterface(
          loss, method='L-BFGS-B', options={'maxiter': 300})
        
    # Optimization
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
       
        print('Started optimization.')
        opt.minimize(sess)
    
        print 'Final loss:', loss.eval()
        result = x.eval()


# ### Invert spectrogram and save the result

# In[ ]:


a = np.zeros_like(a_content)
a[:N_CHANNELS,:] = np.exp(result[0,0].T) - 1

# This code is supposed to do phase reconstruction
p = 2 * np.pi * np.random.random_sample(a.shape) - np.pi
for i in range(500):
    S = a * np.exp(1j*p)
    x = librosa.istft(S)
    p = np.angle(librosa.stft(x, N_FFT))

OUTPUT_FILENAME = 'outputs/out.wav'
librosa.output.write_wav(OUTPUT_FILENAME, x, fs)


# In[ ]:


## print OUTPUT_FILENAME
display(Audio(OUTPUT_FILENAME))


# ### Visualize spectrograms

# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title('Content')
plt.imshow(a_content[:400,:])
plt.subplot(1,3,2)
plt.title('Style')
plt.imshow(a_style[:400,:])
plt.subplot(1,3,3)
plt.title('Result')
plt.imshow(a[:400,:])
plt.show()


# In[ ]:


plt.show()

