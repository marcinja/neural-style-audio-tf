
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


CONTENT_FILENAME = "inputs/spongebob.mp3"
STYLE_FILENAME = "inputs/imperial.mp3"


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

# Set up filter dimensions:
# filter shape is "[filter_height, filter_width, in_channels, out_channels]"
std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
kernel = np.random.randn(1, FILTER_WIDTH, N_CHANNELS, N_FILTERS)*std

std_mel = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS_MEL + N_FILTERS_MEL) * 11))
kernel_mel = np.random.randn(1, MEL_FILTER_WIDTH, N_CHANNELS_MEL, N_FILTERS_MEL)*std

kernel_mel_conv1 = np.random.randn(1, 1, N_FILTERS_MEL, 381)*std
kernel_mel_conv2 = np.random.randn(1, 1, 381, 381)*std
# Filters for dilated convolution taken to be smaller.
kernel_mel_dil1 = np.random.randn(1, MEL_FILTER_WIDTH / 2, N_FILTERS_MEL, N_FILTERS_MEL)*std
kernel_mel_dil2 = np.random.randn(1, MEL_FILTER_WIDTH / 2, 381, 381)*std

g = tf.Graph()
with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
    # data shape is "[batch, in_height, in_width, in_channels]",
    x = tf.placeholder('float32', [1,1,N_SAMPLES,N_CHANNELS], name="x")

    # STFT Net
    kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
    conv = tf.nn.conv2d(
        x,
        kernel_tf,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")

    net = tf.nn.selu(conv)
    content_features = net.eval(feed_dict={x: a_content_tf})
    style_features = net.eval(feed_dict={x: a_style_tf})
    features = np.reshape(style_features, (-1, N_FILTERS))
    style_gram = np.matmul(features.T, features) / N_SAMPLES

    # Convert x to mel spectrogram
    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    sample_rate = 22050 # from librosa docs
    num_spectrogram_bins = x.shape[-1].value
    lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        N_CHANNELS_MEL, N_CHANNELS, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        x, linear_to_mel_weight_matrix, 1)
    # Note: Shape inference for `tf.tensordot` does not currently handle this case.
    mel_spectrograms.set_shape(x.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    # now do mel net
    kernel_tf_mel = tf.constant(kernel_mel, name="kernel_mel", dtype='float32')
    kernel_tf_dil1 = tf.constant(kernel_mel_dil1, name="kernel_mel_dil1", dtype='float32')
    kernel_tf_dil2 = tf.constant(kernel_mel_dil2, name="kernel_mel_dil2", dtype='float32')
    kernel_tf_conv1 = tf.constant(kernel_mel_conv1, name="kernel_mel_conv1", dtype='float32')
    kernel_tf_conv2 = tf.constant(kernel_mel_conv2, name="kernel_mel_conv2", dtype='float32')
    conv_mel = tf.nn.conv2d(
        mel_spectrograms,
        kernel_tf_mel,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv_mel")

    # Add first residual block
    DILATION_RATE = 2

    first_res_shortcut = tf.nn.selu(conv_mel) 
    dilated_conv1 = tf.nn.atrous_conv2d(first_res_shortcut, kernel_tf_dil1, DILATION_RATE, padding="SAME", name="dilated_conv1")
    activated_dil1 = tf.nn.selu(dilated_conv1)
    add_residual = tf.add(first_res_shortcut, activated_dil1)
    std_conv1 = tf.nn.conv2d(
        add_residual,
        kernel_tf_conv1,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="std_conv1")

    # Second residual block for mel net
    second_res_shortcut = tf.nn.selu(std_conv1) 
    dilated_conv2 = tf.nn.atrous_conv2d(second_res_shortcut, kernel_tf_dil2, DILATION_RATE, padding="SAME", name="dilated_conv2")
    activated_dil2 = tf.nn.selu(dilated_conv2)
    add_residual2 = tf.add(second_res_shortcut, activated_dil2)
    std_conv2 = tf.nn.conv2d(
        add_residual2,
        kernel_tf_conv2,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="std_conv2")

    mel_net = tf.nn.selu(std_conv2)
    mel_content_features = mel_net.eval(feed_dict={x: a_content_tf})
 
    mel_style_features = mel_net.eval(feed_dict={x: a_style_tf})
    mel_features = np.squeeze(mel_style_features) #np.reshape(mel_style_features, (-1, N_FILTERS_MEL))
    mel_style_gram = np.matmul(mel_features.T, mel_features) / N_SAMPLES_MEL


# ### Optimize

# In[ ]:


from sys import stderr
import sys

ALPHA= 1e-2
BETA= 1e-2
learning_rate= 1e-3
iterations = 100

result = None
with tf.Graph().as_default():

    # Build graph with variable input
    x = tf.Variable(np.random.randn(1,1,N_SAMPLES,N_CHANNELS).astype(np.float32)*1e-3, name="x")

    # Convert x to mel spectrogram
    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    sample_rate = 22050 # from librosa docs
    num_spectrogram_bins = x.shape[-1].value
    lower_edge_hertz, upper_edge_hertz = 80.0, 7600.0
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        N_CHANNELS_MEL, N_CHANNELS, sample_rate, lower_edge_hertz,
    upper_edge_hertz)
    mel_spectrograms = tf.tensordot(
        x, linear_to_mel_weight_matrix, 1)
    # Note: Shape inference for `tf.tensordot` does not currently handle this case.
    mel_spectrograms.set_shape(x.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))


    kernel_tf = tf.constant(kernel, name="kernel", dtype='float32')
    conv = tf.nn.conv2d(
        x,
        kernel_tf,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    
    
    net = tf.nn.selu(conv)

    # now do mel net
    kernel_tf_mel = tf.constant(kernel_mel, name="kernel_mel", dtype='float32')
    kernel_tf_dil1 = tf.constant(kernel_mel_dil1, name="kernel_mel_dil1", dtype='float32')
    kernel_tf_dil2 = tf.constant(kernel_mel_dil2, name="kernel_mel_dil2", dtype='float32')
    kernel_tf_conv1 = tf.constant(kernel_mel_conv1, name="kernel_mel_conv1", dtype='float32')
    kernel_tf_conv2 = tf.constant(kernel_mel_conv2, name="kernel_mel_conv2", dtype='float32')
    conv_mel = tf.nn.conv2d(
        mel_spectrograms,
        kernel_tf_mel,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv_mel")

    # Add first residual block
    DILATION_RATE = 2

    first_res_shortcut = tf.nn.selu(conv_mel) 
    dilated_conv1 = tf.nn.atrous_conv2d(first_res_shortcut, kernel_tf_dil1, DILATION_RATE, padding="SAME", name="dilated_conv1")
    activated_dil1 = tf.nn.selu(dilated_conv1)
    add_residual = tf.add(first_res_shortcut, activated_dil1)
    std_conv1 = tf.nn.conv2d(
        add_residual,
        kernel_tf_conv1,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="std_conv1")

    # Second residual block for mel net
    second_res_shortcut = tf.nn.selu(std_conv1) 
    dilated_conv2 = tf.nn.atrous_conv2d(second_res_shortcut, kernel_tf_dil2, DILATION_RATE, padding="SAME", name="dilated_conv2")
    activated_dil2 = tf.nn.selu(dilated_conv2)
    add_residual2 = tf.add(second_res_shortcut, activated_dil2)
    std_conv2 = tf.nn.conv2d(
        add_residual2,
        kernel_tf_conv2,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="std_conv2")

    mel_net = tf.nn.selu(std_conv2)

    content_loss = ALPHA * 2 * tf.nn.l2_loss(
            net - content_features)

    mel_content_loss = BETA * 2 * tf.nn.l2_loss(
            mel_net - mel_content_features)

    style_loss = 0

    _, height, width, number = map(lambda i: i.value, net.get_shape())
    _, height_mel, width_mel, number_mel = map(lambda i: i.value, mel_net.get_shape())

    size = height * width * number
    feats = tf.reshape(net, (-1, number))
    gram = tf.matmul(tf.transpose(feats), feats)  / N_SAMPLES
    style_loss = 2 * tf.nn.l2_loss(gram - style_gram)

    size_mel = height_mel * width_mel * number_mel
    feats_mel = tf.reshape(mel_net, (-1, number_mel))
    gram_mel = tf.matmul(tf.transpose(feats_mel), feats_mel)  / N_SAMPLES_MEL
    style_loss_mel = 2 * tf.nn.l2_loss(gram_mel - mel_style_gram)

   # Overall loss
    loss = content_loss + mel_content_loss + style_loss + style_loss_mel

    opt = tf.contrib.opt.ScipyOptimizerInterface(
          loss, method='L-BFGS-B', options={'maxiter': 300})
        
    # Optimization
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
       
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

OUTPUT_FILENAME = sys.argv[1]# 'outputs/out.wav'
librosa.output.write_wav(OUTPUT_FILENAME, x, fs)


# In[ ]:


## print OUTPUT_FILENAME


# ### Visualize spectrograms

# In[ ]:

if len(sys.argv) > 2:
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






