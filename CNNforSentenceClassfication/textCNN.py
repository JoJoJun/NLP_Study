'''
Author: JoJoJun
'''
import tensorflow as tf
import numpy as np

tf.reset_default_graph()
# Parameters
embedding_size = 2
seqence_length = 3
num_classes = 2  # 2 kind of sentiment class
filter_sizes = [2,2,2]
num_filters = 3
sentences = ['i love you','he likes me','this is good','i hate you','sorry for that','that is awful']
labels = [1,1,1,0,0,0]
words_list = list(set(' '.join(sentences).split()))
word_dict = {w:i for i,w in enumerate(words_list)}
vocab_size = len(word_dict)
batch_size = 6

inputs = []
for sen in sentences:
    # !!! the inner [] is crutial, or will be generator
    inputs.append(np.asarray([word_dict[n] for n in sen.split()]))
outputs = []
# for sent in sentences:
#     inputs.append(np.asarray( word_dict[w] for w in sent.split()))
for l in labels:
    outputs.append(np.eye(num_classes)[l])


# model
X = tf.placeholder(tf.int32,[None,seqence_length])
Y = tf.placeholder(tf.int32,[None,num_classes])
# word embedding
W = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0))
embedded_chars = tf.nn.embedding_lookup(W,X)  # inputs sentence--> embedding [batch_size,sequence_length,embedding_size]
embedded_chars = tf.expand_dims(embedded_chars,-1)  # [batch_size,sequence_length,embedding_size,1] in_channel=1

# CNN
pooled_outputs = []
for i ,filter_size in enumerate(filter_sizes):
    filter_shape = [filter_size,embedding_size,1,num_filters]
    W = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1))  # the values of filters
    b = tf.Variable(tf.constant(0.1,shape=[num_filters]))

    conv = tf.nn.conv2d(embedded_chars,W,strides=[1,1,1,1],padding='VALID')
    h = tf.nn.relu(tf.nn.bias_add(conv,b))
    # 第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
    # [batch_size, filter_height, filter_width, channel]
    pooled = tf.nn.max_pool(h,ksize=[1,seqence_length-filter_size+1,1,1],strides=[1,1,1,1],padding='VALID')
    pooled_outputs.append(pooled)  # [batch_size(=6), output_height(=1), output_width(=1), channel(=1)]

num_filters_total = num_filters*len(filter_sizes)
h_pool = tf.concat(pooled_outputs,num_filters)  # [batch_size(=6), output_height(=1), output_width(=1), channel(=1) * 3]
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total]) # [3,6]

# dense
weight = tf.get_variable('W',shape = [num_filters_total,num_classes],initializer=tf.contrib.layers.xavier_initializer())
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
model = tf.nn.xw_plus_b(h_pool_flat, weight, bias)
# train
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model,labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# predict
hypothis_outputs = tf.nn.softmax(model)
prediction = tf.argmax(hypothis_outputs,1)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for epoch in range(5000):
    _, loss = sess.run([optimizer, cost], feed_dict={X: inputs, Y: outputs})
    if (epoch + 1)%1000 == 0:
        print('Epoch:', '%06d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

text_sent = 'you hate me'
input_sent = [np.asarray([word_dict[w] for w in text_sent.split(' ')])]
predict = sess.run([prediction], feed_dict={X: input_sent})
result = predict[0][0]
if result == 0:
    print(text_sent ,"is Bad Mean...")
else:
    print(text_sent ,"is Good Mean!!")
sess.close()