'''
Author: JoJoJun
'''
import tensorflow as tf
import  numpy as np

tf.reset_default_graph()

sentences = ['我 想 喝 酒 P', 'S i want a beer','i want a beer E']
word_list = list(set(' '.join(sentences).split(' ')))
word_dict = {w:i for i,w in enumerate(word_list)}
number_dict = {i:w for i,w in enumerate(word_list)}
n_class = len(word_dict)

# Parameters
lr = 0.001
max_sent_len = 5
n_step = max_sent_len
n_hidden = 128
batch_size = 1

# placeholder & variables
encoder_inputs = tf.placeholder(dtype=tf.float32,shape=[None,max_sent_len,n_class])  # [batch_size, max_sent_len,n_class]
decoder_inputs = tf.placeholder(dtype=tf.float32,shape=[None,max_sent_len,n_class])  # [batch_size,max_sent_len,n_class]
targets = tf.placeholder(dtype=tf.int64,shape=[None,n_step])  # [batch_size,n_step]

attn_p = tf.Variable(tf.random_normal(shape = [n_hidden,n_hidden]))
out_P = tf.Variable(tf.random_normal(shape=[2*n_hidden,n_class]))

def make_batch(sentences):
    input_batch = [np.eye(n_class)[[word_dict[w] for w in sentences[0].split(' ')]]]
    output_batch = [np.eye(n_class)[[word_dict[w] for w in sentences[1].split(' ')]]]
    target_batch = [[word_dict[w] for w in sentences[2].split()]]
    return input_batch, output_batch,target_batch

def get_attn_score(dec_output,enc_output):
    # dec_output [batch_size,1,n_hidden]  enc_output [batch_size,n_hidden]
    score = tf.matmul(enc_output,attn_p)  # [batch_size,n_hidden]
    score = tf.squeeze(score,0)  # [n_hidden]
    dec_output = tf.squeeze(dec_output,[0,1])  # [n_hidden]
    score = tf.tensordot(dec_output,score,1)
    return score
#  enc_output: [batch_size,n_step,n_hidden]  dec_output [batch_size, 1,n_hidden]
def get_attn_weights(dec_output,enc_output):
    attn_scores = []
    enc_output = tf.transpose(enc_output,[1,0,2])
    for j in range(n_step):
        attn_scores.append(get_attn_score(dec_output,enc_output[j]))
    # [n_step,batch_size,1]
    return tf.reshape(tf.nn.softmax(attn_scores),shape=[batch_size,1,n_step])

model = []
Attention = []
with tf.variable_scope('Encode'):
    cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell,output_keep_prob=0.5)
    # outputs: [batchsize, n_step, n_hidden]; state [batchsize,n_hidden]
    enc_ouputs, enc_state = tf.nn.dynamic_rnn(cell,inputs=encoder_inputs,dtype=tf.float32)
with tf.variable_scope('Decode'):
    cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell= cell,output_keep_prob=0.5)
    #  decoder_inputs [batch_size, max_sent_len, n_class] ==>
    inputs = tf.transpose(decoder_inputs,[1,0,2])
    for i in range(n_step):
        # dec_outputs: [batchsize,1,n_hidden]
        dec_outputs, dec_state = tf.nn.dynamic_rnn(cell=cell,inputs=tf.expand_dims(inputs[i],1),
                                                  initial_state=enc_state,dtype=tf.float32, time_major=True)
        attn_weights = get_attn_weights(dec_outputs,enc_ouputs)  # [batchsize,1,n_step]
        Attention.append(attn_weights)
        # [batchsize,1,n_step] [batchsize,n_step,n_hidden] ==> [batchsize,1,n_hidden]
        context =  tf.matmul(attn_weights,enc_ouputs )
        context = tf.squeeze(context,axis=1)  # [batchsize,n_hidden]
        dec_outputs = tf.squeeze(dec_outputs,axis=1)
        concat = tf.concat((dec_outputs,context),axis=1)  # [batchsize,2*n_hidden]  [2*hidden,n_class]
        # [batchsize,n_class]
        model.append(tf.matmul(concat,out_P))

trained_attn = tf.stack([Attention[0], Attention[1], Attention[2], Attention[3], Attention[4]], 0)  # to show attention matrix
# model:[n_step,batchsize,n_class]
model = tf.transpose(model,[1,0,2])

prediction = tf.argmax(model,2)
#  reducemean 用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model,labels=targets))
optimizer = tf.train.AdamOptimizer(lr).minimize(cost)


# Training
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for epoch in range(2000):
        inputs_batch ,outputs_batch ,targets_batch = make_batch(sentences)
        _,loss,attention = sess.run([optimizer,cost,trained_attn],
                                    feed_dict={encoder_inputs:inputs_batch,decoder_inputs:outputs_batch,targets:targets_batch})

        if (epoch+1) % 400 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
    print("traing finished")
    # test
    predict_batch = [np.eye(n_class)[[word_dict[w] for w in 'P P P P P'.split()]]]
    result = sess.run(prediction,feed_dict={encoder_inputs:inputs_batch,decoder_inputs:predict_batch})
    print(sentences[0].split(), '->', [number_dict[n] for n in result[0]])





