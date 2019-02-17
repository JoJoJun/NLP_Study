'''
Author: JoJoJun
'''
import tensorflow as tf
t1=[1,2]
t2 = [2,3]
t = tf.tensordot(t1,t2,1)
with tf.Session() as sess:
    print(sess.run(t))