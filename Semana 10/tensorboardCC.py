
# coding: utf-8

# In[1]:


import tensorflow as tf
# create graph
a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)
# launch the graph in a session
with tf.Session() as sess:
    print(sess.run(c))


# In[2]:


import tensorflow as tf
tf.reset_default_graph()   
x_scalar = tf.get_variable('xscalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
init = tf.global_variables_initializer()
# lanza la sumatoria
with tf.Session() as sess:
    # writer
    for step in range(100):
        # loop
        sess.run(init)
        # crea un resumen
        # guarda el resumen


# In[3]:


import tensorflow as tf
tf.reset_default_graph()   
#Variables
x_scalar = tf.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
x_matrix = tf.get_variable('x_matrix', shape=[30, 40], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
# resumenes
# histogramas
init = tf.global_variables_initializer()
# sesion
with tf.Session() as sess:
    # writer
    for step in range(100):
        # loop
        sess.run(init)
        # two summaries
        # out to writer
        # out to writer


# In[4]:


import tensorflow as tf
tf.reset_default_graph()
#variables
w_gs = tf.get_variable('W_Grayscale', shape=[30, 10], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
w_c = tf.get_variable('W_Color', shape=[50, 30], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
# reshape
w_gs_reshaped = tf.reshape(w_gs, (3, 10, 10, 1))
w_c_reshaped = tf.reshape(w_c, (5, 10, 10, 3))
# summaries
# merge summaries
# initialize variables
init = tf.global_variables_initializer()
# session
with tf.Session() as sess:
    # writer
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    # run init
    sess.run(init)
    # run merged
    #writer

