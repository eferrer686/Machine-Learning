
# coding: utf-8

# In[7]:


import tensorflow as tf
# create graph
a = tf.constant(2, name='a')
b = tf.constant(3, name='b')
c = tf.add(a, b, name='addo')
# launch the graph in a session
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(c))


# In[10]:


import tensorflow as tf
tf.reset_default_graph()   
x_scalar = tf.get_variable('xscalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
#Ver los datos
summary = tf.summary.scalar(name='vScalar', tensor=x_scalar)
init = tf.global_variables_initializer()
# lanza la sumatoria
with tf.Session() as sess:
    # writer
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for step in range(100):
        # loop
        sess.run(init)
        exsummary = sess.run(summary)
        # crea un resumen
        # guarda el resumen
        writer.add_summary(exsummary, step)
        writer.flush()


# In[12]:


import tensorflow as tf
tf.reset_default_graph()   
#Variables
x_scalar = tf.get_variable('x_scalar', shape=[], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
x_matrix = tf.get_variable('x_matrix', shape=[30, 40], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
# resumenes
summary = tf.summary.scalar(name='vScalar', tensor=x_scalar)
histogram_summary = tf.summary.histogram( 'histogram', x_matrix)
# histogramas
init = tf.global_variables_initializer()
# sesion
with tf.Session() as sess:
    # writer
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    for step in range(100):
        # loop
        sess.run(init)
        # two summaries
        sScalar, sHistogram = sess.run([summary, histogram_summary])
        # out to writer
        # out to writer
        writer.add_summary(sScalar, step)
        writer.add_summary(sHistogram, step)
        writer.flush()


# In[14]:


import tensorflow as tf
tf.reset_default_graph()
#variables
w_gs = tf.get_variable('W_Grayscale', shape=[30, 10], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
w_c = tf.get_variable('W_Color', shape=[50, 30], initializer=tf.truncated_normal_initializer(mean=0, stddev=1))
# reshape
w_gs_reshaped = tf.reshape(w_gs, (3, 10, 10, 1))
w_c_reshaped = tf.reshape(w_c, (5, 10, 10, 3))
# summaries
graySummary = tf.summary.image('gray', w_gs_reshaped)
colorSummary = tf.summary.image('color', w_c_reshaped)
# merge summaries
merged = tf.summary.merge_all()
# initialize variables
init = tf.global_variables_initializer()
# session
with tf.Session() as sess:
    # writer
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    # run init
    sess.run(init)
    # run merged
    summary = sess.run(merged)
    #writer
    writer.add_summary(summary)
    writer.flush()

