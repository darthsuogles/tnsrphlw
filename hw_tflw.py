""" Tensorflow
"""
import tensorflow as tf

server = tf.train.Server.create_local_server()

""" Create this on separate machines """
# cluster = tf.train.ClusterSpec({
#     "local": ["localhost:2222", "localhost:2223"]
# })
# server = tf.train.Server(cluster, job_name="local", task_index=0)

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

W = tf.Variable([+.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
linear_model = W * x + b
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# Must initialize the variables
_init = tf.global_variables_initializer()
#sess = tf.Session(server.target)
sess = tf.InteractiveSession()
sess.run(_init) # reset values to wrong

print(sess.run(loss, {x: [1,2,3,4], y: [-1,1,-1,1]}))

# const = tf.constant('salut tnsrphlw')
# 
# sess.run(const)

