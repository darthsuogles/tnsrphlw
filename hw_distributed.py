""" Distributed TensorFlow
"""
import tensorflow as tf
import numpy as np
import webbrowser

# With a simple local server
local_server = tf.train.Server.create_local_server()
with tf.Session(local_server.target) as sess_local:
    # Input
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    # Model
    W = tf.Variable([+.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    linear_model = W * x + b

    # Task
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)

    _init = tf.global_variables_initializer()
    sess_local.run(_init)
    res = sess_local.run(loss, {x: [1,2,3,4], y: [-1,1,-1,1]})
    print(res)

""" Distributed training
"""
cluster = tf.train.ClusterSpec({
    "worker": [
        "localhost:2222",
        "localhost:2223",
        "localhost:2224"
    ],
    "ps": [
        "localhost:2225",
        "localhost:2226"
    ]})

worker_servers = [tf.train.Server(cluster, job_name='worker', task_index=idx)
                  for idx in range(3)]
params_servers = [tf.train.Server(cluster, job_name='ps', task_index=idx)
                  for idx in range(2)]

# for srv in worker_servers:
#     srv.join()
# for srv in params_servers:
#     srv.join()

with tf.device("/job:ps/task:0"):
    W0 = tf.Variable([+.3], tf.float32)
    b0 = tf.Variable([-.3], tf.float32)

with tf.device("/job:ps/task:1"):
    W1 = tf.Variable([+.4], tf.float32)
    b1 = tf.Variable([+.2], tf.float32)

with tf.device("/job:worker/task:1"):
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    linear_model = W0 * x + b0 + W1 * x + b1
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    

with tf.Session(worker_servers[1].target) as sess:
    global_step = tf.contrib.framework.get_or_create_global_step()
    train_op = tf.train.AdagradOptimizer(0.01).minimize(
        loss, global_step=global_step)

    _init = tf.global_variables_initializer()
    sess.run(_init)
    print(sess.run(loss, {x: [1,2,3,4], y: [-1,1,-1,1]}))


""" Models trained on ImageNet
"""
from pathlib import Path 

fp_data = Path.home() / 'local' / 'data'
fp_inception = fp_data / 'inception5h' / 'tensorflow_inception_graph.pb'
graph = tf.Graph()
sess_intr = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(str(fp_inception), 'rb') as fin:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(fin.read())

t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preproc = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preproc})

_ops = graph.get_operations()

def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    _tfb_url = "https://tensorboard.appspot.com/tf-graph-basic.build.html"
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()

    def strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add() 
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = str.encode("<stripped {} bytes>".format(size))
        return strip_def

    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="{tfb}" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), 
               id='graph' + str(np.random.rand()),
               tfb=_tfb_url)

    # Construct the graph def board and open it
    fp_out = Path.cwd() / 'graph_def.html'
    with open(str(fp_out), 'w') as fout:
        fout.write(code)
    wb = webbrowser.get('chrome')
    wb.open('file://{}'.format(fp_out))


