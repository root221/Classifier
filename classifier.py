import tensorflow as tf,sys

image_path = sys.argv[1]

image_data = tf.gfile.FastGFile(image_path,'rb').read()
labels = [line[:-1] for line in tf.gfile.FastGFile('retrained_labels.txt','rb').readlines()]
with tf.gfile.FastGFile('/usr/local/lib/python3.6/site-packages/tensorflow/retrained_graph.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,name='')
    
with tf.Session() as session:
    softmax_tensor = session.graph.get_tensor_by_name('final_result:0')
    predictions = session.run(softmax_tensor,{'DecodeJpeg/contents:0':image_data})
    top_k = predictions[0].argsort()
    print(predictions[0]) 
    print(labels[top_k[len(top_k)-1]]) 
