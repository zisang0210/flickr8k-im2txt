import argparse
import tensorflow as tf
from model import CaptionGenerator
from config import Config

def export_graph(model_folder,model_name,config):
  config.phase = 'test'
  config.train_cnn = False
  config.beam_size = 3

  graph = tf.Graph()
  with graph.as_default():
    model = CaptionGenerator(config)
    

    # input tensor can't use tf.identity() to rename
    # inputs = {}
    outputs = {}
    # # input
    # inputs['contexts'] = tf.identity(model.contexts, name='contexts')
    # inputs['last_word'] = tf.identity(model.last_word, name='last_word')
    # inputs['last_memory'] = tf.identity(model.last_memory, name='last_memory')
    # inputs['last_output'] = tf.identity(model.last_output, name='last_output')
    # outputs
    outputs['initial_memory'] = tf.identity(model.initial_memory, name='initial_memory')
    outputs['initial_output'] = tf.identity(model.initial_output, name='initial_output')
    
    # results
    outputs['conv_feats'] = tf.identity(model.conv_feats, name='conv_feats')
    outputs['alpha'] = tf.identity(model.alpha, name='alpha')
    outputs['memory'] = tf.identity(model.memory, name='memory')
    outputs['output'] = tf.identity(model.output, name='output')
    outputs['probs'] = tf.identity(model.probs, name='probs')
    # logits = model.inference(input_image)
    # y_conv = tf.nn.softmax(logits,name='outputdata')
    # restore_saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    # sess.run(tf.global_variables_initializer())
    # latest_ckpt = tf.train.latest_checkpoint(model_folder)
    # restore_saver.restore(sess, latest_ckpt)
    model.load(sess, model_folder)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), list(outputs.keys()))

#    tf.train.write_graph(output_graph_def, 'log', model_name, as_text=False)
    with tf.gfile.GFile(model_name, "wb") as f:  
        f.write(output_graph_def.SerializeToString()) 

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--model_folder", type=str, help="Model folder to export")  
    parser.add_argument("--output_path", type=str, help="Path to save exported graph") 
    parser.add_argument("--attention", type=str, help="Attention mechanism of the exported graph") 
    args = parser.parse_args()  

  
    config = Config()
    # config.batch_size = 1
    config.attention_mechanism = args.attention

    export_graph(args.model_folder,args.output_path,config)  
