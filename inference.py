import tensorflow as tf
import cv2
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
from utils.misc import CaptionData, TopN, ImageLoader

def load_image_into_numpy_array(filename):
  try:
    image = Image.open(filename)
  except FileNotFoundError:
    print("FileNotFoundError")
    return None
  
  try:  
    (im_width, im_height) = image.size

    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)
  except ValueError:
    print("ValueError")
    return None

class GraphLoader(object):
  """Helper class for decoding images in TensorFlow."""
  
  def load_graph(self, graph_path):
    graph = tf.Graph()
    with graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(graph_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    return graph

  def open_session(self, graph):
    # Create a single TensorFlow Session for all image decoding calls.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    return tf.Session(config=sess_config, graph=graph)

class FasterRcnnEncoder(GraphLoader):

  def __init__(self,graph_path):

    self._graph = self.load_graph(graph_path)
    with self.open_session(self._graph) as self._sess:
      self._image_tensor = self._graph.get_tensor_by_name('image_tensor:0')
      self._proposal_boxes = self._graph.get_tensor_by_name('proposal_boxes:0')
      self._feature = self._graph.get_tensor_by_name('proposal_feature:0')
      # self._feature = self._graph.get_tensor_by_name('SecondStageBoxPredictor/AvgPool:0')

  def extract_faster_rcnn_feature(self, image_np):

    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes,feat) = self._sess.run(
                        [self._proposal_boxes,self._feature],
                        feed_dict={self._image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(feat)
  
  def encode(self, image_np):
    return self.extract_faster_rcnn_feature(image_np)

class BeamSearch(object):
  def __init__(self, beam_size, max_caption_length, start_id, end_id, batch_size):
    self.beam_size = beam_size
    self.max_caption_length = max_caption_length
    self.batch_size = batch_size
    self.start_id = start_id
    self.end_id = end_id

  def init_search_sentence(self, initial_memory, initial_output):
    self.partial_caption_data = []
    self.complete_caption_data = []
    for k in range(self.batch_size):
      initial_beam = CaptionData(sentence = [self.start_id],
                     memory = initial_memory[k],
                     output = initial_output[k],
                     score = 0,
                     alphas =[])
      self.partial_caption_data.append(TopN(self.beam_size))
      self.partial_caption_data[-1].push(initial_beam)
      self.complete_caption_data.append(TopN(self.beam_size))

  def init_search_word(self):
    self.partial_caption_data_lists = []
    for k in range(self.batch_size):
      data = self.partial_caption_data[k].extract()
      self.partial_caption_data_lists.append(data)
      self.partial_caption_data[k].reset()

  def prepare_input(self, beam_idx):
    last_word = np.array([pcl[beam_idx].sentence[-1]
                for pcl in self.partial_caption_data_lists],
                np.int32)

    last_memory = np.array([pcl[beam_idx].memory
                for pcl in self.partial_caption_data_lists],
                np.float32)
    last_output = np.array([pcl[beam_idx].output
                for pcl in self.partial_caption_data_lists],
                np.float32)
    return last_word,last_memory,last_output

  def add_result(self, beam_idx, memory, output, scores, alpha):
    # Find the beam_size most probable next words
    for k in range(self.batch_size):
      caption_data = self.partial_caption_data_lists[k][beam_idx]
      words_and_scores = list(enumerate(scores[k]))
      words_and_scores.sort(key=lambda x: -x[1])
      words_and_scores = words_and_scores[0:self.beam_size+1]

      # Append each of these words to the current partial caption
      for w, s in words_and_scores:
        sentence = caption_data.sentence + [w]
        score = caption_data.score + np.log2(s)
        alphas = caption_data.alphas+ [alpha[k]]
        beam = CaptionData(sentence,
                   memory[k],
                   output[k],
                   score,
                   alphas)
        if w == self.end_id:
          self.complete_caption_data[k].push(beam)
        else:
          self.partial_caption_data[k].push(beam)
  
  def extract_result(self):
    results = []
    for k in range(self.batch_size):
      if self.complete_caption_data[k].size() == 0:
        self.complete_caption_data[k] = self.partial_caption_data[k]
      results.append(self.complete_caption_data[k].extract(sort=True))
    return [r[0] for r in results]

  def search(self, inference_step_fn, initial_memory, initial_output):
    self.init_search_sentence(initial_memory, initial_output)
    
    for idx in range(self.max_caption_length):
      
      self.init_search_word()

      num_steps = 1 if idx == 0 else self.beam_size
      for b in range(num_steps):
        last_word, last_memory, last_output = self.prepare_input(beam_idx = b)

        memory, output, scores, alpha = inference_step_fn(last_word, last_memory, last_output)
        self.add_result(b, memory, output, scores, alpha)

    return self.extract_result()

class LSTMDecoder(GraphLoader):

  def __init__(self, graph_path, vocab, max_caption_length):
    self._vocab = vocab
    self._max_caption_length = max_caption_length
    self._graph = self.load_graph(graph_path)
    with self.open_session(self._graph) as self._sess:
      # inputs
      self._images = self._graph.get_tensor_by_name('image_feed:0')
      self._contexts = self._graph.get_tensor_by_name('contexts:0')
      self._last_word = self._graph.get_tensor_by_name('last_word:0')
      self._last_memory = self._graph.get_tensor_by_name('last_memory:0')
      self._last_output = self._graph.get_tensor_by_name('last_output:0')
      # output
      self._conv_feats = self._graph.get_tensor_by_name('conv_feats:0')
      self._initial_memory = self._graph.get_tensor_by_name('initial_memory:0')
      self._initial_output = self._graph.get_tensor_by_name('initial_output:0')
      self._memory = self._graph.get_tensor_by_name('memory:0')
      self._output = self._graph.get_tensor_by_name('output:0')
      self._probs = self._graph.get_tensor_by_name('probs:0')
      self._alpha = self._graph.get_tensor_by_name('alpha:0')

  def get_sentence(self, result):
    word_idxs = result.sentence
    score = result.score
    score = score/(len(word_idxs)-1)

    caption = self._vocab.get_sentence(word_idxs[1:])
    results={'caption':caption,'score':score}
    return results

  def get_attention(self, result):

    return result.alphas

  def show_attention(self, caption,alphas, bounding_box, image_np, save_path):
    # alphas = result.alphas
    cap = caption['caption'].split()
    plt_w = 4
    plt_h = math.ceil((len(cap)+1)/plt_w)
    im_height,im_width = image_np.shape[0:2]

    plt.figure() 
    plt.subplot(plt_h, plt_w, 1)
    plt.imshow(image_np)
    plt.axis('off')
    # generate attention map for each word
    for idx in range(len(cap)):
      mask = np.zeros([im_height,im_width])
      # assign weights for each region in the picture
      for b in range(bounding_box.shape[0]):
        h_start = int(bounding_box[b,0]*im_height)
        w_start = int(bounding_box[b,1]*im_width)
        h_end = int(bounding_box[b,2]*im_height)
        w_end = int(bounding_box[b,3]*im_width)
        
        mask[h_start:h_end,w_start:w_end] += alphas[idx][b]

      plt.subplot(plt_h, plt_w, idx+2)
      lab = cap[idx]
      plt.text(0, 1, lab, backgroundcolor='white', color='black', fontsize=8)
      plt.imshow(image_np)
      plt.imshow(mask, alpha=0.8)
      plt.set_cmap(cm.Greys_r)
      plt.axis('off')
      plt.subplots_adjust(left=0.08, bottom=0.08, right=0.92, top=0.92, hspace=0.1, wspace=0.1)
    
    plt.savefig(save_path)

  def decode(self, region_poposal_feature):
    """Use beam search to generate the captions for a batch of images."""
    # Feed in the images to get the contexts and the initial LSTM states
    # contexts = np.expand_dims(contexts, axis=0)
    batch_size = 1
    region_poposal_feature = np.tile(region_poposal_feature, (batch_size,1,1,1,1))
    contexts, initial_memory, initial_output = self._sess.run(
      [self._conv_feats, self._initial_memory, self._initial_output],
      feed_dict={self._images: region_poposal_feature})

    def _inference_step_fn(last_word, last_memory, last_output):
      return self._sess.run([self._memory, self._output, self._probs, self._alpha],
                            feed_dict = {self._contexts: contexts,
                                         self._last_word: np.tile(last_word,batch_size),
                                         self._last_memory: np.tile(last_memory,(batch_size,1)),
                                         self._last_output: np.tile(last_output,(batch_size,1))})
    # generate caption for each picture
    bs = BeamSearch(3,self._max_caption_length,self._vocab.start_id,self._vocab.end_id,1)
    # Run beam search
    result = bs.search(_inference_step_fn,initial_memory, initial_output)
    caption = self.get_sentence(result[0])
    attention = self.get_attention(result[0])
    return caption, attention

class ATT_NIC(GraphLoader):

  def __init__(self, graph_path, vocab, max_caption_length):
    self.image_loader = ImageLoader('utils/ilsvrc_2012_mean.npy')
    self._vocab = vocab
    self._max_caption_length = max_caption_length
    self._graph = self.load_graph(graph_path)
    with self.open_session(self._graph) as self._sess:
      # inputs
      self._images = self._graph.get_tensor_by_name('images:0')
      self._contexts = self._graph.get_tensor_by_name('contexts:0')
      self._last_word = self._graph.get_tensor_by_name('last_word:0')
      self._last_memory = self._graph.get_tensor_by_name('last_memory:0')
      self._last_output = self._graph.get_tensor_by_name('last_output:0')
      # output
      self._conv_feats = self._graph.get_tensor_by_name('conv_feats:0')
      self._initial_memory = self._graph.get_tensor_by_name('initial_memory:0')
      self._initial_output = self._graph.get_tensor_by_name('initial_output:0')
      self._memory = self._graph.get_tensor_by_name('memory:0')
      self._output = self._graph.get_tensor_by_name('output:0')
      self._probs = self._graph.get_tensor_by_name('probs:0')
      self._alpha = self._graph.get_tensor_by_name('alpha:0')

  def get_sentence(self, result):
    word_idxs = result.sentence
    score = result.score
    score = score/(len(word_idxs)-1)

    caption = self._vocab.get_sentence(word_idxs)
    results={'caption':caption,'score':score}
    return results

  def get_attention(self, result):

    return result.alphas

  def show_attention(self, caption, alphas, image_np, save_path):
    # alphas = result.alphas
    cap = caption['caption'].split()
    plt_w = 4
    plt_h = math.ceil((len(cap)+1)/plt_w)
    im_width, im_height = image_np.shape[0:2]

    plt.figure(figsize=(10,8)) 
    plt.subplot(plt_h, plt_w, 1)
    plt.imshow(image_np)
    plt.axis('off')
    # generate attention map for each word
    for idx in range(len(cap)):
      # assign weights for each region in the picture
      alpha_image = cv2.resize(alphas[idx].reshape(14,14), (im_height,im_width))
      plt.subplot(plt_h, plt_w, idx+2)
      lab = cap[idx]
      plt.text(0, 1, lab, backgroundcolor='white', color='black', fontsize=8)
      plt.imshow(image_np)
      plt.imshow(alpha_image, alpha=0.8)
      plt.set_cmap(cm.Greys_r)
      plt.axis('off')
      plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, hspace=0.05, wspace=0.05)
    
    plt.savefig(save_path)

  def decode(self, filename):
    """Use beam search to generate the captions for a batch of images."""
    # Feed in the images to get the contexts and the initial LSTM states
    images = self.image_loader.load_images([filename])
    contexts, initial_memory, initial_output = self._sess.run(
        [self._conv_feats, self._initial_memory, self._initial_output],
        feed_dict = {self._images: images})

    def _inference_step_fn(last_word, last_memory, last_output):
      return self._sess.run([self._memory, self._output, self._probs, self._alpha],
                            feed_dict = {self._contexts: contexts,
                                         self._last_word: last_word,
                                         self._last_memory: last_memory,
                                         self._last_output: last_output})
    # generate caption for each picture
    bs = BeamSearch(3,self._max_caption_length,self._vocab.start_id,self._vocab.end_id,1)
    # Run beam search
    result = bs.search(_inference_step_fn,initial_memory, initial_output)
    caption = self.get_sentence(result[0])
    attention = self.get_attention(result[0])
    return caption, attention