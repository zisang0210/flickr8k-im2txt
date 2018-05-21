from inference import *
from utils import vocabulary
from utils import att_nic_vocab
def test_encode():
  image_np = load_image_into_numpy_array('./test/images/3.jpg')
  if image_np is not None:
    faster_rcnn = FasterRcnnEncoder('../data/frozen_faster_rcnn.pb')
    box, feat = faster_rcnn.encode(image_np)
    print(box, feat)
    print(box.shape, feat.shape)
    print("test encode finish!")

def test_decode():
  image_np = load_image_into_numpy_array(
    '/home/zisang/Documents/code/data/Flicker8k/Flicker8k_Dataset/2919459517_b8b858afa3.jpg')
  if image_np is not None:
    faster_rcnn = FasterRcnnEncoder('../data/frozen_faster_rcnn.pb')
    box, feat = faster_rcnn.encode(image_np)

    # build vocabulary file
    vocab = vocabulary.Vocabulary("../data/flickr8k/word_counts.txt")
    lstm = LSTMDecoder('../data/frozen_lstm.pb',vocab,max_caption_length=20)
    caption, attention = lstm.decode(feat)
    lstm.show_attention(caption, attention,box, image_np, './a.jpg')

def test_att_nic():
  image_np = load_image_into_numpy_array('./test/images/2.jpg')
  vocab = att_nic_vocab.Vocabulary(5000, "../output/vocabulary.csv")
  att_nic = ATT_NIC('../data/frozen_att_nic_test.pb',vocab,max_caption_length=20)
  caption, attention = att_nic.decode('./test/images/2.jpg')
  print(attention[0])
  print(attention[1])
  print(attention[0]-attention[1]>0.0003)
  print(attention[0]-attention[1]<-0.0003)
  # print(caption)
  # att_nic.show_attention(caption, attention, image_np, './a.jpg')

# test_encode()
test_decode()
# test_att_nic()