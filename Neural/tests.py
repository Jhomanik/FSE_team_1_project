from fileinput import close
import unittest
from model import *
import sys
from generate import *
from dataset import *
from model import *
import torch

class TestMethods(unittest.TestCase):

  def test_attention(self):
      attn = Attention(ScaledDotProductScore())
      q = torch.randn(2, 3, 5)
      k = torch.randn(2, 4, 5)
      v = torch.randn(2, 4, 7)
      self.assertTrue( attn(q, k, v).shape == (2, 3, 7) )
      q = torch.tensor([[
            [0.01],
            [1],
            [100],
      ]], dtype=torch.float32)
      o = torch.tensor([[
            [-1],
            [0],
            [1],
      ]], dtype=torch.float32) * 1000
      a = attn(q, o, o)
      self.assertTrue( torch.isnan(attn.attention_map).sum() == 0) 
      self.assertTrue( torch.isnan(a).sum() == 0 )

  def test_dict(self):
      captions_tokenized_file = 'Data/captions_tokenized.json'
      word_to_index, vocab = create_dict(captions_tokenized_file)
      n_tokens = len(vocab)
      self.assertTrue(10000 <= n_tokens <= 10500)

  def test_model(self):
      caption_file = 'Data/captions_tokenized.json'
      if torch.cuda.is_available():
          device = torch.device('cuda')
      else:
          device = torch.device('cpu')
      word_to_index, vocab = create_dict(caption_file)
      with open(caption_file) as f:
        captions = json.load(f)
      
      n_tokens = len(vocab)
      network = CaptionNet(n_tokens).to(device)
      network.device = device
      dummy_img_vec = torch.randn(len(captions[0]), 512, 81).to(device)
      dummy_capt_ix = torch.tensor(as_matrix(captions[0], word_to_index), dtype=torch.int64).to(device)

      dummy_logits, dummy_attention_map = network.forward(dummy_img_vec, dummy_capt_ix)
      dummy_logits, dummy_attention_map = network.forward(dummy_img_vec, dummy_capt_ix)

  
      self.assertTrue( dummy_logits.shape == (dummy_capt_ix.shape[0], dummy_capt_ix.shape[1], n_tokens) )
      self.assertTrue( dummy_attention_map.shape == (dummy_capt_ix.shape[0], dummy_capt_ix.shape[1], 81) )


if __name__ == '__main__':
    unittest.main()