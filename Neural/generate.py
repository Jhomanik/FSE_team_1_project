from matplotlib import pyplot as plt
from skimage.transform import resize
from tempfile import mktemp
from os import remove
import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.model_zoo import load_url

from training import as_matrix
from torch import nn
import torch.nn.functional as F
from torchvision.models.vgg import VGG, cfgs as VGG_cfgs, make_layers
from warnings import warn

class BeheadedVGG19(VGG):
    """ Like torchvision.models.inception.Inception3 but the head goes separately """

    def forward(self, x):
        x_for_attn = x= self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = x = self.classifier(x)
        return x_for_attn, logits


def get_features_net():
    features_net = BeheadedVGG19(make_layers(VGG_cfgs['E'], batch_norm=False), init_weights=False)

    
    features_net_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
    features_net.load_state_dict(load_url(features_net_url))

    features_net = features_net.train(False)
    if  torch.cuda.is_available():
        features_net = features_net.cuda()
        features_net = nn.DataParallel(features_net)
    return features_net

def generate_caption(image, network, features_net, vocab, word_to_index, caption_prefix = ("#START#",),
                     t=1, sample=True, max_len=100):

    assert isinstance(image, np.ndarray) and np.max(image) <= 1\
           and np.min(image) >=0 and image.shape[-1] == 3

    image = torch.tensor(image.transpose([2, 0, 1]), dtype=torch.float32)

    vectors_9x9, logits = features_net(image[None])
    caption_prefix = list(caption_prefix)

    attention_maps = []

    for _ in range(max_len):

        prefix_ix = as_matrix([caption_prefix], word_to_index)
        prefix_ix = torch.tensor(prefix_ix, dtype=torch.int64)
        input_features = vectors_9x9.view(vectors_9x9.shape[0], vectors_9x9.shape[1], -1)
        if next(network.parameters()).is_cuda:
            input_features, prefix_ix = input_features.cuda(), prefix_ix.cuda()
        else:
            input_features, prefix_ix = input_features.cpu(), prefix_ix.cpu()
        next_word_logits, cur_attention_map = network(input_features, prefix_ix)
        next_word_logits = next_word_logits[0, -1]
        cur_attention_map = cur_attention_map[0, -1]
        next_word_probs = F.softmax(next_word_logits, -1).detach().cpu().numpy()
        attention_maps.append(cur_attention_map.detach().cpu())

        assert len(next_word_probs.shape) ==1, 'probs must be one-dimensional'
        next_word_probs = next_word_probs ** t / np.sum(next_word_probs ** t) # apply temperature
        
        if sample:
            next_word = np.random.choice(vocab, p=next_word_probs)
        else:
            next_word = vocab[np.argmax(next_word_probs)]

        caption_prefix.append(next_word)

        if next_word=="#END#":
            break

    return caption_prefix, attention_maps

# get and preprocess image
def obtain_image(filename=None, url=None):
    if (filename is None and url is None) or (filename is not None and url is not None):
        raise ValueError('You shoud specify either filename or url')
    if url is not None:
        tmpfilename = mktemp()
       
        img = plt.imread(tmpfilename)
        remove(tmpfilename)
    else:
        img = plt.imread(filename)
    img = resize(img, (299, 299), mode='wrap', anti_aliasing=True).astype('float32')
    return img

def print_possible_captions(img, network, features_net, vocab, word_to_index,  num_captions=10, temperature=5., ):
    for i in range(num_captions):
        print(' '.join(generate_caption(img, t=temperature, network = network, features_net = features_net, word_to_index = word_to_index, vocab = vocab)[0][1:-1]))




#def show_img(img):
#    plt.imshow(img)
#    plt.axis('off')
#def get_possible_captions(img, network, features_net, vocab, word_to_index, num_captions=10, temperature=5., ):
#    for i in range(num_captions):
#        return ' '.join(generate_caption(img, t=temperature, network = network, features_net = features_net, word_to_index = word_to_index, vocab = vocab)[0][1:-1])
#def draw_attention_map(img, caption, attention_map):
#    s = 4
#    n = len(caption)
#    w = 4
#    h = n // w + 1
#   plt.figure(figsize=(w * s, h * s))
#    plt.subplot(h, w, 1)
#    plt.imshow(img)
#    plt.title('INPUT', fontsize=s * 4)
#    plt.axis('off')
#    for i, word, attention in zip(range(n), caption, attention_map):
#        plt.subplot(h, w, 2 + i)
#        attn_map = attention.view(1, 1, 9, 9)
#        attn_map = F.interpolate(attn_map, size=(12, 12), mode='nearest')
#        attn_map = F.interpolate(attn_map, size=(299, 299), mode='bilinear', align_corners=False)
#        attn_map = attn_map[0, 0][:, :, None]
#        attn_map = torch.min(attn_map / attn_map.max(), torch.ones_like(attn_map)).numpy()
#        plt.imshow(img * attn_map)
#        plt.title(word, fontsize=s * 4)
#        plt.axis('off')

#def process_image(img, network, features_net, vocab, word_to_index):
#    print_possible_captions(img, network= network,  features_net = features_net, vocab = vocab, word_to_index = word_to_index)
#    c, am = generate_caption(img, t=5.,network = network, features_net = features_net, vocab = vocab, word_to_index = word_to_index)
#    draw_attention_map(img, c[1:-1], am)