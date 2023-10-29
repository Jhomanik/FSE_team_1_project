import sys
from generate import *
from dataset import *
from model import *
import torch
import subprocess

if __name__ == "__main__":
    img_file = sys.argv[1]
    img = obtain_image(img_file)

    caption_file = sys.argv[2]
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    features_net = get_features_net()
    network_file = sys.argv[3]
    word_to_index, vocab = create_dict(caption_file)
    
    network = torch.load(network_file, map_location=device)
    features_net.to(device)
    network.to(device)
    network.device = device

    subprocess.run(['./hello'])

    print_possible_captions(img, network, features_net, vocab, word_to_index,  num_captions=2, temperature=5.)
