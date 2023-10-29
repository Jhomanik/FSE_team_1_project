import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def as_matrix(sequences, word_to_index, max_len=None):
    """ Convert a list of tokens into a matrix with padding """
    eos_ix = word_to_index['#END#']
    unk_ix = word_to_index['#UNK#']
    pad_ix = word_to_index['#PAD#']
    max_len = max_len or max(map(len,sequences))

    matrix = np.zeros((len(sequences), max_len), dtype='int32') + pad_ix
    for i,seq in enumerate(sequences):
        row_ix = [word_to_index.get(word, unk_ix) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix

    return matrix

def compute_loss(network, image_features, captions_ix, pad_ix):
    """
    :param image_features: torch tensor containing VGG features. shape: [batch, cnn_channels, width * height]
    :param captions_ix: torch tensor containing captions as matrix. shape: [batch, word_i].
        padded with pad_ix
    :returns: crossentropy (neg llh) loss for next captions_ix given previous ones plus
              attention regularizer. Scalar float tensor
    """

    if next(network.parameters()).is_cuda:
        image_features, captions_ix = image_features.cuda(), captions_ix.cuda()

    # captions for input - all except last cuz we don't know next token for last one.
    captions_ix_inp = captions_ix[:, :-1].contiguous()
    captions_ix_next = captions_ix[:, 1:].contiguous()

    # apply the network, get predictions, attnetion map and gates for captions_ix_next
    logits_for_next, attention_map = network.forward(image_features, captions_ix_inp)
    logits_for_next.to()

    
    n_tokens = logits_for_next.shape[-1]

    mask = (captions_ix_next != pad_ix).float()[:,:,None]

    loss = nn.CrossEntropyLoss()( (logits_for_next * mask).view(-1, n_tokens)  ,   captions_ix_next.view(-1))

    # the regularizer for attention - this one requires the attention over each position to sum up to 1,
    # i. e. to look at the whole image during sentence generation process
    mask = (captions_ix_inp != pad_ix)
    masked_attention_map = attention_map * mask[:, :, None].float()
    regularizer = ((1 - masked_attention_map.sum(1)) ** 2).mean()

    return loss + regularizer


from random import choice



def generate_batch(img_codes, captions, batch_size, word_to_index, max_caption_len=None):

    #sample sequential numbers for image/caption indicies (for trainign speed up)
    global last_batch_end
    random_image_ix = np.arange(batch_size, dtype='int') + last_batch_end.get(len(img_codes), 0)
    last_batch_end[len(img_codes)] = last_batch_end.get(len(img_codes), 0) + batch_size
    if last_batch_end[len(img_codes)] + batch_size >= len(img_codes):
        last_batch_end[len(img_codes)] = 0

    #get images
    batch_images = np.vstack([img_codes[i][None] for i in random_image_ix])
    batch_images = batch_images.reshape(batch_images.shape[0], batch_images.shape[1], -1)

    #5-7 captions for each image
    captions_for_batch_images = captions[random_image_ix]

    #pick one from a set of captions for each image
    batch_captions = list(map(choice,captions_for_batch_images))

    #convert to matrix
    batch_captions_ix = as_matrix(batch_captions, word_to_index, max_len=max_caption_len)

    return torch.tensor(batch_images, dtype=torch.float32), torch.tensor(batch_captions_ix, dtype=torch.int64)




def train_model(network, opt, model_file, train_dataset, val_dataset,  word_to_index, batch_size = 128, n_epochs = 64, n_batches_per_epoch = 128, n_validation_batches = 8 ):
    best_error = None
    train_img_codes, train_captions = train_dataset 
    val_img_codes, val_captions = val_dataset
    for epoch in range(n_epochs):
      if torch.cuda.is_available():
          network = network.cuda()

      train_loss=0
      network.train(True)
      with tqdm(range(n_batches_per_epoch)) as iterator:
          for _ in iterator:
              loss_t = compute_loss(network, *generate_batch(train_img_codes, train_captions, word_to_index, batch_size))

              loss_t.backward()
              opt.step()
              opt.zero_grad()

              train_loss += float(loss_t)
      train_loss /= n_batches_per_epoch

      val_loss=0
      network.train(False)
      for _ in range(n_validation_batches):
          loss_t = compute_loss(network, *generate_batch(val_img_codes, val_captions, word_to_index, batch_size))
          val_loss += float(loss_t)
      val_loss /= n_validation_batches

      if torch.cuda.is_available():
          network = network.cpu()


      print('\nEpoch: {}, train loss: {}, val loss: {}'.format(epoch, train_loss, val_loss), flush=True)

      if best_error is None:
          print('\nBest model, saving\n')
          best_error = val_loss
          torch.save(network,model_file )

      else:
          if best_error > val_loss:
              print('\nBest model, saving\n')
              best_error = val_loss
              torch.save(network,model_file )