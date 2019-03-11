import embeddings
from cupy_utils import *

import argparse as ap
import torch
import torch.nn as nn
import numpy as np


class MAP(nn.Module):
    def __init__(self):
        super(MAP, self).__init__()
        self.nn.fc = nn.Linear(300, 300)
    def forward(self, x):
        x = self.nn.fc(x)
        return x

def main():
    parser = ap.ArgumentParser(description='Parse argument')
    parser.add_argument('src_input')
    parser.add_argument('trg_input')
    parser.add_argument('src_output')
    parser.add_argument('trg_output')
    parser.add_argument('seed_dict')
    parser.add_argument('--encoding', default='utf-8')
    parser.add_argument('--precision', choices=['fp16', 'fp32', 'fp64'], default='fp32')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--batch_size', default=10000, type=int)

    args = parser.parse_args()

    # Choose the right dtype for the desired precision
    if args.precision == 'fp16':
        dtype = 'float16'
    elif args.precision == 'fp32':
        dtype = 'float32'
    elif args.precision == 'fp64':
        dtype = 'float64'
    
    # Read input embeddings
    srcfile = open(args.src_input, encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_input, encoding=args.encoding, errors='surrogateescape')
    src_words, x = embeddings.read(srcfile, dtype=dtype)
    trg_words, z = embeddings.read(trgfile, dtype=dtype)
    srcfile.close()
    trgfile.close()

    # Build word to index map
    src_word2ind = {word: i for i, word in enumerate(src_words)}
    trg_word2ind = {word: i for i, word in enumerate(trg_words)}

    #deivce configuration
    device = torch.device('cuda' if args.cuda is True else 'cpu')

    #hyper-parameters
    dictsize = 5000
    learning_rate = 0.001
    num_epochs = 100
    input_size = 300
    output_size = 300

    # load seed dictionary
    dictfile = open(args.seed_dict, encoding=args.encoding, errors='surrogateescape')
    dim = x.shape[1]
    x_train = np.zeros((dictsize, dim))
    z_train = np.zeros((dictsize, dim))
    for i, line in enumrate(dictfile):
        if i == dictsize:
            break
        src, trg = line.strip.split(" ")
        x_train[i, :] = x[src_word2ind[src]]
        z_train[i, :] = z[src_word2ind[src]]

    # get solution
    model = nn.Linear(input_size, output_size)
    input = torch.from_numpy(x_train).to(device)
    target = torch.from_numpy(z_train).to(device)
    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for i in range(num_epochs):
        #forward pass
        output = model(input)
        
        # compute and print loss
        loss = criterion(output, target)
        if (epoch+1) % 5 == 0:
            print('Epoch [{}/{}], Loss:{.4f}'.format(epoch+1, epoch_nums, loss.item()))
        
        #update params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
                  
    # transform embeddings x and z
    with torch.no_grad():
        xw = model(x.to(device))
                  
    # write mapped embeddings
    srcfile = open(args.src_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    trgfile = open(args.trg_output, mode='w', encoding=args.encoding, errors='surrogateescape')
    embeddings.write(src_words, xw.numpy(), srcfile)
    embeddings.write(trg_words, z.numpy(), trgfile)
    srcfile.close()
    trgfile.close()
                  
if __name__ == '__main__':
    main()
