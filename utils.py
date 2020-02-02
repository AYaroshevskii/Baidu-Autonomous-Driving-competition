from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, f1_score
from tqdm import tqdm_notebook as tqdm
from torch.utils import data
import numpy as np
import pandas as pd


#Save and load model checkpoint
def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict()}
    if optimizer:
        state['optimizer'] = optimizer.state_dict()
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

#list of bad samples
bad_list = ['ID_1a5a10365',
            'ID_1db0533c7',
            'ID_53c3fe91a',
            'ID_408f58e9f',
            'ID_4445ae041',
            'ID_bb1d991f6',
            'ID_c44983aeb',
            'ID_f30ebe4d4']
