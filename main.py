import os
import argparse

from solver import Solver
from data_loader import get_audio_loader


# def main(config):
    # Set paths
PATH = '../data'
DATA_PATH = f'{PATH}/mediaeval-2019-jamendo/'
LABELS_TXT = f'{PATH}/moodtheme_split.txt'
TRAIN_PATH = f'{PATH}/autotagging_moodtheme-train.tsv'
VAL_PATH = f'{PATH}/autotagging_moodtheme-validation.tsv'
TEST_PATH = f'{PATH}/autotagging_moodtheme-test.tsv'
    # assert config.mode in {'TRAIN', 'TEST'},\
    #     'invalid mode: "{}" not in ["TRAIN", "TEST"]'.format(config.mode)

    # if not os.path.exists(config.model_save_path):
    #     os.makedirs(config.model_save_path)
CONFIG = {
        'log_dir': './output',
        'batch_size': 8
    }

def get_labels_to_idx(labels_txt):
    labels_to_idx = {}
    tag_list = []
    with open(labels_txt) as f:
        lines = f.readlines()

    for i,l in enumerate(lines):
        tag_list.append(l.strip())
        labels_to_idx[l.strip()] = i

    return labels_to_idx, 

def predict():
    config = CONFIG
    labels_to_idx, tag_list = get_labels_to_idx(LABELS_TXT)

    test_loader = get_audio_loader(DATA_PATH, TEST_PATH, labels_to_idx, batch_size=config['batch_size'], shuffle=False, drop_last=False)

    solver = Solver(test_loader,None, None, tag_list, config)
    predictions = solver.test()

    np.save(f"{CONFIG['log_dir']}/predictions.npy", predictions)


if __name__=="__main__":

    #Train the data
    train()

    #Predict and create submissions
    predict()

# def train():
#     config = CONFIG
#     labels_to_idx, tag_list = get_labels_to_idx(LABELS_TXT)    

#     train_loader1 = get_audio_loader(DATA_PATH, TRAIN_PATH, labels_to_idx, batch_size=config['batch_size'])
#     train_loader2 = get_audio_loader(DATA_PATH, TRAIN_PATH, labels_to_idx, batch_size=config['batch_size'])
#     val_loader = get_audio_loader(DATA_PATH, VAL_PATH, labels_to_idx, batch_size=config['batch_size'], shuffle=False, drop_last=False)
#     solver = Solver(train_loader1,train_loader2, val_loader, tag_list, config)
#     solver.train()


#     if config.mode == 'TRAIN':
#         labels_to_idx, tag_list = get_labels_to_idx(LABELS_TXT)    
#         data_loader = get_audio_loader(config.bc_learning,
#                                         config.audio_path,
#                                         config.subset,
#                                         config.batch_size,
#                                         config.segment_length,
#                                         shuffle = False,
#                                         tr_val = 'train',
#                                         split = config.split)
#         valid_loader = get_audio_loader(config.bc_learning,
#                                         config.audio_path,
#                                         config.subset,
#                                         config.batch_size // 10,
#                                         config.segment_length,
#                                         shuffle = False,
#                                         tr_val='validation',
#                                         split = config.split)
#         solver = Solver(data_loader, valid_loader, config)
#         solver.train()

#     elif config.mode == 'TEST':
#         data_loader = get_audio_loader(config.bc_learning,
#                                         config.audio_path,
#                                         config.subset,
#                                         config.batch_size // 20,
#                                         config.segment_length,
#                                         shuffle = False,
#                                         tr_val = 'test',
#                                         split = config.split)
#         solver = Solver(data_loader, None, config)
#         solver.test()

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     parser.add_argument('--bc_learning', action="store_true")
#     parser.add_argument('--batch_size', type=int, default=8)
#     parser.add_argument('--segment_length', type=int, default=1366)
#     parser.add_argument('--mode', type=str, default='TRAIN')
#     parser.add_argument('--model_save_path', type=str, default='./models')
#     parser.add_argument('--model_name', type=str, default='CNN')
#     parser.add_argument('--gpu_id', type=str, default= "0")

#     parser.add_argument('--audio_path', type=str, default='../path/to')
#     parser.add_argument('--split', type=int, default=0)
#     parser.add_argument('--subset', type=str, default='moodtheme')

#     config = parser.parse_args()

#     print(config)

#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id #gpu id

#     main(config)
