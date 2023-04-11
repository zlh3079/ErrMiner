import random
import time
import os
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from FocalLoss import FocalLoss
from dataloader import dialogDataset
from model import ErrMiner
import logging
import sys
from utils import seed_everything, get_parameter_number, get_dialog_loaders, train_model, evaluate_model, Config

seed = 2021
root_path = os.path.dirname(__file__)

if __name__ == '__main__':
    logging.basicConfig(filename=os.path.join(root_path, "log.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("=" * 50)
    logging.info("running...")
    save_path = os.path.join(root_path, 'cnn_100_graph_256.pth')
    save_path2 = os.path.join(root_path, 'new.pth')

    config = Config()
    cuda = torch.cuda.is_available() and config.cuda
    if cuda:
        logging.info('Running on GPU')
    else:
        logging.info('Running on CPU')

    for project in ['typescript', 'angular', 'appium', 'gitter', 'docker', 'dl4j']:
        seed_everything(seed)
        model = ErrMiner(config.pretrained_model, config.D_bert, config.filter_sizes, config.filter_num,
                            config.D_cnn, config.D_graph, n_speakers=2, graph_class_num=config.graph_class_num,
                            dropout=config.dropout, ifcuda=cuda)
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict)
        # 模型装载至cuda
        if cuda:
            model.cuda()

        graph_loss = FocalLoss(gamma=2)
        # graph_loss = nn.NLLLoss()

        # 冻结bert参数，训练时不更新
        for name, params in model.pretrained_bert.named_parameters():
            params.requires_grad = False

        logging.info(get_parameter_number(model))

        # '', 'appium', 'docker', 'dl4j', 'gitter',, 'appium', 'dl4j', 'docker', 'gitter', 'typescript'

        logging.info("======current_project:{}======".format(project))
        # 过滤掉requires_grad = False的参数

        # print("啊",project)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters(
        )), lr=config.lr, weight_decay=config.l2)

        train_address = os.path.join(
            root_path, 'data/Augmented', project + '_train.json')
        test_address = os.path.join(
            root_path, 'data/Processed', project + '_test.json')

        train_loader, test_loader = get_dialog_loaders(train_address, test_address, config.pretrained_model,
                                                       batch_size=config.batch_size)

        best_f1 = 0
        pre = 0
        rec = 0
        for e in range(config.epochs):
            start_time = time.time()

            train_loss, train_pre, train_rec, train_f1, train_error = \
                train_model(model, graph_loss, train_loader,
                            optimizer, e, cuda, save_path2)

            # test
            test_loss, test_pre, test_rec, test_f1, test_error = \
                evaluate_model(model, graph_loss, test_loader, cuda)

            logging.info('epoch:{},train_loss:{},train_pre:{},train_rec:{},train_f1:{},test_loss:{},test_graph_pre:{},'
                         'test_graph_rec:{},test_graph_f1:{},time:{}sec'.format(e + 1, train_loss, train_pre, train_rec,
                                                                                train_f1,
                                                                                test_loss, test_pre, test_rec, test_f1,
                                                                                round(time.time() - start_time, 2)))
            # print(test_error)
            if test_f1 > best_f1:
                best_f1 = test_f1
                pre = test_pre
                rec = test_rec
                # pth=str(project)+".pth"
                # torch.save(model.state_dict(),os.path.join(root_path, pth))
            # print("train_error", train_error)
            # print("test_error", test_error)
            logging.info(
                "==========================================================================")
            # torch.save(model.state_dict(), 'gitter_result1.pth')

        logging.info("training finish!!!")
        with open(os.path.join(root_path, "cnn_100_graph_256.txt"), 'a') as f:
            f.write(project)
            f.write("pre:" + str(pre))
            f.write("rec:" + str(rec))
            f.write("f1:" + str(best_f1))
            f.write("\n")
