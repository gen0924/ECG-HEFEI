import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim

from adabound import AdaBound
from ecg_model import DenseNet
from ecg_loader import ECGLoader
from sklearn.metrics import f1_score
from ecg_utils import ComboLoss, best_f1_score
from skmultilearn.model_selection import IterativeStratification


class ECGTrainer(object):

    def __init__(self, block_config='small', num_threads=2):
        torch.set_num_threads(num_threads)
        self.n_epochs = 60
        self.batch_size = 128
        self.scheduler = None
        self.num_threads = num_threads
        self.cuda = torch.cuda.is_available()

        if block_config == 'small':
            self.block_config = (3, 6, 12, 8)
        else:
            self.block_config = (6, 12, 24, 16)

        self.__build_model()
        self.__build_criterion()
        self.__build_optimizer()
        self.__build_scheduler()
        return

    def __build_model(self):
        self.model = DenseNet(
            num_classes=55, block_config=self.block_config
        )
        if self.cuda:
            self.model.cuda()
        return

    def __build_criterion(self):
        self.criterion = ComboLoss(
            losses=['mlsml', 'f1', 'focal'], weights=[1, 1, 3]
        )
        return

    def __build_optimizer(self):
        opt_params = {'lr': 1e-3, 'weight_decay': 0.0,
                      'params': self.model.parameters()}
        self.optimizer = AdaBound(amsbound=True, **opt_params)
        return

    def __build_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', factor=0.333, patience=5,
            verbose=True, min_lr=1e-5)
        return

    def run(self, trainset, validset, model_dir):
        print('=' * 100 + '\n' + 'TRAINING MODEL\n' + '-' * 100 + '\n')
        model_path = os.path.join(model_dir, 'model.pth')
        thresh_path = os.path.join(model_dir, 'threshold.npy')

        dataloader = {
            'train': ECGLoader(trainset, self.batch_size, True, self.num_threads).build(),
            'valid': ECGLoader(validset, 64, False, self.num_threads).build()
        }

        best_metric, best_preds = None, None
        for epoch in range(self.n_epochs):
            e_message = '[EPOCH {:0=3d}/{:0=3d}]'.format(epoch + 1, self.n_epochs)

            for phase in ['train', 'valid']:
                ep_message = e_message + '[' + phase.upper() + ']'
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                losses, preds, labels = [], [], []
                batch_num = len(dataloader[phase])
                for ith_batch, data in enumerate(dataloader[phase]):
                    ecg, label = [d.cuda() for d in data] if self.cuda else data

                    pred = self.model(ecg)
                    loss = self.criterion(pred, label)
                    if phase == 'train':
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                    pred = torch.sigmoid(pred)
                    pred = pred.data.cpu().numpy()
                    label = label.data.cpu().numpy()

                    bin_pred = np.copy(pred)
                    bin_pred[bin_pred > 0.5] = 1
                    bin_pred[bin_pred <= 0.5] = 0
                    f1 = f1_score(label.flatten(), bin_pred.flatten())

                    losses.append(loss.item())
                    preds.append(pred)
                    labels.append(label)

                    sr_message = '[STEP {:0=3d}/{:0=3d}]-[Loss: {:.6f} F1: {:.6f}]'
                    sr_message = ep_message + sr_message
                    print(sr_message.format(ith_batch + 1, batch_num, loss, f1), end='\r')

                preds = np.concatenate(preds, axis=0)
                labels = np.concatenate(labels, axis=0)
                bin_preds = np.copy(preds)
                bin_preds[bin_preds > 0.5] = 1
                bin_preds[bin_preds <= 0.5] = 0

                avg_loss = np.mean(losses)
                avg_f1 = f1_score(labels.flatten(), bin_preds.flatten())
                er_message = '-----[Loss: {:.6f} F1: {:.6f}]'
                er_message = '\n\033[94m' + ep_message + er_message + '\033[0m'
                print(er_message.format(avg_loss, avg_f1))

                if phase == 'valid':
                    if self.scheduler is not None:
                        self.scheduler.step(avg_f1)
                    if best_metric is None or best_metric < avg_f1:
                        best_metric = avg_f1
                        best_preds = [labels, preds]
                        best_loss_metrics = [epoch + 1, avg_loss, avg_f1]
                        torch.save(self.model.state_dict(), model_path)
                        print('[Best validation metric, model: {}]'.format(model_path))
                    print()

        best_f1, best_th = best_f1_score(*best_preds)
        np.save(thresh_path, np.array(best_th))
        print('[Searched Best F1: {:.6f}]\n'.format(best_f1))
        res_message = '[VALIDATION PERFORMANCE: BEST F1]' + '\n' \
            + '[EPOCH:{} LOSS:{:.6f} F1:{:.6f} BEST F1:{:.6f}]\n'.format(
                best_loss_metrics[0], best_loss_metrics[1],
                best_loss_metrics[2], best_f1) \
            + '[BEST THRESHOLD:\n{}]\n'.format(best_th) \
            + '=' * 100 + '\n'
        print(res_message)
        return


class ECGTrain(object):

    def __init__(self, cv=5, block_config='small', num_threads=2):

        self.cv = cv
        self.num_threads = num_threads
        self.block_config = block_config
        return

    def __model_dir(self, models_dir, i):
        model_dir = os.path.join(models_dir, str(i))
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        return model_dir

    def __train(self, trainset, validset, model_dir):
        trainer = ECGTrainer(self.block_config, self.num_threads)
        trainer.run(trainset, validset, model_dir)
        return

    def run(self, labels_csv, models_dir):
        dataset_df = pd.read_csv(labels_csv)
        dataset = dataset_df.values.tolist()

        splits = []
        labels = np.array([d[4:] for d in dataset])
        k_fold = IterativeStratification(
            n_splits=self.cv, order=1, random_state=325
        )

        for i, (trainidx, valididx) in enumerate(k_fold.split(dataset, labels)):
            trainset = [dataset[k] for k in trainidx]
            validset = [dataset[k] for k in valididx]
            splits.append([trainset, validset])

            model_dir = self.__model_dir(models_dir, i + 1)
            self.__train(trainset, validset, model_dir)

        return


def main(args):
    pipeline = ECGTrain(cv=args.cv,
                        block_config=args.block_config,
                        num_threads=args.num_threads)
    pipeline.run(labels_csv=args.labels_csv,
                 models_dir=args.models_dir)
    return


if __name__ == '__main__':
    import warnings
    import argparse

    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description='HFECG Competition -Round 1- Training Pipeline'
    )

    parser.add_argument('--labels-csv', '-l', type=str,
                        action='store', dest='labels_csv',
                        help='Label file of input signals')
    parser.add_argument('--models-dir', '-m', type=str,
                        action='store', dest='models_dir',
                        help='Directory of output models')
    parser.add_argument('--block-config', '-b', type=str,
                        action='store', dest='block_config',
                        help='Configuration of number of blocks')
    parser.add_argument('--num-threads', '-nt', type=int, default=2,
                        action='store', dest='num_threads',
                        help='Number of cross validation')
    parser.add_argument('--cv', '-c', type=int,
                        action='store', dest='cv',
                        help='Number of cross validation')
    parser.add_argument('--gpu', '-g', type=str,
                        action='store', dest='gpu',
                        help='Device NO. of GPU')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
