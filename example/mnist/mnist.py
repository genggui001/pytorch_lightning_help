import os
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from torch import nn
from torch.utils.data import DataLoader, random_split

from torchvision import transforms
from torchvision.datasets import MNIST

from pytorch_lightning_help.callbacks import EvalCallback, ProgressBar, ModelCheckpoint

from sklearn.metrics import classification_report, accuracy_score

from argparse import ArgumentParser


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# 定义超参数
parser = ArgumentParser()
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
hparams = parser.parse_args()


# 代码开始
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307, ), (0.3081, ))])

train_dataset = MNIST('./tmp', train=True, download=True, transform=transform)

train_dataset, val_dataset = random_split(train_dataset, [55000, 5000])
test_dataset = MNIST('./tmp', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, num_workers=8)

val_x = []
val_true_y = []

for x, y in iter(val_dataset):
    val_x.append(x)
    val_true_y.append(y)

test_x = []
test_true_y = []

for x, y in iter(test_dataset):
    test_x.append(x)
    test_true_y.append(y)

# 模型定义


class CNNNet(pl.LightningModule):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def test_step(self, batch, batch_idx):
        output = self(batch)
        return {'output': output.tolist()}


class TrainModel(pl.LightningModule):
    def __init__(self, hparams):
        super(TrainModel, self).__init__()
        self.hparams = hparams

        self.cnnnet = CNNNet()

    def forward(self, x):
        raise Exception("训练模型无正向传播")

    def training_step(self, batch, batch_idx):
        data, target = batch
        output = self.cnnnet(data)
        loss = F.nll_loss(output, target)

        self.log('train_loss', loss.tolist())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


def predict(x, runner, model, batch_size):
    # 备份原有模型
    back_trainer_model = runner.model

    data_loader = DataLoader(x, batch_size, num_workers=8)

    y_pre_tmp = runner.test(model, data_loader, verbose=False)

    y_pre = []
    for item in y_pre_tmp:
        y_pre.extend(item['output'])
    del y_pre_tmp

    y_pre = np.array(y_pre).argmax(axis=-1)

    # 还原原有模型
    runner.model = back_trainer_model
    return y_pre

# 模型校验


def evaluation(y_true, y_pre):
    score = accuracy_score(y_true, y_pre)
    other_info = classification_report(y_true, y_pre,
                                       digits=4).replace("\n", "\n    ")

    return score, other_info


class MyEvalCallback(EvalCallback):
    def val_operation(self, trainer, train_model, outputs):
        val_y_pre = predict(val_x, trainer, train_model.cnnnet, batch_size=8)
        now_score, other_info = evaluation(val_true_y, val_y_pre)

        train_model.logger.experiment.add_scalar("val_acc", now_score,
                                                 train_model.current_epoch)
        train_model.logger.experiment.add_text("val_label_score", other_info,
                                               train_model.current_epoch)

        return now_score

    def test_operation(self, trainer, train_model, outputs):
        test_y_pre = predict(test_x, trainer, train_model.cnnnet, batch_size=8)
        now_score, other_info = evaluation(test_true_y, test_y_pre)

        train_model.logger.experiment.add_scalar("test_acc", now_score,
                                                 train_model.current_epoch)
        train_model.logger.experiment.add_text("test_label_score", other_info,
                                               train_model.current_epoch)

        return None


train_model = TrainModel(hparams)
trainer = pl.Trainer(gpus=1,
                     max_epochs=hparams.max_epochs,
                     callbacks=[
                         MyEvalCallback(),
                         ProgressBar(),
                         ModelCheckpoint(
                             monitor='val_score',
                             filename='mnist-{val_score:.4f}',
                             save_top_k=1,
                             save_last=True)
                     ])

trainer.fit(train_model, train_loader)
