import net  
import preprocessing
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch

from pytorch_lightning.callbacks import EarlyStopping, lr_finder
from pytorch_lightning.loggers import TensorBoardLogger

data_file = 'dataset/9947games.npz'
units = 128
layers = 12
lr = 1e-3

batch_size = 256
epochs = 30

model = net.ResNet(units, layers, lr)



torch.set_float32_matmul_precision('medium')

logger = TensorBoardLogger('logs/model',
                            name='logger',
                              version=None)

trainset = preprocessing.Training_Data(data_file)

trainloader = DataLoader(trainset,
                          batch_size=batch_size,
                          shuffle=True)

trainer = pl.Trainer(max_epochs=epochs,
                      callbacks=[EarlyStopping(monitor='train_loss',
                                                check_on_train_epoch_end=False),
                                 lr_finder.LearningRateFinder()], 
                      logger = logger)

trainer.fit(model, trainloader)
model.save()