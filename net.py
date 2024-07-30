import pytorch_lightning as pl 
import torch 
import torch.nn as nn 
import numpy as np 
import torch.optim as optim 
import torch.nn.functional as F
import chess


class ResNet(pl.LightningModule):
    def __init__(self, units, layers, lr):
        super(ResNet, self).__init__()

        #Specifics
        self.model_features = (units, layers, lr)
        device = 'cuda'
        self.value_loss_fn = nn.MSELoss()
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.lr = lr
        
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=units, kernel_size=3, padding='same'), 
            nn.BatchNorm2d(units), 
            nn.ReLU(), 
            *[Resblock(units) for _ in range(layers)] 
        )
        
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=units, out_channels=1, kernel_size=1), 
            nn.BatchNorm2d(1), 
            nn.ReLU(), 
            nn.Flatten(), 
            nn.Linear(64, 256), 
            nn.ReLU(), 
            nn.Linear(256, 1), 
            nn.Tanh()
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=units, out_channels=2, kernel_size=1), 
            nn.BatchNorm2d(2), 
            nn.ReLU()
        )

        self.policy_fc = nn.Linear(in_features=128, out_features=4608)
        self.to(device)



    def forward(self, x):
        x = self.backbone(x)

        value = self.value_head(x)

        policy = self.policy_head(x)
        policy = policy.view(policy.shape[0], 128)
        policy = self.policy_fc(policy)
        return value, policy
    
    def training_step(self, batch, batch_idx):
        val_loss, pol_loss = self._common_step(batch, batch_idx)
        self.log('value_train_loss', val_loss, on_epoch=True, on_step=False)

        if pol_loss != []:
            self.log('train_loss', pol_loss+val_loss, on_epoch=True, on_step=False)
            self.log('policy_train_loss', pol_loss, on_epoch=True, on_step=False)
            loss = val_loss + pol_loss
            return loss
        
        self.log('train_loss', val_loss, on_epoch=True, on_step=False)
        
        return val_loss 
    
    def validation_step(self, batch, batch_idx):
        val_loss, pol_loss = self._common_step(batch, batch_idx)

        self.log('value_validation_loss', val_loss, on_epoch=True, on_step=False)
        self.log('validation_loss', pol_loss+val_loss, on_epoch=True, on_step=False)
        self.log('policy_validation_loss', pol_loss, on_epoch=True, on_step=False)

        loss = val_loss + pol_loss
        return loss
    
    
    
    def _common_step(self, batch, batch_idx):
        x, true_val, true_pol = batch 
        val, policy = self.forward(x)
        
        val_loss = self.value_loss_fn(val, true_val.reshape(val.shape))
        if true_pol == []:
            return val_loss, []
        
        true_pol = torch.flatten(true_pol, 1, -1)
        policy_loss = self.policy_loss_fn(policy, true_pol.reshape(policy.shape))
    
        return val_loss, policy_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    

    def save(self):
        state_dict = self.state_dict()
        torch.save({'state_dict':state_dict},
                    'saved_models/model')


    def load(self):
        file = torch.load('saved_models/model')
        state_dict = file['state_dict']



    
class Resblock(pl.LightningModule):
    def __init__(self, filters):
        super(Resblock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, padding='same'), 
            nn.BatchNorm2d(filters), 
            nn.ReLU(), 
            nn.Conv2d(filters, filters, kernel_size=3, padding='same'), 
            nn.BatchNorm2d(filters)
        )


    def forward(self, x):
        residual = x 
        x = self.model(x)
        x += residual 
        return F.relu(x)

