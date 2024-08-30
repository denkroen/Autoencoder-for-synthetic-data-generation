import torch
import pytorch_lightning as pl

from HARWindows import HARWindows
from torch.utils.data import DataLoader



class LARADataModule(pl.LightningDataModule):

    def __init__(self, datadir, batch_size, num_workers, synt_generation=False, train_on_synthetic=False):
        super().__init__()

        self.data_dir = datadir
        self.synt_generation = synt_generation
        self.train_on_synthetic = train_on_synthetic
        if self.synt_generation:
            self.batch_size = 1
        else:
            self.batch_size = batch_size

        self.num_workers = num_workers
        
        pass

    """def prepare_data(self):
        pass"""

    def setup(self, stage):
        if self.synt_generation:
            self.test_ds = HARWindows(csv_file=self.data_dir+"balanced.csv", root_dir=self.data_dir)
        else:
            self.test_ds = HARWindows(csv_file=self.data_dir+"test.csv", root_dir=self.data_dir)

        
        self.val_ds = HARWindows(csv_file=self.data_dir+"val.csv",root_dir=self.data_dir)
        if self.train_on_synthetic:
            self.train_ds  = HARWindows(csv_file=self.data_dir+"train_final_synthetic.csv",root_dir=self.data_dir)
        else:            
            self.train_ds  = HARWindows(csv_file=self.data_dir+"train_final.csv",root_dir=self.data_dir)




        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )