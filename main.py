from torch.utils.data.dataloader import DataLoader

from option import args
from train import Trainer
from utils.checkpoint import checkpoint
from utils.dataloader import TestData
from utils.utils import setup_seed
setup_seed(20)


test_loader = DataLoader(TestData(args),
                         batch_size=1, shuffle=False)

t = Trainer(args, test_loader, ckp=checkpoint(args))

t.testOnly()
