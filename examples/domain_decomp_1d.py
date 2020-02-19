import os
from tqdm import tqdm

import torch
from torch import optim
from torch.distributed import rpc
from torch.utils.data import DataLoader, Dataset

from luna.rpc.env import Env
from luna.nn.domain_decomposition import DomainDecomp1d


rank = int(os.environ.get('RANK'))
world_size = int(os.environ.get('WORLD_SIZE'))
num_workers = world_size - 1

if rank == world_size - 1:
    rank_name = 'master'
else:
    rank_name = f'worker{rank}'


class RandomDataset(Dataset):

    def __len__(self):
        return 1000

    def __getitem__(self, ix):
        return torch.randn(3, 1000), torch.randint(10, (1,)).item()


class Trainer:

    def __init__(self, dataset, halo_size=1, num_workers=2, batch_size=8):
        self.model = DomainDecomp1d(halo_size, num_workers, rank, 10)
       
        self.dataloader = DataLoader(
            dataset, num_workers=2, batch_size=batch_size, shuffle=True
        )
        self.optim = optim.SGD(self.model.parameters(), lr=1e-3)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, num_epochs=10):
        for ep in range(num_epochs):
            itbar = self.dataloader
            itbar = tqdm(itbar, desc='iter')
            for X, Y in itbar:
                print(f'X shape: {X.shape}')
                self.optim.zero_grad()
                # TODO chunk X and distribute to workers
                tile_len = X.shape[-1] // num_workers
                # this is a pretty inefficient way to do things
                chunks = [rpc.RRef(
                        X[..., tile_len * i : tile_len * (i+1)],
                    ) for i in range(world_size)]

                pred = self.model(chunks)

                loss = self.criterion(pred, Y)
                loss.backward()
                self.optim.step()


if __name__ == '__main__':
    rpc.init_rpc(rank_name, rank=rank, world_size=world_size)

    if rank_name == 'master':
        trainer = Trainer(RandomDataset())
        trainer.train()

    rpc.shutdown()
