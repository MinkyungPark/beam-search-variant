
import numpy as np
from fairseq.data import data_utils, BaseWrapperDataset


class EpochShuffleDataset(BaseWrapperDataset):
    def __init__(self, dataset, num_samples, seed, epoch):
        super().__init__(dataset)
        self.num_samples = num_samples
        self.seed = seed
        self.set_epoch(epoch)

    def set_epoch(self, epoch):
        with data_utils.numpy_seed(self.seed + epoch - 1):
            self.sort_order = np.random.permutation(self.num_samples)

    def ordered_indices(self):
        return self.sort_order

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False