import math
import torch

from torch.utils.data import Sampler


class CustomSampler(Sampler):
    """Sampler to use with `ExtendedWheatDataset`.
    Based on `torch.utils.data.DistributedSampler`, but can be used
    with a single GPU as well.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        orig_images_ratio: The number of original images divided by the total number
            of images to sample.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, orig_images_ratio=0.5, num_replicas=None, rank=None, shuffle=True):
        super(CustomSampler, self).__init__(dataset)
        if not hasattr(dataset, 'num_orig_images'):
            raise AttributeError("{} is not compatible with {}: `num_orig_images` field"
                                 " is missing ".format(type(dataset), self.__name__))
        if num_replicas is None:
            num_replicas = 1
        if rank is None:
            rank = 0

        self.dataset = dataset
        self.orig_images_ratio = orig_images_ratio
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        num_orig = self.dataset.num_orig_images
        num_gen = len(self.dataset) - num_orig

        take_orig = int(round(len(self.dataset) * self.orig_images_ratio))
        take_gen = len(self.dataset) - take_orig

        if self.shuffle:
            def _rnd_indices(length, total):
                idxs = [torch.randperm(length, generator=g)[:total]
                        for _ in range(0, total, length)]
                return torch.cat(idxs)[:total]

            orig_indices = _rnd_indices(num_orig, take_orig)
            gen_indices = _rnd_indices(num_gen, take_gen) + num_orig

            indices = torch.cat([orig_indices, gen_indices])
            ii = torch.randperm(len(indices), generator=g)
            indices = indices[ii].tolist()
        else:
            def _rnd_indices(start, end, total):
                repeats = math.ceil(total / (end - start))
                idxs = idxs = [i for _ in range(repeats) for i in range(start, end)]
                return idxs[:total]

            indices = _rnd_indices(0, num_orig, take_orig)
            indices += _rnd_indices(num_orig, len(self.dataset), take_gen)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
