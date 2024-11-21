import torch
from torch.utils.data import Dataset, DataLoader
from utils.transform import bacteria_train_transform, bacteria_valid_transform

class PointsFolder(Dataset):
    """
    Custom dataset class for loading data and labels with optional transforms.
    """
    def __init__(self, data, label, transform=None) -> None:
        super(PointsFolder, self).__init__()
        self.data = data
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        # Retrieve data and label
        sample = self.data[index]
        target = self.label[index]

        # Apply transform if available
        if self.transform:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return self.data.shape[0]


def make_dataloader(data, label, batch_size=16, shuffle=True, num_workers=4, train=True):
    """
    Create DataLoader for training or validation/testing.
    :param data: Input data (numpy array or torch Tensor)
    :param label: Corresponding labels (numpy array or torch Tensor)
    :param batch_size: Batch size for DataLoader
    :param shuffle: Whether to shuffle the dataset
    :param num_workers: Number of worker threads for DataLoader
    :param train: Whether this is a training loader
    :return: DataLoader instance
    """
    # Select transform based on train/validation mode
    transform = bacteria_train_transform if train else bacteria_valid_transform

    # Create dataset
    dataset = PointsFolder(data=data, label=label, transform=transform)

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader
