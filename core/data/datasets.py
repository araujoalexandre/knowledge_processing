import json
import tarfile
from torch.utils.data import IterableDataset


class TarIterableDataset(IterableDataset):
    
    def __init__(self, tar_path):
        self.tar_path = tar_path
        with tarfile.open(tar_path, 'r') as tar:
            self.lenght = len(tar.getmembers())

    def __len__(self):
        return self.lenght

    def __iter__(self):
        with tarfile.open(self.tar_path, 'r') as tar:
            for member in tar:
                f = tar.extractfile(member)
                if f is not None:
                    content = f.read()
                    data = json.loads(content.decode('utf-8'))
                    yield member.name, data['url'], data['text']


