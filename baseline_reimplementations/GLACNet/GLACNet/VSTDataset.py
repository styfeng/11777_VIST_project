from PIL import Image
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from build_vocab import Vocabulary
import nltk
import torchvision.transforms as transforms

class VSTDataset(Dataset):
    def __init__(self, data_json_path, data_root_dir, vocab, transforms = transforms.ToTensor()):
        with open(data_json_path) as f:
            self.data = json.load(f)
            f.close()
        self.ids = list(self.data.keys())
        self.root_dir = data_root_dir
        self.transforms = transforms
        self.vocab = vocab

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x = []
        y = []
        photo_sequence = self.data[self.ids[idx]]['img_ids']
        album_ids = [self.data[self.ids[idx]]['album_id']]
        for img_id in self.data[self.ids[idx]]['img_ids']:
            img_path = os.path.join(self.root_dir, img_id + ".jpg")
            img = Image.open(img_path)
            x.append(self.transforms(img))
        
        for i in self.data[self.ids[idx]]['sent_orders']:
            y.append(self.data[self.ids[idx]]['sent_texts'][i])

        tokens = []
        for sentence in y:
            try:
                tokens = nltk.tokenize.word_tokenize(sentence.lower())
            except Exception:
                pass

            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            target = torch.Tensor(caption)
            y.append(target)

        return torch.stack(x), y, photo_sequence, album_ids

def collate_fn(data):

    x, y, photo_sequence_set, album_ids_set = zip(*data)

    targets_set = []
    lengths_set = []

    for captions in y:
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        targets_set.append(targets)
        lengths_set.append(lengths)
    return x, targets_set, lengths_set, photo_sequence_set, album_ids_set

    
def get_loader(root, json_path, vocab, transform, batch_size, shuffle, num_workers):
    vist = VSTDataset(json_path, root, vocab, transforms = transform)
    data_loader = DataLoader(dataset = vist, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, collate_fn = collate_fn)
    return data_loader

