import torch
import random
from torch.utils.data import Dataset
from utils import DEVICE


# -------------------------------------------------------------------
class TLOneSubject(Dataset):
    def __init__(self, tgen):

        self.classes = tgen.classes

        # Randomly select one subject
        self.person = random.choice(tgen.persons)

        self.data_dict = {c: tgen.data[self.person][c] for c in self.classes}

        self.data = []
        for c, data in self.data_dict.items():
            for d in data:
                self.data.append((d, c))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx][0]
        class_id = self.data[idx][1]
        return torch.from_numpy(data).to(DEVICE).float(), class_id


# -------------------------------------------------------------------
class TLTrainingSubjects(Dataset):
    def __init__(self, tgen):

        self.classes = tgen.classes

        self.persons = tgen.persons

        self.data_dict = {}
        for p in self.persons:
            for c in self.classes:
                if c in self.data_dict.keys():
                    self.data_dict[c] = torch.cat((self.data_dict[c], torch.from_numpy(tgen.data[p][c])))
                else:
                    self.data_dict[c] = torch.from_numpy(tgen.data[p][c])

        self.data = []
        for c, data in self.data_dict.items():
            for d in data:
                self.data.append((d, c))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx][0]
        class_id = self.data[idx][1]
        return data.to(DEVICE).float(), class_id



