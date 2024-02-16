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

        classes_j = list(set(self.classes).intersection(set(tgen.data[self.person].keys())))
        if self.classes != classes_j:
            self.classes = classes_j
        self.data_dict = {i: tgen.data[self.person][c] for i, c in enumerate(self.classes)}

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

        data_dict_tmp = {}
        for p in self.persons:
            classes_p = list(set(self.classes).intersection(set(tgen.data[p].keys())))
            for c in classes_p:
                if c in data_dict_tmp.keys():
                    data_dict_tmp[c] = torch.cat((data_dict_tmp[c], torch.from_numpy(tgen.data[p][c])))
                else:
                    data_dict_tmp[c] = torch.from_numpy(tgen.data[p][c])

        self.data_dict = {i: v for i, (k, v) in enumerate(data_dict_tmp.items())}
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



