import torch, pickle
import numpy as np, random
from augmentations import augs
from utils import DEVICE, Task


# -------------------------------------------------------------------
def load(dataset_name):
    filenames = {
        "DSA": "data/preprocessed/Daily and Sports Activities Data Set/data.pickle",
        "PAMAP": "data/preprocessed/PAMAP2/data.pickle",
        "WISDM-phone": "data/preprocessed/WISDM/data_accel_phone.pickle",
        "WISDM-watch": "data/preprocessed/WISDM/data_accel_watch.pickle",
        "REALDISP-ideal": "data/preprocessed/REALDISP/data_ideal.pickle",
        "REALDISP-mutual": "data/preprocessed/REALDISP/data_mutual.pickle",
        "REALDISP-self": "data/preprocessed/REALDISP/data_self.pickle",
        "VPA": "data/preprocessed/Vicon Physical Action Data Set/data.pickle",
        "ADL": "data/preprocessed/ADL_HMP_Dataset/data.pickle",
    }

    with open(filenames[dataset_name], 'rb') as handle:
        return pickle.load(handle)


# -------------------------------------------------------------------
class TaskGenerator:
    def __init__(self, n, k, q, background=True, p_split=0.5, dataset_name=0):
        self.n = n
        self.k = k
        self.q = q

        self.data = load(dataset_name)

        persons = list(self.data.keys())
        persons = sorted(persons)
        nb = int(len(persons) * p_split)
        self.persons = persons[:nb] if background else persons[nb:]  # divide persons into meta-train and meta-test

        classes = list(set([c for j in self.data.keys() for c in self.data[j].keys()]))
        classes = sorted(classes)
        nb = int(len(classes) * p_split)
        self.classes = classes[:nb] if background else classes[nb:]  # divide classes into meta-train and meta-test

    # -------------------------------------------------------------------
    # Sample a support set (n*k examples) and a query set (n*q examples)
    def batch(self):
        # Data of a randomly selected person
        j = random.choice(self.persons)
        dico = self.data[j]

        # Randomly select n classes from person j
        classes_j = list(set(self.classes).intersection(set(dico.keys())))
        classes = random.sample(classes_j, min(self.n, len(classes_j)))

        # Randomly map each selected class to a label in {0, ..., n-1}
        labels = random.sample(range(self.n), self.n)
        label_map = dict(zip(classes, labels))

        # Randomly select k support examples and q query examples from each of the selected classes
        X_sp, y_sp, X_qr, y_qr = [], [], [], []
        for c in classes:
            examples = random.sample(list(dico[c]), len(dico[c]))
            Xc_sp = examples[:self.k]
            X_sp += Xc_sp
            y_sp += [label_map[c] for _ in Xc_sp]
            Xc_qr = examples[self.k:self.k + self.q]
            X_qr += Xc_qr
            y_qr += [label_map[c] for _ in Xc_qr]

        # Transform these lists to appropriate tensors and return them
        X_sp, y_sp, X_qr, y_qr = [torch.from_numpy(np.array(lst)).to(DEVICE).float() for lst in [X_sp, y_sp, X_qr, y_qr]]
        y_sp, y_qr = y_sp.long(), y_qr.long()

        return Task(X_sp, y_sp, X_qr, y_qr, 1)


# -------------------------------------------------------------------
def augment_support_set(tsk):
    X_augs = augs(tsk.X_sp)
    nb_augs = 1 + len(X_augs)
    XX_sp = torch.cat([tsk.X_sp] + list(X_augs), dim=0)  # flatten all augmentations in the same vector with size (nb_augs*samples, n_features, n_channels)
    yy_sp = tsk.y_sp.repeat(nb_augs)

    tsk = Task(XX_sp, yy_sp, tsk.X_qr, tsk.y_qr, nb_augs)
    return tsk