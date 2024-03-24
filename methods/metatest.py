from utils import func_call, accuracy, DEVICE
import torch, copy
from collections import defaultdict
from TaskGenerator import augment_support_set


# -------------------------------------------------------------------
def adapt_and_evaluate(model, loss_fn, lr_inner, tsk, steps=20, anil=False, anil_last=False, with_aug=False):
    cmodel = copy.deepcopy(model).to(DEVICE)

    if with_aug:
        tsk = augment_support_set(tsk)

    if anil and anil_last:
        optim_params = {name: value for name, value in dict(cmodel.named_parameters()).items() if 'last' in name}# or 'dense_block2' in name}
    elif anil and not anil_last:
        optim_params = {name: value for name, value in dict(cmodel.named_parameters()).items() if 'last' in name or 'dense_block2' in name}
    else:
        optim_params = dict(cmodel.named_parameters())

    optimizer = torch.optim.SGD(list(optim_params.values()), lr_inner)
    history = defaultdict(list)

    for step in range(steps + 1):
        y_sp_pred, y_qr_pred = func_call(cmodel, optim_params, tsk)

        # Evaluate current model on the test data
        acc = accuracy(y_qr_pred, tsk.y_qr)
        history["pred"].append(y_qr_pred.cpu().detach())
        history["acc"].append(acc)

        # Adapt the model using training data
        if with_aug:
            chunked_y_sp_pred, chunked_y_sp = y_sp_pred.chunk(tsk.nb_augs), tsk.y_sp.chunk(tsk.nb_augs)
            losses = torch.stack([loss_fn(y_pred, y_true) for (y_pred, y_true) in zip(chunked_y_sp_pred, chunked_y_sp)])
            if hasattr(cmodel, 'weights'):
                weights = torch.nn.functional.sigmoid(cmodel.weights)
                loss = torch.sum(losses * weights)
            else:
                loss = torch.sum(losses)
        else:
            loss = loss_fn(y_sp_pred, tsk.y_sp)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return history

