import torch
import numpy as np
import wandb
import logging
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from wandb import Image
from transformers import get_linear_schedule_with_warmup
from lstm_checkpoint import checkpoint

device = (torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu'))

# def checkpoint(model, optimizer, filename):
#     torch.save({'optimizer': optimizer.state_dict(),
#                 'model': model.state_dict(),
#                 }, filename)


def train(model, train_loader, test_loader, optimizer, device, n_epochs, artifact_gcs_filepath):
    wandb.watch(model, log="all", log_freq=1)
    model.train()
    best_val_loss=np.inf
    best_epoch=0
    early_stop_thresh=3
    # setup for lr scheduler
    total_steps = len(train_loader) * n_epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps = total_steps * 0.1, # 10% of train steps for warmup
                                           num_training_steps = total_steps)

    for epoch in range(0, n_epochs):
        train_loss=[]
        val_loss=[]
        train_preds_list=[]
        train_labels_list=[]
        train_correct=0
        train_total=0

        val_preds_list=[]
        val_labels_list=[]
        val_correct=0
        val_total=0
        negatives=0
        positives = 0
        TP=0
        FP=0
        TN=0
        FN=0


        for idx, batch in enumerate(train_loader):
            model.train()
            for key in batch:
                batch[key] = batch[key].to(device)
            optimizer.zero_grad()
            out_dict,_ = model(batch)
            loss, labels, predictions = model.dkt_loss_function(out_dict, batch)
            train_loss.append(loss.item())
            train_total += labels.shape[0]
            train_correct += (torch.gt(predictions,.5).float() ==  labels.float()).sum().item()
            train_preds_list.append(predictions.cpu())
            train_labels_list.append(labels.cpu())
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        model.eval()
        with torch.no_grad():
            for idx, val_batch in enumerate(test_loader):
                for key in val_batch:
                    val_batch[key] = val_batch[key].to(device)
                val_out_dict,_ = model(val_batch)
                v_loss, labels, predictions = model.dkt_loss_function(val_out_dict, val_batch) 
                val_loss.append(v_loss.item())
                val_preds_list.append(predictions.cpu())
                val_labels_list.append(labels.cpu())
                val_total += labels.shape[0]
                val_correct += (torch.gt(predictions,.5).float() ==  labels.float()).sum().item()
                negatives+= labels[labels==0].shape[0]
                positives+= labels[labels==1].shape[0]
                confusion_vector=torch.gt(predictions,.5)/labels # 1=TP, inf=FP, nan=TN, 0=FN,
                FP += torch.sum(confusion_vector == float('inf')).item()
                TN += torch.sum(torch.isnan(confusion_vector)).item()
                FN += torch.sum(confusion_vector == 0).item()
                TP += torch.sum(confusion_vector==1).item()

        print('Epoch: {}, N_training: {}, N_Val: {}, train_loss: {}, val_loss: {}, \
                    train_acc: {}, val_acc: {}, train_auc: {}, val_auc: {}'. format(
                    epoch,
                    torch.cat(train_labels_list).size(),
                    torch.cat(val_labels_list).size(),
                    np.mean(train_loss),
                    np.mean(val_loss),
                    train_correct/train_total,
                    val_correct/val_total,
                    roc_auc_score(torch.cat(train_labels_list).detach(), torch.cat(train_preds_list).detach()),
                    roc_auc_score(torch.cat(val_labels_list).detach(), torch.cat(val_preds_list).detach())
                    ))
        # Checkpoint save and early stopping
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            best_epoch = epoch
            checkpoint(model, optimizer, artifact_gcs_filepath)  
        elif epoch-best_epoch > early_stop_thresh:
            print('Early stopping criterion reached at epoch %d' % epoch) # comment out for artifact run
            break


        wandb.log(
            {
                "train_acc": train_correct/train_total ,
                "test_acc": val_correct / val_total,
                "train_auc": roc_auc_score(torch.cat(train_labels_list).detach(), torch.cat(train_preds_list).detach()),
                "test_auc": roc_auc_score(torch.cat(val_labels_list).detach(), torch.cat(val_preds_list).detach()),
                "train_loss": np.mean(train_loss),
                "test_loss": np.mean(val_loss),
                "test_recall": TP/(TP+FN),
                "test_precision": TP/(TP+FP),
                "val_tn_rate": TN/(FP+TN),
                "val_TN" : TN,
                "val_TP" : TP,
                "val_FN" : FN,
                "val_FP" : FP
                              
            })

        
        
def prediction(model, loader):
    model.eval()
    all_preds=[]
    users=[]
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            for key in batch:
                batch[key] = batch[key].to(device)
            _, pred_vector = model(batch)
            user_id = batch['user_id']
            users.append(user_id)
            length_index=batch['seq_length']-2 # -1 for lost last item in packed_seq() and -1 for 0 indexing
            seq_length_indx=length_index.unsqueeze(-1)
            idx=torch.arange(0,pred_vector.shape[0]).unsqueeze(-1)
            preds=pred_vector[idx, seq_length_indx,:] # one row per student, all items
            all_preds.append(preds)
    return torch.sigmoid(torch.cat(all_preds, dim=0)) #, torch.cat(user_id, dim=0)        
