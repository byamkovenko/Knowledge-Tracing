import torch
import numpy as np
import wandb
import logging
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from wandb import Image

from transformers import get_linear_schedule_with_warmup
from khan_recommender_training.models.checkpoint import checkpoint

# configure the logger for metrics to log out to the console
# while also logging to wandb

logger = logging.getLogger(__name__)

# ONNX_PATH='c:/Users/BogdanYamkovenko/local_python_code/saved_models'

device = (torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu'))

# alternative to save model data locally if artifact_gcs_filepath=None
def local_checkpoint(model, optimizer, filename):
    torch.save({'optimizer': optimizer.state_dict(),
                'model': model.state_dict(),
                }, filename)


def train_and_test(model, train_loader, test_loader, n_epochs, optimizer, device, 
                   artifact_gcs_filepath=None, early_stop_thresh=5):
    wandb.watch(model, log="all", log_freq=1)
    model.train()
    best_val_loss=np.inf
    best_epoch=0
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

        logging.info(f"--Starting forward pass for epoch {epoch}")
        for idx, batch in enumerate(train_loader):
            for key in batch:
                batch[key] = batch[key].to(device)
            optimizer.zero_grad()
            outdict=model(batch)                               ## forward pass one-liner
            prediction = outdict['prediction'].flatten()
            label = outdict['label'].flatten()
            mask = label > -1
            mask = mask.to(device)
            loss = model.loss(batch, outdict)
            train_loss.append(loss.detach().cpu().data.numpy())
            train_total += label[mask].shape[0]
            train_correct += (torch.gt(prediction[mask],.5).float() ==  label[mask].float()).sum().item()
            train_preds_list.append(prediction[mask].cpu())
            train_labels_list.append(label[mask].cpu())
            loss.backward()                                    ## backprop one-liner
            optimizer.step()
            scheduler.step() #adjust lr, make sure it's after opt.step

        for param_group in optimizer.param_groups:
            logger.info("Current learning rate is: {}".format(param_group['lr']))

        model.eval()
        with torch.no_grad():
            for idx, val_batch in enumerate(test_loader):
                for key in val_batch:
                    val_batch[key] = val_batch[key].to(device)
                val_outdict=model(val_batch)
                preds = val_outdict['prediction'].flatten()
                label = val_outdict['label'].flatten()
                mask = label > -1
                mask = mask.to(device)
                val_preds = preds[mask]
                val_label = label[mask]
                val_preds_list.append(val_preds.cpu())
                val_labels_list.append(val_label.cpu())
                val_total += val_label.shape[0]
                val_correct += (torch.gt(val_preds,.5).float() ==  val_label.float()).sum().item()   # threshold to manage true negative rate
                vloss = model.loss(val_batch, val_outdict)
                val_loss.append(vloss.detach().cpu().data.numpy())
                negatives+= val_label[val_label==0].shape[0]
                positives+= val_label[val_label==1].shape[0]
                confusion_vector=torch.gt(val_preds,.5)/val_label # 1=TP, inf=FP, nan=TN, 0=FN,
                FP += torch.sum(confusion_vector == float('inf')).item()
                TN += torch.sum(torch.isnan(confusion_vector)).item()
                FN += torch.sum(confusion_vector == 0).item()
                TP += torch.sum(confusion_vector==1).item()


        logging.info('Epoch: {}, N_training: {}, N_Val: {}, train_loss: {}, val_loss: {}, \
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
        
        # calibration plot with sample sizes in the bins
        val_labels_np=torch.cat(val_labels_list).detach().cpu().numpy()
        val_preds_np=torch.cat(val_preds_list).detach().cpu().numpy()
        bins = np.linspace(0, 1, num=10 + 1)
        bin_ids = np.digitize(val_preds_np, bins) - 1
        prob_true, prob_pred, bin_counts = [], [], []

        # Calculate statistics for each bin
        for bin_id in range(len(bins) - 1):
            # Create a mask for samples in this bin
            mask = (bin_ids == bin_id)

            # Calculate statistics
            prob_true.append( val_labels_np[mask].mean() )
            prob_pred.append( val_preds_np[mask].mean() )
            bin_counts.append( mask.sum() )

        plt.figure(figsize=(10, 10))
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.plot(prob_pred, prob_true, marker='.')
        scaled_bin_counts = [i/500 for i in bin_counts]  # scale up for clarity in plot
        plt.scatter(prob_pred, prob_true, s=scaled_bin_counts)

        # Adding bin_counts as labels to each dot
        for i in range(len(prob_pred)):  
            plt.annotate(str(bin_counts[i]), (prob_pred[i], prob_true[i]),
                        textcoords="offset points", 
                         xytext=(0,10),
                         ha='center')


        plt.ylabel("Fraction of Positives (is_correct)")
        plt.xlabel("Mean Predicted Value")
        plt.title("Calibration Plot, bins=10")
        plt.savefig("calibration_plot.png")
        plt.close()

        # Checkpoint save and early stopping
        if np.mean(val_loss) < best_val_loss:
            best_val_loss = np.mean(val_loss)
            best_epoch = epoch
            if artifact_gcs_filepath:
                checkpoint(model, optimizer, artifact_gcs_filepath)
            else:
                local_checkpoint(model, optimizer, "test_best_model.pth")
        elif epoch-best_epoch > early_stop_thresh:
            logging.info('Early stopping criterion reached at epoch %d' % epoch)
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
                "val_FP" : FP,
                "calibration_plot": wandb.Image("calibration_plot.png")
               
            })
