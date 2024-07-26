from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split


def seq_len_count(dat):
    """computes the length of the sequences in the seq_df, and returns max length"""
    seq_len_list=[]
    for i in dat:
        seq_len_list.append(len(i))
    max_seq_length=np.max(seq_len_list)
    return max_seq_length

# this function converts a dataset with 1 row per student/question to a sequence data frame with 1 row per student
def seq_transformer(df, seq_length, user_column, skill_column, item_column, 
                    correct_column, time_column):
    user_wise_dict = dict()
    cnt, n_inters = 0, 0
    for user, user_df in df.groupby(user_column):
        mydf = user_df[:seq_length]  # consider the first 200 interactions
        user_wise_dict[cnt] = {
            'user_id': user,
            'skill_seq': mydf[skill_column].to_numpy().tolist(),
            'problem_seq': mydf[item_column].to_numpy().tolist(),
            'correct_seq': [round(x) for x in mydf[correct_column].astype(float)],
            'seconds_between_attempts':mydf[time_column].to_numpy().tolist(),
        }
        cnt += 1
        n_inters += len(mydf)
    user_seq_df = pd.DataFrame.from_dict(user_wise_dict, orient='index')
    return user_seq_df, n_inters


class DKT_dataset(Dataset):
    '''class to convert a seq df to a torch dataset'''
    def __init__(self, input_seq_df):
        super().__init__()

        self.data=input_seq_df
        
        # min-max calc excluding the temp_value
        self.min_time = np.min([(torch.tensor(times)[torch.tensor(times)!=-999]).min() 
                                for times in self.data['seconds_between_attempts']])
        self.max_time = np.max([(torch.tensor(times)[torch.tensor(times)!=-999]).max() 
                                for times in self.data['seconds_between_attempts']])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        user_id=torch.tensor(self.data['user_id'][idx]).long()
        skill=torch.tensor(self.data['skill_seq'][idx]).long()
        question=torch.tensor(self.data['problem_seq'][idx]).long()
        correct=torch.tensor(self.data['correct_seq'][idx])
        seq_length = question.shape[0]
        seconds_between_attempts = torch.tensor(self.data['seconds_between_attempts'][idx]) .float()

        # create a mask for temp_values (original zeros)
        mask = (seconds_between_attempts==-999)
        # apply min-max scaling for non-temp_values
        seconds_between_attempts[~mask] = np.clip((seconds_between_attempts[~mask] - self.min_time) / (self.max_time - self.min_time),0,1)
        # convert temp_values back to -1
        seconds_between_attempts[mask] = -1
        
        return {'user_id': user_id, 
                'question': question, 
                'skill': skill,
                'correct': correct, 
                'seq_length': seq_length,
                'seconds_between_attempts': seconds_between_attempts
               } 

# required fun for dataloader, used to combine samples as well as for padding 
def collate_fn(batch):
    '''this fun is an arg to dataloader and is needed to create batches and pad sequences'''
    max_l = max([item['seq_length'] for item in batch])
    user_id = [item['user_id'] for item in batch]
    question = [item['question'] for item in batch]
    skill = [item['skill'] for item in batch]
    correct = [item['correct'] for item in batch]
    seconds_between_attempts = [item['seconds_between_attempts'] for item in batch]
    for i in range(len(batch)):
        question[i] = torch.cat([question[i], torch.zeros(max_l-len(question[i]))]).long()
        skill[i] = torch.cat([skill[i], torch.zeros(max_l-len(skill[i]))]).long()
        correct[i] = torch.cat([correct[i], -1*torch.ones(max_l-len(correct[i]))]).long()
        seconds_between_attempts[i] = torch.cat([seconds_between_attempts[i],
            torch.zeros(max_l-len(seconds_between_attempts[i]))]).float()
    return {'user_id': torch.stack(user_id), 
            'inter_seq': torch.stack(skill), 
            'quest_seq': torch.stack(question), 
            'label_seq': torch.stack(correct),
            'seconds_between_attempts': torch.stack(seconds_between_attempts)}


def splitter(dataset, split=.8, seed=42):
    torch.manual_seed(seed)
    dataset_length = len(dataset)
    train_len = int(dataset_length * split)  # 80% for training
    test_len = dataset_length - train_len  # 20% for testing
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    return train_dataset, test_dataset
