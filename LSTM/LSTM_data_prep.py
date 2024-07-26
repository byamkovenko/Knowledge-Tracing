from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split

# version from before adding time_between_attempts
# class DKT_dataset(Dataset):
#     def __init__(self, input_seq_df):
#         super().__init__()
#         self.data=input_seq_df

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         seq_length = torch.tensor(self.data['seq_length'][idx])
#         user_id=torch.tensor(self.data['user_id'][idx]).long()
#         question=torch.tensor(self.data['problem_seq'][idx]).long()
#         correct=torch.tensor(self.data['correct_seq'][idx])
        
#         return {'seq_length': seq_length, 'user_id': user_id,  
#                 'question':question, 'correct' : correct, 
#                 } 

# added time_between_attempts with zero handling and MaxMin scaling
# added error handling 
class DKT_dataset(Dataset):
    def __init__(self, input_seq_df):
        super().__init__()
        
        self.data = input_seq_df

        # replace nan (first time btw attempts value for each student) with temporary value
        # temp_value = -999
        # for idx in range(len(input_seq_df)):
        #     # mask = input_seq_df['seconds_between_attempts'][idx]==-999
        #     # input_seq_df['seconds_between_attempts'][idx][mask] = temp_value
        #     arr=np.array(input_seq_df.loc[idx, 'seconds_between_attempts'], dtype=float)
        #     #arr = np.array(input_seq_df['seconds_between_attempts'][idx])
        #     mask = np.isnan(arr)
        #     arr[mask] = temp_value
        #     input_seq_df['seconds_between_attempts'][idx] = arr.tolist()


        # min-max calc excluding the temp_value
        
        self.min_time = np.min([(torch.tensor(times)[torch.tensor(times)!=-999]).min() for times in self.data['seconds_between_attempts']])
        self.max_time = np.max([(torch.tensor(times)[torch.tensor(times)!=-999]).max() for times in self.data['seconds_between_attempts']])
        
        # self.min_time = np.min([np.min(times[times!=-999]) for times in self.data['seconds_between_attempts']])
        # self.max_time = np.max([np.max(times[times!=-999]) for times in self.data['seconds_between_attempts']])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            seq_length = torch.tensor(self.data['seq_length'][idx])
            user_id = torch.tensor(self.data['user_id'][idx]).long()
            question = torch.tensor(self.data['problem_seq'][idx]).long()
            correct = torch.tensor(self.data['correct_seq'][idx])
            seconds_between_attempts = torch.tensor(self.data['seconds_between_attempts'][idx]).float()
            
            # create a mask for temp_values (original zeros)
            mask = (seconds_between_attempts==-999)
            # apply min-max scaling for non-temp_values
            seconds_between_attempts[~mask] = np.clip((seconds_between_attempts[~mask] - self.min_time) / (self.max_time - self.min_time),0,1)
            # convert temp_values back to -1
            seconds_between_attempts[mask] = -1

            return {'seq_length': seq_length,
                    'user_id': user_id, 
                    'question':question,
                    'correct' : correct, 
                    'seconds_between_attempts':seconds_between_attempts
                    }
        except Exception as e:
            print(f"Error in loading data at index {idx}: {str(e)}")
            return None

    
    
def seq_transformer(df, seq_length, user_column, skill_column, item_column, correct_column, time_column):
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


def seq_len_count(dat):
    """computes the length of the sequences in the seq_df, and returns max length"""
    seq_len_list=[]
    for i in dat:
        seq_len_list.append(len(i))
    max_seq_length=np.max(seq_len_list)
    return max_seq_length


def collate_fn(batch):
    # Get original positions before sorting
    batch_with_pos = [(item, i) for i, item in enumerate(batch)]
    
    # Sort a list by sequence length (in descending order) in each batch
    batch_with_pos.sort(key=lambda x: x[0]['seq_length'], reverse=True)

    # use pad_sequence for each sequence in batch
    user_id = torch.stack([item['user_id'] for item, _ in batch_with_pos])
    question = torch.nn.utils.rnn.pad_sequence([item['question'] for item, _ in batch_with_pos], batch_first=True)  
    correct = torch.nn.utils.rnn.pad_sequence([item['correct'] for item, _ in batch_with_pos], batch_first=True)  
    seconds_between_attempts=torch.nn.utils.rnn.pad_sequence([item['seconds_between_attempts'] for item, _ in batch_with_pos], batch_first=True)
    seq_length = torch.stack([item['seq_length'] for item, _ in batch_with_pos])

    # Get original indices
    orig_idx = torch.tensor([pos for _, pos in batch_with_pos]) 

    return {'seq_length': seq_length, 'user_id': user_id, 'question': question, 
            'correct': correct, 'orig_idx': orig_idx, 'seconds_between_attempts':seconds_between_attempts}

def splitter(dataset, split=.8, seed=42):
    torch.manual_seed(seed)
    dataset_length = len(dataset)
    train_len = int(dataset_length * split)  # 80% for training
    test_len = dataset_length - train_len  # 20% for testing
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    return train_dataset, test_dataset



# Q-Q Influence
# for each pair of questions compute the influence of question i on question j
# inf_mat row represents i'th question, column represents j'th question.
# input pred_matrix is a full prediction vector from LSTM for each batch
# where each batch is a sequence of length 1 for item i being correct. 

# get pred_matrix
def get_preds(model, n_questions, device):
    pred_input = pd.DataFrame(data={'item_cat_code': np.arange(n_questions),
                     'correct': np.repeat(1,n_questions),
                     # lines below only needed to make the seq_transformer fun work
                    'user_id':np.arange(n_questions),         
                    'time_between_attempts':np.random.rand(n_questions),                
                    'course_exer_order': np.arange(n_questions)})
    item_seq,_=seq_transformer(pred_input, 1, 'user_id', 'course_exer_order', 'item_cat_code', 'correct', 'seconds_between_attempts')
    # adjust length by one otherwise errors out in pack_padded_seq()
    item_seq['seq_length']=item_seq['problem_seq'].apply(lambda x: len(x)+1) 
    pr_dataset=DKT_dataset(item_seq)
    pr_loader = torch.utils.data.DataLoader(pr_dataset, batch_size = 1, collate_fn = collate_fn, shuffle=False)
    
    model.eval()
    pred_matrix = []
    for idx, batch in enumerate(pr_loader):
        for key in batch:
            batch[key] = batch[key].to(device)
        _, pred_vector = model(batch)
        pred_matrix.append(pred_vector)
    return pred_matrix


def generate_influence_mat(pred_matrix):
    pred_matrix=torch.sigmoid(torch.cat(pred_matrix).squeeze())
    num_questions = pred_matrix.shape[0]
    inf_mat = np.zeros((num_questions, num_questions))
    pred_matrix=pred_matrix.cpu().detach().numpy()
    for i in range(num_questions):
        for j in range(num_questions):
        # calculate the influence of question i on question j
            inf_mat[i,j] = pred_matrix[i,j] / np.sum(pred_matrix[:,j]) 
    return inf_mat
