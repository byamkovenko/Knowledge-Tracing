import torch
import pandas as pd
import logging
from google.cloud import bigquery
import wandb
import importlib
import LSTM_train_test_loop
from LSTM_model import DKT_Model
from LSTM_data_prep_script import seq_transformer, seq_len_count, collate_fn, DKT_dataset, splitter
from LSTM_train_test_loop import train #, checkpoint
importlib.reload(LSTM_train_test_loop)
from lstm_checkpoint import generate_gcs_filepath, send_model_artifact_to_gcs, generate_model_gcs_uri_artifact


wandb.login()
torch.cuda.empty_cache()

device = (torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu'))

MODEL_NAME="LSTM"

# load the data
print(f"Querying data")
#client=bigquery.Client(project='ml-infra-test')
client=bigquery.Client(project='khanacademy.org:deductive-jet-827')

# query="""
# SELECT * FROM `khanacademy.org:deductive-jet-827.bogdan.v2_six_grade_licensed_item_attempts_22_23` 
# order by kaid, item_order 
# """
query="""
SELECT kaid, course_exer_order, item_id, item_order, item_cat_code, is_correct, seconds_between_attempts
FROM `khanacademy.org:deductive-jet-827.maya.sy22_grade6_item_attempts` 
WHERE data_version_date = "2024-04-11"
  AND segment = "TD"  
  AND seconds BETWEEN 2 and 2000
  AND prefilter_skill_count > 4
  AND prefilter_attempts_count > 16
  AND kaid_rand<.03

order by kaid, item_order 
"""
job=client.query(query)
inter_df=job.to_dataframe()
inter_df['user_id']=inter_df['kaid'].astype('category').cat.codes
inter_df['item_cat_code']=inter_df['item_id'].astype('category').cat.codes


# add question embedding data 
# with open('embedding_dict.json', 'r') as fp:
#     embedding_dict = json.load(fp)
#inter_df['embedding'] = inter_df['item_id'].map(embedding_dict)

# convert to sequences
print(f"Converting data to sequences")
seqs,_=seq_transformer(inter_df, 
                       200, 
                       'user_id', 
                       'course_exer_order', 
                       'item_cat_code', 
                       'is_correct',
                       'seconds_between_attempts')
seqs['seq_length']=seqs['problem_seq'].apply(lambda x: len(x)) # this will be needed for packing/ordering sequences

# prepare dataloaders
print(f"Preparing dataloaders")
batch_size=100
dt=DKT_dataset(seqs)
train_dataset, test_dataset = splitter(dt)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, collate_fn = collate_fn, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, collate_fn = collate_fn, shuffle=False)

# initialize the model
print(f"Initializing the model")
question_num=max(inter_df['item_cat_code'])+1
skill_num=max(inter_df['course_exer_order'])+1
emb_size=100
num_layer=1
hidden_size=200
n_epochs = 2
dropout_rate=.2
lr=0.003
time_model=False

model=DKT_Model(emb_size=emb_size, skill_num=skill_num, question_num=question_num, hidden_size=hidden_size, num_layer=num_layer, batch_size=batch_size, dropout_rate=dropout_rate, time_model=time_model).to(device)  
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# run
print(f"Starting the training run")
#out=train(model, train_loader, test_loader, optimizer=optimizer, device=device, n_epochs=n_epochs)


config=dict(
    question_num=question_num,
    skill_num=skill_num,
    epochs = n_epochs,
    learning_rate = lr,
    embedding_size = emb_size,
    num_layer = num_layer,
    batch_size = batch_size,
    dataset="TD_6th_1_year",
    seq_input = "questions",
    architecture = 'LSTM',
    dropout_rate=dropout_rate,
    time_model=time_model,
 
)

def model_pipeline(hyperparameters):
    with wandb.init(project="LSTM_TD_bogdan", config=hyperparameters):
        artifact_gcs_filepath = generate_gcs_filepath(wandb.run, MODEL_NAME)
        config=wandb.config
        send_model_artifact_to_gcs(DKT_Model, artifact_gcs_filepath)
        train(model, train_loader, test_loader, n_epochs=n_epochs, optimizer=optimizer, device=device, artifact_gcs_filepath=artifact_gcs_filepath)
        generate_model_gcs_uri_artifact(MODEL_NAME, wandb.run, artifact_gcs_filepath)

    return(model)

print(f"Initiating the training run")
model = model_pipeline(config)
