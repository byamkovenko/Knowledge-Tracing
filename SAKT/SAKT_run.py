# this is simplified SAKT script without all the artifacting and cat mappings.
# this takes a little over 4 hours on TD dataset to train for 20 epochs.
# this same training run with DI aritfacting takes over 12 hours.


import torch
import pandas as pd
import logging
from google.cloud import bigquery
import wandb
import importlib
import train_test_loop
from SAKT_model import SAKT
from sakt_data_prep_scripts import seq_transformer, seq_len_count, collate_fn, DKT_dataset, splitter
from train_test_loop import train_and_test
importlib.reload(train_test_loop)
logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s', level=logging.INFO)

# script parameters up top for easy manipulation
QUERY_LIMIT = None  # change to "None" for full data
BQ_PROJECT = 'ml-infra-test' # alt: 'khanacademy.org:deductive-jet-827'
DATASET_DESCRIPTOR = 'TD_6th_grade_sy22-23'
WANDB_PROJECT_NAME = 'mmv-testing-time-model'
SKILLS_AS_KEYS = True # If False will use Questions for keys and values in attention layer insted

# set up
wandb.login()
torch.cuda.empty_cache()
device = (torch.device('cuda') if torch.cuda.is_available()
    else torch.device('cpu'))
logging.info(f"Training on device {device}.")

# pull the training data
# khanacademy.org:deductive-jet-827.bogdan.v2_six_grade_licensed_item_attempts_22_23 for licensed ~3.4m interactions
# khanacademy.org:deductive-jet-827.bogdan.six_grade_TD_item_attempts_22_23 for teacher directed ~97m interactions
logging.info(f"Querying training data")
client=bigquery.Client(project=BQ_PROJECT)
query="""
SELECT kaid, item_id, exercise_id, exercise_slug, course_exer_order, item_cat_code, is_correct, seconds_between_attempts 
FROM `khanacademy.org:deductive-jet-827.maya.sy22_grade6_item_attempts` 
WHERE data_version_date = '2024-04-11' 
    AND segment = 'TD' 
    AND seconds BETWEEN 2 AND 2000 
    AND prefilter_skill_count > 4 
    AND prefilter_attempts_count > 16
    AND kaid_rand<.03
ORDER BY kaid, item_order
"""
if QUERY_LIMIT:
    query += f"LIMIT {QUERY_LIMIT}"

job=client.query(query)
inter_df=job.to_dataframe()
inter_df['user_id']=inter_df['kaid'].astype('category').cat.codes
inter_df['item_cat_code']=inter_df['item_id'].astype('category').cat.codes

# Convert data into sequences and prepare training and testing batches by padding sequences and converting to tensor
logging.info(f"Converting into sequences")
batch_size=100
user_seq_df, n_inters= seq_transformer(inter_df,
                                    seq_length=200,
                                    user_column = 'user_id',
                                    skill_column= 'course_exer_order',
                                    item_column='item_cat_code',
                                    correct_column='is_correct',
                                    time_column='seconds_between_attempts')

seq_dataset = DKT_dataset(user_seq_df)
train_dataset, test_dataset = splitter(seq_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, collate_fn = collate_fn, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, collate_fn = collate_fn, shuffle=False)

# Initialize model params
logging.info(f"Creating a model instance")
max_seq_length = seq_len_count(user_seq_df['correct_seq'])
question_num = max(inter_df['item_cat_code']) + 1
skill_num = max(inter_df['course_exer_order'])+1
num_head = 4 # number of attention heads
emb_size = 63 # emb dimension for question/skill input
# d_model = emb_size # dimension of feed forward layers and linear projection of k/q/v, typically same as emb_size
# d_feature = int(emb_size/num_head) # if multihead attention is used, needed to specify shape of input in attention layer and denominator in attention score calc
# d_k = d_feature
num_layer = 1 # number of transformer blocks

# NOTE: BCE Loss is coded in the model class
lr=.003
model = SAKT(num_layer=num_layer,
            max_seq_length=max_seq_length,
            num_head=num_head,
            question_num=question_num,
            skill_num=skill_num,
            emb_size=emb_size,
            dropout=.1,
            skills_as_keys=SKILLS_AS_KEYS,
            time_model=True).to(device)
model=model.double()

# optimizer to work with lr scheduler
optimizer = torch.optim.Adam(model.parameters(), lr = lr)

n_epochs=2
config=dict(
    epochs = n_epochs,
    learning_rate = lr,
    embedding_size = emb_size,
    attention_heads = num_head,
    transformer_blocks = num_layer,
    batch_size = batch_size,
    dataset=DATASET_DESCRIPTOR,
    key_value_input = "skills" if SKILLS_AS_KEYS else "questions",
    query_input = "questions",
    max_sequence = max_seq_length,
    architecture = 'SAKT'

)

# I think this is the main wrapper to run the pipeline.
# for this to work we need to define, model, training/testing funct, and the data loaders
def model_pipeline(hyperparameters):
    with wandb.init(project=WANDB_PROJECT_NAME, config=hyperparameters):
        config=wandb.config

        train_and_test(model, train_loader, test_loader, n_epochs=n_epochs, optimizer=optimizer, device=device)

    return(model)

logging.info(f"Initiating the training run")
model = model_pipeline(config)
