# This class is used to run the RoBERTa model on our dataset, and perfrom appropriate measurments

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import seaborn as sns
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import logging
from sklearn.metrics import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import pdb
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
import random


from sklearn.metrics import classification_report


from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

MAX_LEN = 512
TRAIN_BATCH_SIZE = 10
TEST_BATCH_SIZE = 10
LEARNING_RATE = 5e-6
ADD_TRAIN = True
EPOCHS = 4
WEIGHT_DECAY = 0.01

SAVE_PATH = "./models"
directory_path = "./test-mlm"

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if is_tf_available():
        import tensorflow as tf

        tf.random.set_seed(seed)

class DreadditData(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = self.data.label
        self.max_len = max_len
        self.data_id = dataframe.id

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        
        data_id = self.data_id[index]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float),
            'data_id': torch.tensor(data_id, dtype=torch.int)
        }
    
class RobertaClass(torch.nn.Module):
    def __init__(self, add_train = True):
        super(RobertaClass, self).__init__()
        if (add_train is True):
            print("Initializing additionaly trained roberta model")
            self.roberta_layer = RobertaModel.from_pretrained(directory_path)
        else:
            print("Initializing basic roberta model")
            self.roberta_layer = RobertaModel.from_pretrained('roberta-base')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

    # TODO: mozda train metodu stavi ovdje?
    def forward(self, input_ids, attention_mask, token_type_ids):
        roberta_out = self.roberta_layer(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = roberta_out[0]
        pooler = hidden_state[:, 0]
        s = self.pre_classifier(pooler)
        h = torch.nn.ReLU()(s)
        h_dropout = self.dropout(h)
        logits = self.classifier(h_dropout)
        return logits

def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

def train(epoch, model, training_loader):
    loss_fun = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    acc_list = []
    model.train()
    for i,data in tqdm(enumerate(training_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_fun(outputs, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if (i%500 == 0):
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print(f"Training Loss per 500 steps: {loss_step}")
            print(f"Training Accuracy per 500 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")
    
    torch.save(model.state_dict(), SAVE_PATH)

    return 

def test(model, testing_loader):
    model.eval()
    loss_fun = torch.nn.CrossEntropyLoss()
    n_correct = 0; n_wrong = 0; total = 0; test_loss=0; nb_tr_steps=0; nb_tr_examples=0
    loss_list = []
    acc_list = []
    f1_list = []
    recall_list = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids).squeeze()

            loss = loss_fun(outputs, targets)
            loss_list.append(loss.item())
            big_val, big_idx = torch.max(outputs.data, dim=1)
            y_true = targets.detach().cpu().numpy()
            y_pred = big_idx.detach().cpu().numpy()
            acc_list.append(accuracy_score(y_pred, y_true)*100)
            f1_list.append(f1_score(y_pred, y_true, zero_division=1))
            recall_list.append(recall_score(y_pred, y_true, zero_division=1))

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)
            

    epoch_loss = np.nanmean(loss_list)
    accuracy = np.nanmean(acc_list)
    f1 = np.nanmean(f1_list)
    recall = np.nanmean(recall_list)
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {accuracy}")
    
    return accuracy, f1, recall

# metoda na osnovu prediction dataloadera vraća rječnik u kojem
# je predikcija zadanog modela sa id-om posta
def predict(model, prediction_loader):
    prediction_dict = dict()
    test_acc_list = []
    for i, data in tqdm(enumerate(prediction_loader, 0)):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype = torch.long)
        
        post_ids = data['data_id'].cpu().numpy()
        outputs = model(ids, mask, token_type_ids).squeeze()
        #pdb.set_trace()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        y_true = targets.detach().cpu().numpy()
        y_pred = big_idx.detach().cpu().numpy()
        print("\ny_true", y_true)
        print("y_pred", y_pred)

        for i in range(len(y_pred)):
            post_id = post_ids[i]
            y_out = y_pred[i]
            test_acc_list.append(accuracy_score(y_pred, y_true))
            prediction_dict.update({post_id:y_out})
    
    print("Accuracy on prediction: ", np.nanmean(test_acc_list))    
    return prediction_dict


def main():
    train_data = pd.read_csv('data/dreaddit-train.csv')
    test_data = pd.read_csv('data/dreaddit-test.csv')
    
    set_seed(1)

    if (ADD_TRAIN is True):
        print("Using modified RoBERTa version")
        # roberta custom trained on our unlabeled dataset
        tokenizer = RobertaTokenizer.from_pretrained(directory_path, padding = True, truncation=True, do_lower_case=True)
    else:
        # non modified roberta
        print("Using NON-modified RoBERTa version")
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', padding = True, truncation=True, do_lower_case=True)
    
    train_data = train_data[['text', 'label', 'id']]
    test_data = test_data[['text', 'label', 'id']]
        
    training_set = DreadditData(train_data, tokenizer, MAX_LEN)
    testing_set = DreadditData(test_data, tokenizer, MAX_LEN)
    
    print("Train Dataset: {}".format(train_data.shape))
    print("Test Dataset: {}".format(test_data.shape))

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': TEST_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }
    
    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    
    model = RobertaClass(ADD_TRAIN)
    model.to(device)
    
    for epoch in range(EPOCHS):
        train(epoch, model, training_loader)
    
    print("Training finished. Validating model")
    
    accuracy, f1, recall = test(model, testing_loader)
    f1 = f1*100
    recall = recall*100
    print("Accuracy on test data = %0.2f%%" % accuracy)
    print("F1 on test data = %0.2f%%" % f1)
    print("Recall on test data = %0.2f%%" % recall)

    prediction_loader_train = DataLoader(training_set, **train_params)
    prediction_loader_test = DataLoader(testing_set, **test_params)

    predictions_train = predict(model, prediction_loader_train)
    predictions_test = predict(model, prediction_loader_test)

    pred_df_train = pd.DataFrame(predictions_train.items(), columns=['id', 'roberta_prediction'])
    pred_df_test = pd.DataFrame(predictions_test.items(), columns=['id', 'roberta_prediction'])
   
    train_df = pd.read_csv('data/dreaddit-train.csv')
    test_df = pd.read_csv('data/dreaddit-test.csv')

    train_df = train_df.merge(pred_df_train, how='inner', on='id')
    test_df = test_df.merge(pred_df_test, how='inner', on='id')
    
    train_df.to_csv("train_pred.csv", index=False)
    test_df.to_csv("test_pred.csv", index=False)    

if __name__ == "__main__":
    main()