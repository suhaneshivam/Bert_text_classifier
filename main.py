import tez
import torch
import joblib
import torch.nn as nn
import pandas as pd
import transformers
from sklearn import metrics, model_selection, preprocessing
from transformers import AdamW, get_linear_schedule_with_warmup

class BertDataset:
    def __init__(self, texts, targets, max_len = 64):
        self.texts = texts
        self.targets = targets
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case = True)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        text = " ".join(text.split())
        targets = self.targets[item]
        inputs = self.tokenizer.encode_plus(text = text, text_pair = None, add_special_tokens = True,
                                            padding = "max_length", truncation = True, max_length = self.max_len)
        resp = {
        "ids" : torch.tensor(inputs["input_ids"], dtype = torch.long),
        "mask" : torch.tensor(inputs["attention_mask"], dtype = torch.long),
        "token_type_ids" : torch.tensor(inputs["token_type_ids"], dtype = torch.long),
        "targets" : torch.tensor(targets, dtype = torch.long)}
        return resp

class TextModel(tez.Model):
    def __init__(self, num_classes, num_train_steps):
        super(TextModel, self).__init__()
        self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case = True)
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased", return_dict=False, output_hidden_states = True)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(3072, num_classes)
        self.train_steps = num_train_steps
        self.step_scheduler_after = "batch"

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]

        optimizer_parameters = [{"params" : [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay" : 0.001}, {"params" : [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay" : 0.0} ]

        opt = AdamW(optimizer_parameters, lr = 3e-5)

        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps = 0, num_training_steps = self.train_steps)
        return sch

    def loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.CrossEntropyLoss()(outputs, targets)

    def moniter_metrics(self, outputs, targets):
        if targets is None:
            return {}
        outputs = torch.argmax(outputs, dim = 1).cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()
        accuracy = metrics.accuracy_score(targets, outputs)
        return {"accuracy" : accuracy}

    def forward(self, ids, mask, token_type_ids, targets = None):
        #pooling_output dim : (batch_size x 4*768)
        #b_o dim : (batch_size x 4*768)
        #output dim : (batch_size x num_classes)
        #targets dim : (batch_size,)
        _, _, hidden_states = self.bert(ids, attention_mask = mask, token_type_ids = token_type_ids)
        pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
        pooled_output = pooled_output[:, 0, :]
        b_o = self.bert_drop(pooled_output)
        outputs = self.out(b_o)
        loss = self.loss(outputs, targets)
        acc = self.monitor_metrics(outputs, targets)

        return outputs, loss, acc

if __name__ == '__main__':
    dfx = pd.read_csv("bbc-text.csv")
    dfx = dfx.dropna().reset_index(drop = True)
    lbl_enc = preprocessing.LabelEncoder()
    dfx.category = lbl_enc.fit_transform(dfx.category.values)

    df_train, df_valid = model_selection.train_test_split(dfx, test_size = 0.1, random_state = 42, stratify = dfx.category.values)

    df_train = df_train.reset_index(drop = True)
    df_valid = df_valid.reset_index(drop = True)

    num_train_steps = int(len(df_train)/32 * 10)
    meta = {"lbl_enc" : lbl_enc, "num_train_steps" : num_train_steps}
    joblib.dump(meta, "meta.bin")

    train_dataset = BertDataset(texts = df_train.text.values, targets = df_train.category.values)
    valid_dataset = BertDataset(texts = df_valid.text.values, targets = df_valid.category.values)

    model = TextModel(num_classes = len(lbl_enc.classes_), num_train_steps = num_train_steps)

    tb_logger = tez.callbacks.TensorBoardLogger(log_dir = ".logs/")
    es = tez.callbacks.EarlyStopping(monitor = "valid_loss", model_path = "model.bin")

    model.fit(train_dataset, valid_dataset = valid_dataset, train_bs = 32, device = "cpu", epochs = 10, callbacks = [tb_logger, es], fp16 = True)

    model.save("model.bin")
