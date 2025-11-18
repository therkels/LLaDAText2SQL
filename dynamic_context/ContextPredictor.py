from transformers import AutoTokenizer, AutoModel
import torch
from torch import optim
import torch.nn as nn
from dotenv import load_dotenv
import wandb
from tqdm import tqdm, trange
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel

MAX_SCALE = 512
load_dotenv()
class ContextPredictor(nn.Module):
    def __init__(self, dropout = 0.3, bert_requires_grad = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = bert_requires_grad

        self.seq = nn.Sequential(
            nn.Linear(
                in_features=768,
                out_features=256
            ),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(
                in_features=256,
                out_features=64
            ),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64,1),
            nn.Softplus()
        )
    
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_out.last_hidden_state[:, 0, :]
        return self.seq(cls_output)
    
def get_device(default = 'auto'):
    if default == 'auto':
        if torch.cuda.is_available():
            print("using CUDA")
            return 'cuda'
        elif torch.backends.mps.is_available():
            print("using MPS")
            return 'mps'
        else:
            return 'cpu'
    return default

def create_dataloaders(dataset_path, label_tokenizer, input_tokenizer, batch_size = 8):
    reloaded = load_from_disk(dataset_path)


def train_predictor(model, training_data, validation_data):
    device = get_device('cuda')
    model = ContextPredictor()
    loss_fn = nn.SmoothL1Loss()

    
    learning_rate = 1e-6
    epochs = 1

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    model.to(device)
    print(f"model loaded on {device}")
    with wandb.init(name="predictor_loss") as run:
        run.watch(model)

        for epoch in trange(epochs):
            for step, data in tqdm(enumerate(training_data), total=len(training_data)):
                optimizer.zero_grad()
                
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                labels = data['labels'].to(device) / MAX_SCALE
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.squeeze(), labels)
                loss.backward()
                optimizer.step()
                wandb.log({"train_loss": loss.item()})
    model.to('cpu')
    model.eval()
    return model

def main():
    dataset_path = "./data"
    reloaded = load_from_disk(dataset_path)
    input_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    pred_tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    def tokenize_function(example):
        tokenized = input_tokenizer(
            example["sql_prompt"],
            example["sql_context"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        tokenized_target = pred_tokenizer(
            example["sql"],
            truncation=False,
            padding=False
        )
        sql_len = float(len(tokenized_target["input_ids"]))
        tokenized["labels"] = sql_len
        return tokenized

    tokenized_datasets = reloaded.map(tokenize_function, batched=False, remove_columns=reloaded["train"].column_names)

    tokenized_datasets.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, batch_size=64
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=64
    )

    model = ContextPredictor(bert_requires_grad=True)
    train_predictor(model, train_dataloader, eval_dataloader)

if __name__ == "__main__":
    main()