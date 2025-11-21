from transformers import AutoTokenizer, AutoModel
import torch
from torch import optim
import torch.nn as nn
from dotenv import load_dotenv
import wandb
from tqdm import tqdm, trange
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import bisect

FIXED_BUCKETS = [32, 64, 128, 256]
load_dotenv()
class ContextPredictor(nn.Module):
    def __init__(self, num_classes, dropout = 0.3, bert_requires_grad = False):
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
            nn.Dropout(dropout),

            nn.Linear(
                in_features=256,
                out_features=64
            ),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64,num_classes),
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
    loss_fn = nn.CrossEntropyLoss()

    
    learning_rate = 1e-4
    epochs = 1
 
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=epochs * len(training_data)
    )
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
                labels = data['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.squeeze(), labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                wandb.log({"train_loss": loss.item()})

                if (step + 1) % 100 == 0:
                    val_losses = []
                    model.eval()
                    with torch.no_grad():
                        for val_data in validation_data:
                            input_ids = val_data['input_ids'].to(device)
                            attention_mask = val_data['attention_mask'].to(device)
                            labels = val_data['labels'].to(device)
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            val_loss = loss_fn(outputs.squeeze(), labels)
                            val_losses.append(val_loss.item())
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    wandb.log({"val_loss": avg_val_loss})
                    print(f"Epoch {epoch+1}, Step {step+1}, Val Loss: {avg_val_loss}")
                    model.train()
            # end of epoch
            torch.save(model.state_dict(), f"saved_models/predictor_epoch_{epoch+1}.pt")
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
        bucket_idx = bisect.bisect_left(FIXED_BUCKETS, sql_len)
        tokenized["labels"] = bucket_idx
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

    print(eval_dataloader)

    model = ContextPredictor(num_classes=len(FIXED_BUCKETS), bert_requires_grad=False)
    train_predictor(model, train_dataloader, eval_dataloader)

if __name__ == "__main__":
    main()