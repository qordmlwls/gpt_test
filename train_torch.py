# -*- coding: utf-8 -*-
import argparse
import logging
import os
import json

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader, Dataset
# from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel


parser = argparse.ArgumentParser(description='Simsimi based on KoGPT-2')

parser.add_argument('--chat',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--sentiment',
                    type=str,
                    default='0',
                    help='sentiment for system. 0 is neutral, 1 is negative, 2 is positive.')

parser.add_argument('--model_params',
                    type=str,
                    default='model_chp/model_-last.ckpt',
                    help='model binary for starting chat')

parser.add_argument('--train',
                    action='store_true',
                    default=False,
                    help='for training')

# parser.add_argument('--gpus',
#                     default=1)

# parser.add_argument('--max_epochs',
#                     default=3)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

U_TKN = '<usr>'
S_TKN = '<sys>'
BOS = '<|endoftext|>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

TOKENIZER = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-1.3b",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 

# TOKENIZER = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2",
#             bos_token=BOS, eos_token=EOS, unk_token='<unk>',
#             pad_token=PAD, mask_token=MASK) 

TOKENIZER.add_special_tokens({'additional_special_tokens':['<|sept|>', U_TKN, S_TKN]})

def preprocess(data, bos_token, sept_token, answer):
  protagonist = data['characters'][0]['id']
  sentences = []
  length = len(data['scenes'][0]['items'])
  scences = data['scenes'][0]['items'].copy()
  idx = 0
  tmp_id = 0
  while True:
    tmp_sentence = ''
    for content in scences:
      if content['object_type'] == 'text' and content['character_id'] == protagonist and bos_token not in tmp_sentence:
        tmp_sentence = bos_token + tmp_sentence + content['object']['text']
        if idx < length - 1:
          if data['scenes'][0]['items'][idx + 1]['object_type'] != 'text':
            tmp_sentence = ''
            idx = idx + 1
            break
          elif data['scenes'][0]['items'][idx + 1]['character_id'] != protagonist:
            tmp_id = data['scenes'][0]['items'][idx + 1]['character_id']
        if idx == length - 1:
          tmp_sentence = ''
          idx = idx + 1
          break
        idx = idx + 1
        continue
      
      if content['object_type'] == 'text' and content['character_id'] == protagonist and bos_token in tmp_sentence:
        tmp_sentence = tmp_sentence + sept_token + content['object']['text']
        # 대화만 넣어야 됨
        if idx < length - 1:
          if data['scenes'][0]['items'][idx + 1]['object_type'] != 'text':
            tmp_sentence = ''
            idx = idx + 1
            break
          elif data['scenes'][0]['items'][idx + 1]['character_id'] != protagonist:
            tmp_id = data['scenes'][0]['items'][idx + 1]['character_id']
        if idx == length - 1 and answer not in tmp_sentence:
          tmp_sentence = ''
          idx = idx + 1
          break
        idx = idx + 1
        continue
      if data['scenes'][0]['items'][idx - 1]['object_type'] != 'text':
        idx = idx + 1
        break
      if content['object_type'] == 'text' and idx != 0 and content['character_id'] == tmp_id and data['scenes'][0]['items'][idx - 1]['character_id'] == protagonist and answer not in tmp_sentence:
        tmp_sentence = tmp_sentence + answer + content['object']['text']
        if idx < length - 1:
          if data['scenes'][0]['items'][idx + 1]['object_type'] == 'text' and data['scenes'][0]['items'][idx + 1]['character_id'] == protagonist:
            idx = idx + 1
            break
        idx = idx + 1
        continue
      if content['object_type'] == 'text' and idx != 0 and content['character_id'] == tmp_id and data['scenes'][0]['items'][idx - 1]['character_id'] == protagonist and answer in tmp_sentence:
        tmp_sentence = tmp_sentence + sept_token + content['object']['text']
        if idx < length - 1:
          if data['scenes'][0]['items'][idx + 1]['object_type'] == 'text' and data['scenes'][0]['items'][idx + 1]['character_id'] == protagonist:
            idx = idx + 1
            break
        idx = idx + 1
        continue
      idx = idx + 1
      break
    if tmp_sentence != '':
      sentences.append(tmp_sentence)
    scences = data['scenes'][0]['items'][idx:]
    # idx = idx + 1
    if idx == length:
      break
  return sentences


sentence_sum = []
for file in os.listdir('./drive/chat'):
  if 'chatie' in file:
    with open(os.path.join('./drive/chat/', file), 'r') as f:
      test = f.readline()
    data = json.loads(test)
    sentences = preprocess(data, bos_token=BOS, sept_token='<|sept|>', answer=S_TKN)
    sentence_sum = sentence_sum + sentences


class CharDataset(Dataset):
    def __init__(self, chats, max_len=32):
        self._data = chats
        self.first = True
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.pad = PAD
        self.max_len = max_len
        self.tokenizer = TOKENIZER 

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # turn = self._data.iloc[idx]
        # q = turn['Q']
        # a = turn['A']
        
        turn = self._data[idx]
        q = turn.split('<sys>')[0]
        a = turn.split('<sys>')[1]
        # sentiment = str(turn['label'])
        q_toked = self.tokenizer.tokenize(self.q_token + q \
                                          # + self.sent_token + \
                                          # sentiment   
                                          )
        q_len = len(q_toked)
        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            # 길면 자른다.
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len/2)):]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'
        # [mask, mask, ...., mask, ..., <bos>,..A.. <eos>, <pad>....]
        labels = [
            self.mask,
        ] * q_len + a_toked[1:]
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        self.max_len
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]
        return(token_ids, np.array(mask),
               labels_ids)


class KoGPT2Chat(LightningModule):
    def __init__(self, hparams, **kwargs):
        super(KoGPT2Chat, self).__init__()
        # self.args = hparams
        self.args = hparams
        self.neg = -1e18
        self.kogpt2 = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        # self.kogpt2 = AutoModelForCausalLM.from_pretrained("EleutherAI/polyglot-ko-1.3b")
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='none')

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--max-len',
                            type=int,
                            default=32,
                            help='max sentence length on input (default: 32)')

        parser.add_argument('--batch-size',
                            type=int,
                            default=96,
                            help='batch size for training (default: 96)')
        parser.add_argument('--lr',
                            type=float,
                            default=5e-5,
                            help='The initial learning rate')
        # parser.add_argument('--warmup_ratio',
        #                     type=float,
        #                     default=0.1,
        #                     help='warmup ratio')
        return parser

    def forward(self, inputs):
        # (batch, seq_len, hiddens)
        output = self.kogpt2(inputs, return_dict=True)
        return output.logits

    def training_step(self, batch, batch_idx):
        token_ids, mask, label = batch
        out = self(token_ids)
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, self.neg * torch.ones_like(out))
        loss = self.loss_function(mask_out.transpose(2, 1), label)
        loss_avg = loss.sum() / mask.sum()
        self.log('train_loss', loss_avg)
        return loss_avg

    # def configure_optimizers(self):
    #     # Prepare optimizer
    #     param_optimizer = list(self.named_parameters())
    #     no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #     optimizer_grouped_parameters = [
    #         {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #     ]
    #     optimizer = AdamW(optimizer_grouped_parameters,
    #                       lr=self.args.lr, correct_bias=False)
    #     # warm up lr
    #     num_train_steps = len(self.train_dataloader()) * self.args.max_epochs
    #     num_warmup_steps = int(num_train_steps * self.args.warmup_ratio)
    #     scheduler = get_cosine_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
    #     lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
    #                     'monitor': 'loss', 'interval': 'step',
    #                     'frequency': 1}
    #     return [optimizer], [lr_scheduler]
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'loss'
            }
        }

    def _collate_fn(self, batch):
        data = [item[0] for item in batch]
        mask = [item[1] for item in batch]
        label = [item[2] for item in batch]
        return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

    def train_dataloader(self):
        # data = pd.read_csv('Chatbot_data/ChatbotData.csv')
        self.train_set = CharDataset(sentence_sum, max_len=self.args.max_len)
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.args.batch_size, num_workers=2,
            shuffle=True, collate_fn=self._collate_fn)
        return train_dataloader

    # def chat(self, sent='0'):
    def chat(self):
        tok = TOKENIZER
        # sent_tokens = tok.tokenize(sent)
        with torch.no_grad():
            while 1:
                q = input('user > ').strip()
                if q == 'quit':
                    break
                a = ''
                while 1:
                    input_ids = torch.LongTensor(tok.encode(U_TKN + q + \
                                                            # SENT + sent + \
                                                            S_TKN + a)).unsqueeze(dim=0)
                    pred = self(input_ids)
                    gen = tok.convert_ids_to_tokens(
                        torch.argmax(
                            pred,
                            dim=-1).squeeze().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace('▁', ' ')
                print("Simsimi > {}".format(a.strip()))


parser = KoGPT2Chat.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

if __name__ == "__main__":
    if args.train:
        checkpoint_callback = ModelCheckpoint(
            dirpath='model_chp',
            filename='{epoch:02d}-{train_loss:.2f}',
            verbose=True,
            save_last=True,
            monitor='train_loss',
            mode='min',
            # prefix='model_'
        )
        # python train_torch.py --train --gpus 1 --max_epochs 3
        args.gpus = -1
        args.max_epochs = 3
        args.amp_level = 'apex'
        args.accelerator='dp' if torch.cuda.is_available() else None
        args.max_len = 16
        args.batch_size = 50
        model = KoGPT2Chat(args)
        model.train()
        trainer = Trainer.from_argparse_args(
            args,
            callbacks=[checkpoint_callback],
            gradient_clip_val=1.0)
        trainer.fit(model)
        logging.info('best model path {}'.format(checkpoint_callback.best_model_path))
    if args.chat:
        args.model_params = 'model_chp/last.ckpt'
        model = KoGPT2Chat(args)
        model = model.load_from_checkpoint(args.model_params, hparams=args.__dict__)
        model.chat()
