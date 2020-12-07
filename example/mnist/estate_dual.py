import os
import torch
import pickle
import numpy as np
import transformers
import pickle
from tqdm import tqdm
from transformers import BertForMaskedLM, BertConfig, BertTokenizer
from torch.utils.data import Dataset, DataLoader, TensorDataset, SequentialSampler, BatchSampler, IterableDataset
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, f1_score
import pytorch_lightning as pl
from pytorch_lightning_help.callbacks import EvalCallback, ProgressBar, ModelCheckpoint
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICIES'] = '1'

path = "/home/xuekui/nas/code/bert4keras/data/estate_dual"

max_len = 128
train_batch_size = 32
eval_batch_size = 128

p = os.path.join(path, 'train', 'train.query.tsv')

bert_path = '/home/xuekui/nas/code/bert4keras/pretrain_weights/bert-zh/'
bert_config = BertConfig.from_pretrained(
    os.path.join(bert_path, 'bert_config.json'))
bert_tokenizer = BertTokenizer.from_pretrained(bert_path)


# def load_data(train_test='train'):
#     D = {}
#     with open(os.path.join(path, train_test, train_test + '.query.tsv')) as f:
#         for l in f:
#             span = l.strip().split('\t')
#             D[span[0]] = {'query': span[1], 'reply': []}

#     with open(os.path.join(path, train_test, train_test + '.reply.tsv')) as f:
#         for l in f:
#             span = l.strip().split('\t')
#             if len(span) == 4:
#                 q_id, r_id, r, label = span
#                 label = int(label)
#             else:
#                 label = None
#                 q_id, r_id, r = span
#             D[q_id]['reply'].append([r_id, r, label])
#     d = []
#     for k, v in D.items():
#         q_id = k
#         q = v['query']
#         reply = v['reply']

#         for i, r in enumerate(reply):
#             r_id, rc, label = r
#             if rc == '':
#                 rc = '。'
#             d.append([q_id + '-' + r_id, q, rc, label])
#     return d


# train_data = load_data('train')
# test_data = load_data('test')

# np.random.shuffle(train_data)
# n = int(len(train_data) * 0.8)


# train_test_list = train_data[: n] + test_data
# dev_list = train_data[n:]
# test_list = test_data

# # patter在前面, idx大于零；反之小于0
# pattern = '直接回答问题'
# mask_idx = 0
# label_to_id = {'直': 1, '间': 0}

# # pattern = '上述对话不通顺。'
# # mask_idx = -4
# # label_to_id = {'很':1, '不':0}

# id_to_label = {v: k for k, v in label_to_id.items()}

# pos_id = bert_tokenizer.encode(id_to_label[1], add_special_tokens=False)
# neg_id = bert_tokenizer.encode(id_to_label[0], add_special_tokens=False)
# label_to_token_id = {k: bert_tokenizer.encode(k, add_special_tokens=False)[
#     0] for k, v in label_to_id.items()}

# # 如果前缀,mask_idx加1（cls),如果是后缀,mask_idx减1(sep)
# mask_idx = mask_idx + 1 if mask_idx >= 0 else mask_idx - 1


# def add_pattern(query, reply):
#     if query[-1] not in ['。', '，', '？']:
#         query += '？'

#     if reply[-1] not in ['。', '，', '？']:
#         reply += '。'

#     if len(query) + len(reply) + len(pattern) + 3 > max_len:
#         if len(reply) > len(query):
#             reply = reply[:(max_len - len(pattern) - 3 - len(query))]
#         else:
#             query = query[:(max_len - len(pattern) - 3 - len(reply))]

#     # 小于0说明是后缀
#     if mask_idx < 0:
#         reply += pattern

#     else:
#         query = pattern + query
#     return query, reply


# def get_pattern_index_id(l):
#     if mask_idx >= 0:
#         pattern_index_id = [0] + [1] * len(pattern)
#         pattern_index_id += [0] * (l - len(pattern_index_id))
#     else:
#         pattern_index_id = [0] * (l - len(pattern) - 1)
#         pattern_index_id += [1] * len(pattern) + [0]
#     return pattern_index_id


# def self_mlm(pattern_index_id, input_id):
#     rands = np.random.random(len(input_id))
#     source, target = [], []
#     special_token_ids = [
#         bert_tokenizer.mask_token_id,
#         bert_tokenizer.sep_token_id,
#         bert_tokenizer.cls_token_id,
#         bert_tokenizer.unk_token_id,
#         bert_tokenizer.pad_token_id
#     ]
#     for r, t, pi in zip(rands, input_id, pattern_index_id):
#         if pi != 1 and t not in special_token_ids:
#             if r < 0.15 * 0.8:
#                 source.append(bert_tokenizer.mask_token_id)
#                 target.append(t)
#             elif r < 0.15 * 0.9:
#                 source.append(t)
#                 target.append(t)
#             elif r < 0.15:
#                 source.append(np.random.choice(
#                     bert_tokenizer.vocab_size - 1) + 1)
#                 target.append(t)
#             else:
#                 source.append(t)
#                 target.append(-100)
#         else:
#             source.append(t)
#             target.append(-100)
#     return source, target


# def collate_fn(batch, use_pattern, mlm):
#     input_ids, attention_mask, token_type_ids, labels, targets = [], [], [], [], []

#     for item in tqdm(batch):
#         id_, query, reply, label = item
#         has_label = label is not None

#         # 如果加pattern, 处理max_len,加入pattern
#         if has_label or use_pattern:
#             query, reply = add_pattern(query, reply)

#         # 不加pattern进行max_len处理
#         else:
#             if len(query) + len(reply) + 3 > max_len:
#                 if len(reply) > len(query):
#                     reply = reply[:(max_len - 3 - len(query))]
#                 else:
#                     query = query[:(max_len - 3 - len(reply))]
#         assert len(query) + len(reply) + 3 <= max_len, print(query,
#                                                              reply, has_label, len(query), len(reply), id_)
#         encode_dict = bert_tokenizer.encode_plus(
#             query, reply, add_special_tokens=True)
#         input_id = encode_dict['input_ids']
#         mask = encode_dict['attention_mask']
#         token_type_id = encode_dict['token_type_ids']

#         pattern_index_id = get_pattern_index_id(len(input_id))

#         if mlm:
#             input_id, target = self_mlm(pattern_index_id, input_id)
#         else:
#             target = [-100] * len(input_id)

#         # 把pattern处的label修改, 无论是前缀还是后缀
#         if has_label:
#             input_id[mask_idx] = bert_tokenizer.mask_token_id
#             target[mask_idx] = label_to_token_id[id_to_label[label]]

#         input_id += [0] * (max_len - len(input_id))
#         mask += [0] * (max_len - len(mask))
#         token_type_id += [0] * (max_len - len(token_type_id))
# #             token_type_id = [0] * max_len
#         target += [-100] * (max_len - len(target))

#         input_ids.append(input_id)
#         attention_mask.append(mask)
#         token_type_ids.append(token_type_id)
#         targets.append(target)
#         labels.append(label if label is not None else -1)

#     return {
#         'input_ids': np.array(input_ids),
#         'attention_mask': np.array(attention_mask),
#         'token_type_ids': np.array(token_type_ids),
#         'labels': np.array(labels),
#         'targets': np.array(targets)
#     }


# train_data = collate_fn(train_test_list, use_pattern=False, mlm=True)
# dev_data = collate_fn(dev_list, use_pattern=False, mlm=False)
# test_data = collate_fn(test_list, use_pattern=True, mlm=False)

# with open(os.path.join(path, "deal_data.pkl"), "wb") as f:
#     pickle.dump(train_data, f)
#     pickle.dump(dev_data, f)
#     pickle.dump(test_data, f)


with open(os.path.join(path, "deal_data.pkl"), "rb") as f:
    train_data = pickle.load(f)
    dev_data = pickle.load(f)
    test_data = pickle.load(f)

# def to_tensor(data):
#     input_ids = torch.tensor(data['input_ids']).to(device)
#     attention_mask = torch.tensor(data['attention_mask']).to(device)
#     token_type_ids = torch.tensor(data['token_type_ids']).to(device)
#     labels = torch.tensor(data['labels']).to(device)
#     targetes = torch.tensor(data['labels'])

class Mydataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(list(self.data.values())[0])
    def __getitem__(self, index):
        d = {}
        for k, v in self.data.items():
            d[k] = v[index]
        return d
    def collate_fn(self, batch):
        d = {}
        for item in batch:
            for k, v in item.items():
                x = d.get(k, [])
                x.append(v)
                d[k] = x
        for k, v in d.items():
            d[k] = np.array(v)
        return d

# train_set = TensorDataset(*train_data.values())
train_set = Mydataset(train_data)
train_loader = DataLoader(train_set, batch_size=train_batch_size, collate_fn=train_set.collate_fn)

print(123)
print(next(iter(train_loader)))

print(next(iter(train_loader)))
