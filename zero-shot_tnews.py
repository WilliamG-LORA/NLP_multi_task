import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from arguments import get_args
from data_utils.tokenization_gpt2 import GPT2Tokenizer
import mpu
import json
import random

from data.samplers import DistributedBatchSampler, RandomSampler

from torch.utils.data import TensorDataset
from utils import initialize_distributed, set_random_seed, setup_model_and_optimizer, yprint
from generate_samples import get_model, load_checkpoint_model


def setup_model(args):
    """Setup model."""

    model = get_model(args)

    args.iteration = load_checkpoint_model(model, args)

    return model

def get_masks_and_position_ids(data,
                               eod_token,
                               reset_position_ids,
                               reset_attention_mask):
    # Extract batch size and sequence length.
    batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=data.device)).view(
            att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i+1):, :(i+1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i+1):] -= (i + 1 - prev_index)
                    prev_index = i + 1

    return attention_mask, loss_mask, position_ids


def get_batch(context_tokens, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        args.reset_position_ids,
        args.reset_attention_mask)

    return tokens, attention_mask, position_ids

def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
 
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
        
    if top_p > 0.0:
        #convert to 1D
        logits=logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        #going back to 2D
        logits=logits.view(1, -1).contiguous()
	
    return logits


def load_tnews_data(data_path, data_type, tokenizer, few_shot=False):
    args = get_args()

    filename = os.path.join(data_path, data_type+'.json')
    objs = []
    with open(filename) as fin:
        for line in fin:
            objs.append(json.loads(line.strip()))

    pad_id = tokenizer.encoder['<pad>']
    args.eod_token = tokenizer.encoder['<eod>']

    labels = []
    label_map = {}
    label_reverse = {}
    with open(os.path.join(data_path, 'labels.json')) as fin:
        for i, line in enumerate(fin):
            obj = json.loads(line.strip())
            labels.append(obj['label_desc'])
            label_map[obj['label_desc']] = i
            label_reverse[obj['label']] = obj['label_desc']

    all_tokens = []
    all_masks = []
    all_labels = []
    for _, obj in enumerate(objs):
        sentence = obj['sentence']
        tokenized_sentence = tokenizer.encode(sentence)[:args.seq_length-20]
        obj['label_desc'] = label_reverse[obj['label']]

        if few_shot:
            cur_labels = random.sample(labels, 3)
            while obj['label_desc'] in cur_labels:
                cur_labels = random.sample(labels, 3)
            cur_labels.append(obj['label_desc'])
            cur_label = cur_labels.index(obj['label_desc'])
            assert cur_label != -1
        else:
            cur_labels = labels
            cur_label = label_map[obj['label_desc']]

        all_labels.append(cur_label)

        for _, label in enumerate(cur_labels):
            prompt = "这是关于{}的文章：".format(label)
            prompt_tokens = tokenizer.encode(prompt)
            prompt_len = len(prompt_tokens)
            tokens = prompt_tokens + tokenized_sentence
            second_mask = [0] * (args.seq_length-1)
            for idx in range(prompt_len-1, len(tokens)-1):
                second_mask[idx] = 1
            all_masks.append(second_mask)
            token_length = len(tokens)
            assert token_length < args.seq_length
            tokens.extend([pad_id] * (args.seq_length - token_length))
            all_tokens.append(tokens)
    
    all_tokens = torch.tensor(all_tokens, dtype=torch.long)
    all_masks = torch.tensor(all_masks, dtype=torch.float)
    dataset = TensorDataset(all_tokens, all_masks)

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    global_batch_size = args.batch_size * world_size
    num_workers = args.num_workers

    sampler = torch.utils.data.SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=True,
                                            rank=rank,
                                            world_size=world_size)
    
    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=num_workers,
                                       pin_memory=True), all_labels


def evaluate(model, dev_dataloader, all_labels, device, args):
    model.eval()

    if torch.distributed.get_rank() == 0:
        res = []

    with torch.no_grad():
        for batch in tqdm.tqdm(dev_dataloader):
            tokens, masks = [x.to(device) for x in batch]

            tokens, attention_mask, position_ids = get_batch(tokens, args)
            output, _ = model(tokens, position_ids, attention_mask)
            losses = mpu.vocab_parallel_cross_entropy(output[:, :-1, :].contiguous().float(), tokens[:, 1:])

            output = torch.sum(losses * masks, 1) / torch.sum(masks, -1)

            tensor_list = [torch.zeros_like(output) for _ in range(mpu.get_data_parallel_world_size())]
            torch.distributed.all_gather(tensor_list, output, mpu.get_data_parallel_group())
            output = torch.stack(tensor_list, 0).view(-1).cpu().detach().numpy()

            if torch.distributed.get_rank() == 0:
                for v in output:
                    res.append(v)

    if torch.distributed.get_rank() == 0:
        cnt = 0
        label_size = max(all_labels) + 1
        num_inst = len(res) // label_size
        for x in range(num_inst):
            label = all_labels[x]
            cur_res = res[x*label_size:(x+1)*label_size]
            pos = np.argmin(cur_res)
            if pos == label:
                cnt += 1
        print("EVAL", cnt, num_inst)

def train(model, train_dataloader, all_labels, device, args):
    model.train()

    with torch.no_grad():
        for batch in tqdm.tqdm(train_dataloader):
            tokens, masks = [x.to(device) for x in batch]

            tokens, attention_mask, position_ids = get_batch(tokens, args)
            output, _ = model(tokens, position_ids, attention_mask)
            losses = mpu.vocab_parallel_cross_entropy(output[:, :-1, :].contiguous().float(), tokens[:, 1:])

            loss = torch.mean(losses)
            model.backward(loss)
            model.step()


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    #get the tokenizer
    tokenizer = GPT2Tokenizer(os.path.join(args.tokenizer_path, 'vocab.json'), os.path.join(args.tokenizer_path, 'chinese_vocab.model'))

    # load data
    assert args.eval_data_path is not None

    args.eod_token = tokenizer.encoder['<eod>']

    # Model
    args.parallel_output = True
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    device = torch.cuda.current_device()
    
    if args.task == "tnews":
        train_dataloader, train_labels = load_tnews_data(args.eval_data_path, 'train', tokenizer)
        train(model, train_dataloader, train_labels, device, args)

        dev_dataloader, dev_labels = load_tnews_data(args.eval_data_path, 'dev', tokenizer)
        evaluate(model, dev_dataloader, dev_labels, device, args)
    else:
        print("Unknown task!")

if __name__ == "__main__":
    main()