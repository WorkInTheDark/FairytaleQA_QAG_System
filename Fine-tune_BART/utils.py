import itertools
import json
import linecache
import math
import os
import pickle
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Union
import sys

import time
import git
import numpy as np
import torch
import torch.distributed as dist
from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
from torch import nn
from torch.utils.data import Dataset, Sampler

from transformers import BartTokenizer
from transformers.file_utils import cached_property


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def encode_line(tokenizer, line, max_length, pad_to_max_length=True, return_tensors="pt"):
    """Only used by LegacyDataset"""
    extra_kw = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
    return tokenizer(
        [line],
        max_length=max_length,
        padding="max_length" if pad_to_max_length else None,
        truncation=True,
        return_tensors=return_tensors,
        **extra_kw,
    )


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def calculate_bleu(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": round(corpus_bleu(output_lns, [refs_lns], **kwargs).score, 4)}


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
    ):
        super().__init__()
        self.src_file = Path(data_dir).joinpath(type_path + ".source")
        self.tgt_file = Path(data_dir).joinpath(type_path + ".target")
        self.src_lens = self.get_char_lens(self.src_file)
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        assert min(self.src_lens) > 0, f"found empty line in {self.src_file}"
        self.tokenizer = tokenizer
        self.prefix = prefix if prefix is not None else ""

        if n_obs is not None:
            self.src_lens = self.src_lens[:n_obs]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.add_prefix_space = isinstance(self.tokenizer, BartTokenizer)

    def __len__(self):
        return len(self.src_lens)

    @staticmethod
    def get_char_lens(data_file):
        return [len(x) for x in Path(data_file).open().readlines()]

    def make_sortish_sampler(self, batch_size, distributed=False, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, **kwargs)
        else:
            return SortishSampler(self.src_lens, batch_size)

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")


class LegacySeq2SeqDataset(AbstractSeq2SeqDataset):
    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        """Call tokenizer on src and tgt_lines"""
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        source_inputs = encode_line(self.tokenizer, source_line, self.max_source_length)
        target_inputs = encode_line(self.tokenizer, tgt_line, self.max_target_length)

        source_ids = source_inputs["input_ids"].squeeze()
        target_ids = target_inputs["input_ids"].squeeze()
        src_mask = source_inputs["attention_mask"].squeeze()
        return {
            "input_ids": source_ids,
            "attention_mask": src_mask,
            "labels": target_ids,
        }

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["input_ids"] for x in batch])
        masks = torch.stack([x["attention_mask"] for x in batch])
        target_ids = torch.stack([x["labels"] for x in batch])
        pad_token_id = self.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": y,
        }
        return batch


class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch.""" 

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index - 1}

    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            src_lang=self.src_lang,
            tgt_texts=[x["tgt_texts"] for x in batch],
            tgt_lang=self.tgt_lang,
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            add_prefix_space=self.add_prefix_space,
        ).data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding


#########################################
class Seq2SeqDatasetForFID(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        type_path="train",
        n_obs=None,
        src_lang=None,
        tgt_lang=None,
        prefix="",
        fuse_num=10,
        type_embedding=False
    ):
        super().__init__(
                    tokenizer,
                    data_dir,
                    max_source_length,
                    max_target_length,
                    type_path=type_path,
                    n_obs=None,
                    src_lang=None,
                    tgt_lang=None,
                    prefix="",
                )
        self.fuse_num = fuse_num
        self.type_embed_enabled = type_embedding
        self.timer_collate_sum = 0
        self.timer_collate_counter = 0
        self.tic = 0
        self.toc = 0
        self.print_counter = 0
        

    def __getitem__(self, index) -> Dict[str, str]:
        index = index + 1  # linecache starts at 1
        source_line = self.prefix + linecache.getline(str(self.src_file), index).rstrip("\n")
        tgt_line = linecache.getline(str(self.tgt_file), index).rstrip("\n")
        assert source_line, f"empty source line for index {index}"
        assert tgt_line, f"empty tgt line for index {index}"
        # original code
        return {"tgt_texts": tgt_line, "src_texts": source_line.split('\t')[:self.fuse_num], "id": index - 1}
        # ARTHUR'S TESTING CODE
        '''
        new_source_line = []
        for i in source_line.split('\t'):
            new_source_line.append(i.split('<SEP>')[0])
        return {"tgt_texts": tgt_line, "src_texts": new_source_line, "id": index -1}
        '''


    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""
        
        #self.tic = time.time()
        
        sample_src_list = list()
        sample_att_list = list()
        sample_tgt_list = list()
        sample_typ_list = list()

        if not self.type_embed_enabled:
            for sample in batch:
                sample_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
                    sample["src_texts"],
                    src_lang=self.src_lang,
                    tgt_texts=[sample["tgt_texts"]],
                    tgt_lang=self.tgt_lang,
                    max_length=self.max_source_length,
                    max_target_length=self.max_target_length,
                    return_tensors="pt",
                    add_prefix_space=self.add_prefix_space,
                ).data
                sample_src_list.append(sample_encoding['input_ids'])
                sample_att_list.append(sample_encoding['attention_mask'])
                sample_tgt_list.append(sample_encoding['labels'].squeeze(0))

            batch_encoding: Dict[str, torch.Tensor] = dict()
            max_len                     = max([x.shape[-1] for x in sample_src_list])
            max_segment                 = max([x.shape[-2] for x in sample_src_list])
            sample_src_list             = [torch.nn.functional.pad(x, pad=(0, max_len - x.shape[-1], 0, max_segment-x.shape[-2]), mode='constant', value=1) for x in sample_src_list]
            batch_encoding["input_ids"] = torch.stack(sample_src_list)  #[:,:,:350]
            max_len                             = max([x.shape[-1] for x in sample_att_list])
            max_segment                         = max([x.shape[-2] for x in sample_att_list])
            sample_att_list                     = [torch.nn.functional.pad(x, pad=(0, max_len - x.shape[-1], 0, max_segment-x.shape[-2]), mode='constant', value=0) for x in sample_att_list]
            batch_encoding["attention_mask"]    = torch.stack(sample_att_list)   #[:,:,:350]
            max_len                     = max([x.shape[-1] for x in sample_tgt_list])
            sample_tgt_list             = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=1) for x in sample_tgt_list]
            batch_encoding["labels"]    = torch.stack(sample_tgt_list)
            
            #batch_encoding["input_ids"] = batch_encoding["input_ids"][:,:64,:128]
            #batch_encoding["attention_mask"] = batch_encoding["attention_mask"][:,:64,:128]
            
            if self.print_counter == 0:
                self.print_counter += 1
                print(batch_encoding["input_ids"].shape, " input_ids: ", batch_encoding["input_ids"])
            
        else:
            for sample in batch:
                part_inputids_list = list()
                part_mask_list = list()
                part_type_list = list()

                # sample_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
                #     [sample["src_texts"][0]],
                #     src_lang=self.src_lang,
                #     tgt_texts=[sample["tgt_texts"]],
                #     tgt_lang=self.tgt_lang,
                #     max_length=self.max_source_length,
                #     max_target_length=self.max_target_length,
                #     return_tensors="pt",
                #     add_prefix_space=self.add_prefix_space,
                # ).data
                # sample_tgt_list.append(sample_encoding['labels'].squeeze(0))
                # part_inputids_list.append(sample_encoding['input_ids'].squeeze(0))
                # part_mask_list.append(sample_encoding['attention_mask'].squeeze(0))
                sample_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
                    [sample["tgt_texts"]],
                    src_lang=self.src_lang,
                    # tgt_texts=[sample["tgt_texts"]],
                    # tgt_lang=self.tgt_lang,
                    # max_length=self.max_source_length,
                    # max_target_length=self.max_target_length,
                    return_tensors="pt",
                    add_prefix_space=self.add_prefix_space,
                ).data
                sample_tgt_list.append(sample_encoding['input_ids'].squeeze(0))
                
                for part_idx in range(int((len(sample['src_texts']))/2)):
                    type_context        = sample["src_texts"][part_idx * 2 + 1]
                    type_context_toks   = type_context.split()
                    type_context_toks   = [int(tok) for tok in type_context_toks if tok not in ['<SEP>', '<RES>']]
                    raw_context         = sample["src_texts"][part_idx * 2]
                    raw_context_toks    = raw_context.replace('<SEP> ', '<SEP>').replace('<RES> ', '<RES>').split()
                    raw_context_toks    = [tok.replace('<SEP>', '<SEP> ').replace('<RES>', '<RES> ') for tok in raw_context_toks]

                    per_word_outputs: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
                        raw_context_toks,
                        src_lang=self.src_lang,       
                        return_tensors="pt",
                        add_prefix_space=self.add_prefix_space,
                    ).data
                    masks  = per_word_outputs['attention_mask']
                    # input_ids
                    masks[:, 0] = 0
                    indices  = torch.LongTensor([[i, s] for i, s in enumerate(torch.sum(masks, dim=1))])
                    masks = torch.index_put_(masks, tuple(indices.t()), torch.zeros((indices.shape[0],), dtype=torch.int64))
                    input_ids = per_word_outputs['input_ids'][torch.nonzero(masks, as_tuple=True)]
                    input_ids = torch.cat([torch.LongTensor([0]), input_ids[:1022], torch.LongTensor([50265])])
                    part_inputids_list.append(input_ids)
                    # attention_mask
                    attention_mask = torch.LongTensor([1]*len(input_ids))
                    part_mask_list.append(attention_mask)
                    # type_ids
                    token_nums = torch.sum(masks, dim=1)
                    type_tags = list()
                    for raw_tok, type_tok, tok_len in zip(raw_context_toks, type_context_toks, token_nums):
                        if '<SEP>' in raw_tok:
                            type_tags.append([101])     # 50265
                            type_tags.append([type_tok] * (tok_len - 1))
                        elif '<RES>' in raw_tok:
                            type_tags.append([102])     # 50266
                            type_tags.append([type_tok] * (tok_len - 1))
                        else:
                            type_tags.append([int(type_tok)] * tok_len)
                    type_ids  = [tag for tags in type_tags for tag in tags]
                    type_ids  = [0] + type_ids[:1022] + [101]
                    part_type_list.append(torch.tensor(type_ids))

                max_len                     = max([x.shape[-1] for x in part_inputids_list])
                part_inputids_list          = [torch.nn.functional.pad(x, pad=(0, max_len - x.shape[-1]), mode='constant', value=1) for x in part_inputids_list]
                sample_src_list.append(torch.stack(part_inputids_list))
                max_len                     = max([x.shape[-1] for x in part_mask_list])
                part_mask_list              = [torch.nn.functional.pad(x, pad=(0, max_len - x.shape[-1]), mode='constant', value=0) for x in part_mask_list]
                sample_att_list.append(torch.stack(part_mask_list))
                max_len                     = max([x.shape[-1] for x in part_type_list])
                part_type_list              = [torch.nn.functional.pad(x, pad=(0, max_len - x.shape[-1]), mode='constant', value=0) for x in part_type_list]
                sample_typ_list.append(torch.stack(part_type_list))
                
            batch_encoding: Dict[str, torch.Tensor] = dict()
            max_len                     = max([x.shape[-1] for x in sample_src_list])
            max_segment                 = max([x.shape[-2] for x in sample_src_list])
            sample_src_list             = [torch.nn.functional.pad(x, pad=(0, max_len - x.shape[-1], 0, max_segment-x.shape[-2]), mode='constant', value=1) for x in sample_src_list]
            batch_encoding["input_ids"] = torch.stack(sample_src_list) #[:,:,:350]
            max_len                             = max([x.shape[-1] for x in sample_att_list])
            max_segment                         = max([x.shape[-2] for x in sample_att_list])
            sample_att_list                     = [torch.nn.functional.pad(x, pad=(0, max_len - x.shape[-1], 0, max_segment-x.shape[-2]), mode='constant', value=0) for x in sample_att_list]
            batch_encoding["attention_mask"]    = torch.stack(sample_att_list) #[:,:,:350]
            max_len                     = max([x.shape[-1] for x in sample_tgt_list])
            sample_tgt_list             = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=1) for x in sample_tgt_list]
            batch_encoding["labels"]    = torch.stack(sample_tgt_list)
            # max_len                     = max([x.shape[-1] for x in sample_typ_list])
            # sample_typ_list             = [torch.nn.functional.pad(x, pad=(0, sample_src_list[0].shape[-1] - x.numel()), mode='constant', value=1) for x in sample_typ_list]
            sample_typ_list             = [torch.nn.functional.pad(x, pad=(0, sample_src_list[0].shape[-1] - x.shape[-1], 0, sample_src_list[0].shape[-2]-x.shape[-2]), mode='constant', value=0) for x in sample_typ_list]
            batch_encoding["event_ids"] = torch.stack(sample_typ_list)

        batch_encoding["ids"]       = torch.tensor([x["id"] for x in batch])
    
        '''
        # ==== ARTHUR'S TEST CODE ====
        # shape of input_ids and attention_mask are the same => 3-d tensors. 
        # need to truncate if one dimension exceeds the limit 
        #print("==== ARTHUR'S TEST CODE ====")
        print("input_ids: ", batch_encoding["input_ids"].shape)
        print("attention_mask: ", batch_encoding["attention_mask"].shape)
        #print("labels: ", batch_encoding["labels"].shape)
        # print("event_ids: ", batch_encoding["event_ids"].shape)
        #print("==== END OF TEST CODE ====")
        # sys.exit("debug")
        '''
        
        
        '''
        self.toc = time.time()
        self.timer_collate_counter += 1
        #print("collate timer: ",  self.timer_collate_counter)
        
        self.timer_collate_sum = (self.toc - self.tic)
        if self.timer_collate_counter == 1000:
            print("collate 1000", self.timer_collate_sum, ", ", self.timer_collate_sum/1000) 
        if self.timer_collate_counter == 3000:
            print("collate 3000", self.timer_collate_sum, ", ", self.timer_collate_sum/3000) 
        '''
        
        
        return batch_encoding

#########################################



class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size):
        self.data, self.bs = data, batch_size

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs))


def sortish_sampler_indices(data: List, bs: int) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.src_lens[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


logger = getLogger(__name__)


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_git_info(folder_path: str) -> None:
    """Save git information to output_dir/git_log.json"""
    repo_infos = get_git_info()
    save_json(repo_infos, os.path.join(folder_path, "git_log.json"))


def save_json(content, path):
    with open(path, "w") as f:
        json.dump(content, f, indent=4)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    repo_infos = {
        "repo_id": str(repo),
        "repo_sha": str(repo.head.object.hexsha),
        "repo_branch": str(repo.active_branch),
    }
    return repo_infos


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def calculate_rouge(output_lns: List[str], reference_lns: List[str], use_stemmer=True) -> Dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}


# Utilities for freezing parameters and checking whether they are frozen


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


# CLI Parsing utils


def parse_numeric_cl_kwargs(unparsed_args: List[str]) -> Dict[str, Union[int, float]]:
    """Parse an argv list of unspecified command line args to a dict. Assumes all values are numeric."""
    result = {}
    assert len(unparsed_args) % 2 == 0, f"got odd number of unparsed args: {unparsed_args}"
    num_pairs = len(unparsed_args) // 2
    for pair_num in range(num_pairs):
        i = 2 * pair_num
        assert unparsed_args[i].startswith("--")
        try:
            value = int(unparsed_args[i + 1])
        except ValueError:
            value = float(unparsed_args[i + 1])  # this can raise another informative ValueError

        result[unparsed_args[i][2:]] = value
    return result


def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()
