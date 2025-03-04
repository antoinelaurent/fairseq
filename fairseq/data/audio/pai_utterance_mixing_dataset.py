# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import sys
import io
import json
import time

import h5py
from typing import Any, List, Optional, Union

import numpy as np

import torch
import torch.nn.functional as F
from fairseq.data import data_utils
from fairseq.data.fairseq_dataset import FairseqDataset
from fairseq.data.audio.audio_utils import (
    parse_path,
    read_from_stored_zip,
    is_sf_audio_data,
)

from pyannote.audio import Audio
from pyannote.core import Segment

logger = logging.getLogger(__name__)



def load_label(label_path, inds, tot):
    with open(label_path) as f:
        labels = [line.rstrip() for line in f]
        assert (
            len(labels) == tot
        ), f"number of labels does not match ({len(labels)} != {tot})"
        labels = [labels[i] for i in inds]
    return labels


def load_label_offset(label_path, inds, tot):
    with open(label_path) as f:
        code_lengths = [len(line.encode("utf-8")) for line in f]
        assert (
            len(code_lengths) == tot
        ), f"number of labels does not match ({len(code_lengths)} != {tot})"
        offsets = list(itertools.accumulate([0] + code_lengths))
        offsets = [(offsets[i], offsets[i + 1]) for i in inds]
    return offsets


def verify_label_lengths(
    audio_sizes,
    audio_rate,
    label_path,
    label_rate,
    inds,
    tot,
    tol=0.1,  # tolerance in seconds
):
    if label_rate < 0:
        logger.info(f"{label_path} is sequence label. skipped")
        return

    with open(label_path) as f:
        lengths = [len(line.rstrip().split()) for line in f]
        assert len(lengths) == tot
        lengths = [lengths[i] for i in inds]
    num_invalid = 0
    for i, ind in enumerate(inds):
        dur_from_audio = audio_sizes[i] / audio_rate
        dur_from_label = lengths[i] / label_rate
        if abs(dur_from_audio - dur_from_label) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{dur_from_audio} - {dur_from_label}| > {tol}) "
                    f"in line {ind+1} of {label_path}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_sizes[i]}; "
                    f"label length = {lengths[i]}"
                )
            )
            num_invalid += 1
    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )


class PAIUtteranceMixingDataset(FairseqDataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: float,
        label_paths: List[str],
        label_rates: Union[List[float], float],  # -1 for sequence labels
        pad_list: List[str],
        eos_list: List[str],
        label_processors: Optional[List[Any]] = None,
        max_keep_sample_size: Optional[int] = None,
        min_keep_sample_size: Optional[int] = None,
        max_sample_size: Optional[int] = None,
        shuffle: bool = True,
        pad_audio: bool = False,
        normalize: bool = False,
        store_labels: bool = True,
        random_crop: bool = False,
        single_target: bool = False,
        multitask: bool = False,
        mixing_max_len: int = -1,
        mixing_prob: float = 0.2,
        mixing_num: int = 1,
        mixing_noise: bool = False,
        mixing_noise_prob: float = 0.0,
        mixing_noise_num: int = 1,
        noise_path: Optional[str] = None,
        balance: bool = False,
        dataset_log_duration: bool = False,
    ):
        self.sample_rate = sample_rate
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.balance = balance
        self.dataset_log_duration = dataset_log_duration

        self.num_labels = len(label_paths)
        self.pad_list = pad_list
        self.eos_list = eos_list
        self.label_processors = label_processors
        self.single_target = single_target
        self.multitask = multitask
        self.epoch = 0

        self.chunk_names = []
        self.chunk_indices = []


        self.dataset_indices = dict()
        self.ind_dataset = []

        n_long, n_short = 0, 0
        names, inds, sizes = [], [], []
        bnds = []
        bnd_path = manifest_path.replace('tsv', 'bnd')
        if os.path.exists(bnd_path):
            with open(bnd_path) as f:
                bnds = f.readlines()
        new_bnds = []


        with open(manifest_path) as f:
            root = f.readline().strip()
            for (ind, line) in enumerate(f):
                items = line.strip().split("\t")

                if self.balance:
                    if len(items) != 3:
                        logger.info("tsv file should contain the dataset name:\naudio\tnframes\tdataset")
                    assert len(items) == 3
                    dataset = items[2]
                    if not dataset in self.dataset_indices:
                        self.dataset_indices[dataset] = []
                    self.dataset_indices[dataset].append(ind)
                    self.ind_dataset.append(dataset)

                sz = int(items[1])
                if min_keep_sample_size is not None and sz < min_keep_sample_size:
                    n_short += 1
                elif max_keep_sample_size is not None and sz > max_keep_sample_size:
                    n_long += 1
                else:
                    fname = items[0].split(":")
                    if len(fname) > 1:
                        if len(self.chunk_names) == 0 or fname[0] != self.chunk_names[-1]:
                            self.chunk_names.append(fname[0])
                            self.chunk_indices.append(len(names))
                    names.append(items[0])
                    inds.append(ind)
                    sizes.append(sz)
                    if len(bnds) > 0:
                        new_bnds.append(list(map(int, bnds[ind].strip().split())))

        tot = ind + 1

        logger.info(
            (
                f"max_keep={max_keep_sample_size}, min_keep={min_keep_sample_size}, "
                f"loaded {len(names)}, skipped {n_short} short and {n_long} long, "
                f"longest-loaded={max(sizes)}, shortest-loaded={min(sizes)}"
            )
        )



        self.audio_root = root
        self.audio_names = names
        self.sizes = sizes
        self.bnds = new_bnds
    
        self.label_rates = (
            [label_rates for _ in range(len(label_paths))]
            if isinstance(label_rates, int)
            else label_rates
        )
        self.store_labels = store_labels
        if store_labels:
            self.label_list = [load_label(p, inds, tot) for p in label_paths]
        else:
            self.label_paths = label_paths
            self.label_offsets_list = [
                load_label_offset(p, inds, tot) for p in label_paths
            ]
        assert (
            label_processors is None
            or len(label_processors) == self.num_labels
        )
        for label_path, label_rate in zip(label_paths, self.label_rates):
            verify_label_lengths(
                self.sizes, sample_rate, label_path, label_rate, inds, tot
            )

        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.pad_audio = pad_audio
        self.normalize = normalize
        logger.info(
            f"pad_audio={pad_audio}, random_crop={random_crop}, "
            f"normalize={normalize}, max_sample_size={self.max_sample_size}"
        )

        self.mixing_max_len = mixing_max_len
        self.mixing_prob = mixing_prob
        self.mixing_num = mixing_num

        self.mixing_noise = mixing_noise
        self.mixing_noise_prob = mixing_noise_prob
        self.mixing_noise_num = mixing_noise_num

        self.noise_path = noise_path
        if self.mixing_noise:
            assert os.path.exists(self.noise_path), f"Invalid noise path {self.noise_path}"
            self.noise_list = json.load(open(self.noise_path, 'r'))
            self.noise_container = {}
        else:
            self.noise_list = []

        logger.info(
            f"mixing_max_len={mixing_max_len}, mixing_prob={mixing_prob},  mixing_num={mixing_num},"
            f"mixing_noise={mixing_noise}, mixing_noise_prob={mixing_noise_prob},  mixing_noise_num={mixing_noise_num},"
            f"noise_path={noise_path}, noise_list_len={len(self.noise_list)},"
        )

        if self.balance:
            self.prep_balance_indices()



    def set_epoch(self, epoch):
        self.epoch = epoch

    def sectotime(self, seconds):
        heures = int(seconds // 3600)
        seconds = seconds % 3600
        minutes = int(seconds // 60)
        seconds = seconds % 60
        return f"{heures:02d}:{minutes:02d}:{seconds:05.2f}"


    def prep_balance_indices(self):
        self.datasets = []
        logger.info(f"{len(self.dataset_indices)} dataset loaded")
        durations = np.zeros(len(self.dataset_indices))
        audio_dataset_durations = dict()
        self.audio_dataset_cum_prob_duration = dict()

        for (dind, dataset) in enumerate(self.dataset_indices):
            logger.info(f"i'm here ==> {dataset} / {dind}")
            if not dataset in audio_dataset_durations:
                audio_dataset_durations[dataset] = np.zeros(len(self.dataset_indices[dataset]))
                self.audio_dataset_cum_prob_duration[dataset] = np.zeros(len(self.dataset_indices[dataset]))

            logger.info(f"i'm here 2 ==> {dataset} / {dind}")
            logger.info(f"self.audio_dataset_cum_prob_duration[dataset] => {self.audio_dataset_cum_prob_duration[dataset].shape}")
            logger.info(
                f"audio_dataset_durations[dataset] => {audio_dataset_durations[dataset].shape}")

            for (ind, ind_datasets) in enumerate(self.dataset_indices[dataset]):
                logger.info(f"i'm here 3 ==> {ind} / {ind_datasets}")
                audio_dataset_durations[dataset][ind] = self.sizes[ind_datasets]
                logger.info(f"audio_dataset_durations[dataset][ind] = {audio_dataset_durations[dataset][ind]}")
                self.audio_dataset_cum_prob_duration[dataset][ind] = self.sizes[ind_datasets]
                if ind > 0:
                    self.audio_dataset_cum_prob_duration[dataset][ind] += self.audio_dataset_cum_prob_duration[dataset][ind - 1]


            durations[dind] = np.sum(audio_dataset_durations[dataset])
            self.datasets.append(dataset)

        self.total_duration = np.sum(durations)
        logger.info(f"i'm here 1")

        for (ind, dataset) in enumerate(self.datasets):
            logger.info(f"dataset={dataset}, duration={durations[ind]/self.sample_rate:.2f}s"
                        f"({self.sectotime(durations[ind]/self.sample_rate)})")

        if self.dataset_log_duration:
            durations = np.log(durations)
            logger.info(f"datasets: {self.datasets}, Log durations: {durations}")
        else:
            logger.info(f"datasets: {self.datasets}, durations: {durations}")

        # convert duration of each dataset into probabilities according to log durations
        self.cum_prob_duration = np.cumsum(
            durations / np.sum(durations)
        )

        # convert duration of each audiofile into probabilities according to durations
        for dataset in self.datasets:
            self.audio_dataset_cum_prob_duration[dataset] = self.audio_dataset_cum_prob_duration[dataset] / np.sum(
                audio_dataset_durations[dataset])

    def get_balance_indices(self):
        # convert duration of each dataset into probabilities
        indices = []
        selected_duration = 0

        nb_stats = dict()
        while selected_duration < self.total_duration:
            dataset = self.datasets[self.cum_prob_duration.searchsorted(np.random.random())]
            nb_stats[dataset] = nb_stats.get(dataset, 0) + 1
            ind_select = self.audio_dataset_cum_prob_duration[dataset].searchsorted(np.random.random())
            indices.append(self.dataset_indices[dataset][ind_select])
            selected_duration += self.max_sample_size

        logger.info(f"get_balance_indices: {len(indices)} indices")
        for dataset in nb_stats:
            logger.info(f"get_balance_indices: {nb_stats[dataset]} example from {dataset} ({nb_stats[dataset]/len(indices)*100:.2f}%)")
        return indices

    def batch_by_size(self, indices, max_tokens=None, max_sentences=None, required_batch_size_multiple=1):
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self.required_batch_size_multiple = required_batch_size_multiple

        if isinstance(indices[0], list):
            batch_list = []
            for indice in indices:
                batch = super(PAIUtteranceMixingDataset, self).batch_by_size(indice, max_tokens, max_sentences, required_batch_size_multiple)
                batch_list.append(batch)
            return batch_list
        else:
            batch_list = super(PAIUtteranceMixingDataset, self).batch_by_size(indices, max_tokens, max_sentences, required_batch_size_multiple)
            #for (i, batch) in enumerate(batch_list):
            #    ds = [self.ind_dataset[ind] for ind in batch]
            #    logger.info(f"bach:{i} - {ds}")
            return batch_list

    def shuffle_batches(self, batches, seed):
        if isinstance(batches[0], list):
            new_batches = []
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
                for batch in batches:
                    np.random.shuffle(batch)
                    new_batches.extend(batch)
            return new_batches
        else:
            with data_utils.numpy_seed(seed):
                np.random.shuffle(batches)
        return batches

    def reset_batch_sampler(self):
        indices = self.ordered_indices()
        batch_sampler = self.batch_by_size(
                indices,
                self.max_tokens,
                self.max_sentences,
                self.required_batch_size_multiple
        )
        return batch_sampler

    def get_audio(self, index):
        import soundfile as sf

        wav_path = os.path.join(self.audio_root, self.audio_names[index])
        _path, slice_ptr = parse_path(wav_path)
        if len(slice_ptr) == 2:
            byte_data = read_from_stored_zip(_path, slice_ptr[0], slice_ptr[1])
            assert is_sf_audio_data(byte_data)
            wav_path = io.BytesIO(byte_data)

        #we don't want to fully read the wavfiles
        logger.info(f"[ERROR] je suis ici dans get audio index:{index} !!")
        wav, cur_sample_rate = sf.read(wav_path)
        wav = torch.from_numpy(wav).float()
        wav = self.postprocess(wav, cur_sample_rate)
        return wav

    def get_label(self, index, label_idx):
        if self.store_labels:
            label = self.label_list[label_idx][index]
        else:
            with open(self.label_paths[label_idx]) as f:
                offset_s, offset_e = self.label_offsets_list[label_idx][index]
                f.seek(offset_s)
                label = f.read(offset_e - offset_s)

        if self.label_processors is not None:
            label = self.label_processors[label_idx](label)
        return label

    def get_labels(self, index):
        return [self.get_label(index, i) for i in range(self.num_labels)]

    def __getitem__(self, index):
        #wav = self.get_audio(index)
        #we don't want to read the wav here
        wav = None
        labels = self.get_labels(index)
        if len(self.bnds) > 0:
            bnd = self.bnds[index]
        else:
            bnd = []

        #source is audio length instead of source
        return {"id": index, "length": self.sizes[index], "label_list": labels, "boundary": bnd}

    def __len__(self):
        return len(self.sizes)

    def crop_to_max_size(self, wav_id, target_size):
        size = self.sizes[wav_id]
        diff = size - target_size
        #logger.info(f"dans crop to max size ... size:{size} target_size:{target_size} diff:{diff} wav_id:{wav_id}")
        wav_path = os.path.join(self.audio_root, self.audio_names[wav_id])

        if diff <= 0:
            audio = Audio(sample_rate=self.sample_rate, mono='downmix')
            wav, sr = audio({"audio": wav_path})
            wav = wav.squeeze()
            #logger.info(f"diff <= 0 ({size/16000}s total audio duration) {wav.shape}")
            return wav, 0

        start, end = 0, target_size
        #logger.info(f"start:{start} end:{end}")
        if self.random_crop:
            start = np.random.randint(0, diff + 1)
            end = size - diff + start
            audio = Audio(sample_rate=self.sample_rate, mono='downmix')

            wav, sr = audio.crop(wav_path, Segment(start/self.sample_rate, end/self.sample_rate), duration=target_size/self.sample_rate)
            wav = wav.squeeze()
            #logger.info(f"start:{start_s}s end:{end_s}s duration={target_size/16000}s "
            #            f"({size_s}s total audio duration) {wav.shape}")
            #logger.info(f"stats_selected_files:{wav_id}\t{self.ind_dataset[wav_id]}\t{start/self.sample_rate:.2f}\t"
            #            f"{end/self.sample_rate:.2f}\t{size/self.sample_rate:.2f}")
            assert wav.shape[0] == target_size

        #logger.info(f"SHAPE:{wav.shape}, START:{start}")
        return wav, start

    def collater(self, samples):
        # target = max(sizes) -> random_crop not used
        # target = max_sample_size -> random_crop used for long


        chrono_start = time.time()
        #logger.info(f"dans collater ... samples:{samples}")


        samples = [s for s in samples if s["length"] is not None]
        if len(samples) == 0:
            return {}


        audios_ids = [s["id"] for s in samples]
        audio_sizes = [s["length"] for s in samples]

        bnds = [s["boundary"] for s in samples]

        #logger.info(f"self.pad_audio ? {self.pad_audio}, max(audio_sizes): {max(audio_sizes)}, "
        #            f"min(audio_sizes): {min(audio_sizes)}, max_sample_size:{self.max_sample_size}")

        if self.pad_audio:
            audio_size = min(max(audio_sizes), self.max_sample_size)
        else:
            audio_size = min(min(audio_sizes), self.max_sample_size)

        #logger.info(f"dans collater ... audio_size:{audio_size} ({audio_size/self.sample_rate:.2f}s)")

        collated_audios, padding_mask, audio_starts = self.collater_audio(
            audios_ids, audio_size
        )

        #logger.info(f"dans collater ... collated_audios:{collated_audios}, shape: {collated_audios.shape}")

        if self.mixing_prob > 0:
            collated_audios = self.mixing_collated_audios(collated_audios)

        targets_by_label = [
            [s["label_list"][i] for s in samples]
            for i in range(self.num_labels)
        ]
        targets_list, lengths_list, ntokens_list = self.collater_label(
            targets_by_label, audio_size, audio_starts
        )

        net_input = {"source": collated_audios, "padding_mask": padding_mask, "boundary": bnds}
        batch = {
            "id": torch.LongTensor([s["id"] for s in samples]),
            "net_input": net_input,
        }

        if self.single_target:
            batch["target_lengths"] = lengths_list[0]
            batch["ntokens"] = ntokens_list[0]
            batch["target"] = targets_list[0]
        else:
            batch["target_lengths_list"] = lengths_list
            batch["ntokens_list"] = ntokens_list
            batch["target_list"] = targets_list

        if self.multitask:
            batch["task"] = "multitask"
        else:
            batch["task"] = "wavlm"

        chrono_end = time.time()
        #logger.info(f"Time to prepare batch:{chrono_end - chrono_start:.5f}s (batch:{batch})")
        return batch

    def mixing_collated_audios(self, source):
        # mixing utterance or noise within the current batch

        B = source.shape[0]
        T = source.shape[1]
        mixing_max_len = T // 2 if self.mixing_max_len < 0 else T // self.mixing_max_len
        mixing_max_len = T if mixing_max_len > T else mixing_max_len
        for i in range(B):
            if np.random.random() < self.mixing_prob:
                if self.mixing_noise and np.random.random() < self.mixing_noise_prob:
                    # mixing with noise
                    choices = np.random.choice(self.noise_list, self.mixing_noise_num)
                    for c in choices:
                        path, key, start, end = c["loc"].split("\t")
                        if path not in self.noise_container:
                            self.noise_container[path] = h5py.File(path, "r")["wav"]
                        noise = self.noise_container[path][int(start): int(end)]
                        noise = noise.astype(np.float32) / np.iinfo(np.int16).max

                        ref_pow = np.mean(source[i].numpy() ** 2)
                        noise_pow = np.mean(noise ** 2)
                        if noise_pow == 0:
                            scale = 0
                        else:
                            snr = np.random.uniform(-5, 20)
                            scale = (ref_pow / (noise_pow * 10 ** (snr / 10))) ** 0.5
                        noise = scale * noise
                        noise = torch.from_numpy(noise).type_as(source)

                        c_len = np.random.randint(0, mixing_max_len + 1)
                        c_len = min(c_len, noise.shape[0])

                        c_end = np.random.randint(c_len, noise.shape[0] + 1)
                        c_start = c_end - c_len
                        s_end = np.random.randint(c_len, T + 1)
                        s_start = s_end - c_len

                        source[i, s_start:s_end] += noise[c_start:c_end]

                else:
                    # mixing with utterance
                    choices = np.random.choice(range(B), self.mixing_num, replace=True)
                    for c in choices:
                        c_len = np.random.randint(0, mixing_max_len + 1)

                        c_end = np.random.randint(c_len, T + 1)
                        c_start = c_end - c_len
                        s_end = np.random.randint(c_len, T + 1)
                        s_start = s_end - c_len

                        ref_pow = np.mean(source[i].numpy() ** 2)
                        noise_pow = np.mean(source[c].numpy() ** 2)
                        if noise_pow == 0:
                            scale = 0
                        else:
                            snr = np.random.uniform(-5, 5)
                            scale = (ref_pow / (noise_pow * 10 ** (snr / 10))) ** 0.5

                        source[i, s_start:s_end] += source[c, c_start:c_end].clone() * scale

                if self.normalize:
                    with torch.no_grad():
                        source[i] = F.layer_norm(source[i], source[i].shape)

        return source

    def collater_audio(self, audios_ids, audio_size):
        collated_audios = torch.zeros(len(audios_ids), audio_size)
        #print(f"collated_audios:{collated_audios}, shape:{collated_audios.shape}")

        padding_mask = (
            torch.BoolTensor(collated_audios.shape).fill_(False)
            # if self.pad_audio else None
        )
        audio_starts = [0 for _ in audios_ids]
        #logger.info(f"Collater audio: {audio_size} / {audio_starts}")

        for i, audio_id in enumerate(audios_ids):
            diff = self.sizes[audio_id] - audio_size

            audio = Audio(sample_rate=self.sample_rate, mono='downmix')
            wav_path = os.path.join(self.audio_root, self.audio_names[audio_id])

            if diff == 0:
                wav, sr = audio({"audio": wav_path})
                collated_audios[i] = wav.squeeze()
                #logger.info("diff==0 collater.audio exact same size")
                assert self.sizes[audio_id] == len(collated_audios[i])
            elif diff < 0:
                assert self.pad_audio
                wav, sr = audio({"audio": wav_path})
                wav = wav.squeeze()
                #logger.info(f"diff<0 ({diff}) wav.shape:{wav.shape} / {wav_path} / len(wav): {len(wav)}, size={self.sizes[audio_id]}")
                assert self.sizes[audio_id] == len(wav)
                collated_audios[i] = torch.cat(
                    [wav, wav.new_full((-diff,), 0.0)]
                )
                #logger.info(f"diff<0 ({diff}) after pad wav.shape:{collated_audios[i].shape}")
                padding_mask[i, diff:] = True
            else:
                collated_audios[i], audio_starts[i] = self.crop_to_max_size(
                    audio_id, audio_size
                )

        return collated_audios, padding_mask, audio_starts

    def collater_frm_label(
        self, targets, audio_size, audio_starts, label_rate, pad
    ):
        assert label_rate > 0
        s2f = label_rate / self.sample_rate
        frm_starts = [int(round(s * s2f)) for s in audio_starts]
        frm_size = int(round(audio_size * s2f))
        if not self.pad_audio:
            rem_size = [len(t) - s for t, s in zip(targets, frm_starts)]
            frm_size = min(frm_size, *rem_size)
        targets = [t[s: s + frm_size] for t, s in zip(targets, frm_starts)]
        logger.debug(f"audio_starts={audio_starts}")
        logger.debug(f"frame_starts={frm_starts}")
        logger.debug(f"frame_size={frm_size}")

        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_seq_label(self, targets, pad):
        lengths = torch.LongTensor([len(t) for t in targets])
        ntokens = lengths.sum().item()
        targets = data_utils.collate_tokens(
            targets, pad_idx=pad, left_pad=False
        )
        return targets, lengths, ntokens

    def collater_label(self, targets_by_label, audio_size, audio_starts):
        targets_list, lengths_list, ntokens_list = [], [], []
        itr = zip(targets_by_label, self.label_rates, self.pad_list)
        for targets, label_rate, pad in itr:
            if label_rate == -1:
                targets, lengths, ntokens = self.collater_seq_label(
                    targets, pad
                )
            else:
                targets, lengths, ntokens = self.collater_frm_label(
                    targets, audio_size, audio_starts, label_rate, pad
                )
            targets_list.append(targets)
            lengths_list.append(lengths)
            ntokens_list.append(ntokens)
        return targets_list, lengths_list, ntokens_list

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        #if self.pad_audio:
        #    return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.balance:
            return self.get_balance_indices()

        elif self.shuffle:
            if len(self.chunk_names) > 0:
                with data_utils.numpy_seed(self.epoch):
                    self.chunk_order = np.random.permutation(len(self.chunk_names))
                chunk_count = 0
                tmp_sizes = []
                tmp_indices = []
                indice = []
                for i in self.chunk_order:
                    chunk_count += 1
                    start = self.chunk_indices[i]
                    end = self.chunk_indices[i+1] if i < len(self.chunk_names) - 1 else len(self)
                    size = list(self.sizes[start:end])
                    tmp_indices.extend(list(np.arange(start, end)))
                    tmp_sizes.extend(size)
                    if chunk_count % 10 == 0 or i == self.chunk_order[0]:
                        order = [np.random.permutation(len(tmp_indices))]
                        order.append(
                            np.minimum(
                                np.array(tmp_sizes),
                                self.max_sample_size,
                            )
                        )
                        sort_idx = np.lexsort(order)[::-1]
                        indice.append([tmp_indices[k] for k in sort_idx])
                        tmp_indices = []
                        tmp_sizes =[]
                return indice
            else:
                order = [np.random.permutation(len(self))]
                order.append(
                    np.minimum(
                        np.array(self.sizes),
                        self.max_sample_size,
                    )
                )
                return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav
