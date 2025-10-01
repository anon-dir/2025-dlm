"""
Here we provide the TTS data.
Approx 75.000 hours.
"""

from __future__ import annotations
import os
from functools import cache, partial
import math
import re
import functools
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

from i6_core.util import instanciate_delayed
from i6_experiments.users.zeyer.datasets.librispeech import (
    LibrispeechOggZip,
    get_vocab_by_str,
)
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from returnn_common.datasets_old_2022_10.interface import DatasetConfigStatic
from sisyphus import tk

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.datasets.task import Task


NUM_CORPUS_FILES = 750


def py():  # when run directly as Sis config
    get_tts_oggzip_count_num_seqs()
    get_lm_tts_seq_list(seq_tag_format="lm")
    get_lm_tts_seq_list(seq_tag_format="tts")


@cache
def _get_tts_oggzips_base_dir():
    dirs = [
        ".../denoising_lm/lm_tts_data",  # TODO...
    ]
    for d in dirs:
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"Could not find TTS data directory from {dirs}")


@cache
def get_tts_oggzips() -> List[tk.Path]:
    base_dir = _get_tts_oggzips_base_dir()
    ds = []
    for i in range(1, NUM_CORPUS_FILES + 1):
        ds.append(
            tk.Path(
                f"{base_dir}/lm_data_part{i}_ogg.zip",
                hash_overwrite=f".../TTS-data/{i}",  # TODO hash...
            )
        )
    return ds


def get_tts_oggzip_count_num_seqs(*, subset: Literal["train", "dev", "test", "all"] = "train"):
    from i6_experiments.users.zeyer.datasets.utils.extract_seq_list import ExtractNumSeqsFromReturnnDatasetJob
    from i6_experiments.users.zeyer.utils.write_delayed_job import DelayedToVariableJob
    from i6_experiments.users.zeyer.utils.delayed_sum import DelayedSum

    prefix = get_setup_prefix_for_module(__name__)

    oggzips = get_tts_oggzips()
    num_seqs_ = []
    for oggzip in oggzips:
        num_seqs_.append(
            ExtractNumSeqsFromReturnnDatasetJob(
                returnn_dataset={
                    "class": "OggZipDataset",
                    "path": oggzip,
                    "resolve_symlink_for_name": True,
                    "audio": None,
                    "targets": None,
                }
            ).out_num_seqs
        )
    subsets = {
        "all": DelayedToVariableJob(DelayedSum(num_seqs_)).out,
        "train": DelayedToVariableJob(DelayedSum(num_seqs_[LibrispeechTtsOggZip.NUM_EVAL_SETS :])).out,
        "dev": DelayedToVariableJob(DelayedSum(num_seqs_[: LibrispeechTtsOggZip.NUM_EVAL_SETS // 2])).out,
        "test": DelayedToVariableJob(
            DelayedSum(num_seqs_[LibrispeechTtsOggZip.NUM_EVAL_SETS // 2 : LibrispeechTtsOggZip.NUM_EVAL_SETS])
        ).out,
    }
    # orig LM text: 40418261
    # all TTS: 40418261 (same as orig LM text)
    # train TTS: 40310479
    for key, num_seqs in subsets.items():
        tk.register_output(prefix + f"/tts_oggzip.num_seqs_{key}.txt", num_seqs)
    return subsets[subset]


@cache
def get_lm_tts_seq_list(
    *, subset: Union[Literal["train", "dev", "test", "all"], str] = "train", seq_tag_format: Literal["lm", "tts"]
) -> tk.Path:
    """
    Get the seq tag list (and seq order) for the LM text data.
    This function is independent of the TTS Ogg files.

    This seq list is consistent to the TTS data (via the OggZips),
    and all datasets which descent from this pipeline.

    Note that some of the pipeline might further filter out some seqs (e.g. empty, or too long).

    :param subset: we partition the LM (TTS) data into train, dev, test. here you can select which part to take
    :param seq_tag_format: "lm" or "tts". "lm" is like "line_0", "line_1", matching the original LM text (LmDataset).
        "tts" is like "librispeech-lm-part1/recording_0/line_0", ... matching the TTS OggZips (OggZipDataset).
    :return: seq list file. the order is like the TTS (via ShuffleAndSplitSegmentsJob).
    """
    # in tts oggzip, seq tags are like: "librispeech-lm-part1/recording_7566/line_7566"
    # example entry from lm_data_part1_ogg.zip:
    # {'text': 'YAP', 'speaker_name': None, 'file': 'librispeech-lm-part1_recording_36235_line_36235/0.0000_1.0125.ogg',
    #  'seq_name': 'librispeech-lm-part1/recording_36235/line_36235', 'duration': 1.0125},
    # that originates from:
    #     from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
    #     lm_data = get_librispeech_normalized_lm_data()
    #
    #     # misuse shuffle and split segments
    #     from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
    #     shuffle_job = ShuffleAndSplitSegmentsJob(
    #         segment_file=lm_data,
    #         split={"part%i" % (i + 1): 1.0/750.0 for i in range(750)}
    #     )
    #     shuffle_job.add_alias(prefix + "/shuffle_job")
    # ShuffleAndSplitSegmentsJob job:
    # .../work/i6_core/corpus/segments/ShuffleAndSplitSegmentsJob.jbIOxTeGptOy
    #   ShuffleAndSplitSegmentsJob.jbIOxTeGptOy/output/part1.segments:
    #     first line: OH THE FARMER'S LIFE FOR ME
    # in lm_data_part1_ogg.zip:
    # {'text': "OH THE FARMER'S LIFE FOR ME", 'speaker_name': None,
    #  'file': 'librispeech-lm-part1_recording_0_line_0/0.0000_1.9625.ogg',
    #  'seq_name': 'librispeech-lm-part1/recording_0/line_0', 'duration': 1.9625},
    from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
    from i6_experiments.users.zeyer.datasets.utils.extract_seq_list import (
        ExtractNumLinesFromTextFileJob,
        WriteSeqListFromShuffledJob,
    )

    prefix = get_setup_prefix_for_module(__name__)
    lm_data = get_librispeech_normalized_lm_data()
    # skip_empty_lines=True because the original TTS pipeline did not filter empty lines.
    # It used ShuffleAndSplitSegmentsJob, and then created Bliss corpora from that,
    # which includes the empty seqs from the original LM text.
    # (I think there is only a single empty seq, right in the first line of the LM text file.)
    lm_num_lines = ExtractNumLinesFromTextFileJob(text_file=lm_data, skip_empty_lines=False).out_num_lines
    # We can now use WriteLmDatasetSeqListJob + ShuffleAndSplitSegmentsJob to generate matching seq list
    # pointing back to the original LM text.
    # We can also use WriteSeqListInOrigOrderFromShuffledJob to get a seq list of the new seq names
    # in the order of the orig LM text.
    # We however write out a seq list in the new order, matching the new seq names.

    if seq_tag_format == "lm":
        seq_tag_template = "line-{orig_seq_idx}"
    elif seq_tag_format == "tts":
        seq_tag_template = "librispeech-lm-{split_key}/recording_{split_seq_idx}/line_{split_seq_idx}"
    else:
        raise ValueError(f"invalid seq_tag_format {seq_tag_format!r}")

    selected_split_indices = LibrispeechTtsOggZip.get_indices(subset)

    lm_seq_list_after_tts_train = WriteSeqListFromShuffledJob(
        seq_tag_template=seq_tag_template,
        num_seqs=lm_num_lines,
        split={"part%i" % (i + 1): 1 / NUM_CORPUS_FILES for i in range(NUM_CORPUS_FILES)},
        selected_splits=["part%i" % (i + 1) for i in selected_split_indices],
    ).out_segments
    lm_seq_list_after_tts_train.creator.add_alias(prefix + f"/seq-list_ls-lm/{seq_tag_format}_seq_list_{subset}")
    tk.register_output(
        prefix + f"/seq-list_ls-lm/{seq_tag_format}_seq_list_{subset}.segments", lm_seq_list_after_tts_train
    )
    return lm_seq_list_after_tts_train


def get_sub_epoch_dataset(
    files_subepoch: List[str],
    *,
    epoch: Optional[int] = None,
    template: Dict,
    cache_files: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    from returnn.util.file_cache import CachedFile
    from pathlib import Path

    d = template.copy()
    d["targets"]["model_file"] = instanciate_delayed(d["targets"]["model_file"])

    if epoch is not None:
        # Epoch starts at one. We could also do modulo but this way we throw errors when we go out of bounds (which is good).
        real_files = [files_subepoch[epoch - 1]]
    else:
        real_files = files_subepoch

    if cache_files:
        d["path"] = [CachedFile(str(Path(fn).resolve())) for fn in real_files]
    else:
        d["path"] = [str(Path(fn).resolve()) for fn in real_files]
    return d


class LibrispeechTtsOggZip(LibrispeechOggZip):
    """
    Dataset for the TTS files.
    First file is the dev set, second file is the test set (both further subsetted to improve speed), all other files are the training set.
    """

    NUM_EVAL_SETS = 2  # only for code readability, don't change this
    TRAINSPLIT_PATTERN = re.compile(r"train_split_(\d+)_of_(\d+)")  # train_split_{i}_of_{num_parts}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, train_epoch_split=1, **kwargs)

    @staticmethod
    def get_num_training_parts(files_per_part: Optional[int] = None) -> int:
        if files_per_part is None:
            return 1
        num_oggs = len(get_tts_oggzips())
        assert num_oggs - LibrispeechTtsOggZip.NUM_EVAL_SETS > 0
        return int(math.ceil((num_oggs - LibrispeechTtsOggZip.NUM_EVAL_SETS) / files_per_part))

    def copy_trainsplit_as_static(self, name: str, type: Literal["All", "TextOnly"] = "All") -> DatasetConfigStatic:
        # TODO: this method is almost identical to LibrispeechOggZip.copy_train_as_static, maybe refactor the subclass method to do this?
        assert LibrispeechTtsOggZip.TRAINSPLIT_PATTERN.match(name), (
            f"invalid name for trainsplit {name!r}, expected train_split_(i)_of_(num_parts)"
        )
        return DatasetConfigStatic(
            main_name=name,
            main_dataset=self.get_dataset(name, type=type),
            extern_data=self.get_extern_data(),
            default_input=self.get_default_input(),
            default_target=self.get_default_target(),
        )

    def copy_as_static(
        self,
        *,
        with_main_dataset: bool = True,
        type: Literal["All", "TextOnly"] = "All",
    ) -> DatasetConfigStatic:
        return DatasetConfigStatic(
            main_name=self.get_main_name(),
            main_dataset=self.get_main_dataset(type=type) if with_main_dataset else None,
            extern_data=self.get_extern_data(),
            default_input=self.get_default_input(),
            default_target=self.get_default_target(),
            train_dataset=self.get_train_dataset(),
            eval_datasets=self.get_eval_datasets(),
        )

    def get_main_dataset(self, type: Literal["All", "TextOnly"] = "All") -> Dict[str, Any]:
        assert self.main_key is not None, f"{self}: main_dataset not defined, main_key is None"
        return self.get_dataset(self.main_key, type=type)

    @classmethod
    def get_oggs(cls, key: str) -> List[tk.Path]:
        oggs = get_tts_oggzips()
        indices = cls.get_indices(key)
        return [oggs[i] for i in indices]

    @classmethod
    def get_indices(cls, key: str) -> range:
        oggs = range(NUM_CORPUS_FILES)

        if key == "dev":
            oggs = oggs[: cls.NUM_EVAL_SETS // 2]
        elif key == "test":
            oggs = oggs[cls.NUM_EVAL_SETS // 2 : cls.NUM_EVAL_SETS]
        elif key == "train":
            oggs = oggs[cls.NUM_EVAL_SETS :]
        elif key == "devtrain":  # devtrain is a subset of train
            oggs = oggs[cls.NUM_EVAL_SETS : cls.NUM_EVAL_SETS + cls.NUM_EVAL_SETS // 2]
        elif m := cls.TRAINSPLIT_PATTERN.match(key):
            # Complicated logic to transform `train_split_{i}_of_{num_parts}` into a list of ogg files
            i, num_parts = map(int, m.groups())
            assert 0 <= i < num_parts
            num_files_in_part = int(math.ceil((len(oggs) - cls.NUM_EVAL_SETS) / num_parts))
            assert cls.NUM_EVAL_SETS + num_parts * num_files_in_part >= len(oggs)  # we shouldn't leave any files out

            start = cls.NUM_EVAL_SETS + i * num_files_in_part
            end = start + num_files_in_part
            if i == num_parts - 1:
                oggs = oggs[start:]
                assert end >= len(oggs)
            else:
                oggs = oggs[start:end]
        else:
            assert False, f"invalid key {key!r}"
        assert len(oggs) > 0, f"no data for key {key!r}"
        return oggs

    def get_dataset(
        self,
        key: str,
        *,
        training: bool = False,
        subset: Optional[int] = None,
        type: Literal["All", "TextOnly"] = "All",
    ) -> Dict[str, Any]:
        oggs = self.get_oggs(key)

        dt: Dict[str, Any] = {
            "class": "OggZipDataset",
        }

        # this is copied from LibrispeechOggZip.get_dataset
        if self.audio is not None:
            dt["audio"] = self.audio.copy()
        else:
            dt["audio"] = None
        if self.vocab is not None:
            vocab = self.train_vocab if training and self.train_vocab else self.vocab
            dt["targets"] = vocab.get_opts().copy()
            assert "seq_postfix" not in dt["targets"], dt  # we are handling this here
            if self.with_eos_postfix:
                eos_id = vocab.get_eos_idx()
                assert eos_id is not None, f"{self}: vocab {vocab} does not define EOS"
                dt["targets"]["seq_postfix"] = [eos_id]
        else:
            dt["targets"] = None
        if training:
            dt["partition_epoch"] = self.train_epoch_split
            if self.train_epoch_wise_filter is not None:
                dt["epoch_wise_filter"] = self.train_epoch_wise_filter
            if self.train_audio_preprocess is not None:
                assert self.audio is not None, "train_audio_preprocess needs audio"
                dt["audio"]["pre_process"] = self.train_audio_preprocess
            if self.train_audio_random_permute:
                assert self.audio is not None, "train_audio_random_permute needs audio"
                dt["audio"]["random_permute"] = self.train_audio_random_permute
            dt["seq_ordering"] = f"laplace:.{self.train_sort_laplace_num_seqs}"
        else:
            dt["fixed_random_seed"] = 1
            dt["seq_ordering"] = "sorted_reverse"

        if key == "dev" or key == "test":
            assert training is False
            dt["fixed_random_subset"] = 2000  # Otherwise dev/test take way too long
            if subset:
                dt["fixed_random_subset"] = min(subset, dt["fixed_random_subset"])

        if subset:
            dt["fixed_random_subset"] = subset

        if self.extra_args:
            dt.update(self.extra_args)

        # TODO: maybe special case for dev/test, where we don't use more than one file?
        if type == "TextOnly":
            dt["audio"] = None

        train = {
            "class": "DistributeFilesDataset",
            "files": oggs,
            "get_sub_epoch_dataset": functools.partial(get_sub_epoch_dataset, template=dt),
            "partition_epoch": len(oggs),
        }

        wrapped = {
            "class": "MultiEpochDataset",
            "dataset": train,
            "multi_epoch": len(oggs),
        }

        return wrapped


_raw_audio_opts = dict(
    features="raw",
    sample_rate=16_000,
    peak_normalization=True,
    preemphasis=None,
)


def get_tts_datasets(
    *,
    vocab: str,
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    audio_opts: Optional[Dict[str, Any]] = None,
    audio_dim: int = 1,
    extra_dataset_common_opts: Optional[Dict[str, Any]] = None,
    with_devtrain: bool = False,  # not sure if you always want this?
) -> tuple[LibrispeechTtsOggZip, Dict[str, LibrispeechTtsOggZip]]:
    assert isinstance(vocab, str)
    vocab_obj = get_vocab_by_str(vocab)

    audio_opts_ = _raw_audio_opts.copy()
    if audio_opts:
        audio_opts_.update(audio_opts)
    dataset_common_opts: Dict[str, Any] = dict(audio=audio_opts_, audio_dim=audio_dim, vocab=vocab_obj)
    if train_vocab_opts:
        dataset_common_opts["train_vocab"] = vocab_obj.copy(**train_vocab_opts)
    if extra_dataset_common_opts:
        dataset_common_opts.update(extra_dataset_common_opts)

    train_dataset = LibrispeechTtsOggZip(**dataset_common_opts)
    eval_datasets = {
        "dev": LibrispeechTtsOggZip(**dataset_common_opts, main_key="dev"),
        "test": LibrispeechTtsOggZip(**dataset_common_opts, main_key="test"),
    }
    if with_devtrain:
        eval_datasets["devtrain"] = LibrispeechTtsOggZip(**dataset_common_opts, main_key="devtrain")

    return train_dataset, eval_datasets


def get_asr_tts_extended_task(
    *,
    vocab: str = "spm10k",
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    train_audio_preprocess: Optional[Any] = None,
    train_tts_partition_epoch: int = 75,
) -> Task:
    """
    Get task. Combine LS ASR data with LS LM TTS data. Speech -> text.

    After visiting one full epoch, we have seen one time all the LS ASR data.

    Epoch split is 1. No more epoch split is currently possible.

    We use DistributeFilesDataset over OggZipDataset, each contains 10 TTS files + the LS ASR data.
    Ratio is approx 1:1.
    See comment below for some discussion.

    Note, for serialization, this should use train_v4 or newer.
    Set "__serialization_version": 2.

    :param vocab: e.g. "spm10k".
        should be handled by :func:`get_vocab_by_str` and :func:`get_librispeech_task_text_only`
    :param train_vocab_opts:
    :param train_audio_preprocess:
    :param train_tts_partition_epoch: after how many epoch should we complete the full TTS data?
        Setting this to 75 means that after 75 epochs, we have seen approx 75,000h of TTS data,
        and 75*960h of LS ASR data.
    :return: task
    """
    import dataclasses
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2

    task = get_librispeech_task_raw_v2(
        vocab=vocab,
        train_epoch_split=1,
        train_vocab_opts=train_vocab_opts,
        train_epoch_wise_filter=None,
        train_audio_preprocess=None,
    )
    task = dataclasses.replace(task)  # makes new copy

    train_ds = task.train_dataset.copy_as_static(with_main_dataset=False)
    train_ds.use_deep_copy = True
    train_ds_dict = train_ds.train_dataset.copy()
    assert train_ds_dict["class"] == "OggZipDataset"
    train_ds_dict.pop("use_cache_manager", None)
    train_ds_dict["content_name"] = "out.ogg"  # fix for TTS data. should also work for ASR data.

    if train_audio_preprocess is not None:
        assert train_ds_dict.get("audio") is not None, "train_audio_preprocess needs audio"
        train_ds_dict["audio"] = train_ds_dict["audio"].copy()
        assert train_ds_dict["audio"].get("pre_process") is None, "train_audio_preprocess already set"
        train_ds_dict["audio"]["pre_process"] = train_audio_preprocess

    # Note: Potential ways how to combine the TTS data:
    # - VariableDataset: but doesn't handle caching and preloading well, cannot handle so much data
    # - ConcatDataset: no proper mixing, just one after the other
    # - MixingDataset: need better implementation. needed once we want different ratios, and better control
    #   (https://github.com/rwth-i6/returnn/issues/1700 about how to implement this)
    # - OggZipDataset with multiple files: fine for now, LS is 960h, 10 TTS files ~same, so 1:1 ratio
    # - DistributeFilesDataset: needed in some way to deal with the huge amount of data
    #
    # Thus, we use here DistributeFilesDataset over the TTS data,
    # and then inside OggZipDataset on the LS ASR data + 10 files of TTS data.
    # So we get approx a 1:1 ratio of LS ASR data to TTS data.
    # Note: LibriSpeech is 960h, TTS is 750 files, with approx 75,000h in total.

    # Note: MultiProcDataset might consume a bit too much memory in this setup,
    # as we have more data as usual, and also the DistributeFilesDataset will always preload the next epoch,
    # which again duplicates the required memory.
    # PostprocessingDataset with multi-processing (https://github.com/rwth-i6/returnn/issues/1701)
    # could be another better solution to avoid MultiProcDataset.

    tts_files = get_tts_oggzips()
    assert len(tts_files) == NUM_CORPUS_FILES
    train_ds_dict_ = {
        "class": "DistributeFilesDataset",
        "files": tts_files,
        "get_sub_epoch_dataset": partial(
            _get_distribute_files_dataset_for_epoch, base_opts=train_ds_dict, multi_proc_dataset={"num_workers": 10}
        ),
        "partition_epoch": train_tts_partition_epoch,
        "seq_ordering": "random",
    }
    train_ds.train_dataset = train_ds_dict_

    task.train_dataset = train_ds
    return task


# bind base_opts via partial
def _get_distribute_files_dataset_for_epoch(
    files: List[str], *, base_opts: Dict[str, Any], multi_proc_dataset: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    from returnn.util.file_cache import CachedFile
    from sisyphus.job_path import AbstractPath
    from i6_experiments.users.zeyer.datasets.utils import multi_proc as mp_ds_utils

    opts = base_opts.copy()
    assert opts["class"] == "OggZipDataset"
    files = opts["path"] + files
    files = [fn.get_path() if isinstance(fn, AbstractPath) else fn for fn in files]
    assert all(isinstance(fn, str) for fn in files)
    files = [CachedFile(fn) for fn in files]
    opts["path"] = files

    if multi_proc_dataset is not None:
        opts = mp_ds_utils.multi_proc_dataset_opts(opts, **multi_proc_dataset)

    return opts
