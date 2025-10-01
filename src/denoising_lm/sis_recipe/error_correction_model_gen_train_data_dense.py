"""
In constrast to :mod:`error_correction_model_gen_train_data`,
where the hyps are sequences of label indices,
here we want to have sequences of probability distributions over labels,
so dense (shape [...,Seq,Vocab] of float value, summed up to one over Vocab)
instead of sparse (shape [...,Seq] of int values).
To safe space, instead of storing the whole dense vector,
we only store the top-K, so effectively we store:
- shape [...,Seq,K] of int values (indices)
- shape [...,Seq,K] of float values (probs)
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Union, Any, Literal, Dict, List
import functools

from sisyphus import tk
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.zeyer.datasets.task import Task
from i6_experiments.users.zeyer.forward_to_hdf import forward_to_hdf
from returnn_common.datasets_old_2022_10.interface import DatasetConfigStatic
from returnn.tensor import TensorDict
from .ctc import GetHypsCfgV3 as GetCtcHypsCfgV3, GetHypsCfgV4 as GetCtcHypsCfgV4
from .tts_model import TtsOpts
from .tts_data import get_lm_tts_seq_list
from .error_correction_model_gen_train_data import (
    GetCtcHypsCfgV5,
    get_real_lm_txt,
    get_real_asr_txt,
    _get_asr_seq_list,
    get_default_hyps_model_by_vocab_and_cfg,
)

TGetHypsCfg = Union[GetCtcHypsCfgV3, GetCtcHypsCfgV4]


@dataclasses.dataclass
class AsrHypsProbsCfg:
    """
    Instead of storing one single hyp from the ASR model and feeding that to the DLM,
    we feed a seq of prob distributions over the vocab per each label-frame to the DLM.

    There are some variants on what to store / what to feed to the DLM
    which are configured here.
    """

    top_k: int  # how many labels/probs to store per label-frame
    extract_method: str  # "label_prefix_search" or "soft_collapse_repeated"

    extract_method_opts: Optional[Dict[str, Any]] = None
    include_blank: bool = False


def py():
    get_error_correction_model_task_via_tts_dense_txt(
        hyps_probs_cfg=AsrHypsProbsCfg(top_k=5, extract_method="label_prefix_search")
    )


def get_error_correction_model_task_via_tts_dense_txt(
    *,
    prefix: Optional[str] = None,
    vocab: str = "spm10k",
    num_hyps: int = 5,
    hyps_probs_cfg: AsrHypsProbsCfg,
    train_epoch_split: Optional[int] = 20,
    train_repeat_asr_data: int = 1,
    train_repeat_asr_data_via_num_hyps: bool = False,
    hyps_model: Optional[ModelWithCheckpoint] = None,
    hyps_cfg: Optional[TGetHypsCfg] = None,
    hyps_tts_opts: Optional[Union[TtsOpts, List[TtsOpts]]] = None,
    additional_eval_sets: Optional[List[str]] = None,
    use_dependency_boundary: bool = True,
    dependency_boundary_hash: Optional[str] = None,
    get_hyps_extra_config: Optional[Dict[str, Any]] = None,
) -> Task:
    """
    Get task. Using hyps from TTS. As txt, via LmDataset.

    :param prefix:
    :param vocab: e.g. "spm10k".
        should be handled by :func:`get_vocab_by_str` and :func:`get_librispeech_task_text_only`
    :param num_hyps: for train
    :param hyps_probs_cfg:
    :param train_epoch_split: for train
    :param train_repeat_asr_data: for train, how often to repeat the ASR (LS) data
    :param train_repeat_asr_data_via_num_hyps:
    :param hyps_model:
    :param hyps_cfg:
    :param hyps_tts_opts: If given, a TTS model is used for on-the-fly generation of the audio for the CTC model
        to generate the hyps.
        If a list is given, allows mixing multiple TTS models.
        It will just call the hyps func the multiple times, and then mix up the hyps,
        interleaving taking one from each.
    :param additional_eval_sets:
    :param use_dependency_boundary:
    :param dependency_boundary_hash:
    :return: task
    """
    from returnn.tensor import Dim, batch_dim
    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_text_only,
        get_vocab_by_str,
    )
    from i6_experiments.users.zeyer.datasets.utils.sclite_generic_score import generic_sclite_score_recog_out

    if use_dependency_boundary:
        from i6_experiments.common.helpers.dependency_boundary import (
            dependency_boundary,
        )

        return dependency_boundary(
            functools.partial(
                get_error_correction_model_task_via_tts_dense_txt,
                prefix=prefix,
                vocab=vocab,
                num_hyps=num_hyps,
                hyps_probs_cfg=hyps_probs_cfg,
                train_epoch_split=train_epoch_split,
                train_repeat_asr_data=train_repeat_asr_data,
                train_repeat_asr_data_via_num_hyps=train_repeat_asr_data_via_num_hyps,
                hyps_model=hyps_model,
                hyps_cfg=hyps_cfg,
                hyps_tts_opts=hyps_tts_opts,
                additional_eval_sets=additional_eval_sets,
                use_dependency_boundary=False,
                get_hyps_extra_config=get_hyps_extra_config,
            ),
            hash=dependency_boundary_hash,
        )

    if additional_eval_sets is None:
        additional_eval_sets = []
    assert isinstance(additional_eval_sets, list)

    in_key, out_key = "hyps", "real"

    if not hyps_model:
        hyps_model = get_default_hyps_model_by_vocab_and_cfg(vocab=vocab, cfg=hyps_cfg)

    vocab_ = get_vocab_by_str(vocab)
    vocab_dict = vocab_.get_opts()

    real_vocab_dim = Dim(vocab_.get_num_classes(), name="vocab")
    if hyps_probs_cfg.include_blank:
        hyps_vocab_dim = Dim(vocab_.get_num_classes() + 1, name="vocab+blank")
    else:
        hyps_vocab_dim = real_vocab_dim
    hyps_vocab_dict = _get_hyps_vocab_dict(vocab_dict=vocab_dict, hyps_probs_cfg=hyps_probs_cfg)
    out_probs_top_k_dim = Dim(hyps_probs_cfg.top_k, name="k")
    hyps_spatial_dim = Dim(None, name="hyps_spatial")
    extern_data = {
        "hyps": {
            "dims": [batch_dim, hyps_spatial_dim, out_probs_top_k_dim],
            "dtype": "int32",
            "sparse_dim": hyps_vocab_dim,
            # Keep existing hashes.
            "vocab": hyps_vocab_dict if hyps_probs_cfg.include_blank else vocab_dict,
        },
        "hyps_k_log_probs": {
            "dims": [batch_dim, hyps_spatial_dim, out_probs_top_k_dim],
            "dtype": "float32",
        },
        "real": {
            "dims": [batch_dim, Dim(None, name="real_spatial")],
            "dtype": "int32",
            "sparse_dim": real_vocab_dim,
            "vocab": vocab_dict,
        },
    }
    make_dataset_common_kwargs = dict(
        prefix=prefix,
        hyps_model_vocab=vocab,
        hyps_model=hyps_model,
        vocab_dict=vocab_dict,
        hyps_probs_cfg=hyps_probs_cfg,
        extern_data=extern_data,
        get_hyps_extra_config=get_hyps_extra_config,
    )
    train_ds = DatasetConfigStatic(
        default_input=in_key,
        default_target=out_key,
        extern_data=extern_data,
        train_dataset=_make_dataset(
            **make_dataset_common_kwargs,
            subset="train",
            num_hyps=num_hyps,
            train=True,
            train_epoch_split=train_epoch_split,
            repeat_asr_data=train_repeat_asr_data,
            repeat_asr_data_via_num_hyps=train_repeat_asr_data_via_num_hyps,
            hyps_cfg=hyps_cfg,
            hyps_tts_opts=hyps_tts_opts,
        ),
        eval_datasets={
            "dev": _make_dataset(
                **make_dataset_common_kwargs,
                subset="train-dev",
                hyps_cfg=hyps_cfg,
            ),
            "devtrain": _make_dataset(
                **make_dataset_common_kwargs,
                subset="train-devtrain",
                hyps_cfg=hyps_cfg,
            ),
        },
        use_deep_copy=True,
    )

    additional = list(filter(lambda x: not x.startswith("trainlike-"), additional_eval_sets))
    eval_dss = {
        key: DatasetConfigStatic(
            default_input=in_key,
            default_target=out_key,
            extern_data=extern_data,
            main_name=key,
            main_dataset=_make_dataset(
                **make_dataset_common_kwargs,
                subset=key,
                # Use hyps_cfg without any sampling for the eval sets.
                hyps_cfg=GetCtcHypsCfgV4() if isinstance(hyps_cfg, GetCtcHypsCfgV4) else GetCtcHypsCfgV3(),
                eval=True,
            ),
            use_deep_copy=True,
        )
        for key in ["dev-clean", "dev-other", "test-clean", "test-other"] + additional  # "lm-dev", "lm-devtrain"
    }
    additional_trainlike = [x[10:] for x in filter(lambda x: x.startswith("trainlike-"), additional_eval_sets)]
    eval_dss.update(
        {
            "trainlike-" + key: DatasetConfigStatic(
                default_input=in_key,
                default_target=out_key,
                extern_data=extern_data,
                main_name=key,
                main_dataset=_make_dataset(
                    **make_dataset_common_kwargs,
                    subset=key,
                    hyps_cfg=hyps_cfg,
                    hyps_tts_opts=hyps_tts_opts,
                    eval=True,
                ),
                use_deep_copy=True,
            )
            for key in ["dev-clean", "dev-other"] + additional_trainlike  # "lm-dev", "lm-devtrain"
        }
    )

    task = get_librispeech_task_text_only(vocab=vocab)
    task = dataclasses.replace(task)  # makes new copy
    task.train_epoch_split = train_epoch_split
    task.train_dataset = train_ds
    task.dev_dataset = eval_dss["dev-other"]
    task.eval_datasets = eval_dss
    task.score_recog_output_func = functools.partial(
        generic_sclite_score_recog_out, post_proc_funcs=task.recog_post_proc_funcs
    )
    return task


def _get_hyps_vocab_dict(*, vocab_dict: Dict[str, Any], hyps_probs_cfg: AsrHypsProbsCfg) -> Dict[str, Any]:
    from i6_experiments.users.zeyer.datasets.utils.vocab import (
        ExtractVocabLabelsJob,
        ExtractVocabSpecialLabelsJob,
        ExtendVocabLabelsByNewLabelJob,
    )

    vocab_file = ExtractVocabLabelsJob(vocab_dict).out_vocab

    if hyps_probs_cfg.include_blank:
        # Note, we could also introduce a new wrapper Vocabulary class in RETURNN,
        # like {class: ExtendByBlankVocab, base_vocab: vocab_dict}.
        vocab_w_blank_file = ExtendVocabLabelsByNewLabelJob(
            vocab=vocab_file, new_label="<blank>", new_label_idx=-1
        ).out_vocab
        return {
            "class": "Vocabulary",
            "vocab_file": vocab_w_blank_file,
            "special_symbols_via_file": ExtractVocabSpecialLabelsJob(vocab_dict).out_vocab_special_labels_dict,
        }
    else:
        return {
            "class": "Vocabulary",
            "vocab_file": vocab_file,
            "special_symbols_via_file": ExtractVocabSpecialLabelsJob(vocab_dict).out_vocab_special_labels_dict,
        }


def _make_dataset(
    *,
    prefix: Optional[str] = None,
    subset: Literal[
        "train",  # LS ASR train + LS LM (TTS) train ("lm-train")
        "train-dev",  # LS ASR train dev (subset of dev-clean and dev-other)
        "train-devtrain",  # LS ASR train devtrain (subset of train)
        "dev-clean",  # LS ASR dev-clean
        "dev-other",  # LS ASR dev-other
        "test-clean",  # LS ASR test-clean
        "test-other",  # LS ASR test-other
        "lm-dev",  # LS LM (TTS) dev
        "lm-test",  # LS LM (TTS) test
        "lm-devtrain",  # LS LM (TTS) devtrain (subset of train)
    ],
    num_hyps: int = 1,
    hyps_probs_cfg: AsrHypsProbsCfg,
    train: bool = False,
    eval: bool = False,
    train_epoch_split: Optional[int] = None,
    repeat_asr_data: int = 1,
    repeat_asr_data_via_num_hyps: bool = False,
    hyps_model_vocab: str,
    hyps_model: ModelWithCheckpoint,
    vocab_dict: Dict[str, Any],
    hyps_cfg: TGetHypsCfg,
    hyps_tts_opts: Optional[Union[TtsOpts, List[TtsOpts]]] = None,
    extern_data: Dict[str, Dict[str, Any]],
    get_hyps_extra_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    import functools
    from .error_correction_model import _rotate_datasets_for_epoch

    hyps: List[List[tk.Path]] = [[] for _ in range(num_hyps)]  # inner is dataset, outer is hyp idx
    real: List[tk.Path] = []
    seq_tags: List[tk.Path] = []

    get_hyps_common_kwargs = dict(
        prefix=prefix,
        model_vocab=hyps_model_vocab,
        model=hyps_model,
        cfg=hyps_cfg,
        hyps_probs_cfg=hyps_probs_cfg,
        get_hyps_extra_config=get_hyps_extra_config,
    )

    asr_subset = subset if not subset.startswith("lm-") else None
    if asr_subset:
        asr_hyps = get_hyps(
            **get_hyps_common_kwargs,
            subset=asr_subset,
            num_hyps=num_hyps * (repeat_asr_data if repeat_asr_data_via_num_hyps else 1),
        )
        if repeat_asr_data_via_num_hyps:
            assert len(asr_hyps) == num_hyps * repeat_asr_data
            for i, hyp in enumerate(asr_hyps):
                hyps[i % num_hyps].append(hyp)
        else:
            assert len(asr_hyps) == num_hyps
            for i, hyp in enumerate(asr_hyps):
                hyps[i] += [hyp] * repeat_asr_data
        asr_real = get_real_asr_txt(subset=asr_subset)
        real += [asr_real] * repeat_asr_data
        seq_tags += [_get_asr_seq_list(subset=asr_subset)] * repeat_asr_data

    lm_subset = {"train": "train", "lm-dev": "dev", "lm-test": "test", "lm-devtrain": "devtrain"}.get(subset, None)
    if lm_subset:
        assert isinstance(hyps_cfg, GetCtcHypsCfgV3) or isinstance(hyps_cfg, GetCtcHypsCfgV4)
        assert hyps_tts_opts
        lm_hyps = get_hyps(
            **get_hyps_common_kwargs,
            subset="lm-" + lm_subset,
            num_hyps=num_hyps,
            tts_opts=hyps_tts_opts,
            train=train,
        )
        assert len(lm_hyps) == num_hyps
        for i, hyp in enumerate(lm_hyps):
            hyps[i].append(hyp)
        lm_real = get_real_lm_txt(subset=lm_subset)
        real.append(lm_real)
        seq_tags.append(get_lm_tts_seq_list(subset=lm_subset, seq_tag_format="lm"))

    for i in range(num_hyps):
        assert hyps[i]
    assert real

    from i6_experiments.users.zeyer.datasets.utils.unwrap_hdf import unwrap_hdf_dataset

    def _make_hyps_ds(_fs):
        assert isinstance(_fs, list)
        ds = {
            "class": "HDFDataset",
            "files": _fs,
            "use_cache_manager": True,
        }
        if train:
            ds.update({"partition_epoch": train_epoch_split, "seq_ordering": "laplace:.1000"})
        else:
            ds.update({"seq_ordering": "sorted_reverse"})
        return unwrap_hdf_dataset(
            ds,
            # See get_hyps.
            extern_data={
                "data": extern_data["hyps"],
                "output_k_lob_probs": extern_data["hyps_k_log_probs"],
            },
        )

    def _make_real_ds(_fs):
        assert isinstance(_fs, list)
        return {
            "class": "LmDataset",
            "corpus_file": _fs,
            "seq_list_file": seq_tags,
            "use_cache_manager": True,
            "skip_empty_lines": False,
            "orth_vocab": vocab_dict,
            "seq_end_symbol": None,  # handled via orth_vocab
            "unknown_symbol": None,  # handled via orth_vocab
        }

    real_ds = _make_real_ds(real)
    hyps_dss = [_make_hyps_ds(hyps_i) for hyps_i in hyps]
    if len(hyps_dss) == 1:
        hyps_ds = hyps_dss[0]
    else:
        assert not eval
        hyps_ds = {
            "class": "VariableDataset",
            "get_dataset": functools.partial(_rotate_datasets_for_epoch, datasets=hyps_dss),
            "always_same_tags": True,
        }
        if train:
            hyps_ds["partition_epoch"] = train_epoch_split

    return {
        "class": "MetaDataset",
        "datasets": {"hyps": hyps_ds, "real": real_ds},
        "data_map": {
            "hyps": ("hyps", "data"),
            "hyps_k_log_probs": ("hyps", "output_k_lob_probs"),
            "real": ("real", "data"),
        },
        # Note: "hyps" is HDFDataset, so having "real" here would be slower,
        # as then it would pass the seq list to HDFDataset.init_seq_order,
        # which is slow (https://github.com/rwth-i6/returnn/issues/1669).
        # But also, we have duplicate seq tags in "hyps" due to different hyps,
        # and if we have "real" here, the HDFDataset.init_seq_order
        # would never use the alternatives but only the first one,
        # which is not what we want.
        "seq_order_control_dataset": "hyps",
    }


def get_hyps(
    *,
    prefix: Optional[str] = None,
    subset: Union[
        Literal[
            "train",
            "train-dev",
            "train-devtrain",
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
            "lm-train",
            "lm-dev",
            "lm-test",
            "lm-devtrain",
        ],
        str,
    ],
    model_vocab: str = "spm10k",
    model: ModelWithCheckpoint,
    num_hyps: int,
    cfg: Optional[TGetHypsCfg] = None,
    tts_opts: Optional[Union[TtsOpts, List[TtsOpts]]] = None,
    train: bool = False,
    hyps_probs_cfg: AsrHypsProbsCfg,
    get_hyps_extra_config: Optional[Dict[str, Any]] = None,
) -> List[tk.Path]:
    """
    Merged new variant of the old get_asr_hyps_txt and get_lm_hyps_txt_v3
    (from .error_correction_model_gen_train_data).
    Also, we do not produce txt files for LmDataset, but keep the data as HDF files.

    :return: collection of hyps txt corresponding to the hyps, i.e. len = num_hyps.
        The text can be used with LmDataset and is independent of the vocab.
        The order of sequences is determined via :func:`_get_asr_seq_list`,
        i.e. it's the order as in the original Ogg files.
        It's the same order as in :func:`get_real_asr_txt`.
    """
    from i6_experiments.users.zeyer.datasets.librispeech import get_vocab_by_str
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2
    from .tts_data import get_lm_tts_seq_list, LibrispeechTtsOggZip
    from .tts_model import get_tts_model_dataset_dict, get_tts_model_dataset_extern_data_data
    from returnn.tensor import Dim, batch_dim

    prefix = prefix or get_setup_prefix_for_module(__name__)

    name = model_vocab
    if num_hyps != 5:
        name += f"-numHyps{num_hyps}"
    if cfg:
        name += f"-{cfg}"
    name += f"-{hyps_probs_cfg.extract_method}.Top{hyps_probs_cfg.top_k}"
    name += "-keepSubwords"

    # Note: task hardcoded... (and also not needed, I just need the train dataset...)
    # Note: Model hardcoded...
    vocab = model_vocab

    vocab_ = get_vocab_by_str(vocab)
    vocab_dict = vocab_.get_opts()

    hyps_vocab_dict = _get_hyps_vocab_dict(vocab_dict=vocab_dict, hyps_probs_cfg=hyps_probs_cfg)
    model_out_dim = Dim(vocab_.get_num_classes(), name="vocab")
    if hyps_probs_cfg.include_blank:
        hyp_vocab_dim = Dim(vocab_.get_num_classes() + 1, name="vocab+blank")
    else:
        hyp_vocab_dim = model_out_dim

    hyps_spatial_dim = Dim(None, name="hyp_seq", kind=Dim.Types.Spatial)
    out_probs_top_k_dim = Dim(hyps_probs_cfg.top_k, name="k")

    model_outputs = {
        "output": {
            "dims": [batch_dim, hyps_spatial_dim, out_probs_top_k_dim],
            "sparse_dim": hyp_vocab_dim,
            "dtype": "int32",
            # We could always use hyps_vocab_dict, but this here keeps existing hashes...
            "vocab": hyps_vocab_dict if hyps_probs_cfg.include_blank else vocab_dict,
        },
        "output_k_lob_probs": {
            "dims": [batch_dim, hyps_spatial_dim, out_probs_top_k_dim],
            "dtype": "float32",
        },
        "log_probs": {"dims": [batch_dim], "dtype": "float32"},  # for the whole seq
        "enc_seq_lens": {"dims": [batch_dim], "dtype": "int32"},
    }

    if subset == "lm-train":
        assert tts_opts is not None
        # Split it up into split parts.
        # Keep consistent.
        num_train_parts = LibrispeechTtsOggZip.get_num_training_parts(files_per_part=10)
        trainsets = [f"lm-train_split_{i}_of_{num_train_parts}" for i in range(num_train_parts)]
        hyps_ = [
            get_hyps(
                prefix=prefix,
                subset=subset_,
                model_vocab=model_vocab,
                model=model,
                num_hyps=num_hyps,
                cfg=cfg,
                tts_opts=tts_opts[train_part_idx % len(tts_opts)] if isinstance(tts_opts, list) else tts_opts,
                train=train,
                hyps_probs_cfg=hyps_probs_cfg,
                get_hyps_extra_config=get_hyps_extra_config,
            )
            for train_part_idx, subset_ in enumerate(trainsets)
        ]
        assert len(hyps_) == num_train_parts and all(len(hyps__) == num_hyps for hyps__ in hyps_)

        from i6_experiments.users.zeyer.datasets.utils.concat_hdfs import concat_hdfs

        extern_data = model_outputs.copy()
        extern_data["data"] = extern_data.pop("output")

        hyps_txts = []
        for hyp_idx in range(num_hyps):
            hyps_txt = concat_hdfs([hyps__[hyp_idx] for hyps__ in hyps_], extern_data=extern_data)
            hyps_txt.creator.add_alias(f"{prefix}/hyps_from_model/{name}/{subset}-hyps{hyp_idx}-txt")
            tk.register_output(f"{prefix}/hyps_from_model/{name}/{subset}-hyps{hyp_idx}.txt.gz", hyps_txt)
            hyps_txts.append(hyps_txt)
        return hyps_txts

    task = get_librispeech_task_raw_v2(vocab=vocab)

    model_with_tts = None
    if tts_opts:
        assert isinstance(cfg, GetCtcHypsCfgV3) or isinstance(cfg, GetCtcHypsCfgV4)
        from .tts_model import get_asr_with_tts_model_def

        model_def = get_asr_with_tts_model_def(asr_model_def=model.definition, tts_opts=tts_opts)
        model_with_tts = ModelWithCheckpoint(definition=model_def, checkpoint=model.checkpoint)

    # Like sis_get_ctc_hyps_split.
    assert isinstance(cfg, GetCtcHypsCfgV3) or isinstance(cfg, GetCtcHypsCfgV4)
    hyps_hdfs = []
    for i in range(num_hyps):
        if subset == "train":
            dataset = task.train_dataset.copy_train_as_static()
        elif subset == "train-devtrain":
            dataset = task.train_dataset.copy_eval_as_static("devtrain")
        elif subset == "train-dev":
            dataset = task.train_dataset.copy_eval_as_static("dev")
        elif (subset.startswith("dev-") or subset.startswith("test-")) and subset in task.eval_datasets:
            dataset = task.eval_datasets[subset]
        elif subset.startswith("lm-"):
            assert isinstance(tts_opts, TtsOpts)
            lm_seq_list = get_lm_tts_seq_list(subset=subset[len("lm-") :], seq_tag_format="lm")
            real_lm_txt = get_real_lm_txt(subset=subset[len("lm-") :])

            dataset = DatasetConfigStatic(
                main_name=subset,
                main_dataset=get_tts_model_dataset_dict(
                    corpus_text=real_lm_txt,
                    seq_list_file=lm_seq_list,
                    train=train,
                    fixed_random_seed=6577 + i * 1023,
                    tts_opts=tts_opts,
                ),
                default_input="data",
                # target not used, except for get_model to know the dim.
                # In case without blank, having None triggering the fallback (using model_outputs) works fine,
                # and we keep that logic to not break hashes.
                default_target="classes" if hyps_probs_cfg.include_blank else None,
                extern_data={
                    # input dataset for the TTS model
                    "data": get_tts_model_dataset_extern_data_data(tts_opts=tts_opts),
                    # See comment above for default_target.
                    **(
                        {
                            "classes": {
                                "dims": [batch_dim, Dim(None, name="classes_spatial")],
                                "dtype": "int32",
                                "sparse_dim": model_out_dim,
                                "vocab": vocab_dict,
                                "available_for_inference": False,
                            }
                        }
                        if hyps_probs_cfg.include_blank
                        else {}
                    ),
                },
                use_deep_copy=True,
            )

        else:
            raise ValueError(f"invalid subset {subset!r}")

        # This code here is inlined sis_get_ctc_hyps_single_split (sis_get_hyps_single_split),
        # modified/extended to extract dense output of the prob distrib.

        from .ctc import model_recog_single_v3_dense_out, model_recog_single_v4_dense_out

        _forward_def = model_recog_single_v3_dense_out
        if isinstance(cfg, GetCtcHypsCfgV4):
            _forward_def = model_recog_single_v4_dense_out
        assert not isinstance(cfg, GetCtcHypsCfgV5)

        hyp_hdf = forward_to_hdf(
            dataset=dataset,
            model=model_with_tts or model,
            forward_def=functools.partial(_forward_def, cfg=cfg, hyps_probs_cfg=hyps_probs_cfg),
            config={
                # We use dropout to get multiple different hypotheses.
                # We introduce the beam dim, and want to get different hypotheses for each beam.
                "rf_dropout_broadcast": False,
                "model_outputs": model_outputs,
                "load_epoch": 1,
                "random_seed": 1023 + i * 17,
                **(get_hyps_extra_config or {}),
            },
            # mem usage: 11gb. mayb we can increase further?
            forward_post_config={
                "batch_size": 100000 * (model_with_tts or model).definition.batch_size_factor,
                "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            },
        )
        hyp_hdf.creator.rqmt["time"] = 24  # might need more time

        tk.register_output(f"{prefix}/hyps_from_model/{name}/{subset}/hyps{i}.hdf", hyp_hdf)
        hyp_hdf.creator.add_alias(f"{prefix}/hyps_from_model/{name}/{subset}/hyps{i}-hdf")
        hyps_hdfs.append(hyp_hdf)

    return hyps_hdfs


# Specialized version of i6_experiments.users.zeyer.train_v4._returnn_train_step,
# to be used as "train_step" in the RETURNN config.
def returnn_train_step_aed_dlm_dense(*, model, extern_data: TensorDict, **_kwargs_unused):
    from returnn.tensor import batch_dim
    from .error_correction_model import aed_training

    hyps = extern_data["hyps"]
    batch_dim_, hyps_spatial_dim, out_probs_top_k_dim = hyps.dims
    assert batch_dim_ == batch_dim and out_probs_top_k_dim.is_static()
    hyps_k_log_probs = extern_data["hyps_k_log_probs"]
    assert hyps_k_log_probs.dims == hyps.dims
    real = extern_data["real"]
    batch_dim_, real_spatial_dim = real.dims
    assert batch_dim_ == batch_dim

    aed_training(
        model=model,
        data=hyps,
        data_spatial_dim=hyps_spatial_dim,
        data_k_log_probs=hyps_k_log_probs,
        data_k_dim=out_probs_top_k_dim,
        targets=real,
        targets_spatial_dim=real_spatial_dim,
    )


# Adapted from i6_experiments.users.zeyer.recog._returnn_v2_forward_step.
def returnn_forward_step_aed_dlm_dense(*, model, extern_data: TensorDict, **_kwargs_unused):
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim, batch_dim
    from .error_correction_model import model_recog

    if rf.is_executing_eagerly():
        batch_size = int(batch_dim.get_dim_value())
        for batch_idx in range(batch_size):
            seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
            print(f"batch {batch_idx + 1}/{batch_size} seq_tag: {seq_tag!r}")

    hyps = extern_data["hyps"]
    batch_dim_, hyps_spatial_dim, out_probs_top_k_dim = hyps.dims
    assert batch_dim_ == batch_dim and out_probs_top_k_dim.is_static()
    hyps_k_log_probs = extern_data["hyps_k_log_probs"]
    assert hyps_k_log_probs.dims == hyps.dims

    recog_out = model_recog(
        model=model,
        data=hyps,
        data_spatial_dim=hyps_spatial_dim,
        data_k_log_probs=hyps_k_log_probs,
        data_k_dim=out_probs_top_k_dim,
    )
    assert len(recog_out) == 4, f"mismatch, got {len(recog_out)} outputs"
    hyps_, scores, out_spatial_dim, beam_dim = recog_out
    assert isinstance(hyps_, Tensor) and isinstance(scores, Tensor)
    assert isinstance(out_spatial_dim, Dim) and isinstance(beam_dim, Dim)
    rf.get_run_ctx().mark_as_output(hyps_, "hyps", dims=[batch_dim, beam_dim, out_spatial_dim])
    rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim, beam_dim])


# Adapted from i6_experiments.users.zeyer.decoding.rescoring._returnn_score_step
def returnn_forward_step_aed_dlm_dense_rescore(*, model, extern_data: TensorDict, **_kwargs_unused):
    # Similar to i6_experiments.users.zeyer.recog._returnn_v2_forward_step,
    # but using score_def instead of recog_def.
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim, batch_dim
    from returnn.config import get_global_config

    if rf.is_executing_eagerly():
        batch_size = int(batch_dim.get_dim_value())
        for batch_idx in range(batch_size):
            seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
            print(f"batch {batch_idx + 1}/{batch_size} seq_tag: {seq_tag!r}")

    hyps = extern_data["hyps"]
    batch_dim_, hyps_spatial_dim, out_probs_top_k_dim = hyps.dims
    assert batch_dim_ == batch_dim and out_probs_top_k_dim.is_static()
    hyps_k_log_probs = extern_data["hyps_k_log_probs"]
    assert hyps_k_log_probs.dims == hyps.dims

    config = get_global_config()
    targets_beam_dim = config.typed_value("_beam_dim")
    targets_flat = extern_data["data_flat"]
    targets_flat_time_dim = config.typed_value("_data_flat_spatial_dim")
    targets_seq_lens = extern_data["data_seq_lens"]  # [B, beam]
    # TODO stupid that targets_seq_lens first is copied CPU->GPU and now back to CPU...
    targets_spatial_dim = Dim(rf.copy_to_device(targets_seq_lens, "cpu"), name="targets_spatial")
    targets = rf.pad_packed(targets_flat, in_dim=targets_flat_time_dim, dims=[targets_beam_dim, targets_spatial_dim])

    rescore_def = config.typed_value("_rescore_def")
    scores = rescore_def(
        model=model,
        data=hyps,
        data_spatial_dim=hyps_spatial_dim,
        data_k_log_probs=hyps_k_log_probs,
        data_k_dim=out_probs_top_k_dim,
        targets=targets,
        targets_spatial_dim=targets_spatial_dim,
        targets_beam_dim=targets_beam_dim,
    )
    assert isinstance(scores, Tensor)
    rf.get_run_ctx().mark_as_output(targets, "hyps", dims=[batch_dim, targets_beam_dim, targets_spatial_dim])
    rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim, targets_beam_dim])


# Specialized version of i6_experiments.users.zeyer.train_v4._returnn_train_step,
# to be used as "train_step" in the RETURNN config.
def returnn_train_step_ctc_dlm_dense(*, model, extern_data: TensorDict, **_kwargs_unused):
    from returnn.tensor import batch_dim
    from ..error_correction_model_ctc import ctc_dlm_train_def

    hyps = extern_data["hyps"]
    batch_dim_, hyps_spatial_dim, out_probs_top_k_dim = hyps.dims
    assert batch_dim_ == batch_dim and out_probs_top_k_dim.is_static()
    hyps_k_log_probs = extern_data["hyps_k_log_probs"]
    assert hyps_k_log_probs.dims == hyps.dims
    real = extern_data["real"]
    batch_dim_, real_spatial_dim = real.dims
    assert batch_dim_ == batch_dim

    ctc_dlm_train_def(
        model=model,
        data=hyps,
        data_spatial_dim=hyps_spatial_dim,
        data_k_log_probs=hyps_k_log_probs,
        data_k_dim=out_probs_top_k_dim,
        targets=real,
        targets_spatial_dim=real_spatial_dim,
    )


# Adapted from i6_experiments.users.zeyer.recog._returnn_v2_forward_step.
def returnn_forward_step_ctc_dlm_dense(*, model, extern_data: TensorDict, **_kwargs_unused):
    import returnn.frontend as rf
    from returnn.tensor import Tensor, Dim, batch_dim
    from returnn.config import get_global_config

    if rf.is_executing_eagerly():
        batch_size = int(batch_dim.get_dim_value())
        for batch_idx in range(batch_size):
            seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
            print(f"batch {batch_idx + 1}/{batch_size} seq_tag: {seq_tag!r}")

    hyps = extern_data["hyps"]
    batch_dim_, hyps_spatial_dim, out_probs_top_k_dim = hyps.dims
    assert batch_dim_ == batch_dim and out_probs_top_k_dim.is_static()
    hyps_k_log_probs = extern_data["hyps_k_log_probs"]
    assert hyps_k_log_probs.dims == hyps.dims

    config = get_global_config()
    recog_def = config.typed_value("_recog_def")  # e.g. ctc_dlm_recog_def
    recog_out = recog_def(
        model=model,
        data=hyps,
        data_spatial_dim=hyps_spatial_dim,
        data_k_log_probs=hyps_k_log_probs,
        data_k_dim=out_probs_top_k_dim,
    )
    assert len(recog_out) == 4, f"mismatch, got {len(recog_out)} outputs"
    hyps_, scores, out_spatial_dim, beam_dim = recog_out
    assert isinstance(hyps_, Tensor) and isinstance(scores, Tensor)
    assert isinstance(out_spatial_dim, Dim) and isinstance(beam_dim, Dim)
    rf.get_run_ctx().mark_as_output(hyps_, "hyps", dims=[batch_dim, beam_dim, out_spatial_dim])
    rf.get_run_ctx().mark_as_output(scores, "scores", dims=[batch_dim, beam_dim])
