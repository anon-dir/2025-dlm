"""
Recipe for generating training data for the correction model.

Multiple ways:

- Via TTS data, then CTC model.
- Error model.

What data do we get:

- Training data:
    - For the ASR corpus train-960, multiple hypotheses.
    - For the LM corpus train, multiple hypotheses.
- Dev/devtrain during training:
    - Use ASR training task dev/devtrain.
- Test sets: dev-clean, dev-other, test-clean, test-other, LM-dev, LM-test

What format to use?
- HDF (HDFDataset): Probably fast. Labels, so dependent on vocab. Contains seq tags.
- Pure text (LmDataset): Also fast. Pure text, can apply any vocab. No orig seq tags. Easy to inspect.
  Potentially bug-prone if not careful with seq order.
In any case, should be consistent with format.
-> Pure text for now.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Literal, Dict, List
from functools import cache, partial

from sisyphus import tk, gs
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.zeyer.datasets.task import Task
from returnn_common.datasets_old_2022_10.interface import DatasetConfigStatic
from .ctc import (
    GetHypsCfgV1 as GetCtcHypsCfgV1,
    GetHypsCfgV2 as GetCtcHypsCfgV2,
    GetHypsCfgV3 as GetCtcHypsCfgV3,
    GetHypsCfgV4 as GetCtcHypsCfgV4,
    GetHypsCfgV5 as GetCtcHypsCfgV5,
    GetHypsCfgV6 as GetCtcHypsCfgV6,
    sis_get_hyps_split as sis_get_ctc_hyps_split,
    sis_get_hyps_single_split as sis_get_ctc_hyps_single_split,
    sis_get_model as sis_get_ctc_model,
)
from .tts_model import EcmTwiceOpts, TtsOpts, TtsOptsOrEcmTwiceOpts

TGetHypsCfg = Union[
    GetCtcHypsCfgV1,
    GetCtcHypsCfgV2,
    GetCtcHypsCfgV3,
    GetCtcHypsCfgV4,
    GetCtcHypsCfgV5,
    GetCtcHypsCfgV6,
]


def py():
    get_error_correction_model_task_via_tts_txt()


def get_error_correction_model_task_via_tts_txt(
    *,
    prefix: Optional[str] = None,
    vocab: str = "spm10k",
    reverse_in_out: bool = False,
    num_hyps: int = 5,
    train_epoch_split: Optional[int] = 20,
    train_repeat_asr_data: int = 1,
    train_repeat_asr_data_via_num_hyps: bool = False,
    train_input_vocab_opts: Optional[Dict[str, Any]] = None,
    hyps_model_vocab: Optional[str] = None,
    hyps_model: Optional[ModelWithCheckpoint] = None,
    hyps_cfg: Optional[TGetHypsCfg] = None,
    hyps_tts_opts: Optional[Union[TtsOptsOrEcmTwiceOpts, List[TtsOptsOrEcmTwiceOpts]]] = None,
    hyps_tts_opts_all_systems_generate_all_data: Optional[bool] = None,
    get_hyps_extra_config: Optional[Dict[str, Any]] = None,
    resplit_subwords: bool = True,
    dataset_use_deep_copy: bool = False,
    version: int = 1,
    additional_eval_sets: Optional[List[str]] = None,
    use_dependency_boundary: bool = True,
    dependency_boundary_hash: Optional[str] = None,
    register_output: bool = True,
) -> Task:
    """
    Get task. Using hyps from TTS. As txt, via LmDataset.

    :param prefix:
    :param vocab: e.g. "spm10k".
        should be handled by :func:`get_vocab_by_str` and :func:`get_librispeech_task_text_only`
    :param reverse_in_out:
        False (default) -> data: corrupted, classes: reference.
        True -> data: reference, classes: corrupted.
    :param num_hyps: for train
    :param train_epoch_split: for train
    :param train_repeat_asr_data: for train, how often to repeat the ASR (LS) data
    :param train_repeat_asr_data_via_num_hyps:
    :param train_input_vocab_opts: for example, apply additional sampling for the inputs (only with resplit_subword)
    :param hyps_model_vocab:
    :param hyps_model:
    :param hyps_cfg:
    :param hyps_tts_opts: If given, a TTS model is used for on-the-fly generation of the audio for the CTC model
        to generate the hyps.
        If a list is given, allows mixing multiple TTS models.
        It will just call the hyps func the multiple times, and then mix up the hyps,
        interleaving taking one from each.
    :param get_hyps_extra_config:
    :param resplit_subwords: if True, we merge subwords from hyps, and then segment to subwords again on-the-fly
    :param dataset_use_deep_copy: for :class:`DatasetConfigStatic`.
        should only be relevant if you did not use :func:`use_instanciate_delayed_copy_instead_of_inplace`.
    :param version: like behavior_version but here for the task/dataset.
        Can enable some new way to represent the data, which might be better for loading or so,
        which you would want to have, but which would break the hash, so that's why you need to specify the version.
    :param additional_eval_sets:
    :param use_dependency_boundary:
    :param dependency_boundary_hash:
    :param register_output: if True, register the output files in the task
    :return: task
    """
    import dataclasses
    from returnn.tensor import Dim, batch_dim
    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_text_only,
        get_vocab_by_str,
    )
    from i6_experiments.users.zeyer.datasets.utils.sclite_generic_score import (
        generic_sclite_score_recog_out,
    )

    if use_dependency_boundary:
        import functools
        from i6_experiments.common.helpers.dependency_boundary import (
            dependency_boundary,
        )

        return dependency_boundary(
            functools.partial(
                get_error_correction_model_task_via_tts_txt,
                prefix=prefix,
                vocab=vocab,
                reverse_in_out=reverse_in_out,
                num_hyps=num_hyps,
                train_epoch_split=train_epoch_split,
                train_repeat_asr_data=train_repeat_asr_data,
                train_repeat_asr_data_via_num_hyps=train_repeat_asr_data_via_num_hyps,
                train_input_vocab_opts=train_input_vocab_opts,
                hyps_model_vocab=hyps_model_vocab,
                hyps_model=hyps_model,
                hyps_cfg=hyps_cfg,
                hyps_tts_opts=hyps_tts_opts,
                hyps_tts_opts_all_systems_generate_all_data=hyps_tts_opts_all_systems_generate_all_data,
                get_hyps_extra_config=get_hyps_extra_config,
                resplit_subwords=resplit_subwords,
                dataset_use_deep_copy=dataset_use_deep_copy,
                version=version,
                additional_eval_sets=additional_eval_sets,
                register_output=register_output,
                use_dependency_boundary=False,
            ),
            hash=dependency_boundary_hash,
        )

    if hyps_tts_opts is not None and isinstance(hyps_tts_opts, TtsOpts) and "phone_info" in hyps_tts_opts.gen_opts:
        # so i dont forget
        assert get_hyps_extra_config is not None
        assert "behavior_version" in get_hyps_extra_config
        assert get_hyps_extra_config["behavior_version"] >= 24
        # this is not implemented for lists, because it would be a bit annoying to handle
        # (coqui doesnt need behavior_version 24, so i just force it in get_lm_hyps)
    if hyps_tts_opts is not None and not isinstance(hyps_tts_opts, list):
        hyps_tts_opts = [hyps_tts_opts]  # always convert to list

    if train_input_vocab_opts:
        assert resplit_subwords  # doesn't make sense otherwise
        assert not reverse_in_out  # we apply it on the hyps, but with reversed in/out, the hyps are not the input
    if additional_eval_sets is None:
        additional_eval_sets = []
    assert isinstance(additional_eval_sets, list)

    in_key, out_key = "hyps", "real"
    if reverse_in_out:
        in_key, out_key = out_key, in_key

    if not hyps_model:
        hyps_model = get_default_hyps_model_by_vocab_and_cfg(vocab=vocab, cfg=hyps_cfg)

    vocab_ = get_vocab_by_str(vocab)
    vocab_dict = vocab_.get_opts()
    train_input_vocab_dict = vocab_.copy(**train_input_vocab_opts).get_opts() if train_input_vocab_opts else None
    hyps_vocab_dict = None
    hyps_merge_subwords = True
    if not resplit_subwords:
        from i6_experiments.users.zeyer.datasets.utils.vocab import (
            ExtractVocabLabelsJob,
            ExtractVocabSpecialLabelsJob,
        )

        hyps_vocab_dict = {
            "class": "Vocabulary",
            "vocab_file": ExtractVocabLabelsJob(vocab_dict).out_vocab,
            "special_symbols_via_file": ExtractVocabSpecialLabelsJob(vocab_dict).out_vocab_special_labels_dict,
        }
        if not (vocab.startswith("bpe") or vocab.startswith("spm")):
            hyps_vocab_dict["single_whitespace_split"] = True
        hyps_merge_subwords = False
    vocab_dim = Dim(vocab_.get_num_classes(), name="vocab")
    extern_data = {
        "hyps": {
            "dim_tags": [batch_dim, Dim(None, name="hyps_spatial")],
            "sparse_dim": vocab_dim,
            "vocab": vocab_dict,
        },
        "real": {
            "dim_tags": [batch_dim, Dim(None, name="real_spatial")],
            "sparse_dim": vocab_dim,
            "vocab": vocab_dict,
        },
    }
    ds_extra_args = {}
    if vocab_.get_num_classes() <= 2**8:
        ds_extra_args["dtype"] = "int32"  # torch.embedding wants int32/int64... it's currently easiest to do that here
    make_dataset_common_kwargs = dict(
        prefix=prefix,
        hyps_model_vocab=hyps_model_vocab or vocab,
        hyps_model=hyps_model,
        vocab_dict=vocab_dict,
        hyps_vocab_dict=hyps_vocab_dict,
        get_hyps_extra_config=get_hyps_extra_config,
        hyps_merge_subwords=hyps_merge_subwords,
        dataset_extra_args=ds_extra_args,
        version=version,
        register_output=register_output,
    )
    train_ds = DatasetConfigStatic(
        default_input=in_key,
        default_target=out_key,
        extern_data=extern_data,
        # TODO special vocab for train with sampling?
        train_dataset=_make_dataset(
            **make_dataset_common_kwargs,
            subset="train",
            num_hyps=num_hyps,
            train=True,
            train_epoch_split=train_epoch_split,
            repeat_asr_data=train_repeat_asr_data,
            repeat_asr_data_via_num_hyps=train_repeat_asr_data_via_num_hyps,
            train_hyps_vocab_dict=train_input_vocab_dict,
            hyps_cfg=hyps_cfg,
            hyps_tts_opts=hyps_tts_opts,
            hyps_tts_opts_all_systems_generate_all_data=hyps_tts_opts_all_systems_generate_all_data,
        ),
        eval_datasets={
            "dev": _make_dataset(
                **make_dataset_common_kwargs,
                # simple hack to disable all unnecessary train data if necessary (for quickly testing new hyp cfg, etc.)
                num_hyps=min(1, num_hyps),
                subset="train-dev",
                hyps_cfg=hyps_cfg,
            ),
            "devtrain": _make_dataset(
                **make_dataset_common_kwargs,
                num_hyps=min(1, num_hyps),
                subset="train-devtrain",
                hyps_cfg=hyps_cfg,
            ),
        },
        use_deep_copy=dataset_use_deep_copy,
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
                hyps_cfg=GetAedHypsCfgV1() if isinstance(hyps_cfg, GetAedHypsCfgV1) else GetCtcHypsCfgV4(),
                eval=True,
            ),
            use_deep_copy=dataset_use_deep_copy,
        )
        for key in ["dev-clean", "dev-other", "test-clean", "test-other", *additional]
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
                    hyps_tts_opts_all_systems_generate_all_data=hyps_tts_opts_all_systems_generate_all_data,
                    eval=True,
                ),
                use_deep_copy=dataset_use_deep_copy,
            )
            for key in ["dev-clean", "dev-other", *additional_trainlike]  # "lm-dev", "lm-devtrain"
        }
    )

    task = get_librispeech_task_text_only(vocab=vocab)
    task = dataclasses.replace(task)  # makes new copy
    task.train_epoch_split = train_epoch_split
    task.train_dataset = train_ds
    task.dev_dataset = eval_dss["dev-other"]
    task.eval_datasets = eval_dss
    task.score_recog_output_func = partial(generic_sclite_score_recog_out, post_proc_funcs=task.recog_post_proc_funcs)
    return task


def get_default_hyps_model_by_vocab_and_cfg(*, vocab: str, cfg: Optional[TGetHypsCfg] = None) -> ModelWithCheckpoint:
    if cfg is None or isinstance(cfg, (GetCtcHypsCfgV1, GetCtcHypsCfgV2, GetCtcHypsCfgV3, GetCtcHypsCfgV4)):
        return sis_get_ctc_model(vocab=vocab)
    elif isinstance(cfg, GetAedHypsCfgV1):
        return sis_get_aed_model(vocab=vocab)
    else:
        raise TypeError(f"invalid cfg {cfg!r} (type {type(cfg)})")


def _make_dataset(
    *,
    prefix: Optional[str] = None,
    subset: Literal[
        "train",  # LS ASR train + LS LM (TTS) train ("lm-train")
        "train-asr",  # LS ASR train
        "train-asr-tts",  # LS ASR train (but audio replaced by TTS audio)
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
    train: bool = False,
    eval: bool = False,
    train_epoch_split: Optional[int] = None,
    repeat_asr_data: int = 1,
    repeat_asr_data_via_num_hyps: bool = False,
    hyps_model_vocab: str,
    hyps_model: ModelWithCheckpoint,
    vocab_dict: Dict[str, Any],
    train_hyps_vocab_dict: Optional[Dict[str, Any]] = None,
    hyps_vocab_dict: Optional[Dict[str, Any]] = None,
    hyps_cfg: Optional[TGetHypsCfg] = None,
    hyps_tts_opts: Optional[Union[TtsOptsOrEcmTwiceOpts, List[TtsOptsOrEcmTwiceOpts]]] = None,
    hyps_tts_opts_all_systems_generate_all_data: Optional[bool] = None,
    get_hyps_extra_config: Optional[Dict[str, Any]] = None,
    hyps_merge_subwords: bool = True,
    dataset_extra_args: Optional[Dict[str, Any]] = None,
    version: int = 1,
    register_output: bool = True,
) -> Dict[str, Any]:
    import functools
    from .error_correction_model import _rotate_datasets_for_epoch

    hyps: List[List[tk.Path]] = [[] for _ in range(num_hyps)]  # inner is dataset, outer is hyp idx
    real: List[tk.Path] = []

    asr_subset = {"train-asr": "train"}.get(subset, subset if not subset.startswith("lm-") else None)
    if asr_subset:
        asr_hyps = get_asr_hyps_txt(
            prefix=prefix,
            subset=asr_subset,
            num_hyps=num_hyps * (repeat_asr_data if repeat_asr_data_via_num_hyps else 1),
            model_vocab=hyps_model_vocab,
            model=hyps_model,
            cfg=hyps_cfg,
            get_hyps_extra_config=get_hyps_extra_config,
            merge_subwords=hyps_merge_subwords,
            register_output=register_output,
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

    lm_subset = {
        "train": "lm-train",
        "lm-dev": "lm-dev",
        "lm-test": "lm-test",
        "lm-devtrain": "lm-devtrain",
        "train-asr-tts": "train",
    }.get(subset, None)
    if lm_subset:
        if hyps_tts_opts is not None and isinstance(hyps_tts_opts, list) and len(hyps_tts_opts) == 0:
            # special path to disable generation!
            pass  # just do nothing, we will not generate any hyps
        elif isinstance(hyps_cfg, (GetCtcHypsCfgV3, GetCtcHypsCfgV4)):
            # on-the-fly tts
            assert hyps_tts_opts is not None
            lm_hyps = get_lm_hyps_txt_v3(
                prefix=prefix,
                subset=lm_subset,
                num_hyps=num_hyps,
                model_vocab=hyps_model_vocab,
                model=hyps_model,
                cfg=hyps_cfg,
                tts_opts=hyps_tts_opts,
                tts_opts_all_systems_generate_all_data=hyps_tts_opts_all_systems_generate_all_data,
                get_hyps_extra_config=get_hyps_extra_config,
                merge_subwords=hyps_merge_subwords,
                train=train,
                register_output=register_output,
            )
            assert len(lm_hyps) == num_hyps
            for i, hyp in enumerate(lm_hyps):
                hyps[i].append(hyp)
            if lm_subset.startswith("lm-"):
                lm_real = get_real_lm_txt(subset=lm_subset[len("lm-") :])
            else:
                lm_real = get_real_asr_txt(subset=lm_subset)
            if hyps_tts_opts_all_systems_generate_all_data and isinstance(hyps_tts_opts, list):
                # If we have multiple TTS systems, we need to repeat the real text for each system.
                # This is because the real text is the same for all systems, but we want to have it
                # in the same order as the hyps.
                real.extend([lm_real] * len(hyps_tts_opts))
            else:
                real.append(lm_real)
        else:
            # This is the path for the old orig TTS .ogg files, we don't use this anymore
            assert hyps_tts_opts is None
            lm_hyps = get_asr_hyps_txt(
                prefix=prefix,
                subset=lm_subset,
                num_hyps=num_hyps,
                model_vocab=hyps_model_vocab,
                model=hyps_model,
                cfg=hyps_cfg,
                get_hyps_extra_config=get_hyps_extra_config,
                merge_subwords=hyps_merge_subwords,
                register_output=register_output,
            )
            assert len(lm_hyps) == num_hyps
            for i, hyp in enumerate(lm_hyps):
                hyps[i].append(hyp)
            lm_real = get_real_asr_txt(subset=lm_subset)
            real.append(lm_real)

    seq_tags: Optional[List[tk.Path]] = None
    if subset.startswith("dev-") or subset.startswith("test-"):
        assert asr_subset and not lm_subset
        seq_tags = [_get_asr_seq_list(subset=asr_subset)]

    for i in range(num_hyps):
        assert hyps[i]
    assert real

    def _make_base_ds(_fs, *, is_hyps: bool):
        assert isinstance(_fs, list)
        return {
            "class": "LmDataset",
            "corpus_file": _fs,
            **({"seq_list_file": seq_tags} if seq_tags is not None else {}),
            "use_cache_manager": True,
            "skip_empty_lines": False,
            "orth_vocab": train_hyps_vocab_dict
            if train and train_hyps_vocab_dict and is_hyps
            else hyps_vocab_dict
            if hyps_vocab_dict and is_hyps
            else vocab_dict,
            "seq_end_symbol": None,  # handled via orth_vocab
            "unknown_symbol": None,  # handled via orth_vocab
            **(dataset_extra_args or {}),
        }

    real_ds = _make_base_ds(real, is_hyps=False)
    if train:
        real_ds.update({"partition_epoch": train_epoch_split, "seq_ordering": "laplace:.1000"})
    else:
        real_ds.update({"seq_ordering": "sorted_reverse"})
    hyps_dss = [_make_base_ds(hyps_i, is_hyps=True) for hyps_i in hyps]
    if len(hyps_dss) == 1:
        hyps_ds = hyps_dss[0]
    else:
        assert not eval
        hyps_ds = {
            "class": "VariableDataset",
            "get_dataset": functools.partial(_rotate_datasets_for_epoch, datasets=hyps_dss),
        }
        if version >= 2:
            hyps_ds["partition_epoch"] = train_epoch_split

    return {
        "class": "MetaDataset",
        "datasets": {"hyps": hyps_ds, "real": real_ds},
        "data_map": {"hyps": ("hyps", "data"), "real": ("real", "data")},
        "seq_order_control_dataset": "hyps" if eval else "real",
    }


def _get_ogg_zips(
    *,
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
) -> List[tk.Path]:
    from i6_experiments.common.datasets import librispeech

    if subset.startswith("lm-"):
        from .tts_data import LibrispeechTtsOggZip

        return LibrispeechTtsOggZip.get_oggs(subset[len("lm-") :])

    # Keep consistent to our LibrispeechOggZip.
    ls_ogg_zip_dict = librispeech.get_ogg_zip_dict()
    if subset in {"train", "train-devtrain"}:
        parts = ["train-clean-100", "train-clean-360", "train-other-500"]
    elif subset == "train-dev":
        parts = ["dev-clean", "dev-other"]
    else:
        assert isinstance(subset, str)
        parts = [subset]
    ogg_zips = []
    for part in parts:
        ogg_zips += [ls_ogg_zip_dict[part]]
    return ogg_zips


@cache
def _get_asr_seq_list(
    *,
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
        ],
        str,
    ],
) -> tk.Path:
    """
    Extracts seq tag list (and its original 'default' order) from the Ogg files.
    """
    from i6_experiments.users.zeyer.datasets.utils.extract_seq_list import ExtractSeqListJob

    prefix = get_setup_prefix_for_module(__name__)

    if subset == "lm-train":
        from .tts_data import LibrispeechTtsOggZip
        from i6_core.text.processing import ConcatenateJob

        # Concatenate subsets. Otherwise, we easily run into memory issues. And this is maybe also cleaner.
        # Keep consistent.
        num_train_parts = LibrispeechTtsOggZip.get_num_training_parts(files_per_part=10)
        trainsets = [f"lm-train_split_{i}_of_{num_train_parts}" for i in range(num_train_parts)]
        seq_list = ConcatenateJob([_get_asr_seq_list(subset=subset_) for subset_ in trainsets]).out

    else:
        extra = {}
        if subset.startswith("train-"):
            extra.update({"fixed_random_subset": 3000, "fixed_random_seed": 1})  # keep consistent to LibrispeechOggZip
        if subset.startswith("lm-"):
            extra.update({"content_name": "out.ogg"})

        ogg_zips = _get_ogg_zips(subset=subset)
        seq_list = ExtractSeqListJob(
            returnn_dataset={
                "class": "OggZipDataset",
                "path": ogg_zips,
                "audio": None,
                "targets": None,
                **extra,
            },
            returnn_dataset_ext_non_hashed={"use_cache_manager": True},
        ).out_seq_list
        if len(ogg_zips) > 100:
            seq_list.creator.rqmt.update(dict(time=4, mem=20, cpu=2))
    tk.register_output(prefix + f"/seq-list/{subset}.segments", seq_list)
    return seq_list


def get_real_asr_txt(
    *,
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
) -> tk.Path:
    """
    Get the real ASR txt. This is the reference, the correct text.
    It returns a gzipped text file, which can be used with LmDataset.

    The order of sequences is determined via :func:`_get_asr_seq_list`,
    i.e. it's the order as in the original Ogg files.
    """
    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextLinesJob

    prefix = get_setup_prefix_for_module(__name__)

    if subset == "lm-train":
        from .tts_data import LibrispeechTtsOggZip
        from i6_core.text.processing import ConcatenateJob

        # Concatenate subsets. Otherwise, we easily run into memory issues. And this is maybe also cleaner.
        # Keep consistent.
        num_train_parts = LibrispeechTtsOggZip.get_num_training_parts(files_per_part=10)
        trainsets = [f"lm-train_split_{i}_of_{num_train_parts}" for i in range(num_train_parts)]
        asr_txt = ConcatenateJob([get_real_asr_txt(subset=subset_) for subset_ in trainsets]).out

    else:
        extra = {}
        if subset.startswith("train-"):  # train-dev, train-devtrain
            extra.update({"fixed_random_subset": 3000, "fixed_random_seed": 1})  # keep consistent to LibrispeechOggZip
        if subset.startswith("lm-"):
            extra.update({"content_name": "out.ogg"})

        vocab = {"class": "Utf8ByteTargets"}
        asr_txt = ReturnnDatasetToTextLinesJob(
            returnn_dataset={
                "class": "OggZipDataset",
                "path": _get_ogg_zips(subset=subset),
                "audio": None,
                "targets": vocab,
                **extra,
            },
            returnn_dataset_ext_non_hashed={"use_cache_manager": True},
            multi_proc_dataset_opts=dict(num_workers=2, buffer_size=10),
            seq_list=_get_asr_seq_list(subset=subset),
            data_key="classes",
            vocab=vocab,
        ).out_txt
        asr_txt.creator.rqmt.update(dict(time=10, mem=20, cpu=8))
        asr_txt.creator.add_alias(prefix + f"/real-txt/{subset}")
    tk.register_output(prefix + f"/real-txt/{subset}.txt.gz", asr_txt)
    return asr_txt


def get_asr_hyps_txt(
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
    get_hyps_extra_config: Optional[Dict[str, Any]] = None,
    merge_subwords: bool = True,
    register_output: bool = True,
) -> List[tk.Path]:
    """
    :return: collection of hyps txt corresponding to the hyps, i.e. len = num_hyps.
        The text can be used with LmDataset and is independent of the vocab.
        The order of sequences is determined via :func:`_get_asr_seq_list`,
        i.e. it's the order as in the original Ogg files.
        It's the same order as in :func:`get_real_asr_txt`.
    """
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2
    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextLinesJob

    prefix = prefix or get_setup_prefix_for_module(__name__)

    name = model_vocab
    if num_hyps != 5:
        name += f"-numHyps{num_hyps}"
    if cfg:
        name += f"-{cfg}"
    if not merge_subwords:
        name += "-keepSubwords"

    if get_hyps_extra_config is not None:
        name += f"-extraConfig({_name_for_dict(get_hyps_extra_config)})"

    ctc_model_name = "unknown"
    from .ctc import _model_cache_by_name

    # reverse lookup
    for _name, _model in _model_cache_by_name.items():
        if model == _model:
            ctc_model_name = _name
            break
    # if model is not the cache, ignore

    # Note: task hardcoded... (and also not needed, I just need the train dataset...)
    # Note: Model hardcoded...
    vocab = model_vocab

    if subset.startswith("lm-"):
        from .tts_data import get_tts_datasets

        train_dataset, eval_datasets = get_tts_datasets(
            vocab=vocab, extra_dataset_common_opts=dict(extra_args={"content_name": "out.ogg"}), with_devtrain=True
        )
        if subset in {"lm-dev", "lm-test", "lm-devtrain"}:
            dataset = eval_datasets[subset[len("lm-") :]]
        elif subset.startswith("lm-train_split_"):
            dataset = train_dataset.copy_trainsplit_as_static(subset[len("lm-") :])
        elif subset == "lm-train":
            # Keep consistent.
            num_train_parts = train_dataset.get_num_training_parts(files_per_part=10)
            trainsets = [f"lm-train_split_{i}_of_{num_train_parts}" for i in range(num_train_parts)]
            hyps_ = [
                get_asr_hyps_txt(
                    prefix=prefix,
                    subset=subset_,
                    model_vocab=model_vocab,
                    model=model,
                    num_hyps=num_hyps,
                    cfg=cfg,
                    get_hyps_extra_config=get_hyps_extra_config,
                    merge_subwords=merge_subwords,
                )
                for subset_ in trainsets
            ]
            assert len(hyps_) == num_train_parts and all(len(hyps__) == num_hyps for hyps__ in hyps_)

            from i6_core.text.processing import ConcatenateJob

            hyps_txts = []
            for hyp_idx in range(num_hyps):
                hyps_txt = ConcatenateJob([hyps__[hyp_idx] for hyps__ in hyps_]).out
                hyps_txt.creator.add_alias(
                    f"{prefix}/hyps_from_model/{ctc_model_name}/{name}/{subset}-hyps{hyp_idx}-txt"
                )
                if register_output:
                    tk.register_output(
                        f"{prefix}/hyps_from_model/{ctc_model_name}/{name}/{subset}-hyps{hyp_idx}.txt.gz", hyps_txt
                    )
                hyps_txts.append(hyps_txt)
            return hyps_txts
        else:
            raise ValueError(f"invalid LM subset {subset!r}")

    else:
        task = get_librispeech_task_raw_v2(vocab=vocab)
        if subset == "train":
            dataset = task.train_dataset.copy_train_as_static()
        elif subset == "train-devtrain":
            dataset = task.train_dataset.copy_eval_as_static("devtrain")
        elif subset == "train-dev":
            dataset = task.train_dataset.copy_eval_as_static("dev")
        elif (subset.startswith("dev-") or subset.startswith("test-")) and subset in task.eval_datasets:
            dataset = task.eval_datasets[subset]
        else:
            raise ValueError(f"invalid subset {subset!r}")

    if cfg is None or isinstance(cfg, (GetCtcHypsCfgV1, GetCtcHypsCfgV2, GetCtcHypsCfgV3, GetCtcHypsCfgV4)):
        hyps_hdfs, _ = sis_get_ctc_hyps_split(
            model, dataset=dataset, num_hyps=num_hyps, cfg=cfg, extra_config=get_hyps_extra_config
        )
    elif isinstance(cfg, GetAedHypsCfgV1):
        assert not get_hyps_extra_config
        hyps_hdfs, _ = sis_get_aed_hyps_split(model, dataset=dataset, num_hyps=num_hyps, cfg=cfg)
    else:
        raise ValueError(f"invalid cfg {cfg!r} (type {type(cfg)})")

    hyps_txts = []
    for i, hyp_hdf in enumerate(hyps_hdfs):
        if register_output:
            tk.register_output(f"{prefix}/hyps_from_model/{ctc_model_name}/{name}/{subset}/hyps{i}.hdf", hyp_hdf)
        hyp_hdf.creator.add_alias(f"{prefix}/hyps_from_model/{ctc_model_name}/{name}/{subset}/hyps{i}-hdf")

        if merge_subwords:
            assert vocab.startswith("spm")  # TODO see below raw_replacement_list, currently we assume SPM here...
        hyps_txt = ReturnnDatasetToTextLinesJob(
            returnn_dataset={"class": "HDFDataset", "files": [hyp_hdf]},
            returnn_dataset_ext_non_hashed={"use_cache_manager": True},
            multi_proc_dataset_opts=dict(num_workers=5, buffer_size=10),
            seq_list=_get_asr_seq_list(subset=subset),
            data_key="data",
            **(dict(raw_replacement_list=[(" ", ""), ("▁", " ")], raw_final_strip=True) if merge_subwords else {}),
        ).out_txt
        hyps_txt.creator.rqmt.update(dict(time=50, mem=40, cpu=16))
        hyps_txt.creator.add_alias(f"{prefix}/hyps_from_model/{ctc_model_name}/{name}/{subset}-hyps{i}-txt")
        if register_output:
            tk.register_output(f"{prefix}/hyps_from_model/{ctc_model_name}/{name}/{subset}-hyps{i}.txt.gz", hyps_txt)
        hyps_txts.append(hyps_txt)
    return hyps_txts


def get_lm_hyps_txt_v3(
    *,
    prefix: Optional[str] = None,
    subset: Union[
        Literal[
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
    cfg: TGetHypsCfg,
    tts_opts: Union[TtsOptsOrEcmTwiceOpts, List[TtsOptsOrEcmTwiceOpts]],
    tts_opts_all_systems_generate_all_data: Optional[bool] = False,
    get_hyps_extra_config: Optional[Dict[str, Any]] = None,
    merge_subwords: bool = True,
    train: bool = False,
    register_output: bool = True,
) -> List[tk.Path]:
    """
    Note: We called this "get_asr_hyps_txt_v3" before, as a successor of "get_asr_hyps_txt_v2".
    But then this was changed to only operate on LM text data, thus the name change.

    :return: collection of hyps txt corresponding to the hyps, i.e. len = num_hyps.
        The text can be used with LmDataset and is independent of the vocab.
        The order of sequences is determined via :func:`_get_asr_seq_list`,
        i.e. it's the order as in the original Ogg files.
        It's the same order as in :func:`get_real_asr_txt`.
    """
    from i6_experiments.users.zeyer.datasets.librispeech import get_vocab_by_str
    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextLinesJob
    from .tts_data import get_lm_tts_seq_list, LibrispeechTtsOggZip
    from returnn.tensor import Dim, batch_dim

    prefix = prefix or get_setup_prefix_for_module(__name__)

    name1 = f"ttsGen-{model_vocab}"
    if num_hyps != 5:
        name1 += f"-numHyps{num_hyps}"
    name1 += f"-{cfg}"

    name2 = ""
    if isinstance(tts_opts, list):
        for tts_opts_ in tts_opts:
            if isinstance(tts_opts_, TtsOpts):
                name2 += "-" + (_name_for_dict(tts_opts_.gen_opts) or "tts")
            elif isinstance(tts_opts_, EcmTwiceOpts):
                name2 += "-ecm"
    else:
        if isinstance(tts_opts, TtsOpts):
            name2 += "-" + (_name_for_dict(tts_opts.gen_opts) or "tts")
        elif isinstance(tts_opts, EcmTwiceOpts):
            name2 += "-ecm"
    if not merge_subwords:
        name2 += "-keepSubwords"

    if get_hyps_extra_config is not None:
        name2 += f"-extraConfig({_name_for_dict(get_hyps_extra_config)})"
    if train != ("train" in subset):
        name2 += f"-train({train})"

    name = name1 + name2
    if len(name) > 250 and len(name2) > 0:
        # filename limit
        name = f"{name1}/{name2[1:]}"  # remove leading dash from name2

    ctc_model_name = "unknown"
    from .ctc import _model_cache_by_name

    # reverse lookup
    for _name, _model in _model_cache_by_name.items():
        if model == _model:
            ctc_model_name = _name
            break
    # if model is not the cache, ignore

    # Note: task hardcoded... (and also not needed, I just need the train dataset...)
    # Note: Model hardcoded...
    hyp_vocab = model_vocab

    if subset == "lm-train":
        # Split it up into split parts.
        # Keep consistent.
        num_train_parts = LibrispeechTtsOggZip.get_num_training_parts(files_per_part=10)
        trainsets = [f"lm-train_split_{i}_of_{num_train_parts}" for i in range(num_train_parts)]
        extra_config_list = []
        for tts_opt in tts_opts if isinstance(tts_opts, list) else [tts_opts]:
            local_get_hyps_extra_config = get_hyps_extra_config
            if (
                isinstance(tts_opt, TtsOpts)
                and "phone_info" in tts_opt.gen_opts
                and (local_get_hyps_extra_config or {}).get("behavior_version") is None
            ):
                local_get_hyps_extra_config = local_get_hyps_extra_config or {}
                local_get_hyps_extra_config["behavior_version"] = 24
            extra_config_list.append(local_get_hyps_extra_config)
        assert len(extra_config_list) == (len(tts_opts) if isinstance(tts_opts, list) else 1)
        if tts_opts_all_systems_generate_all_data:
            tts_opts_list = tts_opts if isinstance(tts_opts, list) else [tts_opts]
            hyps_ = []
            for i, tts_opt in enumerate(tts_opts_list):
                hyps_.extend(
                    [
                        get_lm_hyps_txt_v3(
                            prefix=prefix,
                            subset=subset_,
                            model_vocab=model_vocab,
                            model=model,
                            num_hyps=num_hyps,
                            cfg=cfg,
                            tts_opts=tts_opt,
                            get_hyps_extra_config=extra_config_list[i],
                            merge_subwords=merge_subwords,
                            train=train,
                        )
                        for subset_ in trainsets
                    ]
                )
            assert len(hyps_) == num_train_parts * len(tts_opts_list) and all(
                len(hyps__) == num_hyps for hyps__ in hyps_
            )
        else:
            hyps_ = [
                get_lm_hyps_txt_v3(
                    prefix=prefix,
                    subset=subset_,
                    model_vocab=model_vocab,
                    model=model,
                    num_hyps=num_hyps,
                    cfg=cfg,
                    tts_opts=tts_opts[train_part_idx % len(tts_opts)] if isinstance(tts_opts, list) else tts_opts,
                    get_hyps_extra_config=extra_config_list[train_part_idx % len(extra_config_list)],
                    merge_subwords=merge_subwords,
                    train=train,
                )
                for train_part_idx, subset_ in enumerate(trainsets)
            ]
            assert len(hyps_) == num_train_parts and all(len(hyps__) == num_hyps for hyps__ in hyps_)

        from i6_core.text.processing import ConcatenateJob

        hyps_txts = []
        for hyp_idx in range(num_hyps):
            hyps_txt = ConcatenateJob([hyps__[hyp_idx] for hyps__ in hyps_]).out
            hyps_txt.creator.add_alias(f"{prefix}/hyps_from_model/{ctc_model_name}/{name}/{subset}-hyps{hyp_idx}-txt")
            if register_output:
                tk.register_output(
                    f"{prefix}/hyps_from_model/{ctc_model_name}/{name}/{subset}-hyps{hyp_idx}.txt.gz", hyps_txt
                )
            hyps_txts.append(hyps_txt)
        return hyps_txts

    if isinstance(tts_opts, EcmTwiceOpts):
        from .ecm_twice import get_hyps_single

        assert "tts_opts" in tts_opts.ecm_opts  # hopefully also "cfg"
        hyps_texts, real = get_hyps_single(
            subset,
            name=f"{tts_opts.name}/{name}",
            asr_model=model,
            dlm_model=tts_opts.ecm_model,
            model_vocab=model_vocab,
            num_hyps=num_hyps,
            train=train,
            merge_subwords=merge_subwords,
            **{"cfg": cfg, **tts_opts.ecm_opts},
        )
        assert len(hyps_texts) == 1  # num iterations

        hyps_txts = []
        for i in range(1):
            hyps_txt = hyps_texts[0]
            hyps_txt.creator.add_alias(f"{prefix}/hyps_from_dlm_model/{tts_opts.name}/{name}/{subset}-hyps{i}-txt")
            if register_output:
                tk.register_output(
                    f"{prefix}/hyps_from_dlm_model/{tts_opts.name}/{name}/{subset}-hyps{i}.txt.gz", hyps_txt
                )
            hyps_txts.append(hyps_txt)
        return hyps_txts

    if isinstance(tts_opts, list):
        assert len(tts_opts) == 1
        tts_opts = tts_opts[0]
    assert isinstance(tts_opts, TtsOpts)  # not a list anymore, assume we handled it above

    if subset.startswith("lm-"):
        lm_seq_list = get_lm_tts_seq_list(subset=subset[len("lm-") :], seq_tag_format="lm")
        real_lm_txt = get_real_lm_txt(subset=subset[len("lm-") :])
    else:
        # extract transcriptions from LS ASR train data and make tts with that
        lm_seq_list = _get_asr_seq_list(subset=subset)
        real_lm_txt = get_real_asr_txt(subset=subset)

    from .tts_model import get_tts_model_dataset_dict, get_tts_model_dataset_extern_data_data

    hyp_vocab_ = get_vocab_by_str(hyp_vocab)
    hyp_vocab_dict = hyp_vocab_.get_opts()
    hyp_vocab_dim = Dim(hyp_vocab_.get_num_classes(), name="hyp_vocab")
    hyps_spatial_dim = Dim(None, name="hyp_seq", kind=Dim.Types.Spatial)

    hyps_hdfs = []
    for i in range(num_hyps):
        dataset = DatasetConfigStatic(
            main_name=subset,
            main_dataset=get_tts_model_dataset_dict(
                corpus_text=real_lm_txt,
                seq_list_file=lm_seq_list,
                train=train,
                fixed_random_seed=6577 + i * 1023,
                tts_opts=tts_opts,
                use_extended_lexicon=(subset.startswith("dev-") or subset.startswith("test-")),
            ),
            default_input="data",
            default_target=None,  # no target
            extern_data={
                # input dataset for the TTS model
                "data": get_tts_model_dataset_extern_data_data(tts_opts=tts_opts)
            },
            use_deep_copy=True,
        )

        hyps_hdf = sis_get_ctc_hyps_single_split(
            model,
            dataset=dataset,
            hyp_idx=i,
            cfg=cfg,
            tts_opts=tts_opts,
            extra_config=get_hyps_extra_config,
            hyp_output_tensor_template_dict={
                "dims": [batch_dim, hyps_spatial_dim],
                "sparse_dim": hyp_vocab_dim,
                "dtype": "int32",
                "vocab": hyp_vocab_dict,
            },
        )
        hyps_hdfs.append(hyps_hdf)

    hyps_txts = []
    for i, hyp_hdf in enumerate(hyps_hdfs):
        if register_output:
            tk.register_output(f"{prefix}/hyps_from_model/{ctc_model_name}/{name}/{subset}/hyps{i}.hdf", hyp_hdf)
        hyp_hdf.creator.add_alias(f"{prefix}/hyps_from_model/{ctc_model_name}/{name}/{subset}/hyps{i}-hdf")

        if merge_subwords:
            assert hyp_vocab.startswith("spm")  # TODO see below raw_replacement_list, currently we assume SPM here...
        hyps_txt = ReturnnDatasetToTextLinesJob(
            returnn_dataset={"class": "HDFDataset", "files": [hyp_hdf]},
            returnn_dataset_ext_non_hashed={"use_cache_manager": True},
            multi_proc_dataset_opts=dict(num_workers=5, buffer_size=10),
            seq_list=lm_seq_list,
            data_key="data",
            **(dict(raw_replacement_list=[(" ", ""), ("▁", " ")], raw_final_strip=True) if merge_subwords else {}),
        ).out_txt
        hyps_txt.creator.rqmt.update(dict(time=50, mem=40, cpu=16))
        hyps_txt.creator.add_alias(f"{prefix}/hyps_from_model/{ctc_model_name}/{name}/{subset}-hyps{i}-txt")
        if register_output:
            tk.register_output(f"{prefix}/hyps_from_model/{ctc_model_name}/{name}/{subset}-hyps{i}.txt.gz", hyps_txt)
        hyps_txts.append(hyps_txt)
    return hyps_txts


def get_real_lm_txt(*, subset: Union[Literal["train", "dev", "test"], str] = "train") -> tk.Path:
    """
    Get the real LM txt (via :func`get_librispeech_normalized_lm_data`) (without LS ASR transcriptions).
    This is the reference, the correct text.
    Reorderes the original LM data based on the seq list.
    Uses the seq list via :func:`get_lm_tts_seq_list`.

    :return: gzipped text file, which can be used with LmDataset. reordered by new seq list order, specified subset.
    """
    from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextLinesJob
    from .tts_data import get_lm_tts_seq_list

    prefix = get_setup_prefix_for_module(__name__)

    lm_seq_list = get_lm_tts_seq_list(seq_tag_format="lm", subset=subset)
    lm_data = get_librispeech_normalized_lm_data()
    vocab = {"class": "Utf8ByteTargets"}
    lm_reordered = ReturnnDatasetToTextLinesJob(
        returnn_dataset={
            "class": "LmDataset",
            "corpus_file": lm_data,
            "skip_empty_lines": False,
            "orth_vocab": vocab,
        },
        returnn_dataset_ext_non_hashed={"use_cache_manager": True},
        multi_proc_dataset_opts=dict(num_workers=4, buffer_size=10),
        seq_list=lm_seq_list,
        data_key="data",
        vocab=vocab,
    ).out_txt
    lm_reordered.creator.rqmt.update(dict(time=50, mem=40, cpu=8))
    lm_reordered.creator.add_alias(prefix + f"/seq-list_ls-lm/lm_{subset}_reordered.txt")
    tk.register_output(prefix + f"/seq-list_ls-lm/lm_{subset}_reordered.txt.gz", lm_reordered)
    return lm_reordered


def get_tts_ctc_hyps_txt(*, subset: Literal["train", "dev", "test"], num_hyps: int, model_vocab: str) -> List[tk.Path]:
    """
    :return: seq list, txt
    """
    # .../hyps_from_model_spm10k_tts/hyps_combined.hdf
    #   ->
    #     .../i6_core/returnn/forward/ReturnnForwardJobV2.GCIyN2Bbe1jC/output/out.hdf
    # 40310479 seqs. (orig LM has 40418261 seqs)
    # Thus we need to filter based on the given seqs.

    from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextLinesJob

    prefix = get_setup_prefix_for_module(__name__)

    from .tts_data import get_lm_tts_seq_list

    lm_tts_seq_list = get_lm_tts_seq_list(seq_tag_format="tts", subset=subset)

    if subset == "train":
        assert model_vocab == "spm10k"
        # TODO...
        tts_hyps_combined = generic_job_output(
            "i6_core/returnn/forward/ReturnnForwardJobV2.GCIyN2Bbe1jC/output/out.hdf"
        )
        assert tts_hyps_combined.get_path() == (
            gs.BASE_DIR + "/work/" + "i6_core/returnn/forward/ReturnnForwardJobV2.GCIyN2Bbe1jC/output/out.hdf"
        )
        tts_hyps_combined = ReturnnDatasetToTextLinesJob(
            returnn_dataset={"class": "HDFDataset", "files": [tts_hyps_combined]},
            returnn_dataset_ext_non_hashed={"use_cache_manager": True},
            multi_proc_dataset_opts=dict(num_workers=5, buffer_size=10),
            seq_list=lm_tts_seq_list,
            data_key="data",
            raw_replacement_list=[(" ", ""), ("▁", " ")],
            raw_final_strip=True,
        ).out_txt
        tts_hyps_combined.creator.rqmt.update(dict(time=50, mem=40, cpu=16))
        tts_hyps_combined.creator.add_alias(prefix + f"/tts_{subset}_hyps_combined.txt")
        tk.register_output(prefix + f"/tts_{subset}_hyps_combined.txt.gz", tts_hyps_combined)
    else:
        # TODO...
        raise NotImplementedError(f"get_tts_ctc_hyps_txt: subset {subset!r} not implemented")

    assert num_hyps == 1  # TODO don't have that yet otherwise...
    return [tts_hyps_combined]


def _name_for_dict(d: Dict[str, Any]) -> str:
    parts = []
    for k, v in d.items():
        k = "".join((part[0] if len(part) > 0 else "") for part in k.split("_"))  # some shortening
        if isinstance(v, (tuple, list)):
            v = "_".join(str(v_) for v_ in v)
        elif isinstance(v, dict):
            v = "{" + _name_for_dict(v) + "}"  # recursive
        parts.append(f"{k}{v}")
    return "-".join(parts)
