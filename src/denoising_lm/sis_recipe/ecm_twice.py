from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Dict

from .tts_model import EcmTwiceOpts
from .error_correction_model import recog_input_eval_datasets
from .error_correction_model_gen_train_data import (
    GetCtcHypsCfgV4,
    _get_asr_seq_list,
    get_asr_hyps_txt,
    get_error_correction_model_task_via_tts_txt,
    get_lm_hyps_txt_v3,
    get_real_asr_txt,
    get_real_lm_txt,
)
from i6_experiments.users.dorian_koch.jobs.fastwer import FastButInaccurateWer
from i6_experiments.users.zeyer.datasets.librispeech import get_vocab_by_str
from i6_experiments.users.zeyer.datasets.utils.sclite_generic_score import ScoreResult, sclite_score_recog_out_to_ref
from i6_experiments.users.zeyer.datasets.utils.vocab import ExtractVocabLabelsJob, ExtractVocabSpecialLabelsJob
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from returnn.datasets.basic import Vocabulary
from .ctc import sis_get_model as sis_get_ctc_model
from .tts_model import get_tts_opts_default_model
from .tts_data import LibrispeechTtsOggZip, get_lm_tts_seq_list
from i6_experiments.users.zeyer.datasets.utils.serialize import (
    ReturnnDatasetToTextDictJob,
    ReturnnDatasetToTextLinesJob,
)
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import use_instanciate_delayed_copy_instead_of_inplace
from returnn.tensor import single_step_dim
from sisyphus import tk

from i6_experiments.users.zeyer.datasets.score_results import JoinScoreResultsJob, RecogOutput
from returnn_common.datasets_old_2022_10.interface import (
    DatasetConfig,
    DatasetConfigStatic,
)

from ..model.error_correction_model import Model, model_recog
import returnn.frontend as rf
from returnn.tensor import batch_dim

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint


def dlm_forward_greedy(
    data: rf.Tensor,
    *,
    in_spatial_dim: rf.Dim,
    model: Model,
):
    """
    Greedy DLM decoding (beam size 1)
    """
    from returnn.frontend.tensor_array import TensorArray
    from returnn.config import get_global_config

    config = get_global_config()

    input_add_bos = config.typed_value("input_add_bos", False)
    input_add_eos = config.typed_value("input_add_eos", False)
    data_spatial_dim = in_spatial_dim
    if input_add_bos:
        data, (data_spatial_dim,) = rf.pad(
            data, axes=[data_spatial_dim], padding=[(int(input_add_bos), 0)], value=model.bos_idx
        )
    if input_add_eos:
        data, (data_spatial_dim_,) = rf.pad(
            data, axes=[data_spatial_dim], padding=[(0, int(input_add_eos))], value=model.eos_idx
        )
        data_spatial_dim = data_spatial_dim_

    batch_dims = data.remaining_dims(data_spatial_dim)

    enc = model.encode(data, spatial_dim=data_spatial_dim)

    max_seq_len = data_spatial_dim.get_size_tensor() * 2
    print("** max seq len (2x):", max_seq_len.raw_tensor)
    neg_inf = float("-inf")

    # Initial state.
    decoder_state = model.decoder.default_initial_state(batch_dims=batch_dims)
    target = rf.constant(model.bos_idx, dims=batch_dims, sparse_dim=model.target_dim, dtype="int32")
    ended = rf.constant(False, dims=batch_dims)
    out_seq_len = rf.constant(0, dims=batch_dims)
    seq_log_prob = rf.constant(0.0, dims=batch_dims)

    i = 0
    seq_targets = TensorArray(target)
    # seq_backrefs = []
    while True:
        logits, decoder_state = model.decoder(target, spatial_dim=single_step_dim, encoder=enc, state=decoder_state)
        label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=neg_inf),
            label_log_prob,
        )

        target = rf.reduce_argmax(logits, axis=model.target_dim)
        target = rf.cast(target, dtype="int32")
        label_log_prob = rf.gather(label_log_prob, indices=target, axis=model.target_dim)
        seq_log_prob = seq_log_prob + label_log_prob  # Batch
        i += 1

        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        seq_targets = seq_targets.push_back(target)
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

    expected_output = rf.get_run_ctx().expected_outputs["output"]
    labels_spatial_dim = expected_output.dims[-1]
    if labels_spatial_dim.dyn_size_ext is None:
        labels_spatial_dim.dyn_size_ext = out_seq_len
    elif labels_spatial_dim.dyn_size_ext is not None and labels_spatial_dim.dyn_size_ext.raw_tensor is None:
        labels_spatial_dim.dyn_size_ext.raw_tensor = out_seq_len.raw_tensor
    else:
        raise ValueError(
            f"Expected output labels_spatial_dim {labels_spatial_dim} has unexpected dyn_size_ext {labels_spatial_dim.dyn_size_ext}"
        )

    seq_targets = seq_targets.stack(axis=labels_spatial_dim)
    assert labels_spatial_dim in seq_targets.dims_set

    print("** output seq len:", labels_spatial_dim.get_size_tensor().raw_tensor)

    # print the first seq target
    first_seq_target = rf.gather(seq_targets, indices=0, axis=batch_dim)
    assert len(first_seq_target.dims) == 1
    first_seq_target_len = rf.gather(first_seq_target.dims[0].get_size_tensor(), indices=0, axis=batch_dim)
    print("** first seq target len:", first_seq_target_len.raw_tensor.v)
    vocab: Vocabulary = model.target_dim.vocab
    try:
        print("Labels (text):", vocab.get_seq_labels(first_seq_target.raw_tensor.cpu().detach().numpy().tolist()))
    except Exception:
        pass
    print("** first seq target:", first_seq_target.raw_tensor.v)
    seq_targets = rf.where(labels_spatial_dim.get_mask(), seq_targets, model.eos_idx)

    rf.get_run_ctx().mark_as_output(seq_targets, "output")
    rf.get_run_ctx().mark_as_output(seq_log_prob, "log_probs")
    rf.get_run_ctx().mark_as_output(in_spatial_dim.dyn_size_ext, "enc_seq_lens")


def model_recog_wrapper(data: rf.Tensor, *, in_spatial_dim: rf.Dim, model: Model):
    seq_targets, seq_log_prob, out_spatial_dim, beam_dim = model_recog(
        model=model, data=data, data_spatial_dim=in_spatial_dim
    )

    max_seqs = rf.reduce_argmax(seq_log_prob, axis=beam_dim)
    seq_targets = rf.gather(seq_targets, indices=max_seqs, axis=beam_dim)
    seq_log_prob = rf.gather(seq_log_prob, indices=max_seqs, axis=beam_dim)

    out_spatial_dim_size = out_spatial_dim.get_size_tensor()
    out_spatial_dim_size = rf.gather(
        out_spatial_dim_size, indices=max_seqs, axis=beam_dim
    )  # Get the size of the output sequence for the best beam

    expected_output = rf.get_run_ctx().expected_outputs["output"]
    labels_spatial_dim = expected_output.dims[-1]
    if labels_spatial_dim.dyn_size_ext is None:
        labels_spatial_dim.dyn_size_ext = out_spatial_dim_size
    elif labels_spatial_dim.dyn_size_ext is not None and labels_spatial_dim.dyn_size_ext.raw_tensor is None:
        labels_spatial_dim.dyn_size_ext.raw_tensor = out_spatial_dim_size.raw_tensor
    else:
        raise ValueError(
            f"Expected output labels_spatial_dim {labels_spatial_dim} has unexpected dyn_size_ext {labels_spatial_dim.dyn_size_ext}"
        )

    seq_targets = rf.replace_dim_v2(seq_targets, in_dim=out_spatial_dim, out_dim=labels_spatial_dim)
    seq_targets = rf.cast(seq_targets, dtype="int32")

    rf.get_run_ctx().mark_as_output(seq_targets, "output")
    rf.get_run_ctx().mark_as_output(seq_log_prob, "log_probs")
    rf.get_run_ctx().mark_as_output(in_spatial_dim.dyn_size_ext, "enc_seq_lens")


def single(
    model: ModelWithCheckpoint,
    *,
    dataset: DatasetConfig,
    hyp_idx: int,
    hyp_output_tensor_template_dict: Dict[str, Any],
    extra_config: Optional[Dict[str, Any]] = None,
    beam_size: int = 1,
) -> tk.Path:
    """
    Get hyps from CTC model, for given hyp_idx (split idx).

    :param model:
    :param dataset:
    :param hyp_idx:
    :param hyp_output_tensor_template_dict:
    :param cfg: config for the hyps generation (e.g. dropout, specaugment, etc)
    :param tts_opts: TTS model, if we want to generate TTS data on-the-fly. requires cfg to be a GetHypsCfgV3
    :param extra_config:
    :return: hyps (list of HDF paths, len is ``num_hyps``), reference sequences path (HDF)
    """
    from i6_experiments.users.zeyer.forward_to_hdf import forward_to_hdf
    from returnn.tensor import batch_dim

    base_config = {
        # We use dropout to get multiple different hypotheses.
        # We introduce the beam dim, and want to get different hypotheses for each beam.
        "rf_dropout_broadcast": False,
        "model_outputs": {
            "output": hyp_output_tensor_template_dict.copy(),
            "log_probs": {"dims": [batch_dim], "dtype": "float32"},
            "enc_seq_lens": {"dims": [batch_dim], "dtype": "int32"},
        },
        "load_epoch": 1,
    }
    if extra_config:
        base_config.update(extra_config)

    hdf = forward_to_hdf(
        dataset=dataset,
        model=model,
        forward_def=dlm_forward_greedy if beam_size == 1 else model_recog_wrapper,
        config={
            "batch_size": 80000 * model.definition.batch_size_factor / beam_size,
            "max_seqs": 2000,
            **base_config,
            **({"beam_size": beam_size} if beam_size > 1 else {}),
            "random_seed": 1023 + hyp_idx * 17,
            "version": 2,
        },
    )
    hdf.creator.rqmt["time"] = 4
    return hdf


def get_hyps_single(
    subset: str,
    *,
    name: Optional[str] = None,
    use_ground_truth_as_hyp: bool = False,
    prefix: Optional[str] = None,
    asr_model: ModelWithCheckpoint,
    dlm_model: ModelWithCheckpoint,
    model_vocab: str = "spm10k",
    merge_subwords: bool = False,
    beam_size: int = 1,
    num_iters: int = 1,
    num_hyps: int = 1,
    **opts,
):
    from i6_experiments.users.zeyer.datasets.librispeech import get_vocab_by_str

    assert num_hyps == 1, "Currently only supports num_hyps=1"

    prefix = prefix or get_setup_prefix_for_module(__name__)

    if "lm-train_split" in subset:
        assert opts.get("train")

    # print(subset, asr_model, merge_subwords)

    if subset.startswith("asr-"):
        subset = subset[len("asr-") :]
        hyp1 = get_asr_hyps_txt(subset=subset, model=asr_model, num_hyps=1, merge_subwords=merge_subwords, **opts)
    else:
        hyp1 = get_lm_hyps_txt_v3(subset=subset, model=asr_model, num_hyps=1, merge_subwords=merge_subwords, **opts)
    assert len(hyp1) == 1, "Expected only one hypothesis for single run"

    if subset.startswith("lm-"):
        lm_seq_list = get_lm_tts_seq_list(subset=subset[len("lm-") :], seq_tag_format="lm")
        real_lm_txt = get_real_lm_txt(subset=subset[len("lm-") :])
    else:
        # extract transcriptions from LS ASR train data and make tts with that
        lm_seq_list = _get_asr_seq_list(subset=subset)
        real_lm_txt = get_real_asr_txt(subset=subset)

    hyp_vocab_ = get_vocab_by_str(model_vocab)
    hyp_vocab_dict = hyp_vocab_.get_opts()
    hyp_vocab_dim = rf.Dim(hyp_vocab_.get_num_classes(), name="hyp_vocab")
    in_hyps_spatial_dim = rf.Dim(None, name="in_hyp_seq", kind=rf.Dim.Types.Spatial)
    out_hyps_spatial_dim = rf.Dim(None, name="out_hyp_seq", kind=rf.Dim.Types.Spatial)

    hyps_vocab_dict = None
    if not merge_subwords:
        from i6_experiments.users.zeyer.datasets.utils.vocab import (
            ExtractVocabLabelsJob,
            ExtractVocabSpecialLabelsJob,
        )

        hyps_vocab_dict = {
            "class": "Vocabulary",
            "vocab_file": ExtractVocabLabelsJob(hyp_vocab_dict).out_vocab,
            "special_symbols_via_file": ExtractVocabSpecialLabelsJob(hyp_vocab_dict).out_vocab_special_labels_dict,
        }
        if not (model_vocab.startswith("bpe") or model_vocab.startswith("spm")):
            hyps_vocab_dict["single_whitespace_split"] = True

    if use_ground_truth_as_hyp:  # try real lm text
        hyp1 = [real_lm_txt]
        hyps_vocab_dict = hyp_vocab_dict

    assert num_iters > 0, "num_iters must be > 0"

    dataset = DatasetConfigStatic(
        main_name=subset,
        main_dataset={
            "class": "LmDataset",
            "corpus_file": hyp1,
            "seq_list_file": lm_seq_list,
            "use_cache_manager": True,
            "skip_empty_lines": False,
            "orth_vocab":  # train_hyps_vocab_dict
            # if train and train_hyps_vocab_dict
            # else
            hyps_vocab_dict if hyps_vocab_dict else hyp_vocab_dict,
            "seq_end_symbol": None,  # handled via orth_vocab
            "unknown_symbol": None,  # handled via orth_vocab
            "seq_ordering": "sorted_reverse",
            # **(dataset_extra_args or {}),
        },
        default_input="data",
        default_target=None,  # no target
        extern_data={
            "data": {
                "dim_tags": [batch_dim, in_hyps_spatial_dim],
                "sparse_dim": hyp_vocab_dim,
                "vocab": hyp_vocab_dict,
            }
        },
        use_deep_copy=True,
    )

    hyp_txts = []
    for i in range(num_iters):
        hyps_hdf = single(
            dlm_model,
            dataset=dataset,
            hyp_idx=0,
            # extra_config=get_hyps_extra_config,
            hyp_output_tensor_template_dict={
                "dims": [batch_dim, out_hyps_spatial_dim],
                "sparse_dim": hyp_vocab_dim,
                "dtype": "int32",
                "vocab": hyp_vocab_dict,
            },
            beam_size=beam_size,
        )

        assert not merge_subwords, "not implemented"
        # for the next iter, load the hdf from last iter
        dataset = DatasetConfigStatic(
            main_name=subset,
            main_dataset={"class": "HDFDataset", "files": [hyps_hdf]},
            default_input="data",
            default_target=None,  # no target
            extern_data={
                "data": {
                    "dim_tags": [batch_dim, in_hyps_spatial_dim],
                    "sparse_dim": hyp_vocab_dim,
                    "vocab": hyp_vocab_dict,
                }
            },
            use_deep_copy=True,
        )
        if merge_subwords:
            assert model_vocab.startswith("spm")  # TODO see below raw_replacement_list, currently we assume SPM here...
        hyps_txt = ReturnnDatasetToTextLinesJob(
            returnn_dataset={"class": "HDFDataset", "files": [hyps_hdf]},
            returnn_dataset_ext_non_hashed={"use_cache_manager": True},
            multi_proc_dataset_opts=dict(num_workers=5, buffer_size=10),
            seq_list=lm_seq_list,
            data_key="data",
            **(dict(raw_replacement_list=[(" ", ""), ("▁", " ")], raw_final_strip=True) if merge_subwords else {}),
        ).out_txt
        hyps_txt.creator.rqmt.update(dict(time=50, mem=40, cpu=16))

        if name is not None:
            hyps_hdf.creator.add_alias(f"{prefix}/hyps_from_dlm_model/{name}_iter{i}/{subset}/hyps{0}-hdf")
        #    hyps_txt.creator.add_alias(f"{prefix}/hyps_from_dlm_model/{name}_iter{i}/{subset}/hyps{0}-txt")

        hyp_txts.append(hyps_txt)

    return hyp_txts, real_lm_txt


# Idea: pass the output of the DLM again as input to the DLM!
def py():
    print("> ecm_twice.py")
    from .error_correction_model import py as dlm_py
    from .error_correction_model import _train_experiments
    from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output

    use_instanciate_delayed_copy_instead_of_inplace()

    from i6_experiments.users.zeyer.tools_paths import monkey_patch_i6_core

    monkey_patch_i6_core()

    if "fileNameTooLong-nEp200" not in _train_experiments:
        with disable_register_output():
            dlm_py()

    ctc_model_tts = sis_get_ctc_model("L16-D1024-spm10k-auxAED-b100k-tts")

    # hyps_cfg = GetCtcHypsCfgV4(
    #     dropout_min=0.1,
    #     dropout_max=0.5,
    #     enable_specaugment=True,
    #     specaugment_opts={"steps": (0, 0, 0), "max_consecutive_spatial_dims": 0},
    # )
    hyps_cfg = GetCtcHypsCfgV4(
        dropout_min=0.0,
        dropout_max=0.0,
        enable_specaugment=False,
        # specaugment_opts={"steps": (0, 0, 0), "max_consecutive_spatial_dims": 0},
    )
    hyps_tts_opts = get_tts_opts_default_model(
        {
            "glow_tts_noise_scale_range": (0.3, 0.9),
            "glow_tts_length_scale_range": (0.7, 1.1),
        },
        compatible_to_nick=True,
    )

    num_train_parts = LibrispeechTtsOggZip.get_num_training_parts(files_per_part=10)

    ecm_model_name = "fileNameTooLong-oclr-nickCompat-nEp200-numHyp1"
    ecm_model = _train_experiments[ecm_model_name].get_last_fixed_epoch()
    # for i in range(num_train_parts):
    #     subset_name = f"lm-train_split_{i}_of_{num_train_parts}"

    #     hyps, real = get_hyps_single(
    #         subset=subset_name,
    #         name=model_name,
    #         dlm_model=model,
    #         asr_model=ctc_model_tts,
    #         cfg=hyps_cfg,
    #         tts_opts=hyps_tts_opts,
    #         train=True,
    #         merge_subwords=False,
    #         use_ground_truth_as_hyp=False,
    #         beam_size=1,
    #         num_iters=1,
    #     )
    #     assert len(hyps) == 1, "Expected only one hypothesis for single run"
    #     tk.register_output(
    #         f"denoising-lm/ecm_twice/{model_name}/{subset_name}_bs{1}",
    #         hyps[0],
    #     )

    task_ecmtwice = get_error_correction_model_task_via_tts_txt(
        train_epoch_split=20,
        vocab="spm10k",
        hyps_model=ctc_model_tts,
        num_hyps=1,
        hyps_cfg=GetCtcHypsCfgV4(
            dropout_min=0.0,
            dropout_max=0.0,
            enable_specaugment=False,
        ),
        hyps_tts_opts=EcmTwiceOpts(
            ecm_model=ecm_model,
            name=ecm_model_name,
            ecm_opts={"tts_opts": hyps_tts_opts, "cfg": hyps_cfg},
        ),
        train_repeat_asr_data=10,
        train_repeat_asr_data_via_num_hyps=True,
        resplit_subwords=False,
        dataset_use_deep_copy=True,
        get_hyps_extra_config={"behavior_version": 24},
        dependency_boundary_hash="3s7O26pAqYgZ",
    )
    recog_input_eval_datasets("spm10k-baseline-ecmTwice", task_ecmtwice)

    # now we trained a model with this ecm twice data, lets see how it does

    for model_name in ["ecmTwice-oclr-nEp100-stdPerturb", ecm_model_name]:
        model = _train_experiments[model_name]
        assert model is not None, "Model should be defined in _train_experiments"

        max_iter = 4
        all_ds = [{} for _ in range(max_iter)]
        for subset_name in [
            "asr-dev-other",
            "asr-test-other",
            "asr-dev-clean",
            "asr-test-clean",
        ]:  # , f"lm-train_split_{0}_of_{num_train_parts}"]:
            for use_real_as_hyp in [False]:  # [False, True]:
                for bs in [64]:  # [1, 2]:
                    hyps, real = get_hyps_single(
                        subset=subset_name,
                        name=model_name,  # fileNameTooLong-nEp200
                        dlm_model=model.get_last_fixed_epoch(),
                        asr_model=ctc_model_tts,
                        cfg=hyps_cfg,
                        **(  # TODO: for compelte train data generation, use different hyps cfg
                            dict(tts_opts=hyps_tts_opts, train=True) if subset_name.startswith("lm-train_split") else {}
                        ),
                        merge_subwords=False,
                        use_ground_truth_as_hyp=use_real_as_hyp,
                        beam_size=bs,
                        num_iters=max_iter,
                    )
                    # tk.register_output("testing", hyp)
                    # tk.register_output("testing2", real)
                    # assuming spm10k...
                    hyp_vocab_ = get_vocab_by_str("spm10k")
                    hyp_vocab_dict = hyp_vocab_.get_opts()
                    if subset_name.startswith("lm-"):
                        lm_seq_list = get_lm_tts_seq_list(subset=subset_name[len("lm-") :], seq_tag_format="lm")
                    elif subset_name.startswith("asr-"):
                        lm_seq_list = _get_asr_seq_list(subset=subset_name[len("asr-") :])
                    else:
                        raise ValueError(f"Unknown subset name {subset_name}")

                    hyp_vocab_dict2 = {
                        "class": "Vocabulary",
                        "vocab_file": ExtractVocabLabelsJob(hyp_vocab_dict).out_vocab,
                        "special_symbols_via_file": ExtractVocabSpecialLabelsJob(
                            hyp_vocab_dict
                        ).out_vocab_special_labels_dict,
                    }

                    realdict = RecogOutput(
                        output=ReturnnDatasetToTextDictJob(
                            returnn_dataset={
                                "class": "LmDataset",
                                "corpus_file": real,
                                "seq_list_file": lm_seq_list,
                                "use_cache_manager": True,
                                "skip_empty_lines": False,
                                "orth_vocab": {"class": "Utf8ByteTargets"},
                            },
                            data_key="data",
                        ).out_txt
                    )
                    for i, hyp in enumerate(hyps):
                        hypdict = RecogOutput(
                            output=ReturnnDatasetToTextDictJob(
                                returnn_dataset={
                                    "class": "LmDataset",
                                    "corpus_file": hyp,
                                    "seq_list_file": lm_seq_list,
                                    "use_cache_manager": True,
                                    "skip_empty_lines": False,
                                    "orth_vocab": hyp_vocab_dict2,
                                    "seq_end_symbol": None,  # handled via orth_vocab
                                    "unknown_symbol": None,  # handled via orth_vocab
                                },
                                data_key="data",
                                vocab=hyp_vocab_dict2,
                                raw_replacement_list=[(" ", ""), ("▁", " ")],
                                raw_final_strip=True,
                            ).out_txt
                        )
                        if subset_name.startswith("lm-") or True:
                            score_out = FastButInaccurateWer(
                                ref=real, hyp_replacement_list=[(" ", ""), ("▁", " ")], hyp=hyp
                            ).out_wer
                        else:
                            score_out = sclite_score_recog_out_to_ref(
                                hypdict, ref=realdict, corpus_name=subset_name
                            ).main_measure_value
                        tk.register_output(
                            f"denoising-lm/ecm_twice/data/{subset_name}_{model_name}_bs{bs}_{i + 1}{'_realInput' if use_real_as_hyp else ''}",
                            score_out,
                        )
                        dsname = subset_name[len("asr-") :]
                        all_ds[i][dsname] = ScoreResult(dataset_name=dsname, main_measure_value=score_out)
        for i in range(max_iter):
            tk.register_output(
                f"denoising-lm/ecm_twice/{model_name}_{i + 1}_score_results.txt",
                JoinScoreResultsJob(all_ds[i]).out_score_results,
            )
    print("< END ecm_twice.py")
