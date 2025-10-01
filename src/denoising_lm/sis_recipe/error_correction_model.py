"""
Error correction model as language model (LM).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union, Generator, Sequence, Dict, Tuple, List

from returnn.frontend.tensor_array import TensorArray
from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase

from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
from i6_experiments.users.zeyer.model_interfaces import RecogDef, RescoreDef
from i6_experiments.users.zeyer.model_interfaces.model import ModelDefWithCfg
from i6_experiments.users.zeyer.model_interfaces.training import TrainDef
from returnn_common.datasets_old_2022_10.interface import (
    DatasetConfig,
)
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.recog import search_dataset

from ..model.error_correction_model import Model, aed_model_def, aed_training, model_recog, dlm_rescore_def
import returnn.frontend as rf
from returnn.frontend.encoder.transformer import TransformerEncoder
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    post_config as baseline_post_config,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
)

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, ModelWithCheckpoint
    from i6_experiments.users.zeyer.datasets.task import Task
    from returnn.datasets.meta import VariableDataset
    from i6_experiments.users.zeyer.decoding.prior_rescoring import Prior


def py():
    from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
    from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import use_instanciate_delayed_copy_instead_of_inplace
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import config_96gb_bf16_accgrad1

    from .error_correction_model_gen_train_data import get_error_correction_model_task_via_tts_txt
    from .ctc import (
        GetHypsCfgV3 as GetCtcHypsCfgV3,
        GetHypsCfgV4 as GetCtcHypsCfgV4,
        GetHypsCfgV6 as GetCtcHypsCfgV6,
        sis_get_model as sis_get_ctc_model,
    )
    from .tts_model import get_tts_opts_default_model, get_tts_opts_coqui_ai_tts_your_tts

    prefix = get_setup_prefix_for_module(__name__)

    use_instanciate_delayed_copy_instead_of_inplace()

    # Base CTC model.
    ctc_model_v2 = sis_get_ctc_model("L16-D1024-spm10k-auxAED-b100k")

    # Better CTC model, trained also on TTS.
    ctc_model_tts = sis_get_ctc_model("L16-D1024-spm10k-auxAED-b100k-tts")

    # Correct TTS setting (nickCompat)
    task_genTts_fm2_spm10k_drop05_01_noBn_lsh10_epSplit20_keepSubwords_ctcTts_nickCompat_numHyp1 = (
        get_error_correction_model_task_via_tts_txt(
            train_epoch_split=20,
            vocab="spm10k",
            hyps_model=ctc_model_tts,
            num_hyps=1,
            hyps_cfg=GetCtcHypsCfgV4(
                dropout_min=0.1,
                dropout_max=0.5,
                enable_specaugment=True,
                specaugment_opts={"steps": (0, 0, 0), "max_consecutive_spatial_dims": 0},
            ),
            hyps_tts_opts=get_tts_opts_default_model(
                {
                    "glow_tts_noise_scale_range": (0.3, 0.9),
                    "glow_tts_length_scale_range": (0.7, 1.1),
                },
                compatible_to_nick=True,
            ),
            train_repeat_asr_data=10,
            train_repeat_asr_data_via_num_hyps=True,
            resplit_subwords=False,
            dataset_use_deep_copy=True,
            additional_eval_sets=["trainlike-lm-devtrain"],
            get_hyps_extra_config={"behavior_version": 24},
            dependency_boundary_hash="Y2syHSPzgCiL",
        )
    )
    eval_task_spm10k_ctcNoTts_nickCompat = get_error_correction_model_task_via_tts_txt(
        train_epoch_split=20,
        vocab="spm10k",
        num_hyps=0,
        hyps_cfg=GetCtcHypsCfgV4(),
        hyps_model=ctc_model_v2,
        hyps_tts_opts=get_tts_opts_default_model(
            {
                "glow_tts_noise_scale_range": (0.3, 0.9),
                "glow_tts_length_scale_range": (0.7, 1.1),
            },
            compatible_to_nick=True,
        ),
        additional_eval_sets=["trainlike-lm-devtrain"],
        register_output=False,
        train_repeat_asr_data=10,
        train_repeat_asr_data_via_num_hyps=True,
        resplit_subwords=False,
        dataset_use_deep_copy=True,
        get_hyps_extra_config={"behavior_version": 24},
        use_dependency_boundary=False,
    )

    # Llama / Transformer++ like
    common_trafo_kwargs = dict(
        model_dim=512,
        pos_enc=None,
        norm=rf.build_dict(rf.RMSNorm),
        ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
        dropout=0.0,
        att_dropout=0.0,
    )
    common_trafo_enc_kwargs = dict(
        **common_trafo_kwargs,
        layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosSelfAttention, with_bias=False)),
    )
    common_trafo_dec_kwargs = dict(
        **common_trafo_kwargs,
        layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    )

    aed_n1024_enc24_dec8_model_def = ModelDefWithCfg(
        aed_model_def,
        {
            "_encoder_model_dict": rf.build_dict(
                TransformerEncoder,
                num_layers=24,
                **dict_update_deep(common_trafo_enc_kwargs, {"model_dim": 1024}),
            ),
            "_decoder_model_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=8,
                **dict_update_deep(common_trafo_dec_kwargs, {"model_dim": 1024}),
            ),
            "input_add_eos": True,
        },
    )

    train_base_cfg = dict_update_deep(
        config_96gb_bf16_accgrad1,
        {
            "optimizer.weight_decay": 1e-2,
            "calculate_exp_loss": True,
        },
    )

    # greedy 3.86, dsr 3.46, dlm sum 3.32
    fntl_nickCompat = train_exp(
        "fileNameTooLong-oclr-nickCompat-nEp200-numHyp1",
        task_genTts_fm2_spm10k_drop05_01_noBn_lsh10_epSplit20_keepSubwords_ctcTts_nickCompat_numHyp1,
        model_def=aed_n1024_enc24_dec8_model_def,
        config=dict_update_deep(
            train_base_cfg,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(200),
                "batch_size": 20_000,
                "max_seqs": 2_000,
                "min_seq_length": 1,
                "__multi_proc_dataset": False,  # not needed here
                "input_swapout_range": (0.1, 0.1),
            },
        ),
        post_config={"log_grad_norm": True},
        asr_model_recog_args=make_asr_ctc_model_recog_args(
            vocab="spm10k", ctc_model=ctc_model_tts, with_lm_rescore=True, calculate_search_errors=True
        ),
    )
    # dlm sum 3.32
    recog_model_ext(
        dlm_task=task_genTts_fm2_spm10k_drop05_01_noBn_lsh10_epSplit20_keepSubwords_ctcTts_nickCompat_numHyp1,
        asr_model=make_asr_ctc_model_recog_args(
            vocab="spm10k",
            ctc_model=ctc_model_tts,
            max_dlm_scale=10.0,
            dlm_sum_extra_args={"length_normalization_exponent": 1.0},
        ),
        dlm=fntl_nickCompat.get_epoch(200),
        prefix="denoising-lm/error_correction_model/fileNameTooLong-oclr-nickCompat-nEp200-numHyp1/length_norm",
    )
    recog_model_ext(
        dlm_task=eval_task_spm10k_ctcNoTts_nickCompat,
        asr_model=make_asr_ctc_model_recog_args(
            vocab="spm10k",
            ctc_model=ctc_model_v2,
            max_dlm_scale=10.0,
            dlm_sum_extra_args={"length_normalization_exponent": 1.0},
        ),
        dlm=fntl_nickCompat.get_epoch(200),
        prefix="denoising-lm/error_correction_model/fileNameTooLong-oclr-nickCompat-nEp200-numHyp1-ctcNoTts/length_norm",
    )

    from i6_experiments.users.zeyer.nn_rf.mixup import MixupOpts

    low_task = get_error_correction_model_task_via_tts_txt(
        train_epoch_split=20,
        vocab="spm10k",
        num_hyps=1,
        hyps_cfg=GetCtcHypsCfgV6(
            dropout_min=0.0,
            dropout_max=0.2,
            enable_specaugment=True,
            specaugment_opts={"steps": (0, 0, 0), "max_consecutive_feature_dims": 0},
            data_perturbation_opts={"mixup": MixupOpts(max_num_mix=2, lambda_min=0.0, lambda_max=0.2, apply_prob=1)},
        ),
        hyps_model=ctc_model_tts,
        hyps_tts_opts=get_tts_opts_default_model(
            {"glow_tts_noise_scale_range": (0.3, 0.9), "glow_tts_length_scale_range": (0.7, 1.1)},
            compatible_to_nick=True,
        ),
        additional_eval_sets=["trainlike-lm-devtrain"],
        train_repeat_asr_data=10,
        train_repeat_asr_data_via_num_hyps=True,
        register_output=False,
        resplit_subwords=False,
        dataset_use_deep_copy=True,
        get_hyps_extra_config={"behavior_version": 24},
        use_dependency_boundary=False,
    )
    # token substitution: (0.1, 0.1)
    # recog_input_eval_datasets("spm10k-puttingItTogether(low)", simulate_token_substitution(low_task, (0.1, 0.1)))
    exp = train_exp(
        "base-puttingItTogether(low)-nEp200",
        low_task,
        model_def=aed_n1024_enc24_dec8_model_def,
        config=dict_update_deep(
            train_base_cfg,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(200),
                "batch_size": 20_000,
                "max_seqs": 2_000,
                "min_seq_length": 1,
                "__multi_proc_dataset": False,
                "input_swapout_range": (0.1, 0.1),
            },
        ),
        post_config={"log_grad_norm": True},
        asr_model_recog_args=make_asr_ctc_model_recog_args(
            vocab="spm10k", ctc_model=ctc_model_tts, with_lm_rescore=True, max_dlm_scale=10
        ),
    )
    # Same DLM, but CTC without TTS.
    # TODO...
    recog_model_ext(
        dlm_task=eval_task_spm10k_ctcNoTts_nickCompat,
        asr_model=make_asr_ctc_model_recog_args(vocab="spm10k", ctc_model=ctc_model_v2, max_dlm_scale=10),
        dlm=exp.get_last_fixed_epoch(),
        prefix="denoising-lm/error_correction_model/base-puttingItTogether(low)-nEp200-ctcNoTts",
    )

    # andYourTts: interleave our TTS with YourTTS
    task_genTts_andYourTts_fm2_spm10k_drop05_lsh10_epSplit20_keepSubwords_ctcTts = (
        get_error_correction_model_task_via_tts_txt(
            prefix=f"{prefix}/ctc-L16-D1024-tts",
            train_epoch_split=20,
            vocab="spm10k",
            hyps_model=ctc_model_tts,  # note: this is different TTS data!
            num_hyps=5,
            hyps_cfg=GetCtcHypsCfgV3(
                dropout_min=0.0,
                dropout_max=0.5,
                enable_specaugment=True,
                specaugment_opts={"steps": (0, 0, 0), "max_consecutive_spatial_dims": 0},
            ),
            hyps_tts_opts=[
                get_tts_opts_default_model(
                    {"glow_tts_noise_scale_range": (0.3, 0.9), "glow_tts_length_scale_range": (0.7, 1.1)}
                ),
                get_tts_opts_coqui_ai_tts_your_tts({"tts_model_opt_sample_ranges": {"length_scale": (1.0, 1.5)}}),
            ],
            train_repeat_asr_data=10,
            train_repeat_asr_data_via_num_hyps=True,
            resplit_subwords=False,
            dataset_use_deep_copy=True,
            version=2,
            use_dependency_boundary=False,
        )
    )
    recog_input_eval_datasets(
        "genTts-andYourTts-fm2-spm10k-hypsV2-drop05-lsh10-epSplit20-keepSubwords-ctcTts",
        task_genTts_andYourTts_fm2_spm10k_drop05_lsh10_epSplit20_keepSubwords_ctcTts,
    )

    # andYourTts
    train_exp(
        "base-genTts-andYourTts-fm2-spm10k-ctcDrop05-ctcTts-n1024-decL16-lsh10-epSplit20-keepSubwords-b2k_20k",
        task_genTts_andYourTts_fm2_spm10k_drop05_lsh10_epSplit20_keepSubwords_ctcTts,
        model_def=ModelDefWithCfg(
            aed_model_def,
            {
                "_encoder_model_dict": rf.build_dict(
                    TransformerEncoder, num_layers=16, **dict_update_deep(common_trafo_enc_kwargs, {"model_dim": 1024})
                ),
                "_decoder_model_dict": rf.build_dict(
                    TransformerDecoder, num_layers=16, **dict_update_deep(common_trafo_dec_kwargs, {"model_dim": 1024})
                ),
                "input_add_eos": True,
            },
        ),
        config=dict_update_deep(
            train_base_cfg,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(100),
                "batch_size": 20_000,
                "max_seqs": 2_000,
                "min_seq_length": 1,
                "__multi_proc_dataset": False,  # not needed here
            },
        ),
        asr_model_recog_args=make_asr_ctc_model_recog_args(vocab="spm10k", ctc_model=ctc_model_tts),
    )


def _rotate_datasets_for_epoch(*, self: VariableDataset, datasets: List[Dict[str, Any]], epoch: int, **_kwargs):
    from i6_core.util import instanciate_delayed

    epoch0 = epoch - 1  # 0-based
    global_epoch0 = epoch0 // (self.partition_epoch or 1)
    ds = datasets[global_epoch0 % len(datasets)]
    return instanciate_delayed(ds)


@dataclass
class AsrModelRecogArgs:
    task: Task
    model: ModelWithCheckpoint
    recog_def: RecogDef
    rescore_def: RescoreDef
    labelwise_prior: Prior
    asr_beam_size: int  # for the individual beam search
    dlm_beam_size: int  # for the individual beam search
    first_pass_recog_beam_size: int
    vocab_num_labels: int
    vocab_file: tk.Path
    vocab_opts_file: tk.Path
    vocab_is_chars: bool = False  # implies special formatting
    orig_vocab_opts: Optional[Dict[str, Any]] = None
    max_dlm_scale: float = 3.0  # This default is not always optimal, should be higher... (10 seems to be ok)
    recog_extra_args: Optional[Dict[str, Any]] = None
    dlm_sum_extra_args: Optional[Dict[str, Any]] = None
    dlm_sum_tune_scales_extra_args: Optional[Dict[str, Any]] = None
    dlm_rescore_extra_config: Optional[Dict[str, Any]] = None
    with_dlm_sum: bool = True  # use DLM sum recog
    dlm_sum_with_prior: bool = True  # DEPRECATED, just use with_prior. Can be removed?
    with_prior: bool = True
    dlm_sum_extra_batch_size_factor: float = 1.0
    with_dlm_greedy_recog: bool = True
    with_lm_rescore: bool = False
    with_dlm_sum_simple: bool = False  # no final ctc prefix score, no prior. Only sum(p_DLM * p_ASR)
    concat_asr_dlm_beams: bool = True  # concat ASR and DLM hyp beams instead of only using DLM beam
    calculate_search_errors: bool = False
    dlm_temperature: float = 1.0  # for softmax temperature scaling
    with_auxiliary_encoder_loss: bool = False  # use auxiliary encoder loss for decoding


# noinspection PyShadowingNames
def train_exp(
    name: str,
    task: Task,
    model_def: ModelDefWithCfg,
    *,
    config: Dict[str, Any],
    post_config: Optional[Dict[str, Any]] = None,
    recog_config: Optional[Dict[str, Any]] = None,
    train_def: Optional[TrainDef[Model]] = None,
    recog_def: Optional[RecogDef[Model]] = None,
    asr_model_recog_args: Optional[AsrModelRecogArgs] = None,
    with_recog: bool = True,
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    from i6_experiments.users.zeyer.train_v4 import train
    from i6_experiments.users.zeyer.recog import recog_training_exp

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/error_correction_model/" + name

    if train_def is None:
        train_def = aed_training

    merged_post_config = dict_update_deep(
        baseline_post_config,
        post_config or {},
    )
    model_with_checkpoint = train(
        prefix,
        task=task,
        config=config,
        post_config=merged_post_config,
        model_def=model_def,
        train_def=train_def,
    )
    _train_experiments[name] = model_with_checkpoint

    if with_recog:
        recog_training_exp(
            prefix,
            task,
            model_with_checkpoint,
            recog_def=recog_def or model_recog,
            search_config={"batch_size": 10_000, "beam_size": 12, **(recog_config or {})},
        )

        # TODO we can later always enable this, but now for testing...
        if asr_model_recog_args:
            # TODO don't do recog_training_exp (greedy decoding) at all, only do this
            # TODO first-pass recog
            # TODO also include other hyps from CTC as seeds for the DLM
            recog_model_ext(
                dlm_task=task,
                asr_model=asr_model_recog_args,
                dlm=model_with_checkpoint.get_last_fixed_epoch(),
                prefix=prefix,
            )

    return model_with_checkpoint


_train_experiments: Dict[str, ModelWithCheckpoints] = {}


def recog_input_eval_datasets(name: str, task: Task, register_outputs: bool = True):
    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = f"{_sis_prefix}/error_correction_model/train_data/{name}/eval_datasets"

    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
    from i6_experiments.users.zeyer.datasets.score_results import RecogOutput

    score_results = {"input": {}, "target": {}}
    for name, ds in task.eval_datasets.items():
        for data_key_name, data_key in {"input": ds.get_default_input(), "target": ds.get_default_target()}.items():
            txt = ReturnnDatasetToTextDictJob(returnn_dataset=ds.get_main_dataset(), data_key=data_key).out_txt
            if register_outputs:
                tk.register_output(f"{prefix}/{data_key_name}/{name}.txt", txt)
            res = RecogOutput(output=txt)
            if task.recog_post_proc_funcs:
                for func in task.recog_post_proc_funcs:
                    res = func(res)
                if register_outputs:
                    tk.register_output(f"{prefix}/{data_key_name}/{name}-pp.txt", res.output)
            score_results[data_key_name][name] = task.score_recog_output_func(ds, res)
    for data_key_name, score_res in score_results.items():
        score_res_col = task.collect_score_results_func(score_res)
        if register_outputs:
            tk.register_output(f"{prefix}/{data_key_name}/score_results.txt", score_res_col.output)

    return score_results


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name


def model_recog_fix_small_vocab(
    *,
    model: Model,
    data: rf.Tensor,
    data_spatial_dim: rf.Dim,
) -> Tuple[rf.Tensor, rf.Tensor, rf.Dim, rf.Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from returnn.config import get_global_config

    config = get_global_config()

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    beam_size = config.int("beam_size", 12)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = rf.Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, Vocab
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    if beam_size <= model.wb_target_dim.get_dim_value():
        label_log_prob_pre_filter, (backrefs_pre_filter,), pre_filter_beam_dim = rf.top_k(
            label_log_prob, k_dim=rf.Dim(beam_size, name="pre-filter-beam"), axis=[model.wb_target_dim]
        )  # seq_log_prob, backrefs_global: Batch, Spatial, PreFilterBeam. backrefs_pre_filter -> Vocab
    else:
        # If we have a small vocab, there is nothing to filter
        label_log_prob_pre_filter = label_log_prob
        backrefs_pre_filter = rf.expand_dim(
            rf.range_over_dim(model.wb_target_dim), enc_spatial_dim
        )  # Spatial, Vocab -> Vocab

        pre_filter_beam_dim = model.wb_target_dim

    label_log_prob_pre_filter_ta = TensorArray.unstack(
        label_log_prob_pre_filter, axis=enc_spatial_dim
    )  # t -> Batch, PreFilterBeam
    backrefs_pre_filter_ta = TensorArray.unstack(backrefs_pre_filter, axis=enc_spatial_dim)  # t -> Batch, PreFilterBeam

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets = []
    seq_backrefs = []
    for t in range(max_seq_len):
        # Filter out finished beams
        seq_log_prob = seq_log_prob + label_log_prob_pre_filter_ta[t]  # Batch, InBeam, PreFilterBeam
        if beam_size <= beam_dim.get_dim_value() * pre_filter_beam_dim.get_dim_value():
            seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
                seq_log_prob, k_dim=rf.Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, pre_filter_beam_dim]
            )  # seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> PreFilterBeam.
        else:
            # just use the smaller beam size here
            seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
                seq_log_prob,
                k_dim=rf.Dim(beam_dim.get_dim_value() * pre_filter_beam_dim.get_dim_value(), name=f"dec-step{t}-beam"),
                axis=[beam_dim, pre_filter_beam_dim],
            )
        target = rf.gather(backrefs_pre_filter_ta[t], indices=target)  # Batch, Beam -> Vocab
        seq_targets.append(target)
        seq_backrefs.append(backrefs)

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = enc_spatial_dim
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


model_recog_fix_small_vocab: RecogDef[Model]
model_recog_fix_small_vocab.output_with_beam = True
model_recog_fix_small_vocab.output_blank_label = "<blank>"
model_recog_fix_small_vocab.batch_size_dependent = False


def make_asr_ctc_model_recog_args(
    *,
    vocab: str,
    ctc_model: Optional[ModelWithCheckpoint] = None,
    register_outputs: bool = True,
    librispeech_opts: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> AsrModelRecogArgs:
    from sisyphus.hash import short_hash
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
        _ctc_model_def_blank_idx,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import (
        ctc_model_rescore,
        get_ctc_prior_probs,
    )
    from i6_experiments.users.zeyer.datasets.utils.vocab import (
        ExtractVocabLabelsJob,
        ExtractVocabSpecialLabelsJob,
        ExtendVocabLabelsByNewLabelJob,
    )
    from i6_experiments.users.zeyer.decoding.prior_rescoring import Prior, PriorRemoveLabelRenormJob
    from i6_experiments.users.zeyer.datasets.librispeech import get_vocab_by_str, get_librispeech_task_raw_v2
    from .ctc import sis_get_model as sis_get_ctc_model
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import model_recog as ctc_model_recog

    _sis_setup_global_prefix()

    task = get_librispeech_task_raw_v2(vocab=vocab, **(librispeech_opts or {}))
    ctc_model_name_prefix = f"ctc-{vocab}" if not ctc_model else f"ctc-{short_hash(ctc_model.checkpoint.path)}-{vocab}"
    ctc_model: ModelWithCheckpoint = ctc_model or sis_get_ctc_model(vocab=vocab)

    vocab_is_chars = vocab in {"char", "utf8"}
    vocab_ = get_vocab_by_str(vocab)
    orig_vocab_opts = vocab_.get_opts()
    vocab_file = ExtractVocabLabelsJob(orig_vocab_opts).out_vocab

    vocab_opts_file = ExtractVocabSpecialLabelsJob(orig_vocab_opts).out_vocab_special_labels_dict

    vocab_w_blank_file = ExtendVocabLabelsByNewLabelJob(
        vocab=vocab_file, new_label=ctc_model_recog.output_blank_label, new_label_idx=_ctc_model_def_blank_idx
    ).out_vocab

    prior = get_ctc_prior_probs(ctc_model, task.train_dataset.copy_train_as_static())

    log_prior_wo_blank = PriorRemoveLabelRenormJob(
        prior_file=prior,
        prior_type="prob",
        vocab=vocab_w_blank_file,
        remove_label=ctc_model_recog.output_blank_label,
        out_prior_type="log_prob",
    ).out_prior

    if register_outputs:
        tk.register_output(f"{_sis_prefix}/vocab/{vocab}.txt.gz", vocab_file)
        tk.register_output(f"{_sis_prefix}/vocab/{vocab}_opts.py", vocab_opts_file)
        tk.register_output(f"{_sis_prefix}/vocab/{vocab}_w_blank.txt.gz", vocab_w_blank_file)
        tk.register_output(f"{_sis_prefix}/vocab/{ctc_model_name_prefix}-prior.txt", prior)
        tk.register_output(f"{_sis_prefix}/vocab/{ctc_model_name_prefix}-log_prior_wo_blank.txt", log_prior_wo_blank)

    # To allow those parameters to be overwritten...
    kwargs = dict_update_deep(
        dict(
            asr_beam_size=64,  # for the individual beam search
            dlm_beam_size=64,  # for the individual beam search
            first_pass_recog_beam_size=128,
        ),
        kwargs,
    )

    return AsrModelRecogArgs(
        task=task,
        model=ctc_model,
        recog_def=ctc_model_recog,
        rescore_def=ctc_model_rescore,
        labelwise_prior=Prior(
            file=log_prior_wo_blank, type="log_prob", vocab=vocab_file, vocab_is_chars=vocab_is_chars
        ),
        vocab_num_labels=vocab_.get_num_classes(),
        vocab_file=vocab_file,
        vocab_opts_file=vocab_opts_file,
        vocab_is_chars=vocab_is_chars,
        orig_vocab_opts=orig_vocab_opts,
        **kwargs,
    )


def recog_model_ext(
    *,
    dlm_task: Task,
    asr_model: AsrModelRecogArgs,
    dlm: ModelWithCheckpoint,
    prefix: str,
    eval_dataset_names: Optional[Sequence[str]] = None,
):
    from i6_experiments.users.zeyer.decoding.concat_hyps import SearchConcatHypsJob
    from i6_experiments.users.zeyer.decoding.rescoring import rescore, combine_scores
    from i6_core.returnn.search import SearchTakeBestJob
    from i6_experiments.users.zeyer.decoding.lm_rescoring import prior_score
    from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob
    from i6_experiments.users.zeyer.datasets.utils.sclite_generic_score import sclite_score_recog_out_to_ref
    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
    from ..recog.ctc_with_dlm import get_ctc_with_dlm_and_labelwise_prior
    from ..recog.dlm_sum import ctc_model_with_dlm_sum_recog

    if not asr_model.dlm_sum_with_prior:
        # We ignore dlm_sum_with_prior, just use with_prior
        assert not asr_model.with_prior

    # Recog pipeline:
    #   - get N-best list from DLM, N-best list from ASR, combine both
    #   - score with DLM, ASR, maybe prior/ILM
    #   - now new job for finding best scales
    #   - opt: do final first-pass decoding with best scales

    asr_task: Task = asr_model.task
    vocab_file = asr_model.vocab_file
    vocab_opts_file = asr_model.vocab_opts_file

    # for the DLM-recog:
    prior_scale: Optional[tk.Variable] = None
    lm_scale: Optional[tk.Variable] = None

    # for LM-rescore like
    asrbeam_prior_scale: Optional[tk.Variable] = None
    asrbeam_lm_scale: Optional[tk.Variable] = None

    # for the DLM-sum recog:
    dlm_sum_prior_scale: Optional[tk.Variable] = None
    dlm_sum_lm_scale: Optional[tk.Variable] = None
    dlm_sum_ctc_scale: Optional[Union[tk.Variable, float]] = None

    outputs = {}
    dlm_recog_outputs = {}  # ASR hyps beam + DLM hyps beam (from greedy ASR hyp), then rescored with only DLM
    dlm_only_errors = {}
    lm_rescore_recog_outputs = {}  # ASR hyps beam, rescored with DLM (so like LM rescoring)
    lm_rescore_errors = {}
    dlm_greedy_recog_outputs = {}  # DLM greedy decoding (from greedy ASR hyp)
    dlm_greedy_errors = {}
    dsr_errors = {}
    dlm_sum_recog_outputs = {}  # partial sum
    dlm_sum_recog_sum_only_outputs = {}  # partial sum, but only the sum (no prior, no separate ASR score)

    dlm_recog_impl = model_recog
    dlm_rescore_def_impl = dlm_rescore_def

    if asr_model.with_auxiliary_encoder_loss:  # joint...
        # from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
        #     model_recog_with_recomb,
        # )
        from ..recog.auxiliary_encoder_loss import (
            dlm_with_ctc_recomb_rescore_def,
            aux_aed_dlm_timesync_recog_recomb_autoscale,
            model_recog_with_recomb_labelsync,
        )

        dlm_recog_impl = model_recog_with_recomb_labelsync
        dlm_rescore_def_impl = dlm_with_ctc_recomb_rescore_def
        dlm = aux_aed_dlm_timesync_recog_recomb_autoscale(dlm=dlm, task=dlm_task, prefix=f"{prefix}/dlm-aux-autoscale")

        asr_model.dlm_rescore_extra_config = asr_model.dlm_rescore_extra_config or {}
        asr_model.dlm_rescore_extra_config["version"] = 2
        assert not asr_model.with_dlm_sum, "Not implemented..."

    for dataset_name in [None] + (
        list(eval_dataset_names) if eval_dataset_names is not None else list(asr_task.eval_datasets.keys())
    ):
        asr_dataset = asr_task.eval_datasets[dataset_name] if dataset_name else asr_task.dev_dataset
        dlm_dataset = dlm_task.eval_datasets[dataset_name] if dataset_name else dlm_task.dev_dataset

        # Note: recog_post_proc_funcs is not set here in search_dataset, i.e. original labels are kept.
        # Except of CTC/blank: This will apply ctc_alignment_to_label_seq when blank is set.

        asr_recog_out = search_dataset(
            dataset=asr_dataset,
            model=asr_model.model,
            recog_def=asr_model.recog_def,  # i.e. model_recog in i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc
            config={
                "batch_size": 10_000 * asr_model.model.definition.batch_size_factor,
                "beam_size": asr_model.asr_beam_size,
                "behavior_version": 24,  # consistent, independent of batch size
            },
            search_alias_name=f"{prefix}/recog-ext/asr/{dataset_name or 'dev-tune-scales'}",
            keep_beam=True,
        )
        if asr_model.vocab_is_chars and asr_model.recog_def.output_blank_label:
            # In that case, the recog output is somewhat broken, looks like: "A T  T H I S  T U R N I N G  ..."
            # We will use rescore_vocab_opts = asr_model.orig_vocab_opts below,
            # and so we expect normal raw text.
            # The recog_post_proc_funcs should handle this
            # (at least currently; that should use the librispeech._char_to_words; see that func for further details).
            for f in dlm_task.recog_post_proc_funcs:
                asr_recog_out = f(asr_recog_out)

        dlm_recog_out = search_dataset(
            dataset=dlm_dataset,
            model=dlm,
            recog_def=dlm_recog_impl,
            config={
                "batch_size": 1_000,
                "beam_size": asr_model.dlm_beam_size,
                "behavior_version": 24,
                **({"temperature": asr_model.dlm_temperature} if asr_model.dlm_temperature != 1.0 else {}),
                **(asr_model.recog_extra_args or {}),
            },
            search_alias_name=f"{prefix}/recog-ext/dlm/{dataset_name or 'dev-tune-scales'}",
            keep_beam=True,
        )

        if asr_model.dlm_beam_size == 1 and dataset_name:
            dlm_greedy_recog_out = RecogOutput(
                output=SearchTakeBestJob(dlm_recog_out.output, output_gzip=True).out_best_search_results
            )
        elif asr_model.with_dlm_greedy_recog and dataset_name:
            dlm_greedy_recog_out = search_dataset(
                dataset=dlm_dataset,
                model=dlm,
                recog_def=dlm_recog_impl,
                config={
                    "batch_size": 1_000,
                    "beam_size": 1,
                    "behavior_version": 24,
                    **(asr_model.recog_extra_args or {}),
                    # no dlm temperature here, doesnt do anything for greedy decoding
                },
                search_alias_name=f"{prefix}/recog-ext/dlm/greedy-{dataset_name}",
                keep_beam=False,
            )
        else:
            dlm_greedy_recog_out = None

        # Both are on label level, potential blanks are already removed.

        # Combine hyps.
        if asr_model.concat_asr_dlm_beams:
            recog_out: RecogOutput = RecogOutput(
                output=SearchConcatHypsJob([asr_recog_out.output, dlm_recog_out.output]).out_search_results
            )
        else:
            recog_out = dlm_recog_out

        if asr_model.vocab_is_chars:
            rescore_vocab_opts = asr_model.orig_vocab_opts
        else:
            rescore_vocab_opts = {
                "class": "Vocabulary",
                "vocab_file": vocab_file,
                "special_symbols_via_file": vocab_opts_file,
            }

        def thrice_rescore(recog_result: RecogOutput):
            # Rescore with the three models: ASR, ASR prior, DLM.
            asr_recog_out = rescore(
                recog_output=recog_result,
                dataset=asr_dataset,  # needed for audio input
                model=asr_model.model,
                vocab_opts=rescore_vocab_opts,
                rescore_def=asr_model.rescore_def,  # e.g. ctc_model_rescore
            )
            if asr_model.with_prior:
                asr_prior_recog_out = prior_score(recog_result, prior=asr_model.labelwise_prior)
            else:
                asr_prior_recog_out = None
            rescore_cfg = asr_model.dlm_rescore_extra_config
            if asr_model.dlm_temperature != 1.0:
                rescore_cfg = {"temperature": asr_model.dlm_temperature, **(rescore_cfg or {})}
            dlm_recog_out = rescore(
                recog_output=recog_result,
                dataset=dlm_dataset,  # needed for greedy hyp from ASR model
                model=dlm,
                vocab_opts=rescore_vocab_opts,
                rescore_def=dlm_rescore_def_impl,
                config=rescore_cfg,
            )
            return asr_recog_out, asr_prior_recog_out, dlm_recog_out

        # try LM-rescore like, so only on ASR beam
        asrbeam_asr_recog_out, asrbeam_asr_prior_recog_out, asrbeam_dlm_recog_out = thrice_rescore(asr_recog_out)

        # Rescore with the three models: ASR, ASR prior, DLM.
        asr_recog_out, asr_prior_recog_out, dlm_recog_out = thrice_rescore(recog_out)

        ref = RecogOutput(
            output=ReturnnDatasetToTextDictJob(
                returnn_dataset=dlm_dataset.get_main_dataset(), data_key=dlm_dataset.get_default_target()
            ).out_txt
        )
        ref_dlm_scores = greedy_scores = None
        ref_asr_recog_out = ref_prior_recog_out = None
        if asr_model.calculate_search_errors:
            from i6_experiments.users.dorian_koch.jobs.scores import TextDictToScoresTextDictJob

            ref_asr_recog_out, ref_prior_recog_out, ref_dlm_scores = thrice_rescore(
                TextDictToScoresTextDictJob(text_dict=ref).get_recog_output()
            )

            if dlm_greedy_recog_out:
                greedy_scores = rescore(
                    recog_output=TextDictToScoresTextDictJob(text_dict=dlm_greedy_recog_out).get_recog_output(),
                    dataset=dlm_dataset,  # needed for greedy hyp from ASR model
                    model=dlm,
                    vocab_opts=rescore_vocab_opts,
                    rescore_def=dlm_rescore_def_impl,
                    config=asr_model.dlm_rescore_extra_config,
                )

        # Post-processing, for example BPE to words.
        # TODO in case of chars, should already be correct? maybe doesn't matter though...
        for ff in dlm_task.recog_post_proc_funcs:
            f = lambda x: x if x is None else ff(x)  # prior is None if not with_prior
            asr_recog_out = f(asr_recog_out)
            asr_prior_recog_out = f(asr_prior_recog_out)
            dlm_recog_out = f(dlm_recog_out)
            if dlm_greedy_recog_out:
                dlm_greedy_recog_out = f(dlm_greedy_recog_out)
            asrbeam_dlm_recog_out = f(asrbeam_dlm_recog_out)
            asrbeam_asr_recog_out = f(asrbeam_asr_recog_out)
            asrbeam_asr_prior_recog_out = f(asrbeam_asr_prior_recog_out)
            ref = f(ref)
            if asr_model.calculate_search_errors:
                ref_dlm_scores = f(ref_dlm_scores)
                ref_asr_recog_out = f(ref_asr_recog_out)
                ref_prior_recog_out = f(ref_prior_recog_out)
                if dlm_greedy_recog_out:
                    greedy_scores = f(greedy_scores)

        if dataset_name:
            # Just from the DLM hyps, take best and evaluate (calc WER).
            # We anyway already have the hyps, and this is good for reference.
            dlm_best_recog_out = RecogOutput(
                output=SearchTakeBestJob(dlm_recog_out.output, output_gzip=True).out_best_search_results
            )
            score_out = sclite_score_recog_out_to_ref(dlm_best_recog_out, ref=ref, corpus_name=dataset_name)
            dlm_recog_outputs[dataset_name] = score_out

            if asr_model.with_dlm_greedy_recog:
                greedy_score_out = sclite_score_recog_out_to_ref(
                    dlm_greedy_recog_out, ref=ref, corpus_name=dataset_name
                )
                dlm_greedy_recog_outputs[dataset_name] = greedy_score_out

            if asr_model.calculate_search_errors:
                from i6_experiments.users.dorian_koch.jobs.scores import CalcSearchErrors

                only_dlm_search_errors = CalcSearchErrors(ref_scores=ref_dlm_scores, hyp_scores=dlm_recog_out)
                dlm_only_errors[dataset_name] = only_dlm_search_errors

                if dlm_greedy_recog_out:
                    greedy_search_errors = CalcSearchErrors(ref_scores=ref_dlm_scores, hyp_scores=greedy_scores)
                    dlm_greedy_errors[dataset_name] = greedy_search_errors

        if dataset_name is None:
            if asr_model.with_prior:
                opt_scales_job = ScaleTuningJob(
                    scores={
                        "am": asr_recog_out.output,
                        "prior": asr_prior_recog_out.output,
                        "lm": dlm_recog_out.output,
                    },
                    ref=ref.output,
                    fixed_scales={"am": 1.0},
                    negative_scales={"prior"},
                    scale_relative_to={"prior": "lm"},
                    max_scales={"lm": asr_model.max_dlm_scale, "prior": 1.0},
                    evaluation="edit_distance",
                )
                tk.register_output(f"{prefix}/recog-ext/opt-grid-plot.pdf", opt_scales_job.out_grid_plot)
            else:
                opt_scales_job = ScaleTuningJob(
                    scores={"am": asr_recog_out.output, "lm": dlm_recog_out.output},
                    ref=ref.output,
                    fixed_scales={"am": 1.0},
                    max_scales={"lm": asr_model.max_dlm_scale},
                    evaluation="edit_distance",
                )
            opt_scales_job.rqmt["engine"] = "short"  # should be fine
            tk.register_output(f"{prefix}/recog-ext/opt-real-scales", opt_scales_job.out_real_scales)
            tk.register_output(f"{prefix}/recog-ext/opt-rel-scales", opt_scales_job.out_scales)

            # We use the real scales.
            prior_scale = opt_scales_job.out_real_scale_per_name["prior"] if asr_model.with_prior else None
            lm_scale = opt_scales_job.out_real_scale_per_name["lm"]

        else:
            if asr_model.with_prior:
                recog_out_scores = combine_scores(
                    [(1.0, asr_recog_out), (prior_scale, asr_prior_recog_out), (lm_scale, dlm_recog_out)]
                )
            else:
                recog_out_scores = combine_scores([(1.0, asr_recog_out), (lm_scale, dlm_recog_out)])

            # Take best.
            recog_out = RecogOutput(
                output=SearchTakeBestJob(recog_out_scores.output, output_gzip=True).out_best_search_results
            )

            # Evaluate (calc WER).
            score_out = sclite_score_recog_out_to_ref(recog_out, ref=ref, corpus_name=dataset_name)
            # outputs[f"{dataset_name}:rescore"] = score_out
            outputs[dataset_name] = score_out

            if asr_model.calculate_search_errors:
                from i6_experiments.users.dorian_koch.jobs.scores import CalcSearchErrors

                if asr_model.with_prior:
                    dsr_ref_recog_out = combine_scores(
                        [
                            (1.0, ref_asr_recog_out),
                            (prior_scale, ref_prior_recog_out),
                            (lm_scale, ref_dlm_scores),
                        ]
                    )
                else:
                    dsr_ref_recog_out = combine_scores(
                        [
                            (1.0, ref_asr_recog_out),
                            (lm_scale, ref_dlm_scores),
                        ]
                    )

                search_errors = CalcSearchErrors(ref_scores=dsr_ref_recog_out, hyp_scores=recog_out_scores)
                dsr_errors[dataset_name] = search_errors

        if asr_model.with_lm_rescore:
            if dataset_name is None:
                if asr_model.with_prior:
                    opt_scales_job = ScaleTuningJob(
                        scores={
                            "am": asrbeam_asr_recog_out.output,
                            "prior": asrbeam_asr_prior_recog_out.output,
                            "lm": asrbeam_dlm_recog_out.output,
                        },
                        ref=ref.output,
                        fixed_scales={"am": 1.0},
                        negative_scales={"prior"},
                        scale_relative_to={"prior": "lm"},
                        max_scales={"lm": asr_model.max_dlm_scale, "prior": 1.0},
                        evaluation="edit_distance",
                    )
                    tk.register_output(f"{prefix}/recog-ext/asrbeam_opt-grid-plot.pdf", opt_scales_job.out_grid_plot)
                else:
                    opt_scales_job = ScaleTuningJob(
                        scores={
                            "am": asrbeam_asr_recog_out.output,
                            "lm": asrbeam_dlm_recog_out.output,
                        },
                        ref=ref.output,
                        fixed_scales={"am": 1.0},
                        max_scales={"lm": asr_model.max_dlm_scale},
                        evaluation="edit_distance",
                    )
                tk.register_output(f"{prefix}/recog-ext/asrbeam_opt-real-scales", opt_scales_job.out_real_scales)
                tk.register_output(f"{prefix}/recog-ext/asrbeam_opt-rel-scales", opt_scales_job.out_scales)

                # We use the real scales.
                asrbeam_prior_scale = opt_scales_job.out_real_scale_per_name["prior"] if asr_model.with_prior else None
                asrbeam_lm_scale = opt_scales_job.out_real_scale_per_name["lm"]

            else:
                if asr_model.with_prior:
                    recog_out_scores = combine_scores(
                        [
                            (1.0, asrbeam_asr_recog_out),
                            (asrbeam_prior_scale, asrbeam_asr_prior_recog_out),
                            (asrbeam_lm_scale, asrbeam_dlm_recog_out),
                        ]
                    )
                else:
                    recog_out_scores = combine_scores(
                        [
                            (1.0, asrbeam_asr_recog_out),
                            (asrbeam_lm_scale, asrbeam_dlm_recog_out),
                        ]
                    )

                # Take best.
                recog_out = RecogOutput(
                    output=SearchTakeBestJob(recog_out_scores.output, output_gzip=True).out_best_search_results
                )

                # Evaluate (calc WER).
                score_out = sclite_score_recog_out_to_ref(recog_out, ref=ref, corpus_name=dataset_name)
                lm_rescore_recog_outputs[dataset_name] = score_out
                if asr_model.calculate_search_errors:
                    from i6_experiments.users.dorian_koch.jobs.scores import CalcSearchErrors

                    if asr_model.with_prior:
                        lmrescore_ref_recog_out = combine_scores(
                            [
                                (1.0, ref_asr_recog_out),
                                (asrbeam_prior_scale, ref_prior_recog_out),
                                (asrbeam_lm_scale, ref_dlm_scores),
                            ]
                        )
                    else:
                        lmrescore_ref_recog_out = combine_scores(
                            [
                                (1.0, ref_asr_recog_out),
                                (asrbeam_lm_scale, ref_dlm_scores),
                            ]
                        )

                    search_errors = CalcSearchErrors(ref_scores=lmrescore_ref_recog_out, hyp_scores=recog_out_scores)
                    lm_rescore_errors[dataset_name] = search_errors

        # Only use this to tune / play around with search params.
        # _recog_hyper_param_tuning_exps(
        #     dlm_task=dlm_task,
        #     asr_model=asr_model,
        #     dlm=dlm,
        #     prefix=prefix,
        #     prior_scale=prior_scale,
        #     lm_scale=lm_scale,
        #     dataset_name=dataset_name,
        #     asr_dataset=asr_dataset,
        #     ref=ref,
        # )

        # Do DLM-sum recog.
        # Originally without prior, LM scale 1 for the DLM sum recog,
        # but can also be with prior.
        if asr_model.with_dlm_sum:
            if dataset_name is None:  # initial scales. will be tuned below
                if asr_model.with_prior:
                    dlm_sum_ctc_scale = 1.0
                    dlm_sum_prior_scale = prior_scale
                    dlm_sum_lm_scale = lm_scale
                else:
                    # We use the optimized LM scale from before as starting point.
                    # This works already good. But then we optimize it further for the new DLM-sum recog.
                    dlm_sum_ctc_scale = lm_scale ** (-1)  # will be further optimized below

            if asr_model.with_prior:
                model_with_dlm = get_ctc_with_dlm_and_labelwise_prior(
                    ctc_model=asr_model.model,
                    language_model=dlm,
                    lm_scale=dlm_sum_lm_scale,
                    prior=asr_model.labelwise_prior.file,
                    prior_type=asr_model.labelwise_prior.type,
                    prior_scale=dlm_sum_prior_scale * (-1),
                )
                model_with_dlm_no_prior = get_ctc_with_dlm_and_labelwise_prior(
                    ctc_model=asr_model.model, language_model=dlm, lm_scale=dlm_sum_lm_scale
                )
            else:
                model_with_dlm = get_ctc_with_dlm_and_labelwise_prior(
                    ctc_model=asr_model.model, language_model=dlm, lm_scale=1.0
                )
                model_with_dlm_no_prior = model_with_dlm

            if dataset_name is None:  # scale tuning
                if (asr_model.dlm_sum_extra_args or {}).get("beam_size", 2) == 1:
                    # beam size 1, scale tuning is pointless. just use the dsr scales
                    continue
                dlm_sum_config = {
                    "ctc_num_hyps": 10,
                    "ctc_soft_collapse_threshold": 0.8,
                    "ctc_soft_collapse_reduce_type": "max_renorm",
                    "ctc_beam_size": min(128, asr_model.vocab_num_labels),
                    "beam_size": min(12, asr_model.vocab_num_labels),
                    "ctc_prefix_score_scale": dlm_sum_ctc_scale,
                    "length_normalization_exponent": 0.0,
                    "dlm_max_seq_len": "ctc",
                    "batch_size": int(
                        20_000
                        * asr_model.model.definition.batch_size_factor
                        * asr_model.dlm_sum_extra_batch_size_factor
                    ),
                    # get the score from DLM alone
                    **(
                        {"return_only_dlm_score": True}
                        if asr_model.with_prior
                        else {"ctc_final_prefix_score_scale": 0.0}  # keep consistent hash
                    ),
                    "recog_version": 4,  # this was 3 before for dlm_sum_with_prior == False... but this would crash the recog?
                    "behavior_version": 24,  # consistent, independent of batch size
                    "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                    **({"dlm_temperature": asr_model.dlm_temperature} if asr_model.dlm_temperature != 1.0 else {}),
                }
                dlm_sum_config = dict_update_deep(dlm_sum_config, asr_model.recog_extra_args)
                dlm_sum_config = dict_update_deep(dlm_sum_config, asr_model.dlm_sum_extra_args)
                dlm_sum_config = dict_update_deep(dlm_sum_config, asr_model.dlm_sum_tune_scales_extra_args)
                dlm_recog_out = search_dataset(
                    dataset=asr_dataset,
                    model=model_with_dlm,
                    recog_def=ctc_model_with_dlm_sum_recog,
                    config=dlm_sum_config,
                    search_alias_name=f"{prefix}/recog-ext/dlm/dev-tune-scales-firstpass-dlm-sum",
                    search_rqmt={"time": 5},
                    keep_beam=True,
                )

                # Rescore with ASR (CTC).
                asr_recog_out = rescore(
                    recog_output=dlm_recog_out,
                    dataset=asr_dataset,  # needed for audio input
                    model=asr_model.model,
                    vocab_opts=rescore_vocab_opts,
                    rescore_def=asr_model.rescore_def,  # e.g. ctc_model_rescore
                )
                if asr_model.with_prior:
                    asr_prior_recog_out = prior_score(dlm_recog_out, prior=asr_model.labelwise_prior)

                # Post-processing, for example BPE to words.
                for f in dlm_task.recog_post_proc_funcs:
                    dlm_recog_out = f(dlm_recog_out)
                    asr_recog_out = f(asr_recog_out)
                    if asr_model.with_prior:
                        asr_prior_recog_out = f(asr_prior_recog_out)

                if asr_model.with_prior:
                    opt_scales_job = ScaleTuningJob(
                        scores={
                            "am": asr_recog_out.output,
                            "prior": asr_prior_recog_out.output,
                            "lm": dlm_recog_out.output,
                        },
                        ref=ref.output,
                        fixed_scales={"am": 1.0},
                        negative_scales={"prior"},
                        scale_relative_to={"prior": "lm"},
                        max_scales={"lm": asr_model.max_dlm_scale, "prior": 1.0},
                        evaluation="edit_distance",
                    )
                else:
                    opt_scales_job = ScaleTuningJob(
                        scores={"dlm": dlm_recog_out.output, "ctc": asr_recog_out.output},
                        ref=ref.output,
                        fixed_scales={"dlm": 1.0},
                        max_scales={"ctc": 1.0},
                        evaluation="edit_distance",
                    )
                opt_scales_job.add_alias(f"{prefix}/recog-ext/opt-dlm-sum-scales")
                tk.register_output(f"{prefix}/recog-ext/opt-dlm-sum-real-scales", opt_scales_job.out_real_scales)
                tk.register_output(f"{prefix}/recog-ext/opt-dlm-sum-rel-scales", opt_scales_job.out_scales)
                if asr_model.with_prior:
                    tk.register_output(f"{prefix}/recog-ext/opt-dlm-sum-grid-plot.pdf", opt_scales_job.out_grid_plot)
                    dlm_sum_prior_scale = opt_scales_job.out_real_scale_per_name["prior"]
                    dlm_sum_lm_scale = opt_scales_job.out_real_scale_per_name["lm"]
                else:
                    dlm_sum_ctc_scale = opt_scales_job.out_real_scale_per_name["ctc"]

            else:  # have dataset_name
                dlm_sum_config = {
                    "ctc_num_hyps": 20,
                    "ctc_soft_collapse_threshold": 0.9,
                    "ctc_soft_collapse_reduce_type": "max_renorm",
                    "ctc_beam_size": min(128, asr_model.vocab_num_labels),
                    "beam_size": min(12, asr_model.vocab_num_labels),
                    "ctc_prefix_score_scale": dlm_sum_ctc_scale,
                    "length_normalization_exponent": 0.0,
                    "dlm_max_seq_len": "ctc",
                    "batch_size": int(
                        10_000
                        * asr_model.model.definition.batch_size_factor
                        * asr_model.dlm_sum_extra_batch_size_factor
                    ),
                    "recog_version": 3 if asr_model.with_prior else 4,
                    "behavior_version": 24,  # consistent, independent of batch size
                    "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                    **({"dlm_temperature": asr_model.dlm_temperature} if asr_model.dlm_temperature != 1.0 else {}),
                }
                dlm_sum_config = dict_update_deep(dlm_sum_config, asr_model.recog_extra_args)
                dlm_sum_config = dict_update_deep(dlm_sum_config, asr_model.dlm_sum_extra_args)

                dlm_recog_out = search_dataset(
                    dataset=asr_dataset,
                    model=model_with_dlm,
                    recog_def=ctc_model_with_dlm_sum_recog,
                    config=dlm_sum_config,
                    search_alias_name=f"{prefix}/recog-ext/dlm/{dataset_name}-firstpass-dlm-sum",
                    search_rqmt={"time": 5},
                )
                dlm_recog_out_no_ctc_final_prefix_score = search_dataset(
                    dataset=asr_dataset,
                    model=model_with_dlm_no_prior,
                    recog_def=ctc_model_with_dlm_sum_recog,
                    config=dict_update_deep(dlm_sum_config, {"ctc_final_prefix_score_scale": 0.0, "recog_version": 4}),
                    search_alias_name=f"{prefix}/recog-ext/dlm/{dataset_name}-firstpass-dlm-sum-no-ctc-final-prefix-score",
                    search_rqmt={"time": 5},
                )
                # Post-processing, for example BPE to words.
                for f in dlm_task.recog_post_proc_funcs:
                    dlm_recog_out = f(dlm_recog_out)
                    dlm_recog_out_no_ctc_final_prefix_score = f(dlm_recog_out_no_ctc_final_prefix_score)
                # Evaluate (calc WER).
                score_out = sclite_score_recog_out_to_ref(dlm_recog_out, ref=ref, corpus_name=dataset_name)
                dlm_sum_recog_outputs[dataset_name] = score_out

                if asr_model.with_dlm_sum_simple:
                    score_out = sclite_score_recog_out_to_ref(
                        dlm_recog_out_no_ctc_final_prefix_score, ref=ref, corpus_name=dataset_name
                    )

                    dlm_sum_recog_sum_only_outputs[dataset_name] = score_out

    # Collect all results.
    res = dlm_task.collect_score_results_func(outputs)
    tk.register_output(f"{prefix}/recog-ext/score_results.txt", res.output)

    res = dlm_task.collect_score_results_func(dlm_recog_outputs)
    tk.register_output(f"{prefix}/recog-ext/dlm_only_score_results.txt", res.output)

    if dlm_only_errors:
        res = dlm_task.collect_score_results_func(
            {ds_name: job.get_search_error(ds_name) for ds_name, job in dlm_only_errors.items()}
        )
        tk.register_output(f"{prefix}/recog-ext/search_errors/dlm_only.txt", res.output)
        res = dlm_task.collect_score_results_func(
            {ds_name: job.get_model_error(ds_name) for ds_name, job in dlm_only_errors.items()}
        )
        tk.register_output(f"{prefix}/recog-ext/model_errors/dlm_only.txt", res.output)
        res = dlm_task.collect_score_results_func(
            {ds_name: job.get_oracle_wer(ds_name) for ds_name, job in dlm_only_errors.items()}
        )
        tk.register_output(f"{prefix}/recog-ext/oracle_wers/dlm_only.txt", res.output)

    if dlm_greedy_recog_outputs:
        res = dlm_task.collect_score_results_func(dlm_greedy_recog_outputs)
        tk.register_output(f"{prefix}/recog-ext/dlm_greedy_score_results.txt", res.output)

    if dlm_greedy_errors:
        res = dlm_task.collect_score_results_func(
            {ds_name: job.get_search_error(ds_name) for ds_name, job in dlm_greedy_errors.items()}
        )
        tk.register_output(f"{prefix}/recog-ext/search_errors/dlm_greedy.txt", res.output)
        res = dlm_task.collect_score_results_func(
            {ds_name: job.get_model_error(ds_name) for ds_name, job in dlm_greedy_errors.items()}
        )
        tk.register_output(f"{prefix}/recog-ext/model_errors/dlm_greedy.txt", res.output)
        res = dlm_task.collect_score_results_func(
            {ds_name: job.get_oracle_wer(ds_name) for ds_name, job in dlm_greedy_errors.items()}
        )
        tk.register_output(f"{prefix}/recog-ext/oracle_wers/dlm_greedy.txt", res.output)

    if dsr_errors:
        res = dlm_task.collect_score_results_func(
            {ds_name: job.get_search_error(ds_name) for ds_name, job in dsr_errors.items()}
        )
        tk.register_output(f"{prefix}/recog-ext/search_errors/dsr.txt", res.output)
        res = dlm_task.collect_score_results_func(
            {ds_name: job.get_model_error(ds_name) for ds_name, job in dsr_errors.items()}
        )
        tk.register_output(f"{prefix}/recog-ext/model_errors/dsr.txt", res.output)
        res = dlm_task.collect_score_results_func(
            {ds_name: job.get_oracle_wer(ds_name) for ds_name, job in dsr_errors.items()}
        )
        tk.register_output(f"{prefix}/recog-ext/oracle_wers/dsr.txt", res.output)

    if lm_rescore_recog_outputs:
        res = dlm_task.collect_score_results_func(lm_rescore_recog_outputs)
        tk.register_output(f"{prefix}/recog-ext/dlm_lm_rescore_score_results.txt", res.output)

    if lm_rescore_errors:
        res = dlm_task.collect_score_results_func(
            {ds_name: job.get_search_error(ds_name) for ds_name, job in lm_rescore_errors.items()}
        )
        tk.register_output(f"{prefix}/recog-ext/search_errors/dlm_lm_rescore.txt", res.output)
        res = dlm_task.collect_score_results_func(
            {ds_name: job.get_model_error(ds_name) for ds_name, job in lm_rescore_errors.items()}
        )
        tk.register_output(f"{prefix}/recog-ext/model_errors/dlm_lm_rescore.txt", res.output)
        res = dlm_task.collect_score_results_func(
            {ds_name: job.get_oracle_wer(ds_name) for ds_name, job in lm_rescore_errors.items()}
        )
        tk.register_output(f"{prefix}/recog-ext/oracle_wers/dlm_lm_rescore.txt", res.output)

    if dlm_sum_recog_outputs:
        res = dlm_task.collect_score_results_func(dlm_sum_recog_outputs)
        tk.register_output(f"{prefix}/recog-ext/dlm_sum_score_results.txt", res.output)

    if dlm_sum_recog_sum_only_outputs:
        res = dlm_task.collect_score_results_func(dlm_sum_recog_sum_only_outputs)
        tk.register_output(f"{prefix}/recog-ext/dlm_sum_no_final_prefix_score_score_results.txt", res.output)
    return res


def _recog_hyper_param_tuning_exps(
    *,
    dlm_task: Task,
    asr_model: AsrModelRecogArgs,
    dlm: ModelWithCheckpoint,
    prefix: str,
    prior_scale: tk.Variable,
    lm_scale: tk.Variable,
    dataset_name: str,
    asr_dataset: DatasetConfig,
    ref: RecogOutput,
):
    """
    Experiments to tune any of the search settings / hyper params.
    """
    from i6_experiments.users.zeyer.datasets.utils.sclite_generic_score import sclite_score_recog_out_to_ref
    from ..recog.ctc_with_dlm import get_ctc_with_dlm_and_labelwise_prior, ctc_model_with_dlm_recog
    from ..recog.dlm_sum import ctc_model_with_dlm_sum_recog

    # also: /base-genTts-fm-spm10k-ctcDrop05-n1024-decL16-lsh10-epSplit20-keepSubwords-b2k_20k
    # /base-errGen-spm10k-errSample10-ctcDrop05-n1024-decL16-lsh10-epSplit20-keepSubwords-b2k_20k
    use_first_pass_recog = prefix.endswith("/base-spm10k-ls1-epSplit20-b2k_80k") or prefix.endswith(
        "/base-spm10k-ctcDrop05-lsh10-epSplit20-keepSubwords-n1024-decL16-b2k_20k"
    )
    if dataset_name != "dev-other":
        return
    if not use_first_pass_recog:
        return

    # { pushd alias/denoising-lm/error_correction_model/base-spm10k-ctcDrop05-lsh10-epSplit20-keepSubwords-n1024-decL16-b2k_20k/recog-ext/dlm/dev-other-firstpass-sum/; for f in *; do echo $f; grep "elapsed: " $f/log.run.1; cat ~/setups/combined/2021-05-31/output/denoising-lm/error_correction_model/base-spm10k-ctcDrop05-lsh10-epSplit20-keepSubwords-n1024-decL16-b2k_20k/recog-ext/firstpass-sum/$f.txt; done; popd }
    # ~/setups/combined/2021-05-31/alias/denoising-lm/error_correction_model/base-spm10k-ctcDrop05-lsh10-epSplit20-keepSubwords-n1024-decL16-b2k_20k/recog-ext/dlm/dev-other-firstpass-sum
    # ctcNumHyps100_ctcBeamSize100_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.70_beamSize24_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 2:07:21.8642
    # 3.99
    # ctcNumHyps100_ctcBeamSize1024_beamSize24
    # elapsed: 1:54:42.1342
    # 4.43
    # ctcNumHyps100_ctcBeamSize1024_beamSize24_ctcPrefixScoreScale0.05_lenNormExp0.00
    # elapsed: 2:03:51.6715
    # 4.07
    # ctcNumHyps100_ctcBeamSize1024_beamSize24_ctcPrefixScoreScale0.10_lenNormExp0.00
    # elapsed: 2:03:17.0448
    # 3.97
    # ctcNumHyps100_ctcBeamSize1024_beamSize24_ctcPrefixScoreScale0.20_lenNormExp0.00
    # elapsed: 2:00:20.6560
    # 3.96
    # ctcNumHyps100_ctcBeamSize1024_beamSize24_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 2:00:46.3245
    # 3.96
    # ctcNumHyps100_ctcBeamSize1024_beamSize24_ctcPrefixScoreScale0.3949447077409163_lenNormExp0.00
    # elapsed: 1:58:02.1930
    # 3.99
    # ctcNumHyps100_ctcBeamSize1024_beamSize24_ctcPrefixScoreScale0.40_lenNormExp0.00
    # elapsed: 1:58:37.7951
    # 3.99
    # ctcNumHyps100_ctcBeamSize1024_beamSize24_ctcPrefixScoreScale0.50_lenNormExp0.00
    # elapsed: 1:57:37.9858
    # 4.11
    # ctcNumHyps100_ctcBeamSize1024_beamSize24_ctcPrefixScoreScale0.70_lenNormExp0.00
    # elapsed: 1:57:29.3076
    # 4.21
    # ctcNumHyps100_ctcBeamSize1024_beamSize24_ctcPrefixScoreScale1.00_lenNormExp0.00
    # elapsed: 1:56:23.4529
    # 4.37
    # ctcNumHyps100_ctcBeamSize1024_beamSize24_lenNormExp0.00
    # elapsed: 1:53:39.4348
    # 4.31
    # ctcNumHyps100_ctcBeamSize1024_beamSize24_lenNormExp0.00_torchAmpbfloat16
    # elapsed: 1:33:10.1094
    # 4.32
    # ctcNumHyps100_ctcBeamSize1024_beamSize32_lenNormExp0.00_torchAmpbfloat16
    # elapsed: 1:57:46.1322
    # 4.3
    # ctcNumHyps100_ctcSoftCollapseThr0.95_ctcBeamSize1024_beamSize8_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:56:49.8188
    # 3.97
    # ctcNumHyps10_ctcBeamSize1024_beamSize128_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 1:11:25.9035
    # 4.34
    # ctcNumHyps10_ctcBeamSize1024_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 0:33:21.5931
    # 3.99
    # ctcNumHyps10_ctcBeamSize1024_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 0:37:42.5751
    # 3.98
    # ctcNumHyps10_ctcBeamSize1024_beamSize32_ctcPrefixScoreScale0.3949447077409163_lenNormExp0.00
    # elapsed: 0:37:11.0422
    # 3.99
    # ctcNumHyps10_ctcBeamSize1024_beamSize32_ctcPrefixScoreScale0.3949447077409163_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 0:37:58.7743
    # 3.99
    # ctcNumHyps10_ctcBeamSize1024_beamSize32_ctcPrefixScoreScale0.3949447077409163_lenNormExp0.00_dlmMaxSeqLenctc_batchSize800000
    # elapsed: 0:30:31.1041
    # 4.08
    # ctcNumHyps10_ctcBeamSize1024_beamSize4_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 0:32:23.8609
    # 4.0
    # ctcNumHyps10_ctcBeamSize1024_beamSize64_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize800000
    # elapsed: 0:57:57.6012
    # 4.07
    # ctcNumHyps10_ctcBeamSize1024_beamSize64_ctcPrefixScoreScale0.3949447077409163_lenNormExp0.00_dlmMaxSeqLenctc_batchSize800000
    # elapsed: 0:56:56.6111
    # 4.09
    # ctcNumHyps10_ctcSoftCollapseThr0.50_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 0:28:26.3801
    # 4.03
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 0:30:24.8438
    # 3.98
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize1600000
    # elapsed: 0:11:14.7514
    # 4.11
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize1600000_EnvUpdatesPytorchCudaAllocConfexpandable_segments:True
    # elapsed: 0:11:14.7514
    # 4.11
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize2400000
    # elapsed: 0:11:00.8019
    # 4.07
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize2400000_EnvUpdatesPytorchCudaAllocConfexpandable_segments:True
    # elapsed: 0:11:00.8019
    # 4.07
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize3200000
    # elapsed: 0:11:00.8443
    # 4.04
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize3200000_EnvUpdatesPytorchCudaAllocConfexpandable_segments:True
    # elapsed: 0:11:00.8443
    # 4.04
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize4800000
    # elapsed: 0:10:53.0245
    # 4.08
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize4800000_EnvUpdatesPytorchCudaAllocConfexpandable_segments:True
    # elapsed: 0:10:53.0245
    # 4.08
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize6400000
    # elapsed: 0:11:30.3220
    # 4.06
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize800000
    # elapsed: 0:15:15.2336
    # 4.06
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize8000000
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize8000000_EnvUpdatesPytorchCudaAllocConfexpandable_segments:True
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize800000_EnvUpdatesPytorchCudaAllocConfexpandable_segments:True
    # elapsed: 0:15:15.2336
    # 4.06
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize24_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize3200000
    # elapsed: 0:20:37.5571
    # 4.05
    # ctcNumHyps10_ctcSoftCollapseThr0.80_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize24_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize3200000_EnvUpdatesPytorchCudaAllocConfexpandable_segments:True
    # elapsed: 0:20:37.5571
    # ctcNumHyps10_ctcSoftCollapseThr0.90_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 0:28:35.5965
    # 3.99
    # ctcNumHyps10_ctcSoftCollapseThr0.95_ctcBeamSize1024_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 0:28:30.5815
    # 4.0
    # ctcNumHyps10_ctcSoftCollapseThr0.95_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 0:28:09.4051
    # 4.0
    # ctcNumHyps10_ctcSoftCollapseThr0.95_ctcBeamSize128_ctcTopKWRndSamp1.00_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 0:29:18.3344
    # 3.99
    # ctcNumHyps10_ctcSoftCollapseThr0.95_ctcBeamSize128_ctcTopKWRndSamp1.00_ctcTopKWRndSampOptsMinNoiseScale1.00_topP0.90_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 0:31:00.7419
    # 4.0
    # ctcNumHyps10_ctcSoftCollapseThr0.95_ctcSoftCollapseReduceTypemax_renorm_ctcBeamSize128_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 0:28:43.1102
    # 3.99
    # ctcNumHyps10_ctcTopKWRndSamp1.00_ctcBeamSize1024_beamSize64_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize800000
    # elapsed: 2:35:14.4436
    # 4.07
    # ctcNumHyps10_ctcTopKWRndSamp1.00_ctcBeamSize64_beamSize64_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize800000
    # elapsed: 1:01:31.3965
    # 4.07
    # ctcNumHyps10_ctcTopKWRndSamp1.00_ctcTopP0.90_ctcBeamSize1024_beamSize64_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize800000
    # elapsed: 2:13:20.0709
    # 4.02
    # ctcNumHyps10_ctcTopKWRndSamp1.00_ctcTopP0.90_ctcBeamSize64_beamSize64_ctcPrefixScoreScale0.30_lenNormExp0.00_dlmMaxSeqLenctc_batchSize800000
    # elapsed: 0:59:00.4756
    # 4.07
    # ctcNumHyps1_batchSize1600000_lenNormExp0.00
    # elapsed: 0:11:43.8139
    # 4.75
    # ctcNumHyps1_ctcBeamSize1024_beamSize32_ctcPrefixScoreScale0.3949447077409163_lenNormExp0.00
    # elapsed: 0:15:50.3221
    # 4.2
    # ctcNumHyps1_ctcPrefixScoreScale0.3949447077409163_lenNormExp0.00_batchSize800000
    # elapsed: 0:55:05.1842
    # 4.25
    # ctcNumHyps200_ctcBeamSize1024_beamSize12_ctcPrefixScoreScale0.10_lenNormExp0.00
    # elapsed: 1:59:36.7334
    # 3.97
    # ctcNumHyps200_ctcBeamSize1024_beamSize12_ctcPrefixScoreScale0.20_lenNormExp0.00
    # elapsed: 2:00:14.0654
    # 3.97
    # ctcNumHyps200_ctcBeamSize1024_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 2:02:20.4414
    # 3.95
    # ctcNumHyps200_ctcBeamSize1024_beamSize12_ctcPrefixScoreScale0.3949447077409163_lenNormExp0.00
    # elapsed: 1:57:29.5336
    # 3.98
    # ctcNumHyps200_ctcBeamSize1024_beamSize16_lenNormExp0.00_torchAmpbfloat16
    # elapsed: 1:54:38.0285
    # 4.3
    # ctcNumHyps200_ctcBeamSize1024_beamSize1_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:52:46.1933
    # 4.09
    # ctcNumHyps200_ctcBeamSize1024_beamSize2_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:57:48.1362
    # 3.97
    # ctcNumHyps200_ctcBeamSize1024_beamSize4_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 1:04:30.9619
    # 3.97
    # ctcNumHyps200_ctcBeamSize1024_beamSize8_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 1:29:45.3708
    # 3.93
    # ctcNumHyps200_ctcSoftCollapseThr0.95_ctcBeamSize1024_beamSize8_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 1:24:36.4202
    # 3.96
    # ctcNumHyps20_ctcBeamSize1024_beamSize32_ctcPrefixScoreScale0.10_lenNormExp0.00
    # elapsed: 0:54:15.1804
    # 4.04
    # ctcNumHyps20_ctcBeamSize1024_beamSize32_ctcPrefixScoreScale0.20_lenNormExp0.00
    # elapsed: 0:53:16.3512
    # 3.99
    # ctcNumHyps20_ctcBeamSize1024_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:56:31.0283
    # 3.97
    # ctcNumHyps20_ctcBeamSize1024_beamSize32_ctcPrefixScoreScale0.3949447077409163_lenNormExp0.00
    # elapsed: 0:53:55.1080
    # 4.01
    # ctcNumHyps20_ctcBeamSize1024_beamSize48_lenNormExp0.00_torchAmpbfloat16
    # elapsed: 0:58:57.5927
    # 4.36
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.50_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.90_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:54:15.1975
    # 4.05
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.70_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.90_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:52:00.9885
    # 4.02
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.80_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.90_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:51:47.4355
    # 3.99
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.90_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.90_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:54:19.0311
    # 4.01
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_beamSize16_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:50:15.2708
    # 3.99
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_beamSize16_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:47:04.2285
    # 3.98
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopKWRndSampOptsMaxNoisePoint16_beamSize16_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:52:33.6038
    # 3.99
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopKWRndSampOptsMaxNoisePoint16_topP0.90_beamSize16_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:49:54.0331
    # 3.99
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopKWRndSampOptsMaxNoisePoint1_beamSize16_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:47:27.8152
    # 4.01
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopKWRndSampOptsMaxNoisePoint1_topP0.90_beamSize16_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:52:15.0147
    # 4.01
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopKWRndSampOptsMaxNoisePoint4_beamSize16_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:46:31.1950
    # 4.02
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopKWRndSampOptsMaxNoisePoint8_beamSize16_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:48:01.9348
    # 4.0
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopKWRndSampOptsMinNoiseScale1.00_beamSize16_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:55:05.1361
    # 4.0
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopKWRndSampOptsMinNoiseScale1.00_topP0.90_beamSize16_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:49:20.7184
    # 3.98
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.90_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:48:57.3429
    # 3.98
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.90_beamSize16_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:46:54.5441
    # 3.98
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.90_beamSize1_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:45:07.9544
    # 4.15
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.90_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:52:42.3363
    # 3.98
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.90_beamSize4_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:46:14.6616
    # 4.01
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.95_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.90_beamSize8_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:48:03.4544
    # 3.97
    # ctcNumHyps20_ctcBeamSize20_ctcSoftCollapseThr0.99_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.90_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 1:01:52.4805
    # 3.98
    # ctcNumHyps20_ctcBeamSize20_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.70_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 1:04:35.0087
    # 3.97
    # ctcNumHyps20_ctcBeamSize20_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.90_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 1:08:02.6210
    # 3.97
    # ctcNumHyps20_ctcBeamSize64_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:53:08.1452
    # 3.97
    # ctcNumHyps20_ctcSoftCollapseThr0.95_ctcBeamSize64_beamSize12_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:41:54.3685
    # 3.99
    # ctcNumHyps20_ctcSoftCollapseThr0.95_ctcBeamSize64_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:46:54.7748
    # 4.0
    # ctcNumHyps20_ctcTopKWRndSamp1.00_ctcBeamSize64_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:56:35.7855
    # 3.96
    # ctcNumHyps20_ctcTopKWRndSamp1.00_ctcTopP0.10_ctcBeamSize64_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 1:00:39.9216
    # 3.96
    # ctcNumHyps20_ctcTopKWRndSamp1.00_ctcTopP0.30_ctcBeamSize64_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:57:50.1340
    # 3.96
    # ctcNumHyps20_ctcTopKWRndSamp1.00_ctcTopP0.50_ctcBeamSize64_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 1:01:27.8108
    # 3.96
    # ctcNumHyps20_ctcTopKWRndSamp1.00_ctcTopP0.60_ctcBeamSize64_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:58:10.3055
    # 3.96
    # ctcNumHyps20_ctcTopKWRndSamp1.00_ctcTopP0.70_ctcBeamSize64_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 1:01:26.0338
    # 3.96
    # ctcNumHyps20_ctcTopKWRndSamp1.00_ctcTopP0.80_ctcBeamSize64_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 1:02:01.6053
    # 3.96
    # ctcNumHyps20_ctcTopKWRndSamp1.00_ctcTopP0.90_ctcBeamSize128_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 1:01:24.6907
    # 3.97
    # ctcNumHyps20_ctcTopKWRndSamp1.00_ctcTopP0.90_ctcBeamSize20_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:59:52.4261
    # 3.96
    # ctcNumHyps20_ctcTopKWRndSamp1.00_ctcTopP0.90_ctcBeamSize32_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 1:00:37.1219
    # 3.97
    # ctcNumHyps20_ctcTopKWRndSamp1.00_ctcTopP0.90_ctcBeamSize64_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 0:58:23.3281
    # 3.96
    # ctcNumHyps50_ctcBeamSize1024_beamSize32_ctcPrefixScoreScale0.3949447077409163_lenNormExp0.00
    # elapsed: 1:29:26.7661
    # 3.99
    # ctcNumHyps5_ctcBeamSize1024_beamSize32_ctcPrefixScoreScale0.3949447077409163_lenNormExp0.00
    # elapsed: 0:24:06.3780
    # 4.1
    # ctcNumHyps5_ctcBeamSize1024_beamSize32_ctcPrefixScoreScale0.3949447077409163_lenNormExp0.00_dlmMaxSeqLenctc
    # elapsed: 0:24:43.5241
    # 4.1
    # ctcNumHyps60_ctcBeamSize60_ctcSoftCollapseThr0.90_initCtcSearchTypelabel_sync_ctcTopKWRndSamp1.00_ctcTopP0.90_beamSize32_ctcPrefixScoreScale0.30_lenNormExp0.00
    # elapsed: 1:41:24.9127
    # 3.99
    # ctcNumHyps80_ctcBeamSize1024_beamSize32
    # elapsed: 2:02:36.7143
    # 4.41

    for opts in (
        [{"recog_recomb": "max"}]
        if "n1024" in prefix
        else [
            {},
            {"beam_size": 512},
            {"recog_recomb": "max"},
            {"recog_recomb": "max", "beam_size": 512},
            {"recog_recomb": "sum"},
            {"recog_recomb": "sum", "beam_size": 512},
            {"recog_recomb": "max", "blank_penalty": 3.0},
            {"recog_recomb": "max", "blank_penalty": 3.0, "beam_size": 512},
            # {"lm_scale": 1.0, "prior_scale": 0.5, "recog_recomb": "max", "blank_penalty": 3.0},
            # {"blank_penalty": 3.0},
            # {"blank_penalty": 5.0},
            # {"lm_scale": 0.0, "prior_scale": 0.0},
            # {"lm_scale": 0.3, "prior_scale": 0.15},
            # {"lm_scale": 0.5, "prior_scale": 0.25},
            # {"lm_scale": 0.7, "prior_scale": 0.35},
            # {"lm_scale": 1.0, "prior_scale": 0.5},
            # {"lm_scale": 1.0, "prior_scale": 0.0},
            # {"lm_scale": 1.0, "prior_scale": 0.0, "blank_penalty": 3.0},
        ]
    ):
        model_with_dlm = get_ctc_with_dlm_and_labelwise_prior(
            ctc_model=asr_model.model,
            prior=asr_model.labelwise_prior.file,
            prior_type=asr_model.labelwise_prior.type,
            prior_scale=opts.get("prior_scale", prior_scale * (-1)),
            language_model=dlm,
            lm_scale=opts.get("lm_scale", lm_scale),
        )

        first_pass_recog_out = search_dataset(
            dataset=asr_dataset,
            model=model_with_dlm,
            recog_def=ctc_model_with_dlm_recog,
            config={
                "batch_size": 20_000 * asr_model.model.definition.batch_size_factor,
                "beam_size": opts.get("beam_size", asr_model.first_pass_recog_beam_size),
                "recog_version": 13,
                **opts,
            },
            search_alias_name=f"{prefix}/recog-ext/dlm/{dataset_name}-firstpass/{_short_repr(opts)}",
            search_rqmt={"time": 24},
        )
        # Post-processing, for example BPE to words.
        for f in dlm_task.recog_post_proc_funcs:
            first_pass_recog_out = f(first_pass_recog_out)

        # Evaluate (calc WER).
        score_out = sclite_score_recog_out_to_ref(first_pass_recog_out, ref=ref, corpus_name=dataset_name)
        tk.register_output(f"{prefix}/recog-ext/firstpass/{_short_repr(opts)}.txt", score_out.main_measure_value)

    # --- Now try with DLM sum recognition (ctc_model_with_dlm_sum_recog) ---

    # No prior, LM scale 1 for the DLM sum recog
    model_with_dlm = get_ctc_with_dlm_and_labelwise_prior(ctc_model=asr_model.model, language_model=dlm, lm_scale=1.0)

    # Note: For controlling memory consumption:
    # * batch_size obviously, but at a certain limit, only one seq would be in a batch,
    #   and then this has no effect anymore
    # * ctc_num_hyps: This is the number of CTC hyps as input to the DLM.
    #   I.e. we keep a separate DLM state for each of these.
    # * beam_size: This is the beam size for the DLM search.
    #   Again the number of DLM states is multiplied by this.
    # * torch_amp: This can reduce memory consumption, but it might not be too much difference.
    #   The parameters are still stored as float32.
    for opts in (
        [
            # base-spm10k-ctcDrop05-lsh10-epSplit20-keepSubwords-n1024-decL16-b2k_20k:
            # "dev-other": 4.21,
            # "dev-other:dlm-only": 4.77,
            # "dev-other:firstpass-{'recog_recomb': 'max'}": 4.19,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 1, 'batch_size': 1600000}": 5.02,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 1, 'ctc_prefix_score_scale': 0.3949447077409163, 'length_normalization_exponent': 0.0, 'batch_size': 800000}": 4.25,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 80, 'ctc_beam_size': 1024, 'beam_size': 32}": 4.41,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 100, 'ctc_beam_size': 1024, 'beam_size': 24}": 4.43,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 100, 'ctc_beam_size': 1024, 'beam_size': 24, 'length_normalization_exponent': 0.0}": 4.31,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 100, 'ctc_beam_size': 1024, 'beam_size': 32, 'length_normalization_exponent': 0.0, 'torch_amp': 'bfloat16'}": 4.3,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 100, 'ctc_beam_size': 1024, 'beam_size': 24, 'ctc_prefix_score_scale': 0.3949447077409163, 'length_normalization_exponent': 0.0}": 3.99,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 200, 'ctc_beam_size': 1024, 'beam_size': 16, 'length_normalization_exponent': 0.0, 'torch_amp': 'bfloat16'}": 4.3,
            {
                "ctc_num_hyps": 1,
                "batch_size": 10_000 * asr_model.model.definition.batch_size_factor,
                "length_normalization_exponent": 0.0,
            },
            {
                "ctc_num_hyps": 1,
                "ctc_prefix_score_scale": lm_scale ** (-1),
                "length_normalization_exponent": 0.0,
                "batch_size": 5_000 * asr_model.model.definition.batch_size_factor,
            },
            {
                "ctc_num_hyps": 5,
                "ctc_beam_size": 1024,
                "beam_size": 32,
                "ctc_prefix_score_scale": lm_scale ** (-1),
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
            },
            {
                "ctc_num_hyps": 10,
                "ctc_beam_size": 1024,
                "beam_size": 32,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
            },
            {
                "ctc_num_hyps": 10,
                "ctc_beam_size": 1024,
                "beam_size": 12,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
            },
            {
                "ctc_num_hyps": 10,
                "ctc_beam_size": 1024,
                "beam_size": 4,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
            },
            {
                "ctc_num_hyps": 10,
                "ctc_soft_collapse_threshold": 0.95,
                "ctc_beam_size": 1024,
                "beam_size": 12,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
            },
            {
                "ctc_num_hyps": 10,
                "ctc_soft_collapse_threshold": 0.95,
                "ctc_beam_size": 128,
                "beam_size": 12,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
            },
            # Note: With low ctc_soft_collapse_threshold, much less mem consumption, can use higher batch size.
            *[
                {
                    "ctc_num_hyps": 10,
                    "ctc_soft_collapse_threshold": p,
                    "ctc_soft_collapse_reduce_type": "max_renorm",
                    "ctc_beam_size": 128,
                    "beam_size": 12,
                    "ctc_prefix_score_scale": 0.3,
                    "length_normalization_exponent": 0.0,
                    "dlm_max_seq_len": "ctc",
                }
                for p in [0.5, 0.8, 0.9, 0.95]
            ],
            *[
                {
                    "ctc_num_hyps": 10,
                    "ctc_soft_collapse_threshold": 0.8,
                    "ctc_soft_collapse_reduce_type": "max_renorm",
                    "ctc_beam_size": 128,
                    "beam_size": 12,
                    "ctc_prefix_score_scale": 0.3,
                    "length_normalization_exponent": 0.0,
                    "dlm_max_seq_len": "ctc",
                    "batch_size": bs * asr_model.model.definition.batch_size_factor,
                    "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
                }
                for bs in [5_000, 10_000, 15_000, 20_000, 30_000, 40_000]
            ],
            {
                "ctc_num_hyps": 10,
                "ctc_soft_collapse_threshold": 0.8,
                "ctc_soft_collapse_reduce_type": "max_renorm",
                "ctc_beam_size": 128,
                "beam_size": 24,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
                "batch_size": 20_000 * asr_model.model.definition.batch_size_factor,
                "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            },
            {
                "ctc_num_hyps": 10,
                "ctc_soft_collapse_threshold": 0.95,
                "ctc_beam_size": 128,
                "ctc_top_k_with_random_sampling": 1.0,
                "beam_size": 12,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
            },
            {
                "ctc_num_hyps": 10,
                "ctc_soft_collapse_threshold": 0.95,
                "ctc_beam_size": 128,
                "ctc_top_k_with_random_sampling": 1.0,
                "ctc_top_k_with_random_sampling_opts": {"min_noise_scale": 1.0, "top_p": 0.9},
                "beam_size": 12,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
            },
            {
                "ctc_num_hyps": 10,
                "ctc_beam_size": 1024,
                "beam_size": 128,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
            },
            {
                "ctc_num_hyps": 10,
                "ctc_beam_size": 1024,
                "beam_size": 32,
                "ctc_prefix_score_scale": lm_scale ** (-1),
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
            },
            {
                "ctc_num_hyps": 10,
                "ctc_beam_size": 1024,
                "beam_size": 32,
                "ctc_prefix_score_scale": lm_scale ** (-1),
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
                "batch_size": 5_000 * asr_model.model.definition.batch_size_factor,
            },
            {
                "ctc_num_hyps": 10,
                "ctc_beam_size": 1024,
                "beam_size": 64,
                "ctc_prefix_score_scale": lm_scale ** (-1),
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
                "batch_size": 5_000 * asr_model.model.definition.batch_size_factor,
            },
            {
                "ctc_num_hyps": 10,
                "ctc_beam_size": 1024,
                "beam_size": 64,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
                "batch_size": 5_000 * asr_model.model.definition.batch_size_factor,
            },
            {
                "ctc_num_hyps": 10,
                "ctc_top_k_with_random_sampling": 1.0,
                "ctc_beam_size": 1024,
                "beam_size": 64,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
                "batch_size": 5_000 * asr_model.model.definition.batch_size_factor,
            },
            {
                "ctc_num_hyps": 10,
                "ctc_top_k_with_random_sampling": 1.0,
                "ctc_top_p": 0.9,
                "ctc_beam_size": 1024,
                "beam_size": 64,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
                "batch_size": 5_000 * asr_model.model.definition.batch_size_factor,
            },
            {
                "ctc_num_hyps": 10,
                "ctc_top_k_with_random_sampling": 1.0,
                "ctc_beam_size": 64,
                "beam_size": 64,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
                "batch_size": 5_000 * asr_model.model.definition.batch_size_factor,
            },
            {
                "ctc_num_hyps": 10,
                "ctc_top_k_with_random_sampling": 1.0,
                "ctc_top_p": 0.9,
                "ctc_beam_size": 64,
                "beam_size": 64,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
                "dlm_max_seq_len": "ctc",
                "batch_size": 5_000 * asr_model.model.definition.batch_size_factor,
            },
            {
                "ctc_num_hyps": 20,
                "ctc_beam_size": 1024,
                "beam_size": 48,
                "length_normalization_exponent": 0.0,
                "torch_amp": "bfloat16",
            },
            *[
                {
                    "ctc_num_hyps": num_hyps,
                    "ctc_beam_size": 1024,
                    "beam_size": 32,
                    "ctc_prefix_score_scale": lm_scale ** (-1),
                    "length_normalization_exponent": 0.0,
                }
                for num_hyps in [1, 5, 10, 20, 50]
            ],
            *[
                {
                    "ctc_num_hyps": 20,
                    "ctc_beam_size": 1024,
                    "beam_size": 32,
                    "ctc_prefix_score_scale": scale,
                    "length_normalization_exponent": 0.0,
                }
                for scale in [0.1, 0.2, 0.3]
            ],
            *[
                {
                    "ctc_num_hyps": 20,
                    "ctc_top_k_with_random_sampling": 1.0,
                    **({"ctc_top_p": p} if p else {}),
                    "ctc_beam_size": 64,
                    "beam_size": 32,
                    "ctc_prefix_score_scale": 0.3,
                    "length_normalization_exponent": 0.0,
                }
                for p in [0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, None]
            ],
            {
                "ctc_num_hyps": 20,
                "ctc_beam_size": 64,
                "beam_size": 32,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
            },
            {
                "ctc_num_hyps": 20,
                "ctc_soft_collapse_threshold": 0.95,
                "ctc_beam_size": 64,
                "beam_size": 32,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
            },
            {
                "ctc_num_hyps": 20,
                "ctc_soft_collapse_threshold": 0.95,
                "ctc_beam_size": 64,
                "beam_size": 12,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
            },
            *[
                {
                    "ctc_num_hyps": 20,
                    "ctc_top_k_with_random_sampling": 1.0,
                    "ctc_top_p": 0.9,
                    "ctc_beam_size": b,
                    "beam_size": 32,
                    "ctc_prefix_score_scale": 0.3,
                    "length_normalization_exponent": 0.0,
                }
                for b in [20, 32, 64, 128]
            ],
            {
                "ctc_num_hyps": 20,
                "ctc_beam_size": 20,
                "initial_ctc_search_type": "label_sync",
                "ctc_top_k_with_random_sampling": 1.0,
                "ctc_top_p": 0.9,
                "beam_size": 32,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
            },
            {
                "ctc_num_hyps": 20,
                "ctc_beam_size": 20,
                "initial_ctc_search_type": "label_sync",
                "ctc_top_k_with_random_sampling": 1.0,
                "ctc_top_p": 0.7,
                "beam_size": 32,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
            },
            {
                "ctc_num_hyps": 20,
                "ctc_beam_size": 20,
                "ctc_soft_collapse_threshold": 0.95,
                "initial_ctc_search_type": "label_sync",
                "beam_size": 16,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
            },
            {
                "ctc_num_hyps": 20,
                "ctc_beam_size": 20,
                "ctc_soft_collapse_threshold": 0.95,
                "initial_ctc_search_type": "label_sync",
                "ctc_top_k_with_random_sampling": 1.0,
                "beam_size": 16,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
            },
            {
                "ctc_num_hyps": 20,
                "ctc_beam_size": 20,
                "ctc_soft_collapse_threshold": 0.95,
                "initial_ctc_search_type": "label_sync",
                "ctc_top_k_with_random_sampling": 1.0,
                "ctc_top_k_with_random_sampling_opts": {"min_noise_scale": 1.0},  # pure sampling
                "beam_size": 16,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
            },
            {
                "ctc_num_hyps": 20,
                "ctc_beam_size": 20,
                "ctc_soft_collapse_threshold": 0.95,
                "initial_ctc_search_type": "label_sync",
                "ctc_top_k_with_random_sampling": 1.0,
                "ctc_top_k_with_random_sampling_opts": {"min_noise_scale": 1.0, "top_p": 0.9},  # like Apple
                "beam_size": 16,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
            },
            *[
                {
                    "ctc_num_hyps": 20,
                    "ctc_beam_size": 20,
                    "ctc_soft_collapse_threshold": 0.95,
                    "initial_ctc_search_type": "label_sync",
                    "ctc_top_k_with_random_sampling": 1.0,
                    "ctc_top_k_with_random_sampling_opts": {"max_noise_point": p},
                    "beam_size": 16,
                    "ctc_prefix_score_scale": 0.3,
                    "length_normalization_exponent": 0.0,
                }
                for p in [1, 4, 8, 16]
            ],
            *[
                {
                    "ctc_num_hyps": 20,
                    "ctc_beam_size": 20,
                    "ctc_soft_collapse_threshold": 0.95,
                    "initial_ctc_search_type": "label_sync",
                    "ctc_top_k_with_random_sampling": 1.0,
                    "ctc_top_k_with_random_sampling_opts": {"max_noise_point": p, "top_p": 0.9},
                    "beam_size": 16,
                    "ctc_prefix_score_scale": 0.3,
                    "length_normalization_exponent": 0.0,
                }
                for p in [1, 16]
            ],
            *[
                {
                    "ctc_num_hyps": 20,
                    "ctc_beam_size": 20,
                    "ctc_soft_collapse_threshold": threshold,
                    "initial_ctc_search_type": "label_sync",
                    "ctc_top_k_with_random_sampling": 1.0,
                    "ctc_top_p": 0.9,
                    "beam_size": 32,
                    "ctc_prefix_score_scale": 0.3,
                    "length_normalization_exponent": 0.0,
                }
                for threshold in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
            ],
            *[
                {
                    "ctc_num_hyps": 20,
                    "ctc_beam_size": 20,
                    "ctc_soft_collapse_threshold": 0.95,
                    "initial_ctc_search_type": "label_sync",
                    "ctc_top_k_with_random_sampling": 1.0,
                    "ctc_top_p": 0.9,
                    "beam_size": bs,
                    "ctc_prefix_score_scale": 0.3,
                    "length_normalization_exponent": 0.0,
                }
                for bs in [1, 4, 8, 12, 16, 32]
            ],
            {
                "ctc_num_hyps": 60,
                "ctc_beam_size": 60,
                "ctc_soft_collapse_threshold": 0.9,
                "initial_ctc_search_type": "label_sync",
                "ctc_top_k_with_random_sampling": 1.0,
                "ctc_top_p": 0.9,
                "beam_size": 32,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
            },
            {
                "ctc_num_hyps": 100,
                "ctc_beam_size": 100,
                "ctc_soft_collapse_threshold": 0.95,
                "initial_ctc_search_type": "label_sync",
                "ctc_top_k_with_random_sampling": 1.0,
                "ctc_top_p": 0.7,
                "beam_size": 24,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
            },
            # {"ctc_num_hyps": 80, "ctc_beam_size": 1024, "beam_size": 32},  # oom
            {"ctc_num_hyps": 100, "ctc_beam_size": 1024, "beam_size": 24},
            {"ctc_num_hyps": 100, "ctc_beam_size": 1024, "beam_size": 24, "length_normalization_exponent": 0.0},
            {
                "ctc_num_hyps": 100,
                "ctc_beam_size": 1024,
                "beam_size": 24,
                "length_normalization_exponent": 0.0,
                "torch_amp": "bfloat16",
            },
            {
                "ctc_num_hyps": 100,
                "ctc_beam_size": 1024,
                "beam_size": 32,
                "length_normalization_exponent": 0.0,
                "torch_amp": "bfloat16",
            },
            {
                "ctc_num_hyps": 100,
                "ctc_beam_size": 1024,
                "beam_size": 24,
                "ctc_prefix_score_scale": lm_scale ** (-1),
                "length_normalization_exponent": 0.0,
            },
            *[
                {
                    "ctc_num_hyps": 100,
                    "ctc_beam_size": 1024,
                    "beam_size": 24,
                    "ctc_prefix_score_scale": scale,
                    "length_normalization_exponent": 0.0,
                }
                for scale in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
            ],
            {
                "ctc_num_hyps": 100,
                "ctc_soft_collapse_threshold": 0.95,
                "ctc_beam_size": 1024,
                "beam_size": 8,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
            },
            {
                "ctc_num_hyps": 200,
                "ctc_beam_size": 1024,
                "beam_size": 16,
                "length_normalization_exponent": 0.0,
                "torch_amp": "bfloat16",
            },
            {
                "ctc_num_hyps": 200,
                "ctc_beam_size": 1024,
                "beam_size": 8,
                "ctc_prefix_score_scale": lm_scale ** (-1),
                "length_normalization_exponent": 0.0,
            },
            *[
                {
                    "ctc_num_hyps": 200,
                    "ctc_beam_size": 1024,
                    "beam_size": 8,
                    "ctc_prefix_score_scale": scale,
                    "length_normalization_exponent": 0.0,
                }
                for scale in [0.1, 0.2, 0.3]
            ],
            *[
                {
                    "ctc_num_hyps": 200,
                    "ctc_beam_size": 1024,
                    "beam_size": bs,
                    "ctc_prefix_score_scale": 0.3,
                    "length_normalization_exponent": 0.0,
                }
                for bs in [1, 2, 4, 8]
            ],
            {
                "ctc_num_hyps": 200,
                "ctc_soft_collapse_threshold": 0.95,
                "ctc_beam_size": 1024,
                "beam_size": 8,
                "ctc_prefix_score_scale": 0.3,
                "length_normalization_exponent": 0.0,
            },
        ]
        if "n1024" in prefix
        else [
            # base-spm10k-ls1-epSplit20-b2k_80k:
            # {"dev-clean": 1.94, "dev-clean:dlm-only": 2.53,
            # "dev-other": 4.64, "dev-other:dlm-only": 5.2,
            # "dev-other:firstpass-{}": 4.86, "dev-other:firstpass-{'beam_size': 512}": 4.66,
            # "dev-other:firstpass-{'recog_recomb': 'max'}": 4.6,
            # "dev-other:firstpass-{'recog_recomb': 'max', 'beam_size': 512}": 4.61,
            # "dev-other:firstpass-{'recog_recomb': 'sum'}": 4.6,
            # "dev-other:firstpass-{'recog_recomb': 'sum', 'beam_size': 512}": 4.61,
            # "dev-other:firstpass-{'recog_recomb': 'max', 'blank_penalty': 3.0}": 4.63,
            # "dev-other:firstpass-{'recog_recomb': 'max', 'blank_penalty': 3.0, 'beam_size': 512}": 4.63,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 1}": 5.23,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 10}": 4.89,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 10, 'ctc_beam_size': 1024}": 4.89,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 20, 'ctc_beam_size': 1024}": 4.9,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 50, 'ctc_beam_size': 1024, 'beam_size': 64}": 4.76,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 100, 'ctc_beam_size': 1024, 'beam_size': 32}": 4.72,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 100, 'ctc_beam_size': 1024, 'beam_size': 32, 'length_normalization_exponent': 0.0}": 4.68,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 100, 'ctc_beam_size': 1024, 'beam_size': 32, 'length_normalization_exponent': 0.5}": 4.68,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 100, 'ctc_beam_size': 1024, 'beam_size': 64, 'torch_amp': 'bfloat16'}": 4.75,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 400, 'ctc_beam_size': 8192, 'beam_size': 8}": 4.73,
            # "dev-other:firstpass-sum-{'ctc_num_hyps': 1024, 'ctc_beam_size': 8192, 'beam_size': 8, 'torch_amp': 'bfloat16'}": 4.75,
            # "test-clean": 2.13, "test-clean:dlm-only": 2.66, "test-other": 5.08, "test-other:dlm-only": 5.77}
            {"ctc_num_hyps": 1},
            {
                "ctc_num_hyps": 1,
                "ctc_prefix_score_scale": lm_scale ** (-1),
                "length_normalization_exponent": 0.0,
                "batch_size": 10_000 * asr_model.model.definition.batch_size_factor,
            },
            # {"ctc_num_hyps": 2},
            # {"ctc_num_hyps": 3},
            # {"ctc_num_hyps": 4},
            {"ctc_num_hyps": 10},
            {"ctc_num_hyps": 10, "ctc_beam_size": 1024},
            {
                "ctc_num_hyps": 10,
                "ctc_beam_size": 1024,
                "ctc_prefix_score_scale": lm_scale ** (-1),
                "length_normalization_exponent": 0.0,
            },
            {"ctc_num_hyps": 20, "ctc_beam_size": 1024},
            {"ctc_num_hyps": 50, "ctc_beam_size": 1024, "beam_size": 64},
            {"ctc_num_hyps": 100, "ctc_beam_size": 1024, "beam_size": 32},
            {"ctc_num_hyps": 100, "ctc_beam_size": 1024, "beam_size": 32, "length_normalization_exponent": 0.0},
            {"ctc_num_hyps": 100, "ctc_beam_size": 1024, "beam_size": 32, "length_normalization_exponent": 0.5},
            # {"ctc_num_hyps": 100, "ctc_beam_size": 1024, "beam_size": 64, "torch_amp": "bfloat16"},  # oom
            {
                "ctc_num_hyps": 100,
                "ctc_beam_size": 1024,
                "beam_size": 32,
                "ctc_prefix_score_scale": lm_scale ** (-1),
                "length_normalization_exponent": 0.0,
            },
            {"ctc_num_hyps": 400, "ctc_beam_size": 1024 * 8, "beam_size": 8},
            # {"ctc_num_hyps": 1024, "ctc_beam_size": 1024 * 8, "beam_size": 8, "torch_amp": "bfloat16"},  # oom
        ]
    ):
        first_pass_recog_out = search_dataset(
            dataset=asr_dataset,
            model=model_with_dlm,
            recog_def=ctc_model_with_dlm_sum_recog,
            config={
                "batch_size": opts.get(
                    "batch_size", 20_000 * asr_model.model.definition.batch_size_factor // opts["ctc_num_hyps"]
                ),
                "beam_size": opts.get("beam_size", asr_model.first_pass_recog_beam_size),
                "ctc_beam_size": opts.get("ctc_beam_size", 128),  # maybe also more?
                "recog_version": 3,
                "behavior_version": 24,
                **opts,
            },
            search_alias_name=f"{prefix}/recog-ext/dlm/{dataset_name}-firstpass-sum/{_short_repr(opts)}",
            search_rqmt={"time": 24},
        )
        # Post-processing, for example BPE to words.
        for f in dlm_task.recog_post_proc_funcs:
            first_pass_recog_out = f(first_pass_recog_out)

        # Evaluate (calc WER).
        score_out = sclite_score_recog_out_to_ref(first_pass_recog_out, ref=ref, corpus_name=dataset_name)
        tk.register_output(f"{prefix}/recog-ext/firstpass-sum/{_short_repr(opts)}.txt", score_out.main_measure_value)


def _short_repr(d: Dict[str, Any], *, len_limit: int = 230, shorten_amount: int = 1) -> str:
    while True:
        parts = []
        for key, value in _iter_dict_deep(d, shorten_amount=shorten_amount):
            value = _short_repr_value(value)
            key_parts = key.split("_")
            key_parts = [_ShortPartNameSubs.get(p, p) for p in key_parts]
            key = key_parts[0] + "".join([p.capitalize() for p in key_parts[1:]])
            parts.append(f"{key}{value}")
        res = "_".join(parts) or "default"
        if len(res) > len_limit:
            shorten_amount += 1
            if shorten_amount >= 2:
                raise ValueError(f"Too long: {d!r}, represented as {res!r} (len {len(res)}), cannot shorten further")
            continue
        return res


# Common names which are hopefully clear enough.
_ShortPartNameSubs = {
    "length": "len",
    "normalization": "norm",
    "exponent": "exp",
    "random": "rnd",
    "sampling": "samp",
    "initial": "init",
    "threshold": "thr",
    "with": "w",
}


def _short_repr_value(n: Any) -> str:
    if isinstance(n, float):
        return f"{n:.2f}"
    if isinstance(n, (list, tuple, set)):
        return "_".join(_short_repr_value(v) for v in n)
    if isinstance(n, DelayedBase):
        return n.__class__.__name__
    return str(n)


def _iter_dict_deep(d: Dict[str, Any], *, shorten_amount: int = 0) -> Generator[Tuple[str, Any], Any, Any]:
    for key, value in d.items():
        if key == "__env_updates":
            continue  # skip
        if isinstance(value, dict):
            for i, (sub_key, sub_value) in enumerate(_iter_dict_deep(value)):
                if shorten_amount >= 1 and i >= 1:
                    yield sub_key, sub_value  # just take the sub-keys
                else:
                    yield f"{key}_{sub_key}", sub_value
        else:
            yield key, value
