from __future__ import annotations
import copy
from typing import Dict, Any, Tuple, Optional
import functools
from .dlm_sum import _generic_seq_label_print
from i6_experiments.users.zeyer.datasets.task import Task
from i6_experiments.users.zeyer.datasets.score_results import ScoreResultCollection, RecogOutput
from i6_experiments.users.zeyer.datasets.utils.vocab import (
    ExtractVocabLabelsJob,
    ExtractVocabSpecialLabelsJob,
    ExtendVocabLabelsByNewLabelJob,
)
from i6_experiments.users.zeyer.recog import recog_model, search_dataset
from i6_experiments.users.zeyer.decoding.rescoring import combine_scores, rescore
from i6_experiments.users.zeyer.decoding.prior_rescoring import prior_score, Prior, PriorRemoveLabelRenormJob
from i6_experiments.users.zeyer.collect_model_dataset_stats import collect_statistics
from i6_experiments.users.zeyer.model_interfaces import (
    ModelDef,
    ModelDefWithCfg,
    RecogDef,
    TrainDef,
)
from returnn.config import get_global_config
from returnn.frontend.loop import TensorArray
from returnn.util.basic import NotSpecified
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim
from ..model.error_correction_model import Model as DLModel
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import Model as AEDModel
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
    aed_ctc_timesync_recog_recomb_auto_scale,
    _get_vocab_opts_from_task,
    get_aed_ctc_and_labelwise_prior,
    model_recog_with_recomb,
    aed_score,
    aed_labelwise_prior_rescore,
)
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from sisyphus import tk
from typing import TYPE_CHECKING, Optional, Union, Any, Callable, Sequence, Tuple, Dict
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_label_sync_espnet import (
    CtcPrefixScorer,
)

if TYPE_CHECKING:
    from returnn_common.datasets_old_2022_10.interface import DatasetConfig


def dlm_recog_only_auxloss(
    *,
    model: DLModel,
    data: Tensor,
    data_spatial_dim: Dim,
    data_k_log_probs: Optional[Tensor] = None,
    **_other,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Small wrapper around the CTC recog which uses the aux loss layer of the DLM model.
    """
    from returnn.config import get_global_config

    config = get_global_config()

    assert data_k_log_probs is None

    input_add_bos = config.typed_value("input_add_bos", False)
    input_add_eos = config.typed_value("input_add_eos", False)

    # these options are not really needed or supported, just assert that they are not specified
    length_normalization_exponent = config.float("length_normalization_exponent", 1.0)
    assert length_normalization_exponent == 1.0
    temperature = config.float("temperature", 1.0)
    assert temperature == 1.0
    nucleus_sampling_opts = config.typed_value("nucleus_sampling", None)
    assert nucleus_sampling_opts is None
    nucleus_sampling_beam_search_opts = config.typed_value("nucleus_sampling_beam_search", None)
    assert nucleus_sampling_beam_search_opts is None
    top_k_and_random_choice_without_replacement_opts = config.typed_value(
        "top_k_and_random_choice_without_replacement", None
    )
    assert top_k_and_random_choice_without_replacement_opts is None

    enc_aux_layers = config.typed_value("dlm_aux_loss_layers") or ()
    assert len(enc_aux_layers) > 0
    chosen_aux_layer = max(enc_aux_layers)

    dummy_feature_dim = Dim(1, name="dummy_feature")
    if data.feature_dim is None:  # ctc model_recog wants a feature dim... add a dummy one
        old_sparse_dim = data.sparse_dim
        data.sparse_dim = None
        data = rf.expand_dim(data, dim=dummy_feature_dim)
        data.feature_dim = dummy_feature_dim

    def fwd(data, in_spatial_dim):
        if data.feature_dim == dummy_feature_dim:
            data = rf.squeeze(data, axis=dummy_feature_dim)
            data.feature_dim = None
            data.sparse_dim = old_sparse_dim
        if input_add_bos:
            data, (in_spatial_dim,) = rf.pad(
                data, axes=[in_spatial_dim], padding=[(int(input_add_bos), 0)], value=model.bos_idx
            )
        if input_add_eos:
            data, (in_spatial_dim,) = rf.pad(
                data, axes=[in_spatial_dim], padding=[(0, int(input_add_eos))], value=model.eos_idx
            )
        collected_outputs = {} if enc_aux_layers else None
        model.encode(data, spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
        assert collected_outputs is not None
        assert model.aux_layer is not None, "aux_layer must be defined for aux loss layers"

        # aux_layer: model_dim -> 2*target_dim
        aux_logits2 = model.aux_layer(collected_outputs[str(chosen_aux_layer - 1)])
        two_dim = rf.Dim(2, name="aux_twodim")
        # [.., 2*target_dim] -> [.., 2, target_dim]
        aux_logits = rf.split_dims(aux_logits2, axis=model.aux_layer.out_dim, dims=[two_dim, model.target_dim_wb])
        # [.., spatial_dim, 2, target_dim] -> [.., 2*spatial_dim, target_dim]
        aux_logits, two_spatial_dim = rf.merge_dims(
            aux_logits, dims=[in_spatial_dim, two_dim]
        )  # IMPORTANT: correct merge order

        aux_logits.feature_dim = model.target_dim_wb

        assert config.int("aux_decoding_version", 1) > 3

        return aux_logits, None, two_spatial_dim

    def logprobs_from_logits(x):
        # we have a problem: the ctc model recog outputs the eos token, but we dont want this for the recog
        # we could manually remove this as a postprocessing step, but we can just do this here now
        # so we just transfer the prob for the eos idx to the blank idx
        # still not clean because we dont consider the case where the model emits bos
        assert not input_add_bos
        x = rf.log_softmax(x, axis=model.target_dim_wb)
        eos_prob = rf.gather(x, indices=model.eos_idx, axis=model.target_dim_wb)
        blank_prob = rf.gather(x, indices=model.blank_idx, axis=model.target_dim_wb)

        # add eos prob to blank prob
        x = rf.where(
            rf.range_over_dim(model.target_dim_wb) == model.blank_idx,
            rf.log_add_exp(blank_prob, eos_prob),
            x,
        )
        # set eos prob to -inf
        x = rf.where(rf.range_over_dim(model.target_dim_wb) == model.eos_idx, float("-inf"), x)
        return x

    fwd.log_probs_wb_from_logits = logprobs_from_logits
    fwd.blank_idx = model.blank_idx
    fwd.wb_target_dim = model.target_dim_wb

    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import model_recog as ctc_model_recog

    return ctc_model_recog(model=fwd, data=data, data_spatial_dim=data_spatial_dim)


# RecogDef API
dlm_recog_only_auxloss: RecogDef[DLModel]
dlm_recog_only_auxloss.output_with_beam = True
dlm_recog_only_auxloss.output_blank_label = "<blank>"
dlm_recog_only_auxloss.batch_size_dependent = False


# for joint decoding we use i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc.aed_ctc_timesync_recog_recomb_auto_scale
# but we need to do some modification to the model def so that it works with our setup
def modified_joint_dec_dlm_def(*, epoch: int, in_dim: Dim, target_dim: Dim, orig_dlm_aed_def: ModelDef):
    from returnn.config import get_global_config

    config = get_global_config()

    enc_aux_layers = config.typed_value("dlm_aux_loss_layers") or ()
    assert len(enc_aux_layers) > 0
    chosen_aux_layer = max(enc_aux_layers)
    input_add_bos = config.typed_value("input_add_bos", False)
    input_add_eos = config.typed_value("input_add_eos", False)

    orig_dlm: DLModel = orig_dlm_aed_def(epoch=epoch, in_dim=in_dim, target_dim=target_dim)

    # so the aed_ctc recog expects our ctc output to have the same spatial dim as the encoder output
    # but in the DLM, the ctc output is twice as long as the encoder spatial dim
    # so we need to modify the model to return the correct out dim
    # TODO maybe just modify the recog function instead to get the spatial dim from somewhere else...

    orig_encode = orig_dlm.encode

    def mdlm_encode(source: Tensor, *, in_spatial_dim: Dim, collected_outputs: Optional[Dict[str, Tensor]] = None):
        if input_add_bos:
            source, (in_spatial_dim,) = rf.pad(
                source, axes=[in_spatial_dim], padding=[(int(input_add_bos), 0)], value=orig_dlm.bos_idx
            )
        if input_add_eos:
            source, (in_spatial_dim,) = rf.pad(
                source, axes=[in_spatial_dim], padding=[(0, int(input_add_eos))], value=orig_dlm.eos_idx
            )
        # we just call the original encode function and then modify the output
        state = orig_encode(source, spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)

        # for use in aux linear layer forward
        mdlm.cur_out_spatial_dim = in_spatial_dim * mdlm.two_dim

        return state, mdlm.cur_out_spatial_dim

    mdlm = orig_dlm
    mdlm.encode = mdlm_encode
    # we expect the following attributes
    mdlm.wb_target_dim = orig_dlm.target_dim_wb
    mdlm.enc_aux_logits = [chosen_aux_layer]
    mdlm.bos_idx = orig_dlm.bos_idx
    mdlm.eos_idx = orig_dlm.eos_idx
    mdlm.blank_idx = orig_dlm.blank_idx
    mdlm.target_dim = orig_dlm.target_dim
    mdlm.two_dim = rf.Dim(2, name="aux_twodim")
    mdlm.out_eos_separated = False  # no idea what this means

    # for the enc aux logits linear layer
    def fwd(x: rf.Tensor):
        assert orig_dlm.aux_layer is not None, "aux_layer must be defined for aux loss layers"
        # we need to find the spatial dimension
        # there are some methods for this in the tensor class, but they seem somewhat deprecated?
        in_spatial_dim = None
        for d in x.dims_set:
            if d.is_batch_dim():
                continue
            if d.is_spatial_dim() or "spatial" in (d.name or ""):
                assert in_spatial_dim is None
                in_spatial_dim = d
        assert in_spatial_dim is not None, f"need spatial dim in {x.dims_set}"

        # aux_layer: model_dim -> 2*target_dim
        aux_logits2 = orig_dlm.aux_layer(x)

        # [.., 2*target_dim] -> [.., 2, target_dim]
        aux_logits = rf.split_dims(
            aux_logits2, axis=orig_dlm.aux_layer.out_dim, dims=[mdlm.two_dim, orig_dlm.target_dim_wb]
        )
        # [.., spatial_dim, 2, target_dim] -> [.., 2*spatial_dim, target_dim]
        aux_logits, two_spatial_dim = rf.merge_dims(
            aux_logits,
            dims=[in_spatial_dim, mdlm.two_dim],
            out_dim=mdlm.cur_out_spatial_dim,  # not sure if this works...
        )  # IMPORTANT: correct merge order

        aux_logits.feature_dim = orig_dlm.target_dim_wb

        assert config.int("aux_decoding_version", 4) > 3

        # one more problem: our ctc may output bos_idx or eos_idx, but the joint decoding does not expect this
        # there exists an option 'use_eos_postfix', but i dont think this is what we want
        # we should add the prob of eos_idx to blank_idx and set eos_idx prob to 0
        # it turns out that the same logsumexp trick from above also works before softmax
        assert not input_add_bos  # would need more changes
        assert input_add_eos  #
        eos_prob = rf.gather(aux_logits, indices=orig_dlm.eos_idx, axis=orig_dlm.target_dim_wb)
        blank_prob = rf.gather(aux_logits, indices=orig_dlm.blank_idx, axis=orig_dlm.target_dim_wb)

        # add eos prob to blank prob
        aux_logits = rf.where(
            rf.range_over_dim(orig_dlm.target_dim_wb) == orig_dlm.blank_idx,
            rf.log_add_exp(blank_prob, eos_prob),
            aux_logits,
        )
        # set eos prob to -inf
        aux_logits = rf.where(rf.range_over_dim(orig_dlm.target_dim_wb) == orig_dlm.eos_idx, float("-inf"), aux_logits)

        return aux_logits

    setattr(mdlm, f"enc_aux_logits_{chosen_aux_layer}", fwd)

    return mdlm


modified_joint_dec_dlm_def: ModelDef[DLModel]
modified_joint_dec_dlm_def.behavior_version = 24
modified_joint_dec_dlm_def.backend = "torch"
modified_joint_dec_dlm_def.batch_size_factor = 1


def model_recog_with_recomb_labelsync(
    *,
    model: AEDModel,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Note, some potential further improvements:
    There are many align label seqs which correspond to the same label seq,
    but the LM score is calculated for each of them.
    We could make this somehow unique depending on the label seq.
    (But unclear how exactly to do this in a GPU friendly, batched way.)

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import returnn
    from returnn.config import get_global_config
    from i6_experiments.users.zeyer.nn_rf.soft_collapse_repeated import soft_collapse_repeated

    config = get_global_config()
    beam_size = config.int("beam_size", 12)
    # recomb = config.typed_value("recog_recomb", "max")  # None, "max", "sum"
    ctc_soft_collapse_threshold = config.typed_value("ctc_soft_collapse_threshold", None)  # e.g. 0.8
    ctc_soft_collapse_reduce_type = config.typed_value("ctc_soft_collapse_reduce_type", "max_renorm")
    aed_scale = config.float("aed_scale", 1.0)
    ctc_scale = config.float("ctc_scale", 1.0)
    ctc_final_prefix_score_scale = ctc_prefix_score_scale = ctc_scale  # TODO i think this is right?
    dlm_max_seq_len_cfg = None  # TODO
    labelwise_prior = None  # TODO
    dlm_temperature = 1.0  # TODO
    length_normalization_exponent = config.float("length_normalization_exponent", 1.0)
    # length_normalization_each_term = False
    return_only_dlm_score = False

    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250125, 0), returnn.__version__

    if data.feature_dim is not None:
        batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    else:
        batch_dims = data.remaining_dims(data_spatial_dim)
    enc_collected_outputs = {}
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=enc_collected_outputs)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    neg_inf = float("-inf")
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    ctc_layer_idx = model.enc_aux_logits[-1]
    linear = getattr(model, f"enc_aux_logits_{ctc_layer_idx}")
    ctc_logits = linear(enc_collected_outputs[str(ctc_layer_idx - 1)])
    ctc_label_log_prob = rf.log_softmax(ctc_logits, axis=model.wb_target_dim)  # Batch, Spatial, VocabWB
    if ctc_soft_collapse_threshold is not None:
        ctc_label_log_prob, enc_spatial_dim = soft_collapse_repeated(
            ctc_label_log_prob,
            spatial_dim=enc_spatial_dim,
            classes_dim=model.wb_target_dim,
            threshold=ctc_soft_collapse_threshold,
            reduce_type=ctc_soft_collapse_reduce_type,
        )
    ctc_label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        ctc_label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=neg_inf),
    )
    if config.bool("use_eos_postfix", False):
        assert False
        # ctc_label_log_prob = rf.where(
        #    rf.range_over_dim(model.wb_target_dim) != model.eos_idx, ctc_label_log_prob, neg_inf
        # )
    # No CTC scale needed.
    beam_dim = Dim(1, name="initial_beam")
    target = rf.constant(
        model.bos_idx, dims=[beam_dim] + batch_dims, sparse_dim=model.target_dim
    )  # Batch, InBeam -> Vocab
    ended = rf.constant(False, dims=[beam_dim] + batch_dims)
    seq_log_prob = rf.constant(0.0, dims=[beam_dim] + batch_dims)  # Batch, InBeam
    prior_seq_log_prob = None
    if labelwise_prior is not None:
        prior_seq_log_prob = rf.constant(0.0, dims=[beam_dim] + batch_dims)  # Batch, InBeam
    lm_state = model.decoder.default_initial_state(batch_dims=[beam_dim] + batch_dims)  # Batch, InBeam, ...
    out_seq_len = rf.constant(0, dims=[beam_dim] + batch_dims)

    ctc_prefix_scorer, ctc_prefix_seq_log_prob, ctc_prefix_log_prob, ctc_prefix_scorer_state = None, None, None, None
    if ctc_prefix_score_scale != 0:
        ctc_prefix_scorer = CtcPrefixScorer(
            log_probs=ctc_label_log_prob,
            batch_dims=batch_dims,
            enc_spatial_dim=enc_spatial_dim,
            vocab_wb_dim=model.wb_target_dim,
            vocab_dim=model.target_dim,
            blank_idx=model.blank_idx,
            eos_idx=model.eos_idx,
        )
        ctc_prefix_seq_log_prob = rf.constant(0.0, dims=[beam_dim] + batch_dims)  # Batch, InBeam

    if dlm_max_seq_len_cfg is None:
        # probably more than necessary...
        max_seq_len = enc_spatial_dim.get_size_tensor(device=target.device)
    elif dlm_max_seq_len_cfg == "ctc":  # this ctc refers to ASR ctc...
        assert False
        # max_seq_len = rf.cast(rf.ceil(rf.cast(labels_max_seq_lens, "float32") * 1.5), "int32")
    else:
        raise ValueError(f"invalid dlm_max_seq_len {dlm_max_seq_len_cfg!r} (type {type(dlm_max_seq_len_cfg)})")
    max_seq_len = rf.copy_to_device(max_seq_len, device=target.device)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        aed_logits, lm_state = model.decoder(
            target,
            spatial_dim=single_step_dim,
            encoder=enc,
            state=lm_state,
        )
        label_log_prob = rf.log_softmax(aed_logits / dlm_temperature, axis=model.target_dim)
        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=neg_inf),
            label_log_prob,
        )
        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab

        length_norm_factor = (
            1.0 / (rf.cast(out_seq_len, dtype=seq_log_prob.dtype) + 1.0)
        ) ** length_normalization_exponent

        seq_log_prob_ = seq_log_prob * aed_scale
        if ctc_prefix_score_scale != 0:
            ctc_prefix_log_prob, ctc_prefix_scorer_state = ctc_prefix_scorer.score_and_update_state(
                prev_label=target, prev_state=ctc_prefix_scorer_state, beam_dim=beam_dim
            )
            # Filter out finished beams
            ctc_prefix_log_prob = rf.where(
                ended,
                rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=neg_inf),
                ctc_prefix_log_prob,
            )
            seq_log_prob_ = (
                seq_log_prob_ + (ctc_prefix_seq_log_prob + ctc_prefix_log_prob) * ctc_prefix_score_scale
            )  # Batch, InBeam, Vocab
        if labelwise_prior is not None:
            seq_log_prob_ -= prior_seq_log_prob
            seq_log_prob_ -= labelwise_prior  # prior already scaled
        if i > 1 and length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob_ *= length_norm_factor

        _, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob_,
            k_dim=Dim(beam_size, name=f"dec_step{i}_beam"),
            axis=[beam_dim, model.target_dim],
        )  # backrefs, target: Batch, Beam
        target = rf.cast(target, dtype=rf.get_default_int_dtype())
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        seq_log_prob = rf.gather(seq_log_prob, indices=backrefs)  # Batch, Beam, Vocab
        seq_log_prob = rf.gather(seq_log_prob, indices=target)  # Batch, Beam
        lm_state = rf.nested.gather_nested(lm_state, indices=backrefs)
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        if ctc_prefix_score_scale != 0:
            ctc_prefix_scorer_state = rf.nested.gather_nested(ctc_prefix_scorer_state, indices=backrefs)
            ctc_prefix_log_prob = rf.gather(ctc_prefix_log_prob, indices=backrefs)  # Batch, Beam, Vocab
            ctc_prefix_log_prob = rf.gather(ctc_prefix_log_prob, indices=target)  # Batch, Beam
            ctc_prefix_seq_log_prob = rf.gather(ctc_prefix_seq_log_prob, indices=backrefs)  # Batch, Beam
            ctc_prefix_seq_log_prob = ctc_prefix_seq_log_prob + ctc_prefix_log_prob  # Batch, Beam
        if labelwise_prior is not None:
            prior_seq_log_prob = rf.gather(prior_seq_log_prob, indices=backrefs)  # Batch, Beam
            prior_seq_log_prob += rf.where(
                ended,
                0.0,
                rf.gather(labelwise_prior, indices=target),
            )  # Batch, beam

        i += 1
        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, i >= max_seq_len)
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

    if return_only_dlm_score:
        pass
    else:
        if aed_scale != 1.0:
            seq_log_prob *= aed_scale
        if ctc_final_prefix_score_scale != 0:
            seq_log_prob += ctc_prefix_seq_log_prob * ctc_final_prefix_score_scale  # Batch, Beam
        if labelwise_prior is not None:
            seq_log_prob -= prior_seq_log_prob

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
    out_spatial_dim = Dim(out_seq_len, name="out_spatial")
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    print("Final seq:")
    _generic_seq_label_print(seq_targets, out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog_with_recomb_labelsync: RecogDef[AEDModel]
model_recog_with_recomb_labelsync.output_with_beam = True
model_recog_with_recomb_labelsync.output_blank_label = "<blank>"
model_recog_with_recomb_labelsync.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def aux_aed_dlm_timesync_recog_recomb_autoscale(dlm: ModelWithCheckpoint, *, task, prefix: str, **kwargs):
    orig_model_def_ = dlm.definition
    if isinstance(orig_model_def_, ModelDefWithCfg):
        config: Dict[str, Any] = orig_model_def_.config.copy()
        orig_model_def_ = orig_model_def_.model_def
    else:
        config = {}

    # Also see: .tts_model.get_asr_with_tts_model_def
    # noinspection PyTypeChecker
    combined_model_def: ModelDef = functools.partial(modified_joint_dec_dlm_def, orig_dlm_aed_def=orig_model_def_)
    # Make it a proper ModelDef
    combined_model_def.behavior_version = max(
        modified_joint_dec_dlm_def.behavior_version, orig_model_def_.behavior_version
    )
    combined_model_def.backend = orig_model_def_.backend
    combined_model_def.batch_size_factor = orig_model_def_.batch_size_factor
    # Need new recog serialization for the partial.
    config["__serialization_version"] = max(2, config.get("__serialization_version", 0))

    new_aed_model = ModelWithCheckpoint(
        definition=ModelDefWithCfg(model_def=combined_model_def, config=config),
        checkpoint=dlm.checkpoint,
    )

    return _impl_dlm_timesync_recog_recomb_autoscale(
        prefix=prefix,
        task=task,
        aed_ctc_model=new_aed_model,
        orig_dlm=dlm,
        # aux_ctc_layer=-1,
        **kwargs,
    )


def _impl_dlm_timesync_recog_recomb_autoscale(
    *,
    prefix: str,
    task: Task,
    aed_ctc_model: ModelWithCheckpoint,
    orig_dlm: ModelWithCheckpoint,
    vocab_file: tk.Path = NotSpecified,
    vocab_opts_file: tk.Path = NotSpecified,
    ctc_soft_collapse_threshold: Optional[float] = 0.8,  # default
    n_best_list_size: int = 64,
    first_pass_recog_beam_size: int = 64,
    first_pass_search_rqmt: Optional[Dict[str, int]] = None,
    recomb_type: str = "max",
    extra_config: Optional[Dict[str, Any]] = None,
) -> ModelWithCheckpoint:
    """
    i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc.aed_ctc_timesync_recog_recomb_auto_scale
    but return the model with the optimal scales
    and lower batch size for 1stpass
    """
    if vocab_file is NotSpecified:
        vocab_file = ExtractVocabLabelsJob(_get_vocab_opts_from_task(task)).out_vocab
        # tk.register_output(f"{prefix}/vocab/{vocab}/vocab.txt.gz", vocab_file)

    if vocab_opts_file is NotSpecified:
        vocab_opts_file = ExtractVocabSpecialLabelsJob(_get_vocab_opts_from_task(task)).out_vocab_special_labels_dict
        # tk.register_output(f"{prefix}/vocab/{vocab}/vocab_opts.py", vocab_opts_file)

    from ..model.error_correction_model import model_recog as dlm_model_recog

    # For CTC-only and then also for joint AED+CTC+prior.
    base_config = {
        "behavior_version": 24,  # should make it independent from batch size
        "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},  # OOM maybe otherwise
        "recog_recomb": recomb_type,
        "ctc_soft_collapse_threshold": ctc_soft_collapse_threshold,
        "aux_loss_layers": [-1],  # keep hash...
        "batch_size": int(1_000 * aed_ctc_model.definition.batch_size_factor),
    }
    if extra_config:
        base_config = dict_update_deep(base_config, extra_config)

    # Only use CTC for first search, no AED, no prior.
    ctc_model_only = get_aed_ctc_and_labelwise_prior(aed_ctc_model=aed_ctc_model, aed_scale=0.0)
    dataset = task.dev_dataset
    # ctc_scores = search_dataset(
    #     dataset=dataset,
    #     model=ctc_model_only,
    #     recog_def=model_recog_with_recomb,
    #     config={**base_config, "beam_size": n_best_list_size},
    #     keep_beam=True,
    # )
    # aed_scores = aed_score(
    #      ctc_scores, dataset=dataset, aed_model=aed_ctc_model, vocab=vocab_file, vocab_opts_file=vocab_opts_file
    # )

    # TODO no length normalization in scale tuning, but we use it later in 1stpass recog?
    aed_scores = search_dataset(
        dataset=dataset,
        model=orig_dlm,
        recog_def=dlm_model_recog,
        config={**base_config, "length_normalization_exponent": 0.0, "beam_size": n_best_list_size},
        keep_beam=True,
    )
    ctc_scores = rescore(
        recog_output=aed_scores,
        dataset=dataset,
        model=ctc_model_only,
        vocab=vocab_file,
        vocab_opts_file=vocab_opts_file,
        rescore_def=dlm_with_ctc_recomb_rescore_def,
        config={**base_config, "beam_size": n_best_list_size, "version": 2},
    )

    # Also register the CTC-only results. (Will not do search again, should be same hash.)
    res = recog_model(
        task=task,
        model=ctc_model_only,
        recog_def=model_recog_with_recomb,
        config={**base_config, "beam_size": n_best_list_size},
    )
    tk.register_output(f"{prefix}/ctc-only-res.txt", res.output)

    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob
    from i6_experiments.users.zeyer.datasets.task import RecogOutput

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )

    for f in task.recog_post_proc_funcs:  # BPE to words or so
        ctc_scores = f(ctc_scores)
        aed_scores = f(aed_scores)
        ref = f(ref)

    from i6_experiments.users.zeyer.decoding.scale_tuning import ScaleTuningJob

    opt_scales_job = ScaleTuningJob(
        scores={"ctc": ctc_scores.output, "aed": aed_scores.output},
        ref=ref.output,
        fixed_scales={"ctc": 1.0},
        max_scales={"aed": 10.0},
        evaluation="edit_distance",
    )
    opt_scales_job.rqmt["engine"] = "short"  # should be fine
    tk.register_output(f"{prefix}/opt-real-scales", opt_scales_job.out_real_scales)
    tk.register_output(f"{prefix}/opt-rel-scales", opt_scales_job.out_scales)
    # We use the real scales.
    aed_scale = opt_scales_job.out_real_scale_per_name["aed"]

    # TODO this is rescore on the ctc hyps... but we want aed hyps rescored with ctc
    # comment out for now
    # res = recog_model(
    #     task=task,
    #     model=ctc_model_only,
    #     recog_def=model_recog_with_recomb,
    #     config={**base_config, "beam_size": n_best_list_size},
    #     recog_pre_post_proc_funcs_ext=[
    #         functools.partial(
    #             aed_labelwise_prior_rescore,
    #             aed_model=aed_ctc_model,
    #             aed_scale=aed_scale,
    #             aed_rescore_rqmt={"cpu": 4, "mem": 30, "time": 24, "gpu_mem": 48},
    #             vocab=vocab_file,
    #             vocab_opts_file=vocab_opts_file,
    #         )
    #     ],
    # )
    # tk.register_output(f"{prefix}/rescore-res.txt", res.output)

    # Now do 1st-pass recog with optimal scales.
    model = get_aed_ctc_and_labelwise_prior(aed_ctc_model=aed_ctc_model, aed_scale=aed_scale)
    first_pass_search_rqmt = first_pass_search_rqmt.copy() if first_pass_search_rqmt else {}
    first_pass_search_rqmt.setdefault("time", 24)
    first_pass_search_rqmt.setdefault("mem", 50)
    res = recog_model(
        task=task,
        model=model,
        recog_def=model_recog_with_recomb_labelsync,
        config={
            **base_config,
            "beam_size": first_pass_recog_beam_size,
            # Batch size was fitted on our small GPUs (1080) with 11GB for beam size 32.
            # So when the beam size is larger, reduce batch size.
            # (Linear is a bit wrong, because the encoder mem consumption is independent, but anyway...)
            "batch_size": int(
                1_000 * aed_ctc_model.definition.batch_size_factor * min(32 / first_pass_recog_beam_size, 1)
            ),
        },
        search_rqmt=first_pass_search_rqmt,
        name=f"{prefix}/recog-opt-1stpass",
    )
    tk.register_output(f"{prefix}/recog-1stpass-res.txt", res.output)

    # now we have computed the best scales, lets return a model with these scales (but scaled down such that they sum to 1 to compare to regular DLM scoring)
    factor = 1.0 / (1.0 + aed_scale)
    model = get_aed_ctc_and_labelwise_prior(
        aed_ctc_model=aed_ctc_model, aed_scale=(factor * aed_scale), ctc_scale=factor
    )
    return model


def dlm_with_ctc_recomb_rescore_def(
    *,
    model: AEDModel,  # we adjusted our DLModel to be compatible with AEDModel
    data: Tensor,
    data_spatial_dim: Dim,
    targets: Tensor,
    targets_beam_dim: Dim,
    targets_spatial_dim: Dim,
    **_other,
):
    from returnn.config import get_global_config

    targets_beam_dim  # noqa  # unused here

    config = get_global_config()  # noqa

    aed_scale = config.float("aed_scale", 1.0)
    ctc_scale = config.float("ctc_scale", 1.0)

    # input_add_eos = config.typed_value("input_add_eos", False)
    # input_add_bos = config.typed_value("input_add_bos", False)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim

    enc_collected_outputs = {}

    # model.encode internally adds eos, bos if needed...
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=enc_collected_outputs)

    ctc_layer_idx = model.enc_aux_logits[-1]
    linear = getattr(model, f"enc_aux_logits_{ctc_layer_idx}")
    ctc_logits = linear(enc_collected_outputs[str(ctc_layer_idx - 1)])
    # ctc_label_log_prob = rf.log_softmax(ctc_logits, axis=model.wb_target_dim)  # Batch, Spatial, VocabWB
    # we manually removed the prob for eos in the "linear" forward, so no need to add them to targets
    ctc_target_probs = -rf.ctc_loss(
        logits=ctc_logits,
        logits_normalized=False,
        targets=targets,
        input_spatial_dim=enc_spatial_dim,  # only works because this isnt actually the encoder spatial dim but the ctc spatial dim
        targets_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx,
        max_approx=False,  # This is wrong! (for now). We should use max reduction here... TODO TODO TODO
    )

    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx, out_dims=[targets_w_eos_spatial_dim]
    )

    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=targets.remaining_dims(targets_spatial_dim)),
    )

    assert not model.out_eos_separated  # joint distrib, std case
    log_prob = rf.log_softmax(logits, axis=model.target_dim)
    log_prob_targets = rf.gather(
        log_prob, indices=targets_w_eos, axis=model.target_dim
    )  # [batch,beam,targets_spatial_w_eos]
    log_prob_targets_seq = rf.reduce_sum(log_prob_targets, axis=targets_w_eos_spatial_dim)  # [batch,beam]
    assert log_prob_targets_seq.dims_set == set(targets.remaining_dims(targets_spatial_dim))

    return aed_scale * log_prob_targets_seq + ctc_scale * ctc_target_probs
