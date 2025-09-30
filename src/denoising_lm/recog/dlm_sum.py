"""
Code for CTC + error correction model

Do first greedy decoding with CTC, then feed that to the error correction model,
then do time-sync beam search over the CTC scores, adding the error correction model scores.
We can do that in a single function, such that the CTC encoder is only called once.

Tuning the scales can be done on dev-other.
"""

from __future__ import annotations
from typing import Optional, Any, Sequence, Generator, Tuple, Dict

from i6_experiments.users.zeyer.model_interfaces import RecogDef
from i6_experiments.users.zeyer.nn_rf.soft_collapse_repeated import soft_collapse_repeated
from i6_experiments.users.zeyer.nn_rf.top_k_and_random_choice_without_replacement import (
    top_k_and_random_choice_without_replacement,
)

from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
    Model as CtcModel,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_label_sync_espnet import (
    CtcPrefixScorer,
)

from ..model.error_correction_model import Model as DenoisingLanguageModel

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray


def ctc_model_with_dlm_sum_recog(
    *,
    model: CtcModel,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Code copied and adapted from
    :func:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc.model_recog`.

    Function is run within RETURNN.

    Note, for debugging, see :func:`model_recog_debug` below.

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
    from returnn.config import get_global_config

    # We usually have TransformerDecoder, but any other type would also be ok when it has the same API.
    # noinspection PyUnresolvedReferences
    dlm: DenoisingLanguageModel = model.lm
    # noinspection PyUnresolvedReferences
    lm_scale: float = model.lm_scale

    # noinspection PyUnresolvedReferences
    labelwise_prior: Optional[rf.Parameter] = model.labelwise_prior

    # noinspection PyUnresolvedReferences
    blank_penalty: float = model.blank_penalty
    assert blank_penalty == 0.0, "custom blank_penalty does not make sense?"

    config = get_global_config()
    version = config.int("recog_version", 1)
    assert version in ({3, 4} if labelwise_prior is not None else {4}), f"invalid recog_version {version}"
    beam_size = config.int("beam_size", 12)
    ctc_beam_size = config.int("ctc_beam_size", beam_size)
    ctc_num_hyps = config.int("ctc_num_hyps", 1)
    assert ctc_num_hyps <= ctc_beam_size
    ctc_soft_collapse_threshold = config.typed_value("ctc_soft_collapse_threshold", None)
    ctc_soft_collapse_reduce_type = config.typed_value("ctc_soft_collapse_reduce_type", "logmeanexp")
    length_normalization_exponent = config.float("length_normalization_exponent", 1.0)
    length_normalization_each_term = config.bool(
        "length_normalization_each_term", False
    )  # if True, we don't length norm the whole prob, only those terms which are actually changing with length
    initial_ctc_search_type = config.typed_value("initial_ctc_search_type", "time_sync")
    ctc_prefix_score_scale = config.float("ctc_prefix_score_scale", 0.0)
    ctc_final_prefix_score_scale = config.float("ctc_final_prefix_score_scale", ctc_prefix_score_scale)
    if ctc_final_prefix_score_scale != 0:
        assert ctc_prefix_score_scale != 0
    return_only_dlm_score = config.bool("return_only_dlm_score", False)
    ctc_top_k_with_random_sampling = config.float(
        "ctc_top_k_with_random_sampling", 0.0
    )  # 0 disabled, 1 enabled. but a smooth transition is possible
    ctc_top_k_with_random_sampling_opts: Optional[Dict[str, Any]] = None
    if ctc_top_k_with_random_sampling:
        ctc_top_k_with_random_sampling_opts = {"max_noise_scale": ctc_top_k_with_random_sampling}
    ctc_top_p = config.typed_value("ctc_top_p", None)  # 1.0 picks all (no effect). e.g. use 0.9.
    if ctc_top_p is not None:
        ctc_top_k_with_random_sampling_opts["top_p"] = ctc_top_p
    if config.typed_value("ctc_top_k_with_random_sampling_opts", None):
        ctc_top_k_with_random_sampling_opts.update(config.typed_value("ctc_top_k_with_random_sampling_opts", None))
    if ctc_top_k_with_random_sampling_opts:
        for k in ["top_p"]:
            v = ctc_top_k_with_random_sampling_opts.get(k, None)
            if v is not None:
                ctc_top_k_with_random_sampling_opts[k] = rf.convert_to_tensor(v, device=data.device)
    dlm_max_seq_len_cfg = config.typed_value("dlm_max_seq_len", None)
    dlm_temperature = config.float("dlm_temperature", 1.0)
    asr_score_method = config.typed_value("asr_score_method", "max")
    assert not config.has("temperature")  # only dlm temperature!

    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext import ctc_debugging

    _seq_label_history_init_state = ctc_debugging._seq_label_history_init_state
    _seq_label_append = ctc_debugging._seq_label_append

    neg_inf = float("-inf")

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    logits, enc_out, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    print("Encoder seq lens:", enc_spatial_dim.get_size_tensor().raw_tensor)
    enc_spatial_dim_original = enc_spatial_dim
    # Eager-mode implementation of beam search.

    # The label log probs include the AM and the (scaled) prior.
    ctc_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    if ctc_soft_collapse_threshold is not None:
        ctc_log_prob, enc_spatial_dim = soft_collapse_repeated(
            ctc_log_prob,
            spatial_dim=enc_spatial_dim,
            classes_dim=model.wb_target_dim,
            threshold=ctc_soft_collapse_threshold,
            reduce_type=ctc_soft_collapse_reduce_type,
        )
    ctc_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        ctc_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=neg_inf),
    )

    if initial_ctc_search_type == "time_sync":
        # Initial state for CTC beam search.
        ctc_beam_dim = Dim(1, name="ctc_initial_beam")
        ctc_seq_log_prob = rf.constant(0.0, dims=[ctc_beam_dim] + batch_dims)  # Batch, Beam
        max_seq_len = int(enc_spatial_dim.get_dim_value())
        ctc_seq_targets = []
        ctc_seq_backrefs = []
        ctc_log_prob_ta = TensorArray.unstack(ctc_log_prob, axis=enc_spatial_dim)  # t -> Batch, VocabWB
        for t in range(max_seq_len):
            ctc_seq_log_prob = ctc_seq_log_prob + ctc_log_prob_ta[t]  # Batch, InBeam, VocabWB
            if ctc_top_k_with_random_sampling:
                ctc_seq_log_prob, (backrefs, target), ctc_beam_dim = top_k_and_random_choice_without_replacement(
                    ctc_seq_log_prob,
                    axis=[ctc_beam_dim, model.wb_target_dim],
                    k=Dim(ctc_beam_size, name=f"ctc_time_step{t}_beam"),
                    **ctc_top_k_with_random_sampling_opts,
                )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> VocabWB.
            else:
                ctc_seq_log_prob, (backrefs, target), ctc_beam_dim = rf.top_k(
                    ctc_seq_log_prob,
                    k_dim=Dim(ctc_beam_size, name=f"ctc_time_step{t}_beam"),
                    axis=[ctc_beam_dim, model.wb_target_dim],
                )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> VocabWB.
            ctc_seq_targets.append(target)
            ctc_seq_backrefs.append(backrefs)

        # Backtrack via backrefs, resolve beams.
        ctc_seq_targets_ = []
        indices = rf.range_over_dim(ctc_beam_dim)  # FinalBeam -> FinalBeam
        for backrefs, target in zip(ctc_seq_backrefs[::-1], ctc_seq_targets[::-1]):
            # indices: FinalBeam -> Beam
            # backrefs: Beam -> PrevBeam
            ctc_seq_targets_.insert(0, rf.gather(target, indices=indices))
            indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

        ctc_seq_targets__ = TensorArray(ctc_seq_targets_[0])
        for target in ctc_seq_targets_:
            ctc_seq_targets__ = ctc_seq_targets__.push_back(target)
        ctc_seq_targets = ctc_seq_targets__.stack(axis=enc_spatial_dim)  # Batch, FinalCtcBeam, EncSpatial

        # Now remove repetitions and the blank label, to get a non-blank label sequence.
        ctc_seq_targets_shifted = rf.shift_right(ctc_seq_targets, axis=enc_spatial_dim, pad_value=model.blank_idx)
        ctc_seq_targets, labels_spatial_dim = rf.masked_select(
            ctc_seq_targets,
            mask=(ctc_seq_targets != model.blank_idx) & (ctc_seq_targets != ctc_seq_targets_shifted),
            dims=[enc_spatial_dim],
        )  # Batch, FinalBeam, LabelsSpatial
        # Set correct sparse_dim. Only works if blank comes after.
        assert model.target_dim.dimension == model.blank_idx
        ctc_seq_targets.sparse_dim = model.target_dim

        # Now, we have lots of duplicates in the CTC beam (ctc_beam_dim).
        # Remove duplicates.
        from .ctc_with_dlm import _same_seq_labels

        ctc_same_seq_labels, ctc_beam_dual_dim = _same_seq_labels(
            ctc_seq_targets, spatial_dim=labels_spatial_dim, beam_dim=ctc_beam_dim
        )  # Batch, CtcBeam, CtcBeamDual
        ctc_seq_log_prob_ext = rf.where(
            ctc_same_seq_labels,
            rf.replace_dim_v2(ctc_seq_log_prob, in_dim=ctc_beam_dim, out_dim=ctc_beam_dual_dim),
            neg_inf,
        )  # Batch, CtcBeam, CtcBeamDual
        argmax_ctc_seq_log_prob = rf.reduce_argmax(
            ctc_seq_log_prob_ext, axis=ctc_beam_dual_dim
        )  # Batch, CtcBeam -> CtcBeamDual
        ctc_seq_mask = argmax_ctc_seq_log_prob == rf.range_over_dim(ctc_beam_dim)  # Batch, CtcBeam
        count_unique_ctc_seqs = rf.reduce_sum(rf.cast(ctc_seq_mask, "int32"), axis=ctc_beam_dim)  # Batch
        print("Count unique CTC seqs:", count_unique_ctc_seqs.raw_tensor.cpu().numpy().tolist())
        # We want to make sure to have at least ctc_num_hyps different hypotheses,
        # but also not more, i.e. having ctc_num_hyps_dim as static dim.
        _, ctc_seq_mask_indices, ctc_num_hyps_dim = rf.top_k(
            # top_k on Torch does not support bool.
            rf.cast(ctc_seq_mask, "int32"),
            axis=ctc_beam_dim,
            k_dim=Dim(ctc_num_hyps, name="ctc_num_hyps"),
        )  # Batch, CtcNumHyps -> CtcBeam
        ctc_seq_targets, ctc_seq_log_prob, labels_spatial_dim = rf.nested.gather_nested(
            (ctc_seq_targets, ctc_seq_log_prob, labels_spatial_dim), indices=ctc_seq_mask_indices
        )  # Batch, CtcNumHyps, (...)

    elif initial_ctc_search_type == "label_sync":
        ctc_beam_dim = Dim(1, name="ctc_initial_beam")
        ctc_prefix_scorer = CtcPrefixScorer(
            log_probs=ctc_log_prob,
            batch_dims=batch_dims,
            enc_spatial_dim=enc_spatial_dim,
            vocab_wb_dim=model.wb_target_dim,
            vocab_dim=model.target_dim,
            blank_idx=model.blank_idx,
            eos_idx=model.eos_idx,
        )
        ctc_prefix_scorer_state = None
        ctc_seq_log_prob = rf.constant(0.0, dims=[ctc_beam_dim] + batch_dims)  # Batch, InBeam
        target = rf.constant(
            model.bos_idx, dims=[ctc_beam_dim] + batch_dims, sparse_dim=model.target_dim
        )  # Batch, InBeam -> Vocab
        ended = rf.constant(False, dims=[ctc_beam_dim] + batch_dims)
        out_seq_len = rf.constant(0, dims=[ctc_beam_dim] + batch_dims)

        max_seq_len = enc_spatial_dim.get_size_tensor(device=data.device)

        i = 0
        seq_targets = []
        seq_backrefs = []
        while True:
            ctc_prefix_log_prob, ctc_prefix_scorer_state = ctc_prefix_scorer.score_and_update_state(
                prev_label=target, prev_state=ctc_prefix_scorer_state, beam_dim=ctc_beam_dim
            )
            # Filter out finished beams
            ctc_prefix_log_prob = rf.where(
                ended,
                rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=neg_inf),
                ctc_prefix_log_prob,
            )
            ctc_seq_log_prob = ctc_seq_log_prob + ctc_prefix_log_prob  # Batch, InBeam, Vocab

            if ctc_top_k_with_random_sampling:
                ctc_seq_log_prob, (backrefs, target), ctc_beam_dim = top_k_and_random_choice_without_replacement(
                    ctc_seq_log_prob,
                    axis=[ctc_beam_dim, model.target_dim],
                    k=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam"),
                    **ctc_top_k_with_random_sampling_opts,
                )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.
            else:
                ctc_seq_log_prob, (backrefs, target), ctc_beam_dim = rf.top_k(
                    ctc_seq_log_prob,
                    k_dim=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam"),
                    axis=[ctc_beam_dim, model.target_dim],
                )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.

            target = rf.cast(target, dtype=rf.get_default_int_dtype())
            seq_targets.append(target)
            seq_backrefs.append(backrefs)
            ended = rf.gather(ended, indices=backrefs)
            out_seq_len = rf.gather(out_seq_len, indices=backrefs)
            ctc_prefix_scorer_state = rf.nested.gather_nested(ctc_prefix_scorer_state, indices=backrefs)

            i += 1
            ended = rf.logical_or(ended, target == model.eos_idx)
            ended = rf.logical_or(ended, i >= max_seq_len)
            if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
                break
            out_seq_len = out_seq_len + rf.where(ended, 0, 1)

        # Backtrack via backrefs, resolve beams.
        seq_targets_ = []
        indices = rf.range_over_dim(ctc_beam_dim)  # FinalBeam -> FinalBeam
        for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
            # indices: FinalBeam -> Beam
            # backrefs: Beam -> PrevBeam
            seq_targets_.insert(0, rf.gather(target, indices=indices))
            indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

        seq_targets__ = TensorArray(seq_targets_[0])
        for target in seq_targets_:
            seq_targets__ = seq_targets__.push_back(target)
        labels_spatial_dim = Dim(out_seq_len, name="ctc_labels_spatial")
        ctc_seq_targets = seq_targets__.stack(axis=labels_spatial_dim)
        # Remove the remaining EOS labels.
        ctc_seq_targets, _ = rf.slice(ctc_seq_targets, axis=labels_spatial_dim, size=labels_spatial_dim)
        assert ctc_num_hyps == ctc_beam_size  # could also support less, not implemented...
        ctc_num_hyps_dim = ctc_beam_dim

    else:
        raise ValueError(f"invalid initial_ctc_search_type {initial_ctc_search_type!r}")

    print("CTC out:")
    _generic_seq_label_print(ctc_seq_targets, labels_spatial_dim)
    labels_seq_lens = labels_spatial_dim.get_size_tensor()  # Batch, CtcNumHyps
    labels_max_seq_lens = rf.reduce_max(labels_seq_lens, axis=labels_seq_lens.remaining_dims(batch_dims))  # Batch
    print("Max CTC label seq lens:", labels_max_seq_lens.raw_tensor)

    if asr_score_method == "ctc":
        prev_dim_set = ctc_seq_log_prob.dims_set
        # rescore with ctc, see ctc_model_rescore
        ctc_log_prob_real = model.log_probs_wb_from_logits(logits)
        neg_log_prob_ = []
        for hyp_idx in range(ctc_num_hyps_dim.get_dim_value()):
            targets_b = rf.gather(ctc_seq_targets, axis=ctc_num_hyps_dim, indices=hyp_idx)
            targets_b_seq_lens = rf.gather(labels_spatial_dim.dyn_size_ext, axis=ctc_num_hyps_dim, indices=hyp_idx)
            targets_b_spatial_dim = Dim(targets_b_seq_lens, name=f"{labels_spatial_dim.name}_hyp{hyp_idx}")
            targets_b, _ = rf.replace_dim(targets_b, in_dim=labels_spatial_dim, out_dim=targets_b_spatial_dim)
            targets_b, _ = rf.slice(targets_b, axis=targets_b_spatial_dim, size=targets_b_spatial_dim)
            neg_log_prob = rf.ctc_loss(
                logits=ctc_log_prob_real,
                logits_normalized=True,
                targets=targets_b,
                input_spatial_dim=enc_spatial_dim_original,
                targets_spatial_dim=targets_b_spatial_dim,
                blank_index=model.blank_idx,
            )
            neg_log_prob_.append(neg_log_prob)
        neg_log_prob, _ = rf.stack(neg_log_prob_, out_dim=ctc_num_hyps_dim)
        ctc_seq_log_prob = -neg_log_prob
        assert ctc_seq_log_prob.dims_set == prev_dim_set
    elif asr_score_method == "max":
        pass  # already done
    else:
        raise ValueError(f"invalid asr_score_method {asr_score_method!r}")

    input_add_bos = config.typed_value("input_add_bos", False)
    input_add_eos = config.typed_value("input_add_eos", False)
    if input_add_bos:
        ctc_seq_targets, (labels_spatial_dim,) = rf.pad(
            ctc_seq_targets, axes=[labels_spatial_dim], padding=[(int(input_add_bos), 0)], value=model.bos_idx
        )
    if input_add_eos:
        ctc_seq_targets, (labels_spatial_dim,) = rf.pad(
            ctc_seq_targets, axes=[labels_spatial_dim], padding=[(0, int(input_add_eos))], value=model.eos_idx
        )

    # Renormalize log probs.
    ctc_seq_log_prob -= rf.reduce_logsumexp(ctc_seq_log_prob, axis=ctc_num_hyps_dim)  # Batch, CtcNumHyps

    beam_dim = Dim(1, name="initial_beam")
    target = rf.constant(
        model.bos_idx, dims=[beam_dim] + batch_dims, sparse_dim=model.target_dim
    )  # Batch, InBeam -> Vocab
    ended = rf.constant(False, dims=[beam_dim] + batch_dims)
    seq_log_prob = rf.constant(0.0, dims=[beam_dim, ctc_num_hyps_dim] + batch_dims)  # Batch, CtcNumHyps, InBeam
    prior_seq_log_prob = None
    if labelwise_prior is not None:
        prior_seq_log_prob = rf.constant(0.0, dims=[beam_dim] + batch_dims)  # Batch, InBeam
    enc = dlm.encode(ctc_seq_targets, spatial_dim=labels_spatial_dim)  # Batch, CtcNumHyps, ...
    lm_state = dlm.decoder.default_initial_state(
        batch_dims=[beam_dim, ctc_num_hyps_dim] + batch_dims
    )  # Batch, CtcNumHyps, InBeam, ...
    out_seq_len = rf.constant(0, dims=[beam_dim] + batch_dims)

    ctc_prefix_scorer, ctc_prefix_seq_log_prob, ctc_prefix_log_prob, ctc_prefix_scorer_state = None, None, None, None
    if ctc_prefix_score_scale != 0:
        ctc_prefix_scorer = CtcPrefixScorer(
            log_probs=ctc_log_prob,
            batch_dims=batch_dims,
            enc_spatial_dim=enc_spatial_dim,
            vocab_wb_dim=model.wb_target_dim,
            vocab_dim=model.target_dim,
            blank_idx=model.blank_idx,
            eos_idx=model.eos_idx,
        )
        ctc_prefix_seq_log_prob = rf.constant(0.0, dims=[beam_dim] + batch_dims)  # Batch, InBeam

    if dlm_max_seq_len_cfg is None:
        # Old default. Note that this is probably way more than needed.
        max_seq_len = enc_spatial_dim.get_size_tensor(device=target.device)
    elif dlm_max_seq_len_cfg == "ctc":
        max_seq_len = rf.cast(rf.ceil(rf.cast(labels_max_seq_lens, "float32") * 1.5), "int32")
    else:
        raise ValueError(f"invalid dlm_max_seq_len {dlm_max_seq_len_cfg!r} (type {type(dlm_max_seq_len_cfg)})")
    max_seq_len = rf.copy_to_device(max_seq_len, device=target.device)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        logits, lm_state = dlm.decoder(
            target,
            spatial_dim=single_step_dim,
            encoder=enc,
            state=lm_state,
        )
        label_log_prob = rf.log_softmax(logits / dlm_temperature, axis=model.target_dim)
        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=neg_inf),
            label_log_prob,
        )
        seq_log_prob = seq_log_prob + label_log_prob  # Batch, CtcNumHyps, InBeam, Vocab

        length_norm_factor = (
            1.0 / (rf.cast(out_seq_len, dtype=seq_log_prob.dtype) + 1.0)
        ) ** length_normalization_exponent
        if length_normalization_each_term:
            seq_log_prob_ = length_norm_factor * seq_log_prob + ctc_seq_log_prob  # Batch, CtcNumHyps, InBeam, Vocab
        else:
            seq_log_prob_ = seq_log_prob + ctc_seq_log_prob  # Batch, CtcNumHyps, InBeam, Vocab
        seq_log_prob_ = rf.reduce_logsumexp(seq_log_prob_, axis=ctc_num_hyps_dim)  # Batch, InBeam, Vocab
        if lm_scale != 1.0:
            seq_log_prob_ *= lm_scale
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
            seq_log_prob_ = seq_log_prob_ + (ctc_prefix_seq_log_prob + ctc_prefix_log_prob) * ctc_prefix_score_scale * (
                length_norm_factor if length_normalization_each_term else 1.0
            )  # Batch, InBeam, Vocab
        if labelwise_prior is not None:
            if length_normalization_each_term:
                seq_log_prob_ -= length_norm_factor * prior_seq_log_prob
                seq_log_prob_ -= length_norm_factor * labelwise_prior
            else:
                seq_log_prob_ -= prior_seq_log_prob
                seq_log_prob_ -= labelwise_prior  # prior already scaled
        if i > 1 and length_normalization_exponent != 0 and not length_normalization_each_term:
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
        seq_log_prob = rf.gather(seq_log_prob, indices=backrefs)  # Batch, CtcNumHyps, Beam, Vocab
        seq_log_prob = rf.gather(seq_log_prob, indices=target)  # Batch, CtcNumHyps, Beam
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

    seq_log_prob = seq_log_prob + ctc_seq_log_prob  # Batch, CtcNumHyps, Beam
    seq_log_prob = rf.reduce_logsumexp(seq_log_prob, axis=ctc_num_hyps_dim)  # Batch, Beam
    if return_only_dlm_score:
        pass
    else:
        if lm_scale != 1.0:
            seq_log_prob *= lm_scale
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
ctc_model_with_dlm_sum_recog: RecogDef[CtcModel]
ctc_model_with_dlm_sum_recog.output_with_beam = True
ctc_model_with_dlm_sum_recog.output_blank_label = None
ctc_model_with_dlm_sum_recog.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def _generic_seq_label_print(labels: Tensor, spatial_dim: Dim):
    labels = rf.copy_to_device(labels, "cpu")
    batch_dims = labels.remaining_dims(spatial_dim)
    for indices in _iter_dims_indices(batch_dims):
        print(" ", end="")
        hist_seq_len_ = spatial_dim.get_size_tensor()
        hist_ = labels
        for dim, i in zip(batch_dims, indices):
            hist_ = rf.gather(hist_, axis=dim, indices=i)
            if dim in hist_seq_len_.dims:
                hist_seq_len_ = rf.gather(hist_seq_len_, axis=dim, indices=i)
            print(f" {dim}={i}", end="")
        hist_, _ = rf.slice(hist_, axis=spatial_dim, size=hist_seq_len_)
        print(
            f": len={hist_seq_len_.raw_tensor}"
            f" {[labels.sparse_dim.vocab.id_to_label(l.item()) for l in hist_.raw_tensor]}"
        )


def _generic_print(tensor: Tensor):
    tensor = rf.copy_to_device(tensor, "cpu")
    for indices in _iter_dims_indices(tensor.dims):
        print(" ", end="")
        tensor_ = tensor
        for dim, i in zip(tensor.dims, indices):
            tensor_ = rf.gather(tensor_, axis=dim, indices=i)
            print(f" {dim}={i}", end="")
        print(f": {tensor_.raw_tensor.item()}")


def _iter_dims_indices(dims: Sequence[Dim]) -> Generator[Tuple[int, ...]]:
    if not dims:
        yield ()
        return
    dim, rest = dims[0], dims[1:]
    for i in range(dim.get_dim_value()):
        for rest_indices in _iter_dims_indices(rest):
            yield (i,) + rest_indices
        break


def ctc_model_with_dlm_sum_recog_debug_batching(
    *,
    model: CtcModel,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Debug :func:`ctc_model_with_dlm_sum_recog`, to test that batching is correct,
    i.e. single seq yields the same result as batched seqs.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim (size shape usually {batch, beam}),
        final beam_dim
    """
    import torch
    from returnn.tensor import batch_dim
    from returnn.util.debug import PyTracer
    from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

    # noinspection PyProtectedMember
    from torch.testing._comparison import make_tensor_mismatch_msg

    eps = 1e-3
    # Torch defaults: rtol: _float = 1e-05, atol: _float = 1e-08
    rtol, atol = eps, eps
    batch_size = int(batch_dim.get_dim_value())
    if batch_size == 1:  # only single seq
        print("(Single seq in batch)")
        return ctc_model_with_dlm_sum_recog(model=model, data=data, data_spatial_dim=data_spatial_dim)

    print(f"(Batched seqs, num seqs: {batch_size})")
    # First do whole batch.
    # Funcs to trace: Maybe also include: ctc_model_with_dlm_sum_recog, ConformerEncoderLayer.__call__
    funcs_to_trace_list = [CtcModel.__call__, ConformerEncoder.__call__, ConformerConvSubsample.__call__]
    with PyTracer(funcs_to_trace_list, Tensor) as tracer_batch:
        output, log_probs, out_spatial_dim, beam_dim = ctc_model_with_dlm_sum_recog(
            model=model, data=data, data_spatial_dim=data_spatial_dim
        )
    # Now do single seqs.
    for batch_idx in range(batch_size):
        data_b, data_spatial_b_dim, output_b, log_probs_b, out_spatial_b_dim, beam_b_dim = rf.nested.gather_nested(
            (data, data_spatial_dim, output, log_probs, out_spatial_dim, beam_dim),
            indices=rf.convert_to_tensor(batch_idx, dims=(), sparse_dim=batch_dim),
        )
        # The dummy dim allows for easier comparison with the batched version but is otherwise not really needed.
        dummy_ext_dim = Dim(1, name="dummy_batch")
        data_b = rf.expand_dim(data_b, dummy_ext_dim)
        data_b = data_b.copy_transpose(
            [dummy_ext_dim, data_spatial_b_dim] + data.remaining_dims((batch_dim, data_spatial_dim))
        )
        with PyTracer(funcs_to_trace_list, Tensor) as tracer_single:
            (
                output_b_single,
                log_probs_b_single,
                out_spatial_b_single_dim,
                beam_b_single_dim,
            ) = ctc_model_with_dlm_sum_recog(model=model, data=data_b, data_spatial_dim=data_spatial_b_dim)
        assert set(tracer_batch.captured_locals.keys()) == set(tracer_single.captured_locals.keys())
        for func in tracer_batch.captured_locals:
            capt_func_batch = tracer_batch.captured_locals[func]
            capt_func_single = tracer_single.captured_locals[func]
            assert len(capt_func_batch) == len(capt_func_single), f"func {func} num calls differs"
            for call_idx in range(len(capt_func_batch)):
                capt_locals_batch = capt_func_batch[call_idx]
                capt_locals_single = capt_func_single[call_idx]
                assert set(capt_locals_batch.keys()) == set(capt_locals_single.keys())
                for name in capt_locals_batch:
                    capt_locals_name_batch = capt_locals_batch[name]
                    capt_locals_name_single = capt_locals_single[name]
                    for idx in range(min(len(capt_locals_name_batch), len(capt_locals_name_single))):
                        capt_tensor_batch = capt_locals_name_batch[idx]
                        capt_tensor_single = capt_locals_name_single[idx]
                        assert isinstance(capt_tensor_batch, Tensor) and isinstance(capt_tensor_single, Tensor)
                        assert len(capt_tensor_batch.dims) == len(capt_tensor_single.dims)
                        dim_map = {data_spatial_dim: data_spatial_b_dim}
                        for dim_batch, dim_single in zip(capt_tensor_batch.dims, capt_tensor_single.dims):
                            if dim_batch == batch_dim:
                                assert dim_single == dummy_ext_dim
                            elif dim_batch == data_spatial_dim:
                                assert dim_single == data_spatial_b_dim
                            elif dim_batch.dimension is not None or dim_single.dimension is not None:
                                assert dim_single.dimension == dim_batch.dimension
                            if dim_batch != dim_single and dim_batch != batch_dim:
                                dim_map[dim_batch] = dim_single
                        capt_tensor_batch = rf.nested.gather_nested(
                            capt_tensor_batch,
                            indices=rf.convert_to_tensor(batch_idx, dims=(), sparse_dim=batch_dim),
                            dim_map=dim_map,
                        )
                        capt_tensor_single = rf.squeeze(capt_tensor_single, dummy_ext_dim)
                        matches = torch.isclose(
                            capt_tensor_single.raw_tensor, capt_tensor_batch.raw_tensor, rtol=rtol, atol=atol
                        )
                        if not matches.all():
                            print(
                                f"func {func} call {call_idx} local {name!r} idx {idx},",
                                f"{capt_tensor_batch} vs {capt_tensor_single}:",
                                make_tensor_mismatch_msg(
                                    capt_tensor_single.raw_tensor,
                                    capt_tensor_batch.raw_tensor,
                                    matches,
                                    rtol=rtol,
                                    atol=atol,
                                    identifier="Tensors",
                                ),
                            )
        output_b_single = rf.squeeze(output_b_single, dummy_ext_dim)
        log_probs_b_single = rf.squeeze(log_probs_b_single, dummy_ext_dim)
        # Note: if any of these fail, and we want to debug further,
        # we can use Python tracing (e.g. via PyTracer) to investigate intermediate outputs.
        assert log_probs_b.dims_set == {beam_b_dim}
        assert log_probs_b_single.dims_set == {beam_b_single_dim}
        assert beam_b_dim.get_size_tensor().dims == beam_b_single_dim.get_size_tensor().dims == ()
        assert beam_b_dim.get_dim_value() == beam_b_single_dim.get_dim_value()
        assert out_spatial_b_dim.dyn_size_ext.dims == (beam_b_dim,)
        out_spatial_b_single_dim_dyn_size_ext = out_spatial_b_single_dim.dyn_size_ext
        if dummy_ext_dim in out_spatial_b_single_dim_dyn_size_ext.dims:
            out_spatial_b_single_dim_dyn_size_ext = rf.squeeze(out_spatial_b_single_dim_dyn_size_ext, dummy_ext_dim)
        assert out_spatial_b_single_dim_dyn_size_ext.dims == (beam_b_single_dim,)
        assert output_b.dims_set == {beam_b_dim, out_spatial_b_dim}
        assert output_b_single.dims_set == {beam_b_single_dim, out_spatial_b_single_dim}
        output_b = output_b.copy_transpose((beam_b_dim, out_spatial_b_dim))
        output_b_single = output_b_single.copy_transpose((beam_b_single_dim, out_spatial_b_single_dim))
        for beam_idx in range(int(beam_b_dim.get_dim_value())):
            log_probs_b_b = log_probs_b.raw_tensor[beam_idx]
            log_probs_b_b_single = log_probs_b_single.raw_tensor[beam_idx]
            seq_len_b_b = out_spatial_b_dim.dyn_size[beam_idx]
            seq_len_b_b_single = out_spatial_b_single_dim_dyn_size_ext.raw_tensor[beam_idx]
            assert abs(log_probs_b_b - log_probs_b_b_single).cpu() < eps, (
                f"batch {batch_idx}, beam {beam_idx}, "
                f"log probs {log_probs_b_b} != {log_probs_b_b_single},"
                f" diff {abs(log_probs_b_b - log_probs_b_b_single)}."
                f" all log probs: {log_probs_b.raw_tensor} vs {log_probs_b_single.raw_tensor}"
            )
            assert seq_len_b_b == seq_len_b_b_single
            assert (
                output_b.raw_tensor[beam_idx, :seq_len_b_b].cpu().numpy().tolist()
                == output_b_single.raw_tensor[beam_idx, :seq_len_b_b].cpu().numpy().tolist()
            )

    return output, log_probs, out_spatial_dim, beam_dim
