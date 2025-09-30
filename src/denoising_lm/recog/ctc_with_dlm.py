"""
Code for CTC + error correction model

Do first greedy decoding with CTC, then feed that to the error correction model,
then do time-sync beam search over the CTC scores, adding the error correction model scores.
We can do that in a single function, such that the CTC encoder is only called once.

Tuning the scales can be done on dev-other.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Tuple, Dict

from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase

from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, ModelDefWithCfg, ModelWithCheckpoint

from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
    Model as CtcModel,
    ctc_model_def,
    _batch_size_factor,
)

from ..model.error_correction_model import aed_model_def as dlm_model_def, Model as DenoisingLanguageModel

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray


def get_ctc_with_dlm_and_labelwise_prior(
    *,
    ctc_model: ModelWithCheckpoint,
    prior: Optional[tk.Path] = None,
    prior_type: str = None,
    prior_scale: Optional[Union[float, tk.Variable, DelayedBase]] = None,
    language_model: ModelWithCheckpoint,
    lm_scale: Union[float, tk.Variable],
    blank_penalty: Optional[Union[float, tk.Variable]] = None,
) -> ModelWithCheckpoint:
    """
    Combined CTC model with DLM and labelwise prior

    :param ctc_model:
    :param prior: path to labelwise prior file
    :param prior_type: "prob" or "log_prob"
    :param prior_scale: scale for the prior
    :param language_model: DLM
    :param lm_scale: scale for the DLM
    :param blank_penalty:
    """
    # Keep CTC model config as-is, extend below for prior and LM.
    ctc_model_def_ = ctc_model.definition
    if isinstance(ctc_model_def_, ModelDefWithCfg):
        assert ctc_model_def_.model_def is ctc_model_def
        config: Dict[str, Any] = ctc_model_def_.config.copy()
    else:
        assert ctc_model_def_ is ctc_model_def
        config = {}

    # Add prior.
    # Then the CTC Model log_probs_wb_from_logits will include the prior.
    if prior is not None:
        assert prior_scale is not None
    if prior_scale is not None:
        assert prior is not None and prior_type is not None
        config.update(
            {
                "labelwise_prior": {"type": prior_type, "file": prior, "scale": prior_scale},
            }
        )
    if blank_penalty is not None:
        config["blank_penalty"] = blank_penalty

    # Add LM.
    # LM has _model_def_dict in config. Put that as _lm_model_def_dict.
    assert language_model.definition.model_def is dlm_model_def
    dlm_model_config = language_model.definition.config.copy()
    assert "_encoder_model_dict" in dlm_model_config and "_decoder_model_dict" in dlm_model_config
    if dlm_model_config.get("_model_dict"):
        dlm_model_dict = dlm_model_config.pop("_model_dict").copy()
        assert "encoder" not in dlm_model_dict and "decoder" not in dlm_model_dict
    else:
        dlm_model_dict = rf.build_dict(DenoisingLanguageModel)
    dlm_model_dict["encoder"] = dlm_model_config.pop("_encoder_model_dict")
    dlm_model_dict["decoder"] = dlm_model_config.pop("_decoder_model_dict")
    for k, v in dlm_model_config.items():
        assert k not in config
        config[k] = v
    config.update(
        {
            "_dlm_model_def_dict": dlm_model_dict,
            "lm_scale": lm_scale,
        }
    )
    config.setdefault("preload_from_files", {})["lm"] = {"prefix": "lm.", "filename": language_model.checkpoint}

    return ModelWithCheckpoint(
        definition=ModelDefWithCfg(model_def=ctc_model_with_dlm_def, config=config),
        checkpoint=ctc_model.checkpoint,
    )


def ctc_model_with_dlm_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> CtcModel:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config
    import numpy

    epoch  # noqa
    config = get_global_config()  # noqa

    dlm_model_dict = config.typed_value("_dlm_model_def_dict")
    dlm_in_dim = target_dim
    dlm_target_dim = target_dim
    lm = rf.build_from_dict(
        dlm_model_dict,
        dlm_in_dim,
        target_dim=dlm_target_dim,
        blank_idx=dlm_target_dim.dimension,
        bos_idx=_get_bos_idx(dlm_target_dim),
        eos_idx=_get_eos_idx(dlm_target_dim),
    )
    lm_scale = config.typed_value("lm_scale", None)
    assert isinstance(lm_scale, (int, float))

    # (framewise) ctc_prior_type / static_prior handled by ctc_model_def.
    model = ctc_model_def(epoch=epoch, in_dim=in_dim, target_dim=target_dim)
    model.lm = lm
    model.lm_scale = lm_scale

    labelwise_prior = config.typed_value("labelwise_prior", None)
    if labelwise_prior:
        assert isinstance(labelwise_prior, dict) and set(labelwise_prior.keys()) == {"type", "file", "scale"}
        v = numpy.loadtxt(labelwise_prior["file"])
        assert v.shape == (
            target_dim.dimension,
        ), f"invalid shape {v.shape} for labelwise_prior {labelwise_prior['file']!r}, expected dim {target_dim}"
        # The `type` is about what is stored in the file.
        # We always store it in log prob here, so we potentially need to convert it.
        if labelwise_prior["type"] == "log_prob":
            pass  # already log prob
        elif labelwise_prior["type"] == "prob":
            v = numpy.log(v)
        else:
            raise ValueError(f"invalid static_prior type {labelwise_prior['type']!r}")
        v *= labelwise_prior["scale"]  # can already apply now
        model.labelwise_prior = rf.Parameter(
            rf.convert_to_tensor(v, dims=[target_dim], dtype=rf.get_default_float_dtype()),
            auxiliary=True,
            non_critical_for_restore=True,
        )
    else:
        model.labelwise_prior = None

    model.blank_penalty = config.typed_value("blank_penalty", None) or 0.0

    return model


ctc_model_with_dlm_def: ModelDef[CtcModel]
ctc_model_with_dlm_def.behavior_version = 21
ctc_model_with_dlm_def.backend = "torch"
ctc_model_with_dlm_def.batch_size_factor = _batch_size_factor


def _get_bos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def _get_eos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.eos_label_id is not None:
        eos_idx = target_dim.vocab.eos_label_id
    else:
        raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
    return eos_idx


def ctc_model_with_dlm_recog(
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
    import returnn
    from returnn.config import get_global_config

    config = get_global_config()
    beam_size = config.int("beam_size", 12)
    version = config.int("recog_version", 1)
    assert version == 13, f"invalid recog_version {version}"
    recomb = config.typed_value("recog_recomb", None)  # None, "max", "sum"

    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext import ctc_debugging

    _generic_print = ctc_debugging._generic_print
    _seq_label_history_init_state = ctc_debugging._seq_label_history_init_state
    _seq_label_append = ctc_debugging._seq_label_append

    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250125, 0), returnn.__version__

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    logits, enc_out, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    neg_inf = float("-inf")

    # The label log probs include the AM and the (scaled) prior.
    ctc_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
    ctc_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        ctc_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=neg_inf),
    )

    ctc_greedy_labels = rf.reduce_argmax(ctc_log_prob, axis=model.wb_target_dim)
    ctc_greedy_labels = rf.cast(ctc_greedy_labels, "int32")

    ctc_greedy_labels_shifted = rf.shift_right(ctc_greedy_labels, axis=enc_spatial_dim, pad_value=model.blank_idx)
    ctc_greedy_labels, labels_spatial_dim = rf.masked_select(
        ctc_greedy_labels,
        mask=(ctc_greedy_labels != model.blank_idx) & (ctc_greedy_labels != ctc_greedy_labels_shifted),
        dims=[enc_spatial_dim],
    )  # [batch_dim,labels_spatial_dim]

    # Set correct sparse_dim. Only works if blank comes after.
    assert model.target_dim.dimension == model.blank_idx
    ctc_greedy_labels.sparse_dim = model.target_dim

    input_add_bos = config.typed_value("input_add_bos", False)
    input_add_eos = config.typed_value("input_add_eos", False)
    if input_add_bos:
        ctc_greedy_labels, (labels_spatial_dim,) = rf.pad(
            ctc_greedy_labels, axes=[labels_spatial_dim], padding=[(int(input_add_bos), 0)], value=model.bos_idx
        )
    if input_add_eos:
        ctc_greedy_labels, (labels_spatial_dim,) = rf.pad(
            ctc_greedy_labels, axes=[labels_spatial_dim], padding=[(0, int(input_add_eos))], value=model.eos_idx
        )

    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(
        model.blank_idx, dims=batch_dims_, sparse_dim=model.wb_target_dim
    )  # Batch, InBeam -> VocabWB

    # We usually have TransformerDecoder, but any other type would also be ok when it has the same API.
    # noinspection PyUnresolvedReferences
    dlm: DenoisingLanguageModel = model.lm
    # noinspection PyUnresolvedReferences
    lm_scale: float = model.lm_scale

    # noinspection PyUnresolvedReferences
    labelwise_prior: Optional[rf.Parameter] = model.labelwise_prior

    # noinspection PyUnresolvedReferences
    blank_penalty: float = model.blank_penalty

    ctc_seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam
    lm_seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam
    enc = dlm.encode(ctc_greedy_labels, spatial_dim=labels_spatial_dim)  # Batch, ...
    lm_state = dlm.decoder.default_initial_state(batch_dims=batch_dims_)  # Batch, InBeam, ...
    lm_logits, lm_state = dlm.decoder(
        target,
        encoder=enc,
        spatial_dim=single_step_dim,
        state=lm_state,
    )  # Batch, InBeam, Vocab / ...
    lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
    lm_log_probs = lm_log_probs.copy_compatible_to_dims(batch_dims_ + [model.target_dim])
    lm_log_probs *= lm_scale
    if labelwise_prior is not None:
        lm_log_probs -= labelwise_prior  # prior scale already applied

    seq_label = _seq_label_history_init_state(vocab_dim=model.target_dim, batch_dims=batch_dims_)

    ctc_label_log_prob_ta = TensorArray.unstack(ctc_log_prob, axis=enc_spatial_dim)  # t -> Batch, VocabWB
    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        ctc_seq_log_prob = ctc_seq_log_prob + ctc_label_log_prob_ta[t]  # Batch, InBeam, VocabWB

        lm_log_prob_wb = rf.where(
            (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
            _target_dense_extend_blank(
                lm_log_probs,
                target_dim=model.target_dim,
                wb_target_dim=model.wb_target_dim,
                blank_idx=model.blank_idx,
                value=-blank_penalty,
            ),
            0.0,
        )  # Batch, InBeam, VocabWB
        lm_seq_log_prob = lm_seq_log_prob + lm_log_prob_wb  # Batch, InBeam, VocabWB

        # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
        seq_log_prob = ctc_seq_log_prob + lm_seq_log_prob  # Batch, InBeam, VocabWB

        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, model.wb_target_dim]
        )
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        seq_targets_wb.append(target_wb)
        seq_backrefs.append(backrefs)

        ctc_seq_log_prob = rf.gather(ctc_seq_log_prob, indices=backrefs)  # Batch, Beam, VocabWB
        ctc_seq_log_prob = rf.gather(ctc_seq_log_prob, indices=target_wb)  # Batch, Beam
        lm_seq_log_prob = rf.gather(lm_seq_log_prob, indices=backrefs)  # Batch, Beam, VocabWB
        lm_seq_log_prob = rf.gather(lm_seq_log_prob, indices=target_wb)  # Batch, Beam

        lm_log_probs = rf.gather(lm_log_probs, indices=backrefs)  # Batch, Beam, Vocab
        lm_state = rf.nested.gather_nested(lm_state, indices=backrefs)
        prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
        prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB
        got_new_label = (target_wb != model.blank_idx) & (target_wb != prev_target_wb)  # Batch, Beam -> 0|1
        target = rf.where(
            got_new_label,
            _target_remove_blank(
                target_wb, target_dim=model.target_dim, wb_target_dim=model.wb_target_dim, blank_idx=model.blank_idx
            ),
            prev_target,
        )  # Batch, Beam -> Vocab
        seq_label = rf.nested.gather_nested(seq_label, indices=backrefs)

        got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
        if got_new_label_cpu.raw_tensor.sum().item() > 0:
            (
                (target_, lm_state_, seq_label_, enc_),
                packed_new_label_dim,
                packed_new_label_dim_map,
            ) = rf.nested.masked_select_nested(
                (target, lm_state, seq_label, enc),
                mask=got_new_label,
                mask_cpu=got_new_label_cpu,
                dims=batch_dims + [beam_dim],
            )
            # packed_new_label_dim_map: old dim -> new dim. see _masked_select_prepare_dims
            assert packed_new_label_dim.get_dim_value() > 0

            lm_logits_, lm_state_ = dlm.decoder(
                target_,
                encoder=enc_,
                spatial_dim=single_step_dim,
                state=lm_state_,
            )  # Flat_Batch_Beam, Vocab / ...
            lm_log_probs_ = rf.log_softmax(lm_logits_, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
            lm_log_probs_ = lm_log_probs_.copy_compatible_to_dims([packed_new_label_dim, model.target_dim])

            lm_log_probs_ *= lm_scale
            if labelwise_prior is not None:
                lm_log_probs_ -= labelwise_prior  # prior scale already applied

            seq_label_ = _seq_label_append(seq_label_, target_)
            # _seq_label_print(f"{t=} packed append", seq_label_)

            lm_log_probs, lm_state, seq_label = rf.nested.masked_scatter_nested(
                (lm_log_probs_, lm_state_, seq_label_),
                (lm_log_probs, lm_state, seq_label),
                mask=got_new_label,
                mask_cpu=got_new_label_cpu,
                dims=batch_dims + [beam_dim],
                in_dim=packed_new_label_dim,
                masked_select_dim_map=packed_new_label_dim_map,
            )  # Batch, Beam, Vocab / ...

        # Recombine paths with the same label seq.
        if not recomb:
            pass
        elif recomb in ("max", "sum"):
            # Set ctc_seq_log_prob for batch entries to neg_inf if they have the same label seq.
            same_seq_labels, beam_dual_dim = _same_seq_labels(
                seq_label.history, spatial_dim=seq_label.hist_dim, beam_dim=beam_dim
            )
            seq_log_prob_ext = rf.where(
                same_seq_labels, rf.replace_dim_v2(seq_log_prob, in_dim=beam_dim, out_dim=beam_dual_dim), neg_inf
            )  # Batch, Beam, BeamDual
            ctc_seq_log_prob_ext = rf.where(
                same_seq_labels, rf.replace_dim_v2(ctc_seq_log_prob, in_dim=beam_dim, out_dim=beam_dual_dim), neg_inf
            )  # Batch, Beam, BeamDual
            if recomb == "sum":
                ctc_seq_log_prob = rf.reduce_logsumexp(ctc_seq_log_prob_ext, axis=beam_dual_dim)  # Batch, Beam
            argmax_seq_log_prob = rf.reduce_argmax(seq_log_prob_ext, axis=beam_dual_dim)  # Batch, Beam -> BeamDual
            ctc_seq_log_prob = rf.where(argmax_seq_log_prob == rf.range_over_dim(beam_dim), ctc_seq_log_prob, neg_inf)
            lm_seq_log_prob = rf.where(argmax_seq_log_prob == rf.range_over_dim(beam_dim), lm_seq_log_prob, neg_inf)
        else:
            raise ValueError(f"invalid recog_recomb {recomb!r}")

    # ctc_seq_log_prob, lm_log_probs: Batch, Beam
    # Add LM EOS score at the end.
    lm_eos_score = rf.gather(lm_log_probs, indices=model.eos_idx, axis=model.target_dim)
    lm_seq_log_prob += lm_eos_score  # Batch, Beam
    seq_log_prob = ctc_seq_log_prob + lm_seq_log_prob  # Batch, Beam

    # TODO feed label seq again through DLM, compare scores

    # Backtrack via backrefs, resolve beams.
    seq_targets_wb_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target_wb in zip(seq_backrefs[::-1], seq_targets_wb[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_wb_.insert(0, rf.gather(target_wb, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets_wb__ = TensorArray(seq_targets_wb_[0])
    for target_wb in seq_targets_wb_:
        seq_targets_wb__ = seq_targets_wb__.push_back(target_wb)
    out_spatial_dim = enc_spatial_dim
    seq_targets_wb = seq_targets_wb__.stack(axis=out_spatial_dim)

    return seq_targets_wb, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
ctc_model_with_dlm_recog: RecogDef[CtcModel]
ctc_model_with_dlm_recog.output_with_beam = True
ctc_model_with_dlm_recog.output_blank_label = "<blank>"
ctc_model_with_dlm_recog.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def _target_remove_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert target.sparse_dim == wb_target_dim
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    return rf.set_sparse_dim(target, target_dim)


def _target_dense_extend_blank(
    target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int, value: float
) -> Tensor:
    assert target_dim in target.dims
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    res, _ = rf.pad(target, axes=[target_dim], padding=[(0, 1)], out_dims=[wb_target_dim], value=value)
    return res


def _same_seq_labels(seq: Tensor, *, spatial_dim: Dim, beam_dim: Dim) -> Tuple[Tensor, Dim]:
    seq_label_dual, beam_dual_dim = rf.replace_dim(seq, in_dim=beam_dim)
    same_seq_labels = rf.compare_bc(seq, "==", seq_label_dual)  # Batch, Beam, BeamDual, Spatial
    same_seq_labels = rf.reduce_all(same_seq_labels, axis=spatial_dim)  # Batch, Beam, BeamDual
    if beam_dim in spatial_dim.get_size_tensor().dims:
        seq_labels_lens = spatial_dim.get_size_tensor(device=same_seq_labels.device)
        seq_labels_dual_lens = rf.replace_dim_v2(
            seq_labels_lens, in_dim=beam_dim, out_dim=beam_dual_dim
        )  # Batch, BeamDual
        same_seq_labels_lens = rf.compare_bc(seq_labels_lens, "==", seq_labels_dual_lens)  # Batch, Beam, BeamDual
        same_seq_labels = rf.logical_and(same_seq_labels, same_seq_labels_lens)
    return same_seq_labels, beam_dual_dim
