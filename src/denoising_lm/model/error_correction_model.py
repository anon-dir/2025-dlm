"""
Error correction model - the denoising language model (DLM)
"""

from __future__ import annotations
import copy
from typing import Dict, Any, Tuple, Optional
import functools
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from returnn.config import get_global_config
from returnn.util.basic import NotSpecified
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim
from returnn.frontend.encoder.transformer import TransformerEncoder, TransformerEncoderLayer
from returnn.frontend.decoder.transformer import TransformerDecoder


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        target_dim: Dim,
        blank_idx: Optional[int] = None,
        eos_idx: int,
        bos_idx: int,
        encoder: Dict[str, Any],
        decoder: Dict[str, Any],
    ):
        super(Model, self).__init__()

        self.in_dim = in_dim
        self.encoder: TransformerEncoder = rf.build_from_dict(encoder, in_dim)
        self.decoder: TransformerDecoder = rf.build_from_dict(decoder, self.encoder.model_dim, target_dim)

        self.target_dim = target_dim
        self.blank_idx = blank_idx  # TODO why do I need this? i think this was just a mistake to put this here...
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx
        self.target_dim_wb = target_dim
        if self.blank_idx is not None and self.blank_idx >= self.target_dim_wb.dimension:
            self.target_dim_wb = self.target_dim_wb + 1
            if target_dim.vocab and not self.target_dim_wb.vocab:
                from returnn.datasets.util.vocabulary import Vocabulary

                # Just assumption for code now, might extend this later.
                assert self.target_dim_wb.dimension == target_dim.dimension + 1 and blank_idx == target_dim.dimension
                vocab_labels = list(target_dim.vocab.labels) + ["<blank>"]
                self.target_dim_wb.vocab = Vocabulary.create_vocab_from_labels(
                    vocab_labels, user_defined_symbols={"<blank>": blank_idx}
                )

        config = get_global_config()

        enc_aux_logits = config.typed_value("dlm_aux_loss_layers") or ()
        self.enc_aux_selected_layers = enc_aux_logits
        if enc_aux_logits:
            self.aux_layer = rf.Linear(self.encoder.model_dim, self.target_dim_wb * 2)
        else:
            self.aux_layer = None

        if config.typed_value("dlm_dense_method", "").startswith("attention"):
            from .modules.dense_combiner import DenseCombinerBlock

            # probably use build from dict here in the future...
            dense_block = DenseCombinerBlock(
                embed_dim=self.encoder.model_dim,
                out_dim=self.encoder.model_dim,
                dropout=encoder.get("dropout", 0.1),
                att_dropout=encoder.get("att_dropout", 0.1),
                num_heads=encoder.get("num_heads", 8),
                ff=encoder.get("ff", NotSpecified),
            )
            self.dense_encoder = rf.Sequential(copy.deepcopy(dense_block) for _ in range(2))
            self.dense_start_embed = rf.Parameter([dense_block.embed_dim])
            self.dense_start_embed.initial = rf.init.Glorot()
        else:
            self.dense_encoder = None
            self.dense_start_embed = None

    def encode(
        self,
        source: Tensor,
        *,
        spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> rf.State:
        """encode, and extend the encoder output for things we need in the decoder"""
        enc = self.encoder(source, spatial_dim=spatial_dim, collected_outputs=collected_outputs)
        return self.decoder.transform_encoder(enc, axis=spatial_dim)

    def encode_k_dense(self, source: Tensor, *, k_log_probs: Tensor, spatial_dim: Dim, k_dim: Dim) -> rf.State:
        """
        For :mod:`.error_correction_model_gen_train_data_dense`.

        :param source: [...,spatial_dim,k_dim]
        :param k_log_probs: [...,spatial_dim,k_dim]
        :param spatial_dim:
        :param k_dim:
        :return: encoder output as state for the decoder
        """
        assert self.encoder.__call__.__func__ is TransformerEncoder.__call__  # currently assumed for the following

        # Inline, simplified, adapted: enc = self.encoder(source, spatial_dim=spatial_dim)
        assert self.encoder.input_embedding is not None
        input_embedding: rf.Embedding = self.encoder.input_embedding
        assert type(input_embedding) is rf.Embedding
        decoded = rf.gather(
            input_embedding.weight, indices=source, axis=input_embedding.in_dim
        )  # [...,spatial_dim,k_dim,emb_dim]
        if self.dense_encoder is not None:
            decoded = self.dense_encoder(
                self.dense_start_embed, k_probs=rf.exp(k_log_probs), k_embeds=decoded, k_dim=k_dim
            )
        else:
            decoded = rf.matmul(decoded, rf.exp(k_log_probs), reduce=k_dim)  # [...,spatial_dim,emb_dim]
        decoded *= self.encoder.input_embedding_scale
        if self.encoder.pos_enc is not None:
            decoded = decoded + self.encoder.pos_enc(spatial_dim=spatial_dim)
        decoded = rf.dropout(decoded, self.encoder.input_dropout)
        if self.encoder.input_embedding_proj is not None:
            decoded = self.encoder.input_embedding_proj(decoded)

        for layer_name, layer in self.encoder.layers.items():
            layer: TransformerEncoderLayer  # or similar
            decoded = layer(decoded, spatial_dim=spatial_dim)

        enc = self.encoder.final_layer_norm(decoded)

        return self.decoder.transform_encoder(enc, axis=spatial_dim)


class ModelWithMultEmbedding(Model):
    """
    Extends the standard Transformer
    """

    def __init__(self, *args, encoder_num_embeddings: int, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.encoder.input_embedding is None
        self.encoder_num_embeddings_dim = Dim(encoder_num_embeddings, name="encoder_num_embeddings")
        self.encoder_input_embedding = rf.Parameter(
            (self.in_dim, self.encoder_num_embeddings_dim, self.encoder.embed_dim or self.encoder.model_dim)
        )
        self.encoder_input_embedding.initial = rf.init.Glorot()

    def encode(
        self,
        source: Tensor,
        *,
        spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> rf.State:
        """encode, and extend the encoder output for things we need in the decoder"""
        embed = rf.gather(self.encoder_input_embedding, indices=source, axis=self.in_dim)
        embed, spatial_dim_ = rf.merge_dims(embed, dims=(spatial_dim, self.encoder_num_embeddings_dim))
        assert spatial_dim_.dyn_size_ext.dims_set == spatial_dim.dyn_size_ext.dims_set
        embed *= self.encoder.input_embedding_scale
        enc = self.encoder(embed, spatial_dim=spatial_dim_, collected_outputs=collected_outputs)
        return self.decoder.transform_encoder(enc, axis=spatial_dim_)


def aed_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    epoch  # noqa
    config = get_global_config()

    model_dict = config.typed_value("_model_dict")
    encoder = config.typed_value("_encoder_model_dict")
    decoder = config.typed_value("_decoder_model_dict")

    if in_dim is None:
        print("** in_dim is None, getting dim from external data")

        default_input_key = config.typed_value("default_input")
        extern_data_dict = config.typed_value("extern_data")
        data = Tensor(name=default_input_key, **extern_data_dict[default_input_key])
        in_dim = data.feature_dim_or_sparse_dim

    print("** in_dim:", in_dim)
    print("** target_dim:", target_dim)
    print("in dim and target equal:", in_dim == target_dim)

    if not model_dict:
        model_dict = rf.build_dict(Model)

    model: Model = rf.build_from_dict(
        model_dict,
        in_dim,
        encoder=encoder,
        decoder=decoder,
        target_dim=target_dim,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        eos_idx=_get_eos_idx(target_dim),
    )
    return model


aed_model_def: ModelDef[Model]
aed_model_def.behavior_version = 21
aed_model_def.backend = "torch"
aed_model_def.batch_size_factor = 1


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


def aed_training(
    *,
    model: Model,
    data: rf.Tensor,
    data_spatial_dim: Dim,
    data_k_log_probs: Optional[Tensor] = None,
    data_k_dim: Optional[Dim] = None,
    targets: rf.Tensor,
    targets_spatial_dim: Dim,
    **_other,
):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()
    label_smoothing = config.float("label_smoothing", 0.1)
    input_swapout_range: Optional[Tuple[float, float]] = config.typed_dict.get("input_swapout_range", None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    enc_aux_layers = config.typed_value("dlm_aux_loss_layers") or ()
    enc_aux_weights = config.typed_value("dlm_aux_loss_weights") or 1.0
    if isinstance(enc_aux_weights, (int, float)):
        enc_aux_weights = [enc_aux_weights] * len(enc_aux_layers)

    if data_k_log_probs is not None and config.bool("dense_prob_renorm", False):
        data_k_log_probs = rf.log_softmax(data_k_log_probs, axis=data_k_dim)

    assert data.sparse_dim  # input text
    assert targets.sparse_dim  # target text
    batch_dims = targets.remaining_dims(targets_spatial_dim)

    if input_swapout_range:
        assert isinstance(input_swapout_range, tuple) and len(input_swapout_range) == 2
        random_labels = rf.random_uniform(
            data.dims,
            sparse_dim=data.sparse_dim,
            dtype=data.dtype,
            device=data.device,
            minval=0,
            maxval=data.sparse_dim.dimension,
        )  # [B, T_in]
        swapout_rate = rf.random_uniform(
            batch_dims, minval=input_swapout_range[0], maxval=input_swapout_range[1]
        )  # [B]
        mask = rf.random_uniform(dims=data.dims) < swapout_rate  # [B, T_in]
        data = rf.where(mask, random_labels, data)  # [B, T_in]

    input_add_bos = config.typed_value("input_add_bos", False)
    input_add_eos = config.typed_value("input_add_eos", False)
    if input_add_bos:
        assert data_k_log_probs is None  # not implemented
        data, (data_spatial_dim,) = rf.pad(
            data, axes=[data_spatial_dim], padding=[(int(input_add_bos), 0)], value=model.bos_idx
        )
    if input_add_eos:
        data, (data_spatial_dim_,) = rf.pad(
            data, axes=[data_spatial_dim], padding=[(0, int(input_add_eos))], value=model.eos_idx
        )
        if data_k_log_probs is not None:
            data_k_log_probs, _ = rf.pad(
                data_k_log_probs,
                axes=[data_spatial_dim],
                padding=[(0, int(input_add_eos))],
                value=rf.log(1.0 / data_k_dim.get_size_tensor(device=data.device)),
                out_dims=[data_spatial_dim_],
            )
        data_spatial_dim = data_spatial_dim_

    if data_k_log_probs is None:
        collected_outputs = {} if enc_aux_layers else None
        enc = model.encode(data, spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    else:
        assert not enc_aux_layers, "not implemented for dense k_log_probs"
        collected_outputs = None
        enc = model.encode_k_dense(data, k_log_probs=data_k_log_probs, spatial_dim=data_spatial_dim, k_dim=data_k_dim)

    if enc_aux_layers:
        assert len(enc_aux_weights) == len(enc_aux_layers)
        assert config.typed_value("dlm_aux_loss_version", 1) > 1  # fixed merge dims issue
        from i6_experiments.users.zeyer.nn_rf.torch_ctc_fixed_grad import ctc_loss_fixed_grad

        assert collected_outputs is not None
        assert model.aux_layer is not None, "aux_layer must be defined for aux loss layers"

        aux_targets = targets
        aux_targets_spatial_dim = targets_spatial_dim
        # same as input, make it as easy as possible to just feed through the data from the input
        if input_add_bos:
            aux_targets, (aux_targets_spatial_dim,) = rf.pad(
                aux_targets, axes=[aux_targets_spatial_dim], padding=[(int(input_add_bos), 0)], value=model.bos_idx
            )
        if input_add_eos:
            aux_targets, (aux_targets_spatial_dim,) = rf.pad(
                aux_targets, axes=[aux_targets_spatial_dim], padding=[(0, int(input_add_eos))], value=model.eos_idx
            )

        for i in enc_aux_layers:
            assert isinstance(i, int)
            # aux_layer: model_dim -> 2*target_dim
            aux_logits2 = model.aux_layer(collected_outputs[str(i - 1)])
            two_dim = rf.Dim(2, name="aux_twodim")
            # [.., 2*target_dim] -> [.., 2, target_dim]
            aux_logits = rf.split_dims(aux_logits2, axis=model.aux_layer.out_dim, dims=[two_dim, model.target_dim_wb])
            # [.., spatial_dim, 2, target_dim] -> [.., 2*spatial_dim, target_dim]
            aux_logits, two_spatial_dim = rf.merge_dims(
                aux_logits, dims=[data_spatial_dim, two_dim]
            )  # IMPORTANT: correct merge order

            aux_logits = rf.log_softmax(aux_logits, axis=model.target_dim_wb)
            aux_logits.feature_dim = model.target_dim_wb

            aux_loss = ctc_loss_fixed_grad(
                logits=aux_logits,
                logits_normalized=True,
                input_spatial_dim=two_spatial_dim,
                targets=aux_targets,
                targets_spatial_dim=aux_targets_spatial_dim,
                blank_index=model.blank_idx,
            )
            aux_loss.mark_as_loss(
                f"aux_{i}",
                scale=enc_aux_weights[i % len(enc_aux_weights)],
                custom_inv_norm_factor=aux_targets_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )

    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets,
        axes=[targets_spatial_dim],
        padding=[(0, 1)],
        value=model.eos_idx,
        out_dims=[targets_w_eos_spatial_dim],
    )

    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=batch_dims),
    )

    logits_packed, pack_dim = rf.pack_padded(
        logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
    )
    targets_packed, _ = rf.pack_padded(
        targets_w_eos,
        dims=batch_dims + [targets_w_eos_spatial_dim],
        enforce_sorted=False,
        out_dim=pack_dim,
    )

    log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
    if label_smoothing:
        log_prob = rf.label_smoothed_log_prob_gradient(log_prob, label_smoothing, axis=model.target_dim)
    loss = rf.cross_entropy(
        target=targets_packed,
        estimated=log_prob,
        estimated_type="log-probs",
        axis=model.target_dim,
    )
    loss.mark_as_loss("ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

    best = rf.reduce_argmax(log_prob, axis=model.target_dim)
    frame_error = best != targets_packed
    frame_error.mark_as_loss(name="fer", as_error=True)


aed_training: TrainDef[Model]
aed_training.learning_rate_control_error_measure = "ce"


def model_recog(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    data_k_log_probs: Optional[Tensor] = None,
    data_k_dim: Optional[Dim] = None,
    max_seq_len: Optional[int] = None,
    **_other,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Recog (beam search) for the DLM, directly operating on a single hyp from the ASR model.

    This is used for DSR decoding.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import tree
    from returnn.frontend.tensor_array import TensorArray
    from returnn.config import get_global_config
    from i6_experiments.users.zeyer.nn_rf.nucleus_sampling import nucleus_sampling, nucleus_sampling_beam_search
    from i6_experiments.users.zeyer.nn_rf.top_k_and_random_choice_without_replacement import (
        top_k_and_random_choice_without_replacement,
    )

    config = get_global_config()

    if data_k_log_probs is not None and config.bool("dense_prob_renorm", False):
        data_k_log_probs = rf.log_softmax(data_k_log_probs, axis=data_k_dim)

    input_add_bos = config.typed_value("input_add_bos", False)
    input_add_eos = config.typed_value("input_add_eos", False)
    beam_size = config.int("beam_size", 12)
    length_normalization_exponent = config.float("length_normalization_exponent", 1.0)
    temperature = config.float("temperature", 1.0)
    nucleus_sampling_opts = config.typed_value("nucleus_sampling", None)
    nucleus_sampling_beam_search_opts = config.typed_value("nucleus_sampling_beam_search", None)
    if nucleus_sampling_opts:
        assert config.int("nucleus_sampling_version", 0) == 2
    top_k_and_random_choice_without_replacement_opts = config.typed_value(
        "top_k_and_random_choice_without_replacement", None
    )

    if input_add_bos:
        assert data_k_log_probs is None  # not implemented
        data, (data_spatial_dim,) = rf.pad(
            data, axes=[data_spatial_dim], padding=[(int(input_add_bos), 0)], value=model.bos_idx
        )
    if input_add_eos:
        data, (data_spatial_dim_,) = rf.pad(
            data, axes=[data_spatial_dim], padding=[(0, int(input_add_eos))], value=model.eos_idx
        )
        if data_k_log_probs is not None:
            data_k_log_probs, _ = rf.pad(
                data_k_log_probs,
                axes=[data_spatial_dim],
                padding=[(0, int(input_add_eos))],
                value=rf.log(1.0 / data_k_dim.get_size_tensor(device=data.device)),
                out_dims=[data_spatial_dim_],
            )
        data_spatial_dim = data_spatial_dim_

    batch_dims = data.remaining_dims(data_spatial_dim if data_k_dim is None else (data_spatial_dim, data_k_dim))

    if data_k_log_probs is None:
        enc = model.encode(data, spatial_dim=data_spatial_dim)
    else:
        enc = model.encode_k_dense(data, k_log_probs=data_k_log_probs, spatial_dim=data_spatial_dim, k_dim=data_k_dim)

    if max_seq_len is None:
        max_seq_len = data_spatial_dim.get_size_tensor() * 2
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)
    neg_inf = float("-inf")

    # Eager-mode implementation of beam search.
    # Initial state.
    if nucleus_sampling_opts:
        beam_dim = Dim(beam_size, name="hyps")  # they always stay separate
    else:
        beam_dim = Dim(1, name="initial_beam")
    batch_dims_ = [beam_dim] + batch_dims
    decoder_state = model.decoder.default_initial_state(batch_dims=batch_dims_)
    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    ended = rf.constant(False, dims=batch_dims_)
    out_seq_len = rf.constant(0, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        logits, decoder_state = model.decoder(target, spatial_dim=single_step_dim, encoder=enc, state=decoder_state)
        label_log_prob = rf.log_softmax(logits / temperature, axis=model.target_dim)
        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=neg_inf),
            label_log_prob,
        )
        if nucleus_sampling_opts:
            seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
            seq_log_prob, target = nucleus_sampling(seq_log_prob, axis=model.target_dim, **nucleus_sampling_opts)
            backrefs = rf.range_over_dim(beam_dim)  # just keep all beams, no pruning
        elif nucleus_sampling_beam_search_opts:
            seq_log_prob, (backrefs, target), beam_dim = nucleus_sampling_beam_search(
                seq_log_prob,
                label_log_prob,
                k_dim=Dim(min(beam_size, beam_dim.dimension * model.target_dim.dimension), name=f"dec_step{i}_beam"),
                axis=[beam_dim, model.target_dim],
                **nucleus_sampling_beam_search_opts,
            )
        elif top_k_and_random_choice_without_replacement_opts:
            seq_log_prob, (backrefs, target), beam_dim = top_k_and_random_choice_without_replacement(
                seq_log_prob + label_log_prob,
                k=Dim(min(beam_size, beam_dim.dimension * model.target_dim.dimension), name=f"dec_step{i}_beam"),
                axis=[beam_dim, model.target_dim],
                **top_k_and_random_choice_without_replacement_opts,
            )
        else:
            seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
                seq_log_prob + label_log_prob,
                k_dim=Dim(min(beam_size, beam_dim.dimension * model.target_dim.dimension), name=f"dec_step{i}_beam"),
                axis=[beam_dim, model.target_dim],
            )  # seq_log_prob, backrefs, target: Batch, Beam
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        decoder_state = tree.map_structure(functools.partial(_gather_backrefs, backrefs=backrefs), decoder_state)
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        i += 1

        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

        if i > 1 and length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob *= rf.where(
                ended,
                (i / (i - 1)) ** length_normalization_exponent,
                1.0,
            )

    if i > 0 and length_normalization_exponent != 0:
        seq_log_prob *= (1 / (i - 1)) ** length_normalization_exponent

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
    out_spatial_dim = Dim(out_seq_len, name="out-spatial")
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = None
model_recog.batch_size_dependent = True


def _gather_backrefs(s, *, backrefs: Tensor):
    if isinstance(s, Tensor):
        if backrefs.sparse_dim in s.dims:
            return rf.gather(s, indices=backrefs)  # really the default case
        return s  # e.g. scalar or so, independent from beam
    if isinstance(s, Dim):
        assert s.dimension or backrefs not in s.dyn_size_ext.dims  # currently not supported, also not expected
        return s
    raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")


# RescoreDef
def dlm_rescore_def(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    data_k_log_probs: Optional[Tensor] = None,
    data_k_dim: Optional[Dim] = None,
    targets: Tensor,
    targets_beam_dim: Dim,
    targets_spatial_dim: Dim,
) -> Tensor:
    """
    RescoreDef for DLM
    """
    import returnn.frontend as rf
    from returnn.config import get_global_config

    targets_beam_dim  # noqa  # unused here

    vocab_dim = model.decoder.vocab_dim
    bos_idx = _get_bos_idx(vocab_dim)
    eos_idx = _get_eos_idx(vocab_dim)

    config = get_global_config()

    if data_k_log_probs is not None and config.bool("dense_prob_renorm", False):
        data_k_log_probs = rf.log_softmax(data_k_log_probs, axis=data_k_dim)

    input_add_bos = config.typed_value("input_add_bos", False)
    input_add_eos = config.typed_value("input_add_eos", False)
    temperature = config.float("temperature", 1.0)

    if input_add_bos:
        assert data_k_log_probs is None  # not implemented
        data, (data_spatial_dim,) = rf.pad(
            data, axes=[data_spatial_dim], padding=[(int(input_add_bos), 0)], value=model.bos_idx
        )
    if input_add_eos:
        data, (data_spatial_dim_,) = rf.pad(
            data, axes=[data_spatial_dim], padding=[(0, int(input_add_eos))], value=model.eos_idx
        )
        if data_k_log_probs is not None:
            data_k_log_probs, _ = rf.pad(
                data_k_log_probs,
                axes=[data_spatial_dim],
                padding=[(0, int(input_add_eos))],
                value=rf.log(1.0 / data_k_dim.get_size_tensor(device=data.device)),
                out_dims=[data_spatial_dim_],
            )
        data_spatial_dim = data_spatial_dim_

    if data_k_log_probs is None:
        enc = model.encode(data, spatial_dim=data_spatial_dim)
    else:
        enc = model.encode_k_dense(data, k_log_probs=data_k_log_probs, spatial_dim=data_spatial_dim, k_dim=data_k_dim)

    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=eos_idx, out_dims=[targets_w_eos_spatial_dim]
    )

    batch_dims = targets.remaining_dims(targets_spatial_dim)
    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=batch_dims),
    )

    log_prob = rf.log_softmax(logits / temperature, axis=vocab_dim)
    log_prob_targets = rf.gather(log_prob, indices=targets_w_eos, axis=vocab_dim)  # [batch,beam,targets_spatial_w_eos]
    log_prob_targets_seq = rf.reduce_sum(log_prob_targets, axis=targets_w_eos_spatial_dim)  # [batch,beam]
    assert log_prob_targets_seq.dims_set == set(batch_dims)
    return log_prob_targets_seq
