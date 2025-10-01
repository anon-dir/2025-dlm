"""
CTC model
"""

from __future__ import annotations
import functools
import math
import contextlib
from typing import TYPE_CHECKING, Literal, Optional, Union, Any, Callable, Sequence, List, Dict, Tuple
from dataclasses import dataclass, is_dataclass, asdict

from i6_experiments.users.dorian_koch.misc.audio_perturbs import generalized_specaugment, save_spectogram
from i6_experiments.users.zeyer.model_interfaces.model_with_checkpoints import (
    ModelWithCheckpoint,
)

from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
    train_exp,
    speed_pert_librosa_config,
    Model,
    _get_cfg_lrlin_oclr_by_bs_nep,
    _batch_size_factor,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
    config_24gb_v6,
    config_96gb_bf16_accgrad1,
    _get_cfg_lrlin_oclr_by_bs_nep_v3,
)

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)
from returnn.util.basic import CollectionReadCheckCovered
from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from i6_experiments.users.zeyer.model_interfaces import ForwardRFDef
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from sisyphus import tk


if TYPE_CHECKING:
    from .tts_model import TtsOpts
    from .error_correction_model_gen_train_data_dense import AsrHypsProbsCfg


def py():
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
        train_exp as ctc_train_exp,
        _raw_sample_rate,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_claix2023 import (
        recog_ext_with_lm,
        recog_ext_labelwise_with_lm,
    )
    from .tts_data import get_asr_tts_extended_task

    for vocab in ["spm10k", "spm512", "spm128"]:
        get_hyps(vocab=vocab)

    prefix = get_setup_prefix_for_module(__name__)

    for vocab in ["spm10k", "spm128", "bpe10k", "bpe128", "utf8", "char"]:
        name = f"time3-L16-D1024-{vocab}-auxAED-b50k"
        ctc_train_exp(
            name,
            config_96gb_bf16_accgrad1,
            prefix=prefix + "/ctc/",
            vocab=vocab,
            model_config={
                "enc_build_dict": rf.build_dict(
                    # ConformerEncoder(in_dim, enc_model_dim, **enc_opts)
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (1, 1)],  # downsampling 3
                    ),
                    num_layers=16,
                    out_dim=1024,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(50_000, 100, batch_size_factor=_batch_size_factor),
                "optimizer.weight_decay": 1e-2,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                # purely used for training
                "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            train_vocab_opts=(
                {"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}}
                if vocab.startswith("spm") or vocab.startswith("bpe")
                else None
            ),
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
        # recog_ext_with_lm(ctc_model_name=name, lm_name="n32-d1024-claix2023")

    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_raw_v2,
    )

    vocab = "spm10k"
    task = get_librispeech_task_raw_v2(vocab=vocab)
    ctc_model = sis_get_model(vocab=vocab)

    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import (
        get_ctc_prior_probs,
    )

    prior = get_ctc_prior_probs(ctc_model, task.train_dataset.copy_train_as_static())
    tk.register_output(f"{prefix}/ctc-prior", prior)

    recog_ext_with_lm(ctc_model_name=f"ctc-{vocab}", ctc_model=ctc_model, lm_name="n32-d1024")
    recog_ext_with_lm(ctc_model_name=f"ctc-{vocab}", ctc_model=ctc_model, lm_name="n32-d1024-claix2023")
    recog_ext_with_lm(ctc_model_name=f"ctc-{vocab}", ctc_model=ctc_model, lm_name="n32-d1280-claix2023")

    task_with_tts = get_asr_tts_extended_task(
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        train_audio_preprocess=speed_pert_librosa_config,
    )

    for num_layers, num_dims, batch_size in [(16, 1024, 100_000)]:
        # Baseline without using TTS.
        # Warning: this keeps aux_loss_layers=[4, 8], not sure if this is optimal...
        ctc_train_exp(
            f"L{num_layers}-D{num_dims}-spm10k-auxAED-b{batch_size // 1000}k",
            config_96gb_bf16_accgrad1,
            model_config={
                "enc_build_dict": rf.build_dict(
                    # ConformerEncoder(in_dim, enc_model_dim, **enc_opts)
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                    ),
                    num_layers=num_layers,
                    out_dim=num_dims,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(batch_size, 100, batch_size_factor=_batch_size_factor),
                "optimizer.weight_decay": 1e-2,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                # purely used for training
                "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

        # Using TTS.
        # Warning: this keeps aux_loss_layers=[4, 8], not sure if this is optimal...
        # Note: there are some minor diffs compared to the above,
        # e.g. speed_pert_librosa_config,
        # __multi_proc_dataset_opts num_workers,
        # SamplingBytePairEncoding.
        name = f"L{num_layers}-D{num_dims}-spm10k-auxAED-b{batch_size // 1000}k-tts"
        ctc_train_exp(
            name,
            config_96gb_bf16_accgrad1,
            prefix=prefix + "/ctc/",
            task=task_with_tts,
            model_config={
                "enc_build_dict": rf.build_dict(
                    # ConformerEncoder(in_dim, enc_model_dim, **enc_opts)
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                    ),
                    num_layers=num_layers,
                    out_dim=num_dims,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(batch_size, 100, batch_size_factor=_batch_size_factor),
                "__serialization_version": 2,
                "optimizer.weight_decay": 1e-2,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                # purely used for training
                "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            },
            post_config_updates={"log_grad_norm": True},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    for name in ["L16-D1024-spm10k-auxAED-b100k", "L16-D1024-spm10k-auxAED-b100k-tts"]:
        for lm_name in ["n32-d1024-claix2023", "n32-d1024-nEp200-claix2023"]:
            recog_ext_with_lm(ctc_model_name=name, lm_name=lm_name, ctc_soft_collapse_threshold=0.9)
            recog_ext_labelwise_with_lm(ctc_model_name=name, lm_name=lm_name, ctc_soft_collapse_threshold=0.9)
        # Some others
        recog_ext_with_lm(ctc_model_name=name, lm_name="n32-d1024-nEp300-claix2023", ctc_soft_collapse_threshold=0.9)
        recog_ext_with_lm(ctc_model_name=name, lm_name="n32-d1024-nEp400-claix2023", ctc_soft_collapse_threshold=0.9)
        recog_ext_labelwise_with_lm(ctc_model_name=name, lm_name="n32-d1280-claix2023", ctc_soft_collapse_threshold=0.9)

    for vocab in ["char", "spm128"]:
        name = f"time3-L16-D1024-{vocab}-auxAED-b50k-tts"
        train_vocab_opts = (
            {"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}}
            if vocab.startswith("spm") or vocab.startswith("bpe")
            else None
        )
        task_ = get_asr_tts_extended_task(
            vocab=vocab,
            train_vocab_opts=train_vocab_opts,
            train_audio_preprocess=speed_pert_librosa_config,
        )
        ctc_train_exp(
            name,
            config_96gb_bf16_accgrad1,
            prefix=prefix + "/ctc/",
            vocab=vocab,
            task=task_,
            model_config={
                "enc_build_dict": rf.build_dict(
                    # ConformerEncoder(in_dim, enc_model_dim, **enc_opts)
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (1, 1)],  # downsampling 3
                    ),
                    num_layers=16,
                    out_dim=1024,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(50_000, 100, batch_size_factor=_batch_size_factor),
                "__serialization_version": 2,
                "optimizer.weight_decay": 1e-2,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                # purely used for training
                "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            train_vocab_opts=train_vocab_opts,
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
        # recog_ext_with_lm(ctc_model_name=name, lm_name="n32-d1024-claix2023")


@dataclass
class GetHypsCfgV1:
    """
    Used for :func:`model_recog_single_dropout`
    (fallback to :func:`model_recog_single` for default args).
    """

    dropout: float = 0.1
    enable_specaugment: bool = True


@dataclass
class GetHypsCfgV2:
    """
    Attached to :func:`model_recog_single_v2`.
    """

    dropout_min: float = 0.0
    dropout_max: float = 0.2


@dataclass
class GetHypsCfgV3:
    """
    Attached to :func:`model_recog_single_v3`.
    This can include optional on-the-fly TTS data generation (see :class:`TtsOpts` for that).
    """

    # specaug: merged with Model._specaugment_opts, used for rf.audio.specaugment.
    enable_specaugment: bool = False
    specaugment_opts: Optional[Dict[str, Any]] = None
    # applied on the Model
    dropout_min: float = 0.0
    dropout_max: float = 0.0


@dataclass
class GetHypsCfgV4:
    """
    Attached to :func:`model_recog_single_v4`.
    This can include optional on-the-fly TTS data generation (see :class:`TtsOpts` for that).
    This is a fixed variant over GetHypsCfgV3 where we can control where the train flag is enabled.
    By default, we will not enable the train flag for batch norm, i.e. this is different from GetHypsCfgV3!
    """

    # specaug: merged with Model._specaugment_opts, used for rf.audio.specaugment.
    enable_specaugment: bool = False
    specaugment_opts: Optional[Dict[str, Any]] = None
    # applied on the Model
    dropout_min: float = 0.0
    dropout_max: float = 0.0
    extra_train_flag_funcs: Sequence[Callable] = ()

    def __str__(self) -> str:
        s = "GetHypsCfgV4("
        s += f"specaug{'On' if self.enable_specaugment else 'Off'}"
        if self.enable_specaugment and self.specaugment_opts:
            s += repr(self.specaugment_opts)
        s += f"-dropout{self.dropout_min}_{self.dropout_max}"
        if len(self.extra_train_flag_funcs) > 0:
            s += f"-extraTrainFlags{self.extra_train_flag_funcs!r}"
        s += ")"
        return s


@dataclass
class GetHypsCfgV5(GetHypsCfgV4):
    """
    Attached to :func:`model_recog_single_v5`.
    This is like V4, but we will randomly sample some
    """

    sample_method: Union[Literal["greedy"], Literal["top_k"], Literal["discriminator"]] = "greedy"
    sample_opts: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        s = "GetHypsCfgV5("
        s += f"specaug{'On' if self.enable_specaugment else 'Off'}"
        if self.enable_specaugment and self.specaugment_opts:
            s += repr(self.specaugment_opts)
        s += f"-dropout{self.dropout_min}_{self.dropout_max}"
        if len(self.extra_train_flag_funcs) > 0:
            s += f"-extraTrainFlags{repr(self.extra_train_flag_funcs)}"
        s += f"-sampleMethod{self.sample_method}"
        if self.sample_opts and len(self.sample_opts) > 0:
            # dont print discriminator as its super long
            sample_opts_without_discriminator = self.sample_opts.copy()
            d = sample_opts_without_discriminator.pop("discriminator", None)
            # if d is not None:
            #    s += f"-discriminator{d}"
            s += "("
            objDescribeStr = ""
            if "discriminator_scale" in sample_opts_without_discriminator:
                objDescribeStr += f"-discScale{sample_opts_without_discriminator.pop('discriminator_scale')}"
            if "use_top_p_sampling" in sample_opts_without_discriminator:
                objDescribeStr += f"-topP{sample_opts_without_discriminator.pop('use_top_p_sampling')}"
            if "length_normalization_exponent" in sample_opts_without_discriminator:
                objDescribeStr += f"-lenNormExp{sample_opts_without_discriminator.pop('length_normalization_exponent')}"
            if len(sample_opts_without_discriminator) > 0:
                objDescribeStr += "-" + repr(sample_opts_without_discriminator)

            if objDescribeStr and objDescribeStr[0] == "-":
                objDescribeStr = objDescribeStr[1:]
            s += objDescribeStr
            s += ")"
        s += ")"
        return s


@dataclass
class GetHypsCfgV6(GetHypsCfgV5):
    # dict so that we dont have to make a new GetHypsCfg class for each new perturbation option
    data_perturbation_opts: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        super_str = super().__str__()

        s = "GetHypsCfgV6("
        s += super_str[super_str.index("(") + 1 : super_str.rindex(")")]
        if self.data_perturbation_opts and len(self.data_perturbation_opts) > 0:
            s += f"-perturb{'{'}{_name_for_dict(self.data_perturbation_opts)}{'}'}"
        s += ")"

        return s


TGetHypsCfg = Union[GetHypsCfgV1, GetHypsCfgV2, GetHypsCfgV3, GetHypsCfgV4, GetHypsCfgV5, GetHypsCfgV6]


# See sis_get_model below.
# We always take the last epoch
# (not necessarily the best (regarding dev-other), but close to it, maybe even better on test-other).
USE_24GB_IF_POSSIBLE = False
# spm10k 11gb last epoch: {"dev-clean": 2.38, "dev-other": 5.67, "test-clean": 2.63, "test-other": 5.93}
# with LM (recog-timesync-labelprior-recomb-beam64-fp64-lm_n32-d1024-claix2023-sct0.8):
# {"dev-clean": 1.93, "dev-other": 4.0, "test-clean": 2.01, "test-other": 4.33}
SPM10k_11gb = "v6-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001"
# spm10k 24gb last epoch: ... (this specific model not trained yet...)
SPM10k_24gb = "v6-relPosAttDef-noBias-aedLoss-bhv20-24gb-bf16-bs40k-accgrad2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001"
_model_name_by_vocab = {
    "spm10k": SPM10k_24gb if USE_24GB_IF_POSSIBLE else SPM10k_11gb,
    # spm512: last epoch: {"dev-clean": 2.36, "dev-other": 6.06, "test-clean": 2.55, "test-other": 6.19}
    "spm512": "v6-relPosAttDef"
    "-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
    "-featBN-speedpertV2-spm512-bpeSample0005",
    # spm128 last epoch: {"dev-clean": 2.52, "dev-other": 6.39, "test-clean": 2.7, "test-other": 6.47}
    "spm128": "v6-relPosAttDef"
    "-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
    "-featBN-speedpertV2-spm128",
    "char": "time3-L16-D1024-char-auxAED-b50k",
    # "spm128": "time3-L16-D1024-spm128-auxAED-b50k",
}

_dep_bound_hash_by_model_name = {
    "v6-relPosAttDef-noBias-aedLoss"
    "-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k"
    "-featBN-speedpertV2-spm10k-bpeSample001": "CzG0AHg5psm5",
    "v6-relPosAttDef"
    "-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
    "-featBN-speedpertV2-spm512-bpeSample0005": "YSvtF9CcL6WF",
    "v6-relPosAttDef"
    "-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k"
    "-featBN-speedpertV2-spm128": "hD9XELdfeFO7",
    # L16-D1024-spm10k-auxAED-b100k last epoch:
    # {"dev-clean": 2.27, "dev-other": 5.08, "test-clean": 2.42, "test-other": 5.32}
    # with LM (recog-timesync-labelprior-recomb-beam64-fp64-lm_n32-d1024-claix2023-sct 0.8):
    # {"dev-clean": 1.85, "dev-other": 3.93, "test-clean": 2.0, "test-other": 4.27}
    "L16-D1024-spm10k-auxAED-b100k": "j9jCdYuRRtCi",
    "L16-D1024-spm10k-auxAED-b100k-tts": "tOoHJgizAlSv",
    # not a clean solution... maybe rewrite sis_get_model to return all checkpoints?
    "L16-D1024-spm10k-auxAED-b100k-tts__epoch10": "TkuLkUwmUT47",
    "L16-D1024-spm10k-auxAED-b100k-tts__epoch20": "DIUBTVBEjNcO",
    "L16-D1024-spm10k-auxAED-b100k-tts__epoch40": "qUzryM7DcZ0b",
    "L16-D1024-spm10k-auxAED-b100k-tts__epoch80": "lL1nzlxeMpYs",
    "L16-D1024-spm10k-auxAED-b100k-tts__epoch100": "tOoHJgizAlSv",
    "time3-L16-D1024-char-auxAED-b50k-tts": "FMBGjQ1AoR3Y",
    "time3-L16-D1024-char-auxAED-b50k": "TINHOlYNVQa2",
    "time3-L16-D1024-spm128-auxAED-b50k-tts": "7ZBdFn3Y5gkS",
    "v6-EBranchformer-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001": "rzm540fxQ5Us",
    "v6-EBranchformer-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001__epoch20": "zTSUUqEr5BuT",
    "v6-EBranchformer-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001__epoch40": "XSMTqbTwo4bE",
    "v6-EBranchformer-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001__epoch80": "QtYFiXeTFT8a",
    "v6-EBranchformer-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001__epoch160": "mWWYsc3M3F8X",
    "v6-EBranchformer-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001__epoch320": "ujCWIsYQ6weQ",
    "v6-EBranchformer-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001__epoch500": "rzm540fxQ5Us",
    "L16-D512-spm10k-auxAED-b100k__epoch100": "BqKSlpXNiVHq",
    "L16-D768-spm10k-auxAED-b100k__epoch63": "8TR7XLRGtVeW",
    "L16-D768-spm10k-auxAED-b100k__epoch100": "ILAgdQpdpZoe",
}


def get_hyps(*, vocab: str, num_hyps: int = 5) -> Dict[str, List[tk.Path]]:
    # TODO...
    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_raw_v2,
    )

    prefix = get_setup_prefix_for_module(__name__)

    # Note: task hardcoded... (and also not needed, I just need the train dataset...)
    # Note: spm10k hardcoded...
    task = get_librispeech_task_raw_v2(vocab=vocab)

    # Model hardcoded...
    model = sis_get_model(_model_name_by_vocab[vocab])

    # Should be without speed pert, partition epoch, as we access this via DatasetConfig.get_main_dataset.
    res = {}
    for key, dataset in {
        "train/train": task.train_dataset.copy_train_as_static(),
        **{f"train/{k}": task.train_dataset.copy_eval_as_static(k) for k in task.train_dataset.get_eval_datasets()},
        "dev": task.dev_dataset,
        **{f"eval/{k}": v for k, v in task.eval_datasets.items()},
    }.items():
        hyps, _ = sis_get_hyps_split(model, dataset=dataset, num_hyps=num_hyps if key == "train/train" else 1)
        for i, hyp in enumerate(hyps):
            tk.register_output(f"{prefix}/hyps_from_model_{vocab}/{key}/hyps{i}.hdf", hyp)
            hyp.creator.add_alias(f"{prefix}/hyps_from_model_{vocab}/{key}/hyps{i}")
        res[key] = hyps

        # Currently we don't use this combined variant:
        # hyps_combined = sis_get_hyps_combined(
        #     model,
        #     # Should be without speed pert, partition epoch, as we access this via DatasetConfig.get_main_dataset.
        #     dataset=train_dataset,
        # )
        # tk.register_output(prefix + "/hyps_combined.hdf", hyps_combined)
        # hyps_combined.creator.add_alias(prefix + "/hyps")

    return res


def sis_get_hyps_combined(model: ModelWithCheckpoint, *, dataset: DatasetConfig, num_hyps: int = 12) -> tk.Path:
    from i6_experiments.users.zeyer.forward_to_hdf import forward_to_hdf

    extern_data_template_dict = dataset.get_extern_data()
    target_template_dict = extern_data_template_dict[dataset.get_default_target()]
    target_template = Tensor(
        name=dataset.get_default_target(),
        # Exclude vocab to not load it at this point.
        **{k: v for k, v in target_template_dict.items() if k != "vocab"},
    )
    batch_dim, out_spatial_dim = target_template.dims
    assert batch_dim.is_batch_dim()
    assert out_spatial_dim.is_spatial_dim()

    num_hyps_dim = Dim(num_hyps, name="hyps")
    hyps_packed_spatial_dim = Dim(Tensor(name="hyps_packed_spatial", dims=[batch_dim], dtype="int32"))

    res = forward_to_hdf(
        dataset=dataset,
        model=model,
        forward_def=model_recog,
        config={
            "num_hyps_dim": num_hyps_dim,
            # We use dropout to get multiple different hypotheses.
            # We introduce the beam dim, and want to get different hypotheses for each beam.
            "rf_dropout_broadcast": False,
            "model_outputs": {
                "output": {
                    "dims": [batch_dim, hyps_packed_spatial_dim],
                    "sparse_dim": target_template.sparse_dim,
                    "dtype": target_template.dtype,
                },
                "seq_lens": {"dims": [batch_dim, num_hyps_dim], "dtype": "int32"},
                "log_probs": {"dims": [batch_dim, num_hyps_dim], "dtype": "float32"},
                "enc_seq_lens": {"dims": [batch_dim], "dtype": "int32"},
            },
        },
        forward_post_config={"batch_size": 10000 * model.definition.batch_size_factor},
        forward_rqmt={"time": 24},
    )
    return res


_SisGetHypsSplitIgnoreExtraConfig = False  # For sis_get_hyps_split


@contextlib.contextmanager
def sis_get_hyps_split_ignore_extra_config_ctx(yes: bool = True, /):
    """
    This is to get back some buggy behavior we had before,
    to not break hashes of some older experiments.
    """
    global _SisGetHypsSplitIgnoreExtraConfig
    old = _SisGetHypsSplitIgnoreExtraConfig
    _SisGetHypsSplitIgnoreExtraConfig = yes
    try:
        yield
    finally:
        _SisGetHypsSplitIgnoreExtraConfig = old


def sis_get_hyps_split(
    model: ModelWithCheckpoint,
    *,
    dataset: DatasetConfig,
    num_hyps: int = 12,
    cfg: Optional[TGetHypsCfg] = None,
    tts_opts: Optional[TtsOpts] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Tuple[List[tk.Path], tk.Path]:
    """
    Get hyps from CTC model, split up over ``num_hyps``.

    :param model:
    :param dataset:
    :param num_hyps: how many hyps per seq to generate. each hyp will be saved separately.
    :param cfg: config for the hyps generation (e.g. dropout, specaugment, etc)
    :param tts_opts: TTS model, if we want to generate TTS data on-the-fly. requires cfg to be a GetHypsCfgV3
    :param extra_config:
    :return: hyps (list of HDF paths, len is ``num_hyps``), reference sequences path (HDF)
    """
    if cfg is None:
        cfg = GetHypsCfgV1(**kwargs)
    else:
        assert not kwargs, f"unexpected kwargs {kwargs!r}"
    if num_hyps > 1:
        if isinstance(cfg, GetHypsCfgV1):
            assert cfg.dropout > 0, "if dropout is disabled, generating multiple hypotheses doesn't make sense"
        elif isinstance(cfg, GetHypsCfgV2):
            assert cfg.dropout_max > 0, "if dropout is disabled, generating multiple hypotheses doesn't make sense"
        elif isinstance(cfg, (GetHypsCfgV3, GetHypsCfgV4)):
            pass  # there can also be TTS here, so the check is not so easy... just accept it
        else:
            raise TypeError(f"invalid cfg {cfg!r} type {type(cfg)}")
    from i6_experiments.users.zeyer.forward_to_hdf import forward_to_hdf

    extern_data_template_dict = dataset.get_extern_data()

    if _SisGetHypsSplitIgnoreExtraConfig:
        extra_config = None

    corrupted_hypotheses = []
    for i in range(num_hyps):
        hdf = sis_get_hyps_single_split(
            model=model, dataset=dataset, hyp_idx=i, cfg=cfg, tts_opts=tts_opts, extra_config=extra_config
        )
        corrupted_hypotheses.append(hdf)

    # this job just forwards the dataset so that we have uncorrupted reference data
    reference_sequences = forward_to_hdf(
        dataset=dataset,
        model=model,  # for backend and behaviour_version
        forward_step=noop_output_classes,
        config={
            "model_outputs": {"output": extern_data_template_dict["classes"]},
            "load_epoch": 1,
            **(extra_config or {}),
        },
        forward_device="cpu",
        forward_rqmt={"cpu": 16},
    )
    # TODO: figure out a more exact value, my jobs were failing because of too little tmp space so I just set it here
    # refs.creator.rqmt["sbatch_args"] = refs.creator.rqmt.get("sbatch_args", []) + [
    #   "--tmp=30G"
    # ]

    return corrupted_hypotheses, reference_sequences


def sis_get_hyps_single_split(
    model: ModelWithCheckpoint,
    *,
    dataset: DatasetConfig,
    hyp_idx: int,
    hyp_output_tensor_template_dict: Optional[Dict[str, Any]] = None,
    cfg: TGetHypsCfg,
    tts_opts: Optional[TtsOpts] = None,
    extra_config: Optional[Dict[str, Any]] = None,
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

    if not hyp_output_tensor_template_dict:
        extern_data_template_dict = dataset.get_extern_data()
        target_template_dict = extern_data_template_dict[dataset.get_default_target()]
        target_template = Tensor(
            name=dataset.get_default_target(),
            # Exclude vocab to not load it at this point.
            **{k: v for k, v in target_template_dict.items() if k != "vocab"},
        )
        batch_dim_, out_spatial_dim = target_template.dims
        assert batch_dim_.is_batch_dim() and batch_dim_ == batch_dim
        assert out_spatial_dim.is_spatial_dim()
        hyps_spatial_dim = Dim(None, name="hyps_spatial")
        hyp_output_tensor_template_dict = {
            "dims": [batch_dim, hyps_spatial_dim],
            "sparse_dim": target_template.sparse_dim,
            "dtype": target_template.dtype,
        }

    if isinstance(cfg, GetHypsCfgV1):
        # messy code because I don't want to break the hash
        if cfg.dropout == 0.1 and cfg.enable_specaugment:
            forward_def = model_recog_single
        else:
            forward_def = functools.partial(
                model_recog_single_dropout,
                dropout=cfg.dropout,
                enable_specaugment=cfg.enable_specaugment,
                break_hash=43,
            )
    elif isinstance(cfg, GetHypsCfgV2):
        forward_def = functools.partial(model_recog_single_v2, cfg=cfg)
    elif isinstance(cfg, GetHypsCfgV3):
        forward_def = functools.partial(model_recog_single_v3, cfg=cfg)
    elif isinstance(cfg, GetHypsCfgV6):  # need to test v6 first, because its a subclass of v5, v4
        forward_def = functools.partial(model_recog_single_v6, cfg=cfg)
    elif isinstance(cfg, GetHypsCfgV5):  # need to test v5 first, because its a subclass of v4
        forward_def = functools.partial(model_recog_single_v5, cfg=cfg)
    elif isinstance(cfg, GetHypsCfgV4):
        forward_def = functools.partial(model_recog_single_v4, cfg=cfg)
    else:
        raise TypeError(f"invalid cfg {cfg!r} type {type(cfg)}")

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

    model_def = model.definition
    if tts_opts:
        assert isinstance(cfg, (GetHypsCfgV3, GetHypsCfgV4))
        from .tts_model import get_asr_with_tts_model_def

        model_def = get_asr_with_tts_model_def(asr_model_def=model_def, tts_opts=tts_opts)

    if isinstance(cfg, GetHypsCfgV5) and cfg.sample_method == "discriminator":
        raise NotImplementedError("sampling with discriminator not implemented yet")

    model = ModelWithCheckpoint(definition=model_def, checkpoint=model.checkpoint)

    post_cfg_update_batchsize = {
        # "__env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    }

    if "batch_size" not in base_config and "max_seqs" not in base_config:
        # Use a larger batch size by default
        # if this is too large, some tensor operations will fail due to 32bit-index limitations...
        # also unclear if this helps the speed much
        # Coqai needs 20k batch size, so cant set this higher after all...
        post_cfg_update_batchsize = {
            **post_cfg_update_batchsize,
            "batch_size": 20_000 * model.definition.batch_size_factor,
            "max_seqs": 2000,
        }

    hdf = forward_to_hdf(
        dataset=dataset,
        model=model,
        forward_def=forward_def,
        config={
            "batch_size": 20000 * model.definition.batch_size_factor,
            "max_seqs": 200,
            **base_config,
            "random_seed": 1023 + hyp_idx * 17,
        },
        forward_post_config=post_cfg_update_batchsize,
    )
    hdf.creator.rqmt["time"] = 24  # might need more time
    return hdf


_model_cache_by_name = {}
_called_base_ctc_py_once = False
_called_tts_ctc_py_once = False


def sis_get_model(
    name: Optional[str] = None,
    *,
    vocab: Optional[str] = None,
    use_dependency_boundary: bool = True,
    epoch: Optional[int] = None,
) -> ModelWithCheckpoint:
    """
    Get some CTC model.
    We currently assume this model already exists
    (even though the pipeline would allow to also retrain that),
    to get deterministic results.

    :return: model with checkpoint
    """

    if not name and vocab:
        name = _model_name_by_vocab[vocab]

    cache_name = name
    if name and epoch is not None:
        cache_name = f"{name}__epoch{epoch}"
    if cache_name in _model_cache_by_name:
        return _model_cache_by_name[cache_name]

    if use_dependency_boundary:
        from i6_experiments.common.helpers.dependency_boundary import dependency_boundary

        model = dependency_boundary(
            functools.partial(sis_get_model, name=name, use_dependency_boundary=False, epoch=epoch),
            hash=_dep_bound_hash_by_model_name.get(cache_name),
        )

    elif name == SPM10k_11gb or name == SPM10k_24gb:
        # TODO: use continous epoch number instead of calculating by hand
        config_opts = (
            _get_cfg_lrlin_oclr_by_bs_nep(40_000, 2000)
            if name == SPM10k_24gb
            else _get_cfg_lrlin_oclr_by_bs_nep(15_000, 500)
        )
        model = train_exp(
            name,
            config_24gb_v6 if name == SPM10k_24gb else config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
            model_config={
                "enc_conformer_layer": rf.build_dict(
                    rf.encoder.conformer.ConformerEncoderLayer,
                    ff=rf.build_dict(
                        rf.encoder.conformer.ConformerPositionwiseFeedForward,
                        activation=rf.build_dict(rf.relu_square),
                        with_bias=False,
                    ),
                    num_heads=8,
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **config_opts,
                "optimizer.weight_decay": 1e-2,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
            },
            vocab="spm10k",
            train_vocab_opts={
                "other_opts": {
                    "class": "SamplingBytePairEncoding",
                    "breadth_prob": 0.01,
                }
            },
        )
        if epoch is None:
            model = model.get_last_fixed_epoch()
        else:
            model = model.get_epoch(epoch)
    else:
        # noinspection PyProtectedMember
        from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
            py as ctc_py,
            _train_experiments as ctc_train_experiments,
        )
        from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_claix2023 import (
            py as ctc_claix2023_py,
        )
        from i6_experiments.users.zeyer.utils.sis_setup import disable_register_output

        global _called_base_ctc_py_once, _called_tts_ctc_py_once

        if not _called_base_ctc_py_once:
            with disable_register_output():
                ctc_py()
                ctc_claix2023_py()
            _called_base_ctc_py_once = True

        if name not in ctc_train_experiments and not _called_tts_ctc_py_once:
            with disable_register_output():
                py()
            _called_tts_ctc_py_once = True

        exp = ctc_train_experiments[name]
        model = exp

        if epoch is None:
            model = model.get_last_fixed_epoch()
        else:
            model = model.get_epoch(epoch)

    # Not sure about the assert.
    # There are cases where we still train the CTC model, i.e. it is not always imported.
    # assert model.checkpoint.exists(), f"model {name} checkpoint {model.checkpoint} does not exist"
    _model_cache_by_name[cache_name] = model

    prefix = get_setup_prefix_for_module(__name__)
    # Note: Register the model checkpoint, such that any potential cleanup script
    # will not accidentally remove the model.
    if epoch is None:
        tk.register_output(f"{prefix}/ctc/baseline/{name}/checkpoint.pt", model.checkpoint.path)
    else:
        tk.register_output(f"{prefix}/ctc/baseline/{name}/checkpoint_epoch{epoch}.pt", model.checkpoint.path)
    model.checkpoint.path.creator.add_alias(f"{prefix}/ctc/baseline/{name}/training")
    return model


def model_recog(source: Tensor, *, in_spatial_dim: Dim, model: Model):
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.
    """
    from returnn.config import get_global_config

    config = get_global_config()
    num_hyps_dim = config.typed_value("num_hyps_dim")
    assert isinstance(num_hyps_dim, Dim)

    # We use dropout to get multiple different hypotheses.
    # We introduce the beam dim, and want to get different hypotheses for each beam (via dropout).
    source = rf.expand_dim(source, num_hyps_dim)

    expected_output = rf.get_run_ctx().expected_outputs["output"]
    hyps_packed_spatial_dim = expected_output.dims[-1]

    # Enable dropout.
    with rf.get_run_ctx().train_flag_ctx(True):
        logits, enc, enc_spatial_dim = model(source, in_spatial_dim=in_spatial_dim)
        labels = rf.reduce_argmax(logits, axis=model.wb_target_dim)
        labels = rf.cast(labels, "int32")
        log_probs = rf.log_softmax(logits, axis=model.wb_target_dim)
        log_probs = rf.gather(log_probs, indices=labels, axis=model.wb_target_dim)
        log_probs = rf.reduce_sum(log_probs, axis=enc_spatial_dim)

        labels_shifted = rf.shift_right(labels, axis=enc_spatial_dim, pad_value=model.blank_idx)
        mask_repeat = labels != labels_shifted
        labels, labels_spatial_dim = rf.masked_select(
            labels,
            mask=(labels != model.blank_idx) & mask_repeat,
            dims=[enc_spatial_dim],
        )

        # Set correct sparse_dim. Only works if blank comes after.
        assert model.target_dim.dimension == model.blank_idx
        labels.sparse_dim = model.target_dim

        labels, _ = rf.pack_padded(
            labels,
            dims=[num_hyps_dim, labels_spatial_dim],
            out_dim=hyps_packed_spatial_dim,
        )

    rf.get_run_ctx().mark_as_output(labels_spatial_dim.dyn_size_ext, "seq_lens")
    rf.get_run_ctx().mark_as_output(labels, "output")
    rf.get_run_ctx().mark_as_output(log_probs, "log_probs")
    rf.get_run_ctx().mark_as_output(enc_spatial_dim.dyn_size_ext, "enc_seq_lens")


# ForwardDef API
model_recog: ForwardRFDef[Model]


def model_recog_single(
    source: Tensor,
    *,
    in_spatial_dim: Dim,
    model: Model,
    enable_train_flag: bool = True,
    train_flag_funcs: Optional[Sequence[Callable]] = None,
):
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.
    """
    expected_output = rf.get_run_ctx().expected_outputs["output"]
    labels_spatial_dim = expected_output.dims[-1]

    # train_flag_ctx would potentially enable dropout etc
    with rf.get_run_ctx().train_flag_ctx(enable_train_flag, func=train_flag_funcs):
        logits, enc, enc_spatial_dim = model(source, in_spatial_dim=in_spatial_dim)
        labels = rf.reduce_argmax(logits, axis=model.wb_target_dim)
        labels = rf.cast(labels, "int32")
        log_probs = rf.log_softmax(logits, axis=model.wb_target_dim)
        log_probs = rf.gather(log_probs, indices=labels, axis=model.wb_target_dim)
        log_probs = rf.reduce_sum(log_probs, axis=enc_spatial_dim)

        labels_shifted = rf.shift_right(labels, axis=enc_spatial_dim, pad_value=model.blank_idx)
        mask_repeat = labels != labels_shifted
        labels, _ = rf.masked_select(
            labels,
            mask=(labels != model.blank_idx) & mask_repeat,
            dims=[enc_spatial_dim],
            out_dim=labels_spatial_dim,
        )

        # Set correct sparse_dim. Only works if blank comes after.
        assert model.target_dim.dimension == model.blank_idx
        labels.sparse_dim = model.target_dim

    rf.get_run_ctx().mark_as_output(labels, "output")
    rf.get_run_ctx().mark_as_output(log_probs, "log_probs")
    rf.get_run_ctx().mark_as_output(enc_spatial_dim.dyn_size_ext, "enc_seq_lens")


# this is technically wrong, but we can ignore the last argument because it has a default
model_recog_single: ForwardRFDef[Model]


def update_dropout(object: rf.Module, old_dropout: float, new_dropout: float):
    names = ["dropout", "input_dropout", "att_dropout"]
    for mod in object.modules():
        for name in names:
            if hasattr(mod, name):
                old = getattr(mod, name)
                if old == old_dropout:  # we don't want to update dropout that was e.g. explicitly set to 0
                    setattr(mod, name, new_dropout)


def model_recog_single_dropout(
    source: Tensor,
    *,
    in_spatial_dim: Dim,
    model: Model,
    dropout: float = 0.1,
    enable_specaugment: bool = True,
    break_hash: int = 0,
):
    assert break_hash == 43, f"break_hash is just a dummy argument to break the hash. got {break_hash}"

    if enable_specaugment and (dropout == 0.1 or dropout == 0):
        return model_recog_single(source, in_spatial_dim=in_spatial_dim, model=model, enable_train_flag=dropout > 0)

    if model.encoder.dropout != dropout:
        update_dropout(model.encoder, old_dropout=model.encoder.dropout, new_dropout=dropout)

    if not enable_specaugment:
        model._specaugment_opts["steps"] = (math.inf, math.inf, math.inf)

    model_recog_single(source, in_spatial_dim=in_spatial_dim, model=model, enable_train_flag=True)


# this is technically wrong, but we can ignore the last argument because it has a default
model_recog_single_dropout: ForwardRFDef[Model]


def model_recog_single_v2(
    source: Tensor,
    *,
    in_spatial_dim: Dim,
    model: Model,
    cfg: GetHypsCfgV2,
):
    import torch

    dropout = cfg.dropout_min + torch.rand(()).item() * (cfg.dropout_max - cfg.dropout_min)
    update_dropout(model.encoder, old_dropout=model.encoder.dropout, new_dropout=dropout)

    # Disable SpecAugment.
    # noinspection PyProtectedMember
    model._specaugment_opts["steps"] = (math.inf, math.inf, math.inf)

    model_recog_single(source, in_spatial_dim=in_spatial_dim, model=model, enable_train_flag=True)


def _update_dropout_v2(root: rf.Module, new_dropout: float):
    names = ["dropout", "input_dropout", "att_dropout"]
    for mod in root.modules():
        for name in names:
            if hasattr(mod, name):
                # Note: No check for the old value. Assume, if there is some dropout attrib, we can use it.
                # Also note, for the Transformer DLM / err model, we usually set dropout=0 for training,
                # so it doesn't make sense to check the old value.
                setattr(mod, name, new_dropout)


def model_recog_single_v3(
    source: Tensor,
    *,
    in_spatial_dim: Dim,
    model: Model,
    cfg: GetHypsCfgV3,
):
    import torch

    tts_model = getattr(model, "tts_model", None)
    if tts_model:  # via :func:`asr_with_tts_model_def`
        # We expect that we get phonemes as input, and we need to convert this to audio via the TTS model.
        from .tts_model import TtsModel

        tts_model: TtsModel
        source, in_spatial_dim = tts_model(source, spatial_dim=in_spatial_dim)

    dropout = cfg.dropout_min + torch.rand(()).item() * (cfg.dropout_max - cfg.dropout_min)
    _update_dropout_v2(model.encoder, new_dropout=dropout)

    if cfg.enable_specaugment:
        # Always enable.
        # noinspection PyProtectedMember
        model._specaugment_opts["steps"] = (0, 0, 0)
        if cfg.specaugment_opts:
            # noinspection PyProtectedMember
            model._specaugment_opts.update(cfg.specaugment_opts)
    else:
        # Disable SpecAugment.
        # noinspection PyProtectedMember
        model._specaugment_opts["steps"] = (math.inf, math.inf, math.inf)

    model_recog_single(source, in_spatial_dim=in_spatial_dim, model=model, enable_train_flag=True)


def _apply_v4_cfg(source: Tensor, in_spatial_dim: Dim, model: Model, cfg: GetHypsCfgV4):
    import torch

    tts_model = getattr(model, "tts_model", None)
    if tts_model:  # via :func:`asr_with_tts_model_def`
        # We expect that we get phonemes as input, and we need to convert this to audio via the TTS model.
        from .tts_model import TtsModel

        tts_model: TtsModel
        source, in_spatial_dim = tts_model(source, spatial_dim=in_spatial_dim)

        # lm-devtrain on ASR trained with the orig TTS data:
        # peak_normalization False: 1.64 WER
        # peak_normalization True:  1.60 WER
        if isinstance(cfg, GetHypsCfgV6) and cfg.data_perturbation_opts.get("peak_normalization", False):
            source_maxnorm = rf.reduce_max(rf.abs(source), axis=in_spatial_dim)
            source = source / (source_maxnorm + 1e-4)  # avoid division by zero

    train_flag_funcs = list(cfg.extra_train_flag_funcs)

    dropout = cfg.dropout_min + torch.rand(()).item() * (cfg.dropout_max - cfg.dropout_min)
    _update_dropout_v2(model.encoder, new_dropout=dropout)
    if cfg.dropout_max > 0:
        train_flag_funcs.append(rf.dropout)

    if cfg.enable_specaugment:
        # Always enable.
        # noinspection PyProtectedMember
        model._specaugment_opts["steps"] = (0, 0, 0)
        if cfg.specaugment_opts:
            # noinspection PyProtectedMember
            model._specaugment_opts.update(cfg.specaugment_opts)
        train_flag_funcs.append(rf.audio.specaugment)
    else:
        # Disable SpecAugment.
        # noinspection PyProtectedMember
        model._specaugment_opts["steps"] = (math.inf, math.inf, math.inf)
    return source, in_spatial_dim, train_flag_funcs


dump_iter = 0


def model_recog_single_v4(
    source: Tensor,
    *,
    in_spatial_dim: Dim,
    model: Model,
    cfg: GetHypsCfgV4,
):
    source, in_spatial_dim, train_flag_funcs = _apply_v4_cfg(source, in_spatial_dim, model, cfg)

    model_recog_single(source, in_spatial_dim=in_spatial_dim, model=model, train_flag_funcs=train_flag_funcs)


def model_recog_single_v5(
    source: Tensor,
    *,
    in_spatial_dim: Dim,
    model: Model,
    cfg: GetHypsCfgV5,
):
    if cfg.sample_method == "greedy":
        assert cfg.sample_opts is None
        model_recog_single_v4(source, in_spatial_dim=in_spatial_dim, model=model, cfg=cfg)
        return

    source, in_spatial_dim, train_flag_funcs = _apply_v4_cfg(source, in_spatial_dim, model, cfg)

    if cfg.sample_method == "top_k":
        model_recog_single_v5_top_k(
            source, in_spatial_dim=in_spatial_dim, model=model, train_flag_funcs=train_flag_funcs, cfg=cfg
        )
    elif cfg.sample_method == "discriminator":
        raise NotImplementedError("sampling with discriminator not implemented yet")
    else:
        raise ValueError(f"unknown sample_method {cfg.sample_method!r}")


def model_recog_single_v5_top_k(
    source: Tensor,
    *,
    in_spatial_dim: Dim,
    model: Model,
    enable_train_flag: bool = True,
    train_flag_funcs: Optional[Sequence[Callable]] = None,
    cfg: GetHypsCfgV5,
):
    import torch

    expected_output = rf.get_run_ctx().expected_outputs["output"]
    labels_spatial_dim = expected_output.dims[-1]
    sampling_opts = cfg.sample_opts or {}

    assert sampling_opts.get("p", 1.0) == 1.0, "top_p sampling not implemented yet"

    # train_flag_ctx would potentially enable dropout etc
    with rf.get_run_ctx().train_flag_ctx(enable_train_flag, func=train_flag_funcs):
        logits, enc, enc_spatial_dim = model(source, in_spatial_dim=in_spatial_dim)

        # first we collapse repeated labels
        labels_argmaxed = rf.reduce_argmax(logits, axis=model.wb_target_dim)
        labels_argmaxed = rf.cast(labels_argmaxed, "int32")

        # these logprobs will not correspond to the correct probs, but i still return them because they are expected ( but never used?)
        log_probs_wrong = rf.log_softmax(logits, axis=model.wb_target_dim)
        log_probs_wrong = rf.gather(log_probs_wrong, indices=labels_argmaxed, axis=model.wb_target_dim)
        log_probs_wrong = rf.reduce_sum(log_probs_wrong, axis=enc_spatial_dim)

        labels_argmax_shifted = rf.shift_right(labels_argmaxed, axis=enc_spatial_dim, pad_value=model.blank_idx)
        mask_repeat = labels_argmaxed != labels_argmax_shifted
        logits_collapsed, _ = rf.masked_select(
            logits,
            mask=(labels_argmaxed != model.blank_idx) & mask_repeat,
            dims=[enc_spatial_dim],
            out_dim=labels_spatial_dim,
        )

        # we collapsed our logits according to the argmax, now reduce the blank prob to zero TODO: maybe don't do this?
        if not sampling_opts.get("include_blank", False):
            logits_collapsed = rf.where(
                rf.range_over_dim(model.wb_target_dim) == model.blank_idx, float("-inf"), logits_collapsed
            )
        # take the top k
        logits_collapsed, (indices,), k_dim = rf.top_k(
            logits_collapsed, axis=[model.wb_target_dim], k_dim=Dim(sampling_opts.get("k", 20), name="cool-beam")
        )
        non_target_dims = logits_collapsed.remaining_dims(k_dim)
        logits_collapsed = logits_collapsed.copy_transpose(perm=[*non_target_dims, k_dim])
        # normalize again (because we took the top k). Theoretically not necessary becasue they are normalized in multinomial, but im concerned about numerical stability
        logits_collapsed -= rf.reduce_logsumexp(logits_collapsed, axis=k_dim)
        # now we sample from the top k according to the ctc logits
        torch_logis: torch.Tensor = rf.exp(logits_collapsed).raw_tensor
        *torch_batch_dims, torch_k_dim = torch_logis.shape
        torch_logis = torch_logis.view(-1, torch_k_dim)
        labels_sampled_idx_torch = torch.multinomial(torch_logis, 1, replacement=True)
        labels_sampled_idx_torch = labels_sampled_idx_torch.view(*torch_batch_dims)
        labels_sampled_idx = rf.convert_to_tensor(labels_sampled_idx_torch, dims=non_target_dims)
        # convert back to vocab indices
        labels = rf.gather(indices, axis=k_dim, indices=labels_sampled_idx)
        labels = rf.cast(labels, "int32")

        # Set correct sparse_dim. Only works if blank comes after.
        assert model.target_dim.dimension == model.blank_idx
        labels.sparse_dim = model.target_dim

    rf.get_run_ctx().mark_as_output(labels, "output")
    rf.get_run_ctx().mark_as_output(log_probs_wrong, "log_probs")
    rf.get_run_ctx().mark_as_output(enc_spatial_dim.dyn_size_ext, "enc_seq_lens")


def model_recog_single_v4_dense_out(
    source: Tensor,
    *,
    in_spatial_dim: Dim,
    model: Model,
    cfg: GetHypsCfgV4,
    hyps_probs_cfg: AsrHypsProbsCfg,
):
    source, in_spatial_dim, train_flag_funcs = _apply_v4_cfg(source, in_spatial_dim, model, cfg)
    with rf.get_run_ctx().train_flag_ctx(True, func=train_flag_funcs):
        logits, enc, enc_spatial_dim = model(source, in_spatial_dim=in_spatial_dim)

    model_recog_dense_out(
        logits,
        source=source,
        in_spatial_dim=in_spatial_dim,
        model=model,
        hyps_probs_cfg=hyps_probs_cfg,
        enc_spatial_dim=enc_spatial_dim,
    )


def model_recog_single_v3_dense_out(
    source: Tensor,
    *,
    in_spatial_dim: Dim,
    model: Model,
    cfg: GetHypsCfgV3,
    hyps_probs_cfg: AsrHypsProbsCfg,
):
    import torch

    tts_model = getattr(model, "tts_model", None)
    if tts_model:  # via :func:`asr_with_tts_model_def`
        # We expect that we get phonemes as input, and we need to convert this to audio via the TTS model.
        from .tts_model import TtsModel

        tts_model: TtsModel
        source, in_spatial_dim = tts_model(source, spatial_dim=in_spatial_dim)

    dropout = cfg.dropout_min + torch.rand(()).item() * (cfg.dropout_max - cfg.dropout_min)
    _update_dropout_v2(model.encoder, new_dropout=dropout)

    if cfg.enable_specaugment:
        # Always enable.
        # noinspection PyProtectedMember
        model._specaugment_opts["steps"] = (0, 0, 0)
        if cfg.specaugment_opts:
            # noinspection PyProtectedMember
            model._specaugment_opts.update(cfg.specaugment_opts)
    else:
        # Disable SpecAugment.
        # noinspection PyProtectedMember
        model._specaugment_opts["steps"] = (math.inf, math.inf, math.inf)

    with rf.get_run_ctx().train_flag_ctx(True):  # Dropout etc
        logits, enc, enc_spatial_dim = model(source, in_spatial_dim=in_spatial_dim)

    model_recog_dense_out(
        logits,
        source=source,
        in_spatial_dim=in_spatial_dim,
        model=model,
        hyps_probs_cfg=hyps_probs_cfg,
        enc_spatial_dim=enc_spatial_dim,
    )


def model_recog_dense_out(
    logits: Tensor,
    *,
    source: Tensor,
    in_spatial_dim: Dim,
    model: Model,
    hyps_probs_cfg: AsrHypsProbsCfg,
    enc_spatial_dim: Dim,
):
    # Inline model_recog_single(source, in_spatial_dim=in_spatial_dim, model=model, enable_dropout=True):
    # See .error_correction_model_gen_train_data_dense.get_hyps.

    from returnn.tensor import batch_dim
    from returnn.frontend.tensor_array import TensorArray
    from returnn.util.basic import CollectionReadCheckCovered

    expected_output = rf.get_run_ctx().expected_outputs["output"]
    batch_dim_, labels_spatial_dim_, out_probs_top_k_dim = expected_output.dims
    assert batch_dim_ == batch_dim  # sanity check
    assert out_probs_top_k_dim.is_static() and out_probs_top_k_dim.dimension == hyps_probs_cfg.top_k  # sanity check

    log_probs = rf.log_softmax(logits, axis=model.wb_target_dim)

    batch_dims = log_probs.remaining_dims((enc_spatial_dim, model.wb_target_dim))
    assert batch_dims == [batch_dim]  # just that it matches the expected output, otherwise not really needed

    if hyps_probs_cfg.extract_method == "soft_collapse_repeated":
        assert hyps_probs_cfg.include_blank  # not implemented currently otherwise

        from i6_experiments.users.zeyer.nn_rf.soft_collapse_repeated import soft_collapse_repeated

        opts = dict(threshold=0.9, reduce_type="max_renorm")  # some reasonable defaults
        if hyps_probs_cfg.extract_method_opts:
            opts.update(hyps_probs_cfg.extract_method_opts)
        log_probs, out_spatial_dim = soft_collapse_repeated(
            log_probs, spatial_dim=enc_spatial_dim, classes_dim=model.wb_target_dim, **opts
        )

        log_probs_, seq_targets, _ = rf.top_k(
            log_probs, axis=model.wb_target_dim, k_dim=out_probs_top_k_dim
        )  # [B,T_out,K]

        seq_targets = rf.cast(seq_targets, "int32")
        assert seq_targets.sparse_dim == model.wb_target_dim
        assert model.wb_target_dim.dimension == expected_output.sparse_dim.dimension
        seq_targets = rf.set_sparse_dim(seq_targets, expected_output.sparse_dim)

        ctc_seq_log_prob = rf.gather(log_probs_, axis=out_probs_top_k_dim, indices=0)  # [B,T_out]
        ctc_seq_log_prob = rf.reduce_sum(ctc_seq_log_prob, axis=out_spatial_dim)  # [B,T_out]

        labels_spatial_dim_.declare_same_as(out_spatial_dim)
        rf.get_run_ctx().mark_as_output(seq_targets, "output")  # [B,T_out,K]
        rf.get_run_ctx().mark_as_output(log_probs_, "output_k_lob_probs")  # [B,T_out,K]
        rf.get_run_ctx().mark_as_output(ctc_seq_log_prob, "log_probs")  # [B]
        rf.get_run_ctx().mark_as_output(enc_spatial_dim.dyn_size_ext, "enc_seq_lens")  # [B]
        return

    assert hyps_probs_cfg.extract_method == "label_prefix_search"
    assert not hyps_probs_cfg.include_blank
    opts = CollectionReadCheckCovered(hyps_probs_cfg.extract_method_opts or {})

    # currently these are not used... we could add some extra_opts to this function,
    # and then add those via functools.partial.
    top_k_with_random_sampling_opts: Optional[Dict[str, Any]] = opts.get("top_k_with_random_sampling", None)
    ctc_beam_size = opts.get("beam_size", 1)

    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.ctc_label_sync_espnet import (
        CtcPrefixScorer,
    )
    from i6_experiments.users.zeyer.nn_rf.top_k_and_random_choice_without_replacement import (
        top_k_and_random_choice_without_replacement,
    )

    ctc_beam_dim = Dim(1, name="ctc_initial_beam")
    ctc_prefix_scorer = CtcPrefixScorer(
        log_probs=log_probs,
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

    max_seq_len = enc_spatial_dim.get_size_tensor(device=source.device)
    neg_inf = float("-inf")

    i = 0
    seq_targets = []
    seq_backrefs = []
    seq_target_probs = []
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

        if top_k_with_random_sampling_opts is not None:
            ctc_seq_log_prob, (backrefs, target), ctc_beam_dim = top_k_and_random_choice_without_replacement(
                ctc_seq_log_prob,
                axis=[ctc_beam_dim, model.target_dim],
                k=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam"),
                **top_k_with_random_sampling_opts,
            )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.
        else:
            ctc_seq_log_prob, (backrefs, target), ctc_beam_dim = rf.top_k(
                ctc_seq_log_prob,
                k_dim=Dim(ctc_beam_size, name=f"ctc_dec_step{i}_beam"),
                axis=[ctc_beam_dim, model.target_dim],
            )  # ctc_seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> Vocab.

        target = rf.cast(target, dtype=rf.get_default_int_dtype())

        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        ctc_prefix_log_prob = rf.gather(ctc_prefix_log_prob, indices=backrefs)
        ctc_prefix_scorer_state = rf.nested.gather_nested(ctc_prefix_scorer_state, indices=backrefs)

        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        seq_target_probs.append(ctc_prefix_log_prob)

        i += 1
        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, i >= max_seq_len)
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    seq_target_probs_ = []
    indices = rf.range_over_dim(ctc_beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target, probs in zip(seq_backrefs[::-1], seq_targets[::-1], seq_target_probs[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        seq_target_probs_.insert(0, rf.gather(probs, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    seq_target_probs__ = TensorArray(seq_target_probs_[0])
    for target, probs in zip(seq_targets_, seq_target_probs_):
        seq_targets__ = seq_targets__.push_back(target)
        seq_target_probs__ = seq_target_probs__.push_back(probs)
    labels_spatial_dim = Dim(out_seq_len, name="ctc_labels_spatial")
    seq_targets___ = seq_targets__.stack(axis=labels_spatial_dim)
    seq_target_probs___ = seq_target_probs__.stack(axis=labels_spatial_dim)
    # Remove the remaining EOS labels.
    seq_targets___, _ = rf.slice(seq_targets___, axis=labels_spatial_dim, size=labels_spatial_dim)
    # Select first (best).
    ctc_seq_log_prob, seq_targets___, seq_target_probs___, labels_spatial_dim = rf.nested.gather_nested(
        (ctc_seq_log_prob, seq_targets___, seq_target_probs___, labels_spatial_dim),
        indices=rf.constant(0, dims=(), sparse_dim=ctc_beam_dim),
    )

    assert labels_spatial_dim.dyn_size_ext.dims_set == {batch_dim}  # sanity check
    assert seq_target_probs___.dims_set == {batch_dim, labels_spatial_dim, model.target_dim}  # sanity check
    seq_target_probs___, seq_targets___, _ = rf.top_k(
        seq_target_probs___, axis=model.target_dim, k_dim=out_probs_top_k_dim
    )
    seq_targets___ = rf.cast(seq_targets___, "int32")
    assert seq_targets___.sparse_dim.dimension == expected_output.sparse_dim.dimension
    seq_targets___ = rf.set_sparse_dim(seq_targets___, expected_output.sparse_dim)

    labels_spatial_dim_.declare_same_as(labels_spatial_dim)
    rf.get_run_ctx().mark_as_output(seq_targets___, "output")
    rf.get_run_ctx().mark_as_output(seq_target_probs___, "output_k_lob_probs")
    rf.get_run_ctx().mark_as_output(ctc_seq_log_prob, "log_probs")
    rf.get_run_ctx().mark_as_output(enc_spatial_dim.dyn_size_ext, "enc_seq_lens")


cur_spectogram_ind = 0


def model_recog_single_v6(
    source: Tensor,
    *,
    in_spatial_dim: Dim,
    model: Model,
    cfg: GetHypsCfgV6,
):
    if cfg.data_perturbation_opts is None or len(cfg.data_perturbation_opts) == 0:
        return model_recog_single_v5(source, in_spatial_dim=in_spatial_dim, model=model, cfg=cfg)

    if not hasattr(model, "cfgv6_init") or not model.cfgv6_init:
        print("Initializing model with cfgv6 data perturbation opts")
        perturb_opts = CollectionReadCheckCovered(cfg.data_perturbation_opts)

        assert model._mixup is None
        mixup_hook = None  # we use model._mixup as our hook for all the data perturbations

        save_spec_count: Optional[int] = perturb_opts.get("debug_save_spectogram_count", None)

        save_ind = 0

        cfg.extra_train_flag_funcs = list(cfg.extra_train_flag_funcs or [])

        def id(src: Tensor, *, spatial_dim: Dim) -> Tensor:
            nonlocal save_ind
            if save_spec_count is not None:
                global cur_spectogram_ind
                save_ind = cur_spectogram_ind
                cur_spectogram_ind += 1
                if save_ind >= save_spec_count:
                    print(f"Saved enough spectograms ({save_spec_count}), not saving more.")
                    import sys

                    sys.exit(0)

                save_spectogram(
                    src,
                    f"spec_{save_ind:06d}.png",
                    feature_dim=model.in_dim,
                    spatial_dim=spatial_dim,
                )
            return src

        mixup_hook = id  # default is no perturbation

        if perturb_opts.get("mixup", None) is not None:  # Mixup
            from i6_experiments.users.zeyer.returnn.models.rf_mixup import MixupOpts
            from i6_experiments.users.dorian_koch.misc.audio_perturbs import MixupWithBugsFixed

            mixup_opts = perturb_opts["mixup"]
            cfg.extra_train_flag_funcs.append(MixupWithBugsFixed.__call__)
            assert isinstance(mixup_opts, MixupOpts), "mixup opts must be a MixupOpts"

            prev_hook = mixup_hook
            _mixup = MixupWithBugsFixed(feature_dim=model.in_dim, opts=mixup_opts)

            def mixup_with_debug(src: Tensor, *, spatial_dim: Dim) -> Tensor:
                src = prev_hook(src, spatial_dim=spatial_dim)

                src_mixed = _mixup(src, spatial_dim=spatial_dim)

                if save_spec_count is not None:
                    save_spectogram(
                        src_mixed,
                        f"spec_{save_ind:06d}_mixed.png",
                        feature_dim=model.in_dim,
                        spatial_dim=spatial_dim,
                    )

                return src_mixed

            mixup_hook = mixup_with_debug

        if perturb_opts.get("gen_sa", None) is not None:  # Generalized SpecAugment
            gensa_opts = perturb_opts["gen_sa"]

            cfg.extra_train_flag_funcs.append(generalized_specaugment)

            prev_hook = mixup_hook

            def gensa(src: Tensor, *, spatial_dim: Dim) -> Tensor:
                src = prev_hook(src, spatial_dim=spatial_dim)

                src_whitenoised = generalized_specaugment(
                    x=src,
                    spatial_dim=spatial_dim,
                    feature_dim=model.in_dim,
                    max_consecutive_spatial_dims=gensa_opts.get("max_consecutive_spatial_dims", 20),
                    max_consecutive_feature_dims=gensa_opts.get("max_consecutive_feature_dims", None),
                    num_spatial_mask_factor=gensa_opts.get("num_spatial_mask_factor", 100),
                )

                if save_spec_count is not None:
                    save_spectogram(
                        src_whitenoised,
                        f"spec_{save_ind:06d}_whitenoised.png",
                        feature_dim=model.in_dim,
                        spatial_dim=spatial_dim,
                    )

                return src_whitenoised

            mixup_hook = gensa

        # this is not max pooling, but something very similar
        if (
            perturb_opts.get("max_pooling", None) is not None
        ):  # "A Perceptually Inspired Data Augmentation Method for Noise Robust CNN Acoustic Models"
            maxpool_opts = perturb_opts["max_pooling"]
            raise NotImplementedError()

        model._mixup = mixup_hook
        model.cfgv6_init = True

        perturb_opts.get("peak_normalization", None)  # this is read at another location
        perturb_opts.assert_all_read()

    return model_recog_single_v5(source, in_spatial_dim=in_spatial_dim, model=model, cfg=cfg)


def returnn_forward_step_with_forwarddef_and_output_reference(
    *, model, extern_data: TensorDict, forward_def, **_kwargs_unused
):
    # This whole function doesn't really do much. It just wraps the forward_def.
    # We might consider to remove this function and just use forward_def directly.
    import returnn.frontend as rf
    from returnn.tensor import batch_dim
    from returnn.config import get_global_config

    if rf.is_executing_eagerly():
        batch_size = int(batch_dim.get_dim_value())
        for batch_idx in range(batch_size):
            seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
            print(f"batch {batch_idx + 1}/{batch_size} seq_tag: {seq_tag!r}")

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    data = extern_data[default_input_key]
    data_spatial_dim = data.get_time_dim_tag()

    rf.get_run_ctx().mark_as_output(extern_data.data["classes"], "classes", dims=extern_data.data["classes"].dims)
    forward_def(data, in_spatial_dim=data_spatial_dim, model=model)


def noop_output_classes(
    *, model, extern_data: TensorDict, output_key: str = "classes", verbose: bool = True, **_kwargs_unused
):
    import returnn.frontend as rf
    from returnn.tensor import batch_dim

    if rf.is_executing_eagerly():
        batch_size = int(batch_dim.get_dim_value())
        if verbose:
            for batch_idx in range(batch_size):
                seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
                print(f"batch {batch_idx + 1}/{batch_size} seq_tag: {seq_tag!r}")
        elif batch_size > 1:
            seq_tag = extern_data["seq_tag"].raw_tensor[0].item()
            print(f"batch 1/{batch_size} seq_tag: {seq_tag!r}, (more output truncated)")

    rf.get_run_ctx().mark_as_output(extern_data.data[output_key], "output", dims=extern_data.data[output_key].dims)


def _name_for_dict(d: Dict[str, Any]) -> str:
    parts = []
    for k, v in d.items():
        k = "".join(part[0] for part in k.split("_"))  # some shortening
        if isinstance(v, (tuple, list)):
            v = "_".join(str(v_) for v_ in v)
        elif isinstance(v, dict):
            v = "{" + _name_for_dict(v) + "}"  # recursive
        elif is_dataclass(v):
            v = "{" + _name_for_dict(asdict(v)) + "}"
        parts.append(f"{k}{v}")
    return "-".join(parts)
