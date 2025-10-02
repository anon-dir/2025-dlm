"""
TTS model

Reference: i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.experiments.tts.glow.ls460_lukas_base.run_flow_tts_460h
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union, Optional, Any, Dict, Tuple
from functools import cache, partial

from i6_core.text.processing import ConcatenateJob
from i6_experiments.users.dorian_koch.jobs.lexicon import MergeLexiconWithoutDuplicatesJob
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoint
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from sisyphus import tk, Path
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg
from i6_experiments.users.zeyer.utils.generic_job_output import generic_job_output
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim

if TYPE_CHECKING:
    import numpy as np


@dataclass
class TtsOpts:
    """
    Opts for TTS model, :class:`TtsModel`.
    """

    model_opts: Dict[str, Any]  # via get_tts_model_config()
    preload_from_files: Dict[str, Any]  # via get_tts_model_preload_from_files()
    gen_opts: Dict[str, Any]
    """
    opts for :class:`TtsModel`, mostly about generation (noise scale etc),
    excluding those for the model (they are given via :func:`get_tts_model_config`).
    """


@dataclass
class EcmTwiceOpts:
    """
    ErrorCorrectionModelTwiceOpts
    We can feed hypotheses through an error correction model to get slightly more correct hypotheses.
    """

    ecm_model: ModelWithCheckpoint
    name: str
    ecm_opts: Dict[str, Any]


TtsOptsOrEcmTwiceOpts = Union[TtsOpts, EcmTwiceOpts]


def get_tts_opts_default_model(gen_opts: Dict[str, Any], compatible_to_nick: Optional[bool] = None) -> TtsOpts:
    """
    Create TTS opts with our default TTS model.

    :param gen_opts: opts for :class:`TtsModel`, mostly about generation (noise scale etc),
        excluding those for the model (they are given via :func:`get_tts_model_config`).
        E.g. ``glow_tts_noise_scale_range`` and ``glow_tts_length_scale_range``.
    :param compatible_to_nick: if True, will use the same phoneme parameters as the orig TTS data
    :return: TTS opts
    """

    if compatible_to_nick:
        # convenience for compatibility with the orig TTS data, we will use this often
        # compatible_to_nick also needs peak normalization, but the difference is very small (~0.04 WER)
        # peak normalization can be enabled via GetHypsCfgV6.data_perturbation_opts
        gen_opts = dict_update_deep(
            {
                "phone_info": {
                    "add_silence_beginning": 1.0,
                    "add_silence_end": 1.0,
                    "phon_pick_strategy": "first",
                }
            },
            gen_opts,
        )
    elif compatible_to_nick is None:
        pass  # Maybe print a warning here in the future?

    return TtsOpts(
        model_opts=get_tts_model_config(),
        preload_from_files=get_tts_model_preload_from_files(),
        gen_opts=gen_opts,
    )


def get_tts_opts_coqui_ai_tts_your_tts(gen_opts: Optional[Dict[str, Any]] = None) -> TtsOpts:
    """
    :param gen_opts: for example: {"tts_model_opt_sample_ranges": {"length_scale": (1.0, 1.5)}}
    """
    from i6_experiments.users.zeyer.external_models import coqui_ai_tts

    model_name = "tts_models/multilingual/multi-dataset/your_tts"
    repo_dir = coqui_ai_tts.get_default_tts_repo_dir()

    return TtsOpts(
        model_opts={
            "class": CoquiAiTtsModel,
            "model_name": model_name,
            "tts_repo_dir": repo_dir,
            "tts_data_dir": coqui_ai_tts.download_model(model_name, tts_repo_dir=repo_dir),
        },
        preload_from_files={
            # ignore the params. CoquiAiTtsModel will load them.
            "tts_model": {"prefix": "tts_model.", "filename": None},
        },
        gen_opts=gen_opts or {},
    )


@cache
def get_tts_model_checkpoint() -> tk.Path:
    # TODO...
    f = generic_job_output("i6_core/returnn/training/ReturnnTrainingJob.Jqv1rStK7xWH/output/models/epoch.400.pt")
    prefix = get_setup_prefix_for_module(__name__)
    tk.register_output(f"{prefix}/tts_model_checkpoint", f)
    return f


@cache
def get_tts_gl_model_checkpoint() -> tk.Path:
    # TODO...
    f = generic_job_output("i6_core/returnn/training/ReturnnTrainingJob.H9EByABag8UN/output/models/epoch.050.pt")
    prefix = get_setup_prefix_for_module(__name__)
    tk.register_output(f"{prefix}/tts_gl_model_checkpoint", f)
    return f


@cache
def get_tts_phoneme_vocab() -> tk.Path:
    # TODO...
    f = generic_job_output("i6_core/returnn/vocabulary/ReturnnVocabFromPhonemeInventory.z2RlZd9Y0jWQ/output/vocab.pkl")
    # Special symbols at the end:
    # ...
    #  '[UNKNOWN]': 39,
    #  '[end]': 40,
    #  '[space]': 41,
    #  '[start]': 42,
    #  '[blank]': 43}
    prefix = get_setup_prefix_for_module(__name__)
    tk.register_output(f"{prefix}/tts_phoneme_vocab", f)
    return f


def get_tts_phoneme_vocab_special_symbols() -> Dict[str, Any]:
    """
    :return: special symbols in the phoneme vocab, matching to :func:`get_tts_phoneme_vocab`
        (But not sure if we really need that? Also not sure on space/blank.)
    """
    return {
        "unknown_label": "[UNKNOWN]",
        "bos_label": "[start]",
        "eos_label": "[end]",
        # "control_symbols": {"space": "[space]", "blank": "[blank]"}, -- not sure?
    }


def get_tts_phoneme_vocab_size() -> int:
    """:return: size of the phoneme vocab, matching to :func:`get_tts_phoneme_vocab`"""
    return 44


@cache
def get_tts_lexicon() -> tk.Path:
    # TODO...
    f = generic_job_output("i6_core/lexicon/modification/MergeLexiconJob.P8go21pxx40e/output/lexicon.xml.gz")
    prefix = get_setup_prefix_for_module(__name__)
    tk.register_output(f"{prefix}/tts_lexicon", f)
    return f


@cache
def get_extended_tts_lexicon() -> tk.Path:
    """
    The orig TTS lexicon doesnt have words from the LS ASR dev/test sets, so we manually add them here
    """
    from i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.data.tts.generation import (
        bliss_from_text,
        get_lexicon,
    )
    from i6_experiments.common.helpers.g2p import G2PBasedOovAugmenter
    from .error_correction_model_gen_train_data import get_real_asr_txt

    # compare with i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.experiments.tts.glow.ls460_lukas_base
    prefix = get_setup_prefix_for_module(__name__)

    ls_sets = ["dev-clean", "dev-other", "test-clean", "test-other"]
    ls_data = [get_real_asr_txt(subset=s) for s in ls_sets]
    ls_data_concat = ConcatenateJob(ls_data).out

    ls_data_bliss = bliss_from_text(prefix=prefix, name="librispeech-asr-evals", lm_text=ls_data_concat)
    ls960_tts_lexicon = get_lexicon(with_blank=False, corpus_key="train-other-960")

    g2p_augmenter = G2PBasedOovAugmenter(
        original_bliss_lexicon=ls960_tts_lexicon,
        train_lexicon=ls960_tts_lexicon,
        train_args={},
        apply_args={"concurrent": 5},
    )
    extended_bliss_lexicon = g2p_augmenter.get_g2p_augmented_bliss_lexicon(
        bliss_corpus=ls_data_bliss,
        corpus_name="ls_asr_evals",
        alias_path=prefix,
        casing="upper",
    )

    basic_lexicon = get_tts_lexicon()
    ls_data_merged_lexicon = MergeLexiconWithoutDuplicatesJob(
        bliss_lexica=[basic_lexicon, extended_bliss_lexicon]
    ).out_bliss_lexicon
    return ls_data_merged_lexicon


def get_tts_model_config() -> Dict[str, Any]:
    """
    From .../i6_core/returnn/forward/ReturnnForwardJobV2.2QuByt0Q3gsm/output/returnn.config
    """
    return {
        "glow_tts_model_config": {
            "feature_extraction_config": {
                "sample_rate": 16000,
                "win_size": 0.05,
                "hop_size": 0.0125,
                "f_min": 0,
                "f_max": 7600,
                "min_amp": 1e-10,
                "num_filters": 80,
                "center": True,
                "norm": (-72.83881497383118, 37.73079669103133),
            },
            "encoder_config": {
                "num_layers": 6,
                "vocab_size": 44,
                "basic_dim": 256,
                "conv_dim": 1024,
                "conv_kernel_size": 3,
                "dropout": 0.1,
                "mhsa_config": {
                    "input_dim": 256,
                    "num_att_heads": 2,
                    "dropout": 0.1,
                    "att_weights_dropout": 0.1,
                    "window_size": 4,
                    "heads_share": True,
                    "block_length": None,
                    "proximal_bias": False,
                    "proximal_init": False,
                },
                "prenet_config": {
                    "input_embedding_size": 256,
                    "hidden_dimension": 256,
                    "kernel_size": 5,
                    "output_dimension": 256,
                    "num_layers": 3,
                    "dropout": 0.5,
                },
            },
            "duration_predictor_config": {
                "num_convs": 2,
                "hidden_dim": 384,
                "kernel_size": 3,
                "dropout": 0.1,
            },
            "flow_decoder_config": {
                "target_channels": 80,
                "hidden_channels": 256,
                "kernel_size": 5,
                "dilation_rate": 1,
                "num_blocks": 12,
                "num_layers_per_block": 4,
                "num_splits": 4,
                "num_squeeze": 2,
                "dropout": 0.05,
                "use_sigmoid_scale": False,
            },
            "num_speakers": 1172,
            "speaker_embedding_size": 256,
            "mean_only": True,
        },
        "simple_gl_net_config": {
            "hidden_size": 512,
            "feature_extraction_config": {
                "sample_rate": 16000,
                "win_size": 0.05,
                "hop_size": 0.0125,
                "f_min": 0,
                "f_max": 7600,
                "min_amp": 1e-10,
                "num_filters": 80,
                "center": True,
                "norm": (-72.83881497383118, 37.73079669103133),
            },
        },
    }


def get_tts_model_dataset_dict(
    *,
    corpus_text: tk.Path,
    seq_list_file: Optional[tk.Path],
    fixed_random_seed: Optional[int] = None,
    train: bool,
    tts_opts: TtsOpts,
    use_extended_lexicon: bool = False,
) -> Dict[str, Any]:
    # see i6_experiments.users.rossenbach.setups.tts.preprocessing.process_corpus_text_with_extended_lexicon
    # see i6_core.corpus.transform.ApplyLexiconToCorpusJob
    opts: Dict[str, Any] = {
        "class": "LmDataset",
        "corpus_file": corpus_text,
        "seq_list_file": seq_list_file,
        "use_cache_manager": True,
        "skip_empty_lines": False,
        "dtype": "int32",
        "seq_end_symbol": None,  # handled via phone_info/orth_vocab
        "unknown_symbol": None,  # handled via phone_info/orth_vocab
        "fixed_random_seed": fixed_random_seed,
    }
    cls_ = tts_opts.model_opts.get("class", None)
    if cls_ is None:  # our default TtsModel
        # Our default TtsModel works on phonemes.
        opts["phone_info"] = {
            "lexicon_file": get_tts_lexicon(),
            "phoneme_vocab_file": get_tts_phoneme_vocab(),
            "allo_num_states": 1,
            "add_silence_beginning": 0.01 if train else 0.0,
            "add_silence_between_words": 0.95 if train else 1.0,
            "add_silence_end": 0.01 if train else 0.0,
            "repetition": 0.01 if train else 0.0,
            "silence_repetition": 0.01 if train else 0.0,
            "silence_lemma_orth": "[space]",
            "extra_begin_lemma": {"phons": [{"phon": "[start]"}]},
            "extra_end_lemma": {"phons": [{"phon": "[end]"}]},
        }
        if tts_opts.gen_opts.get("phone_info", None) is not None:
            opts["phone_info"].update(tts_opts.gen_opts["phone_info"])

        if use_extended_lexicon:
            opts["phone_info"]["lexicon_file"] = get_extended_tts_lexicon()
    elif cls_ is CoquiAiTtsModel:
        from i6_experiments.users.zeyer.external_models import coqui_ai_tts

        model_name = tts_opts.model_opts["model_name"]
        repo_dir = tts_opts.model_opts["tts_repo_dir"]
        data_dir = tts_opts.model_opts["tts_data_dir"]
        assert isinstance(repo_dir, Path)
        assert isinstance(data_dir, Path)
        vocab_file = coqui_ai_tts.ExtractVocabFromModelJob(
            model_name=model_name, tts_model_dir=data_dir, tts_repo_dir=repo_dir
        ).out_vocab_file

        opts["orth_vocab"] = {"class": "CharacterTargets", "vocab_file": vocab_file, "unknown_label": None}
        opts["orth_post_process"] = functools.partial(coqui_ai_tts_model_orth_post_process, tts_repo_dir=repo_dir)
    else:
        raise ValueError(f"unexpected TTS model class {cls_}")
    return opts


def test_tts_model_dataset_dict():
    from i6_core.util import instanciate_delayed
    from returnn.util.basic import better_repr
    from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data

    lm_text = get_librispeech_normalized_lm_data()
    d = get_tts_model_dataset_dict(corpus_text=lm_text, seq_list_file=None, train=True, tts_opts=TtsOpts({}, {}, {}))
    d = instanciate_delayed(d)
    print(better_repr(d))


def get_tts_model_dataset_extern_data_data(*, tts_opts: TtsOpts) -> Dict[str, Any]:
    from returnn.tensor import batch_dim, Dim

    cls_ = tts_opts.model_opts.get("class", None)
    if cls_ is None:  # our default TtsModel
        return {
            "dim_tags": [batch_dim, Dim(None, name="phone_seq", kind=Dim.Types.Spatial)],
            "sparse_dim": Dim(get_tts_phoneme_vocab_size(), name="phone_vocab"),
            "vocab": {
                "class": "Vocabulary",
                "vocab_file": get_tts_phoneme_vocab(),
                **get_tts_phoneme_vocab_special_symbols(),
            },
        }
    elif cls_ is CoquiAiTtsModel:
        from i6_experiments.users.zeyer.external_models import coqui_ai_tts

        model_name = tts_opts.model_opts["model_name"]
        repo_dir = tts_opts.model_opts["tts_repo_dir"]
        data_dir = tts_opts.model_opts["tts_data_dir"]
        assert isinstance(repo_dir, Path)
        assert isinstance(data_dir, Path)
        vocab_file = coqui_ai_tts.ExtractVocabFromModelJob(
            model_name=model_name, tts_model_dir=data_dir, tts_repo_dir=repo_dir
        ).out_vocab_file

        return {
            "dim_tags": [batch_dim, Dim(None, name="char_seq", kind=Dim.Types.Spatial)],
            # sparse_dim will be inferred from vocab
            "vocab": {"class": "CharacterTargets", "vocab_file": vocab_file, "unknown_label": None},
        }
    else:
        raise ValueError(f"unexpected TTS model class {cls_}")


def get_asr_with_tts_model_def(
    *, asr_model_def: Union[ModelDef, ModelDefWithCfg], tts_opts: TtsOpts
) -> ModelDefWithCfg:
    """
    :param asr_model_def: e.g. the CTC model
    :param tts_opts:
    :return: combined model def. the config contains "preload_from_files" for the TTS model.
    """
    config = {}
    if isinstance(asr_model_def, ModelDefWithCfg):
        config.update(asr_model_def.config)
        asr_model_def = asr_model_def.model_def

    tts_model_config = tts_opts.model_opts.copy()
    # add everything except phone_info
    # (phone_info is handled separately in the LmDataset)
    tts_model_config.update({k: v for k, v in tts_opts.gen_opts.items() if k != "phone_info"})

    # noinspection PyTypeChecker
    combined_model_def: ModelDef = partial(
        asr_with_tts_model_def, asr_model_def=asr_model_def, tts_model_config=tts_model_config
    )
    # Make it a proper ModelDef
    combined_model_def.behavior_version = asr_model_def.behavior_version
    combined_model_def.backend = asr_model_def.backend
    combined_model_def.batch_size_factor = 1  # it's not on audio anymore but on the phonemes (or text) sequence

    assert "preload_from_files" not in config
    config["preload_from_files"] = tts_opts.preload_from_files
    return ModelDefWithCfg(model_def=combined_model_def, config=config)


# to be used with functools.partial to bind ctc_model_def and tts_model_config
def asr_with_tts_model_def(*, asr_model_def: ModelDef, tts_model_config: Dict[str, Any], **kwargs):
    from returnn.config import get_global_config

    asr_model = asr_model_def(**kwargs)

    config = get_global_config()

    extern_data_dict = config.typed_value("extern_data")
    default_input_key = config.typed_value("default_input")
    data_templ_dict = {"name": default_input_key, **extern_data_dict[default_input_key]}
    data = Tensor(**data_templ_dict)
    # input data is expected to be phonemes
    assert data.sparse_dim

    tts_model_config = tts_model_config.copy()
    cls_: Optional[type] = tts_model_config.pop("class", None)
    if not cls_:
        tts_model = TtsModel(phoneme_vocab_dim=data.sparse_dim, **tts_model_config)
    else:
        tts_model = cls_(**tts_model_config)
    asr_model.tts_model = tts_model
    return asr_model


asr_with_tts_model_def: ModelDef


def get_tts_model_preload_from_files() -> Dict[str, Dict[str, Any]]:
    """for ``preload_from_files`` in the RETURNN config"""
    return {
        "tts_model": {"prefix": "tts_model.glow_tts_model.", "filename": get_tts_model_checkpoint()},
        "tts_gl_model": {"prefix": "tts_model.gl_model.", "filename": get_tts_gl_model_checkpoint()},
        # no need to load, it's just a persistent buffer which will get initialized anyway
        "tts_griffin_lim": {"pattern": "tts_model.griffin_lim.*", "filename": None},
    }


class TtsModel(rf.Module):
    """
    Adapted from:
    i6_experiments.users.rossenbach.experiments.librispeech.ctc_rnnt_standalone_2024.pytorch_networks.glow_tts.simple_gl_decoder
    """

    def __init__(
        self,
        phoneme_vocab_dim: Dim,
        *,
        glow_tts_model_config: Dict[str, Any],
        simple_gl_net_config: Dict[str, Any],
        gl_iter: int = 32,
        gl_momentum: float = 0.99,
        glow_tts_noise_scale_range: Tuple[float, float],  # e.g. (0.3, 0.9)
        glow_tts_length_scale_range: Tuple[float, float],  # e.g. (0.7, 1.3)
        phone_swapout_rate_range: Tuple[float, float] = (0.0, 0.0),
    ):
        super().__init__()

        import torchaudio
        from i6_experiments.users.zeyer.experiments.nick_ctc_rnnt_standalone_2024.pytorch_networks.glow_tts.glow_tts_v1 import (
            Model as GlowTtsModel,
        )
        from i6_experiments.users.zeyer.experiments.nick_ctc_rnnt_standalone_2024.pytorch_networks.vocoder.simple_gl.blstm_gl_predictor import (
            Model as BlstmGlPredictorModel,
        )

        self.phoneme_vocab_dim = phoneme_vocab_dim
        self.glow_tts_noise_scale_range = glow_tts_noise_scale_range
        self.glow_tts_length_scale_range = glow_tts_length_scale_range
        self.phone_swapout_rate_range = phone_swapout_rate_range

        self.glow_tts_model = GlowTtsModel(config=glow_tts_model_config)
        self.gl_model = BlstmGlPredictorModel(config=simple_gl_net_config)

        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=800,
            n_iter=gl_iter,
            win_length=int(0.05 * 16000),
            hop_length=int(0.0125 * 16000),
            power=1.0,
            momentum=gl_momentum,
        )

    def __call__(self, phonemes: Tensor, *, spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        from returnn.tensor import Dim, batch_dim
        import torch

        assert not self.glow_tts_model.training

        assert phonemes.sparse_dim == self.phoneme_vocab_dim
        batch_dims = phonemes.remaining_dims(spatial_dim)
        assert batch_dims == [batch_dim]  # currently expected below...

        if self.phone_swapout_rate_range[1] > 0:
            random_phonemes = rf.random_uniform(
                phonemes.dims,
                sparse_dim=phonemes.sparse_dim,
                dtype=phonemes.dtype,
                device=phonemes.device,
                minval=0,
                maxval=phonemes.sparse_dim.dimension,
            )  # [B, T_phon]
            phone_swapout_rate = rf.random_uniform(
                batch_dims, minval=self.phone_swapout_rate_range[0], maxval=self.phone_swapout_rate_range[1]
            )  # [B]
            mask = rf.random_uniform(dims=phonemes.dims) < phone_swapout_rate  # [B, T_phon]
            phonemes = rf.where(mask, random_phonemes, phonemes)  # [B, T_phon]

        phonemes_pt = phonemes.copy_compatible_to_dims_raw([batch_dim, spatial_dim])  # [B, T_phon] (sparse)
        phonemes_len_pt = spatial_dim.get_size_tensor(device=phonemes.device).copy_compatible_to_dims_raw(
            [batch_dim]
        )  # [B]
        bs = phonemes_pt.size(0)

        speaker_labels = torch.randint(
            0, self.glow_tts_model.num_speakers, (bs, 1), device=phonemes_pt.device
        )  # [B, 1] (sparse)

        # noise_scale should match [B,1,T_freq] or [B,1,1].
        noise_scale = self.glow_tts_noise_scale_range[0] + torch.rand((bs, 1, 1), device=phonemes_pt.device) * (
            self.glow_tts_noise_scale_range[1] - self.glow_tts_noise_scale_range[0]
        )  # [B, 1, 1]
        # length_scale should match [B,1,T_freq] or [B,1,1].
        length_scale = self.glow_tts_length_scale_range[0] + torch.rand((bs, 1, 1), device=phonemes_pt.device) * (
            self.glow_tts_length_scale_range[1] - self.glow_tts_length_scale_range[0]
        )  # [B, 1, 1]

        ((log_mels, z_m, z_logs, logdet, z_mask, y_lengths), _, _) = self.glow_tts_model(
            phonemes_pt,
            phonemes_len_pt,
            g=speaker_labels,
            gen=True,
            noise_scale=noise_scale,
            length_scale=length_scale,
        )
        # log_mels: [B, F_logmel, T_freq]
        assert y_lengths.shape == (bs,)  # [B]

        _, linears = self.gl_model(log_mels.transpose(1, 2), y_lengths)  # [B, T_freq, F_freq]
        linears = linears.transpose(1, 2)  # [B, F_freq, T_freq]

        # !!! this is wrong!
        # need to cut the linears to length before griffin lim!
        # but this would have been fixed by batch size = 1, so this is not our issue...
        wave = self.griffin_lim(linears)  # [B, T_wave], 16kHz
        # griffin_lim uses torch.istft. From istft doc:
        # `T` is the number of frames, `1 + length // hop_length`
        # <=> length = (T - 1) * hop_length
        wave_lengths = (y_lengths - 1) * self.griffin_lim.hop_length
        assert wave_lengths.max() == wave.size(1)
        wave_lengths = wave_lengths.to(torch.int32)
        wave_lengths_rf = rf.convert_to_tensor(wave_lengths.cpu(), dims=[batch_dim])
        wave_lengths_dim = Dim(wave_lengths_rf, name="wave")
        wave_rf = rf.convert_to_tensor(wave, dims=[batch_dim, wave_lengths_dim])

        if os.environ.get("RETURNN_DEBUG", None) == "1":
            for b in range(bs):
                save_audio(
                    wave[b, : wave_lengths[b]].cpu().numpy(), name=f"ttsGen-step{rf.get_run_ctx().step}-batch{b}"
                )

        return wave_rf, wave_lengths_dim


def save_audio(wav: np.ndarray, name: str, sample_rate: int = 16_000):
    import subprocess
    import numpy as np

    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    p1 = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-threads",
            "1",
            "-f",
            "s16le",
            "-ar",
            "%i" % sample_rate,
            "-i",
            "pipe:0",
            "-c:a",
            # "libvorbis",
            "aac",
            "-q",
            "3.0",
            f"{name}.aac",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    p1.communicate(input=wav.astype(np.int16).tobytes())
    p1.terminate()
    if p1.returncode != 0:
        raise subprocess.CalledProcessError(p1.returncode, p1.args)


class CoquiAiTtsModel(rf.Module):
    """
    Use Coqui AI TTS.
    """

    def __init__(
        self,
        *,
        model_name: str,
        tts_repo_dir: Union[str, Path],
        tts_data_dir: Union[str, Path],
        tts_model_opts: Optional[Dict[str, float]] = None,
        tts_model_opt_sample_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        :param model_name: for TTS api
        :param tts_repo_dir: path to the TTS repo
        :param tts_data_dir: path to where the TTS model was downloaded
        :param tts_model_opts: original YourTTS has
            {"length_scale": 1.5, "inference_noise_scale": 0.3, "inference_noise_scale_dp": 0.3}.
            You can overwrite those (or maybe others).
        :param tts_model_opt_sample_ranges: same as tts_model_opts, but will sample from given range
        """
        super().__init__()

        if isinstance(tts_repo_dir, Path):
            tts_repo_dir = tts_repo_dir.get_path()
        assert isinstance(tts_repo_dir, str)
        if isinstance(tts_data_dir, Path):
            tts_data_dir = tts_data_dir.get_path()
        assert isinstance(tts_data_dir, str)

        self.model_name = model_name
        self.tts_repo_dir = tts_repo_dir
        self.tts_data_dir = tts_data_dir

        # See i6_experiments.users.zeyer.external_models.coqui_ai_tts._demo for reference.

        import sys
        import os
        import torch

        if tts_repo_dir not in sys.path:
            sys.path.insert(0, tts_repo_dir)
        os.environ["TTS_HOME"] = tts_data_dir
        os.environ["COQUI_TOS_AGREED"] = "1"

        from TTS.api import TTS
        from TTS.utils.manage import ModelManager
        from TTS.tts.utils.text.cleaners import multilingual_cleaners
        from TTS.tts.models.vits import Vits

        def _disallowed_create_dir_and_download_model(*_args, **_kwargs):
            raise RuntimeError(
                f"Disallowed create_dir_and_download_model({_args}, {_kwargs}),"
                f" model {model_name} not found, model data dir {tts_data_dir} not valid? "
            )

        # patch to avoid any accidental download
        assert hasattr(ModelManager, "create_dir_and_download_model")
        ModelManager.create_dir_and_download_model = _disallowed_create_dir_and_download_model

        # Note: we want to ignore the RETURNN param handling for this. E.g. set in preload_from_files as ignore.
        self.tts = TTS(model_name=self.model_name, progress_bar=False)
        self._tts_config = self.tts.synthesizer.tts_config
        self._tts_model: Vits = self.tts.synthesizer.tts_model
        self._func_wav_seq_lens_from_y_lens = self._get_func_wav_seq_lens_from_y_lens(self._tts_model)
        self._blank_idx: int = self._tts_model.tokenizer.characters.blank_id
        self._language_id: int = self._tts_model.language_manager.name_to_id["en"]  # maybe should be configurable...?
        self._speaker_embeddings = None
        if self._tts_model.speaker_manager:
            self._speaker_embeddings = torch.tensor(
                [v["embedding"] for v in self._tts_model.speaker_manager.embeddings.values()], dtype=torch.float32
            )  # [num_embeddings, emb_dim]
        assert isinstance(self._tts_model, Vits)  # for the following
        if tts_model_opts:
            for k, v in tts_model_opts.items():
                assert hasattr(self._tts_model, k)
                print(f"orig {k} = {getattr(self._tts_model, k)}, setting to {v}")
                setattr(self._tts_model, k, v)
        if tts_model_opt_sample_ranges:
            for k, v in tts_model_opt_sample_ranges.items():
                assert hasattr(self._tts_model, k)
                print(f"orig {k} = {getattr(self._tts_model, k)}, will set to random value in range {v}")
        self.tts_model_opt_sample_ranges = tts_model_opt_sample_ranges

        # we assume that in coqui_ai_tts_model_orth_post_process
        assert self._tts_model.tokenizer.text_cleaner is multilingual_cleaners

    def __call__(self, chars: Tensor, *, spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        """
        :param chars: [B, T_in], sparse. We assume this matches to the vocab of self._tts_model.tokenizer.characters,
            and also has done the same pre-processing.
            See i6_experiments.users.zeyer.external_models.coqui_ai_tts._demo for reference.
        :param spatial_dim: T_in
        """
        # See i6_experiments.users.zeyer.external_models.coqui_ai_tts._demo for reference.
        import torch

        batch_dims = chars.remaining_dims(spatial_dim)
        assert len(batch_dims) == 1  # currently only implemented for this case
        batch_dim = batch_dims[0]

        text_inputs_ = chars.copy_compatible_to_dims_raw([batch_dim, spatial_dim])  # [B, T_in]
        dev = text_inputs_.device
        batch_size = text_inputs_.shape[0]
        text_inputs_lens_ = spatial_dim.dyn_size_ext.copy_compatible_to_dims_raw([batch_dim])  # [B]
        text_inputs_lens_ = text_inputs_lens_.to(dev)

        if self._tts_model.device != dev:
            self._tts_model.to(dev)
        if self._speaker_embeddings is not None and self._speaker_embeddings.device != dev:
            self._speaker_embeddings = self._speaker_embeddings.to(dev)

        # intersperse blank
        text_inputs_lens = text_inputs_lens_ * 2 + 1
        text_inputs = torch.full(
            (batch_size, text_inputs_.shape[1] * 2 + 1), fill_value=self._blank_idx, dtype=torch.int32
        )
        text_inputs[:, 1::2] = text_inputs_
        text_inputs = text_inputs.to(dev)

        speaker_id = None
        speaker_embedding = None
        if self._speaker_embeddings is not None:
            speaker_id = torch.randint(0, self._speaker_embeddings.shape[0], [batch_size], device=dev)  # [B]
            if self._tts_config.use_d_vector_file:
                # the model expects speaker_embedding, and only those, not speaker_id
                speaker_embedding = torch.nn.functional.embedding(speaker_id, self._speaker_embeddings)
                speaker_id = None

        language_id = (
            torch.tensor(self._language_id, device=dev).expand(batch_size) if self._language_id is not None else None
        )  # [B]

        if self.tts_model_opt_sample_ranges:
            for k, v in self.tts_model_opt_sample_ranges.items():
                assert hasattr(self._tts_model, k)
                v_ = v[0] + torch.rand(()).item() * (v[1] - v[0])
                setattr(self._tts_model, k, v_)

        outputs = self._tts_model.inference(
            text_inputs,
            aux_input={
                "x_lengths": text_inputs_lens,
                "speaker_ids": speaker_id,
                "d_vectors": speaker_embedding,
                "language_ids": language_id,
            },
        )
        wav = outputs["model_outputs"]  # for Vits: [B, 1, T_wav]
        wav = wav.squeeze(1)  # [B, T_wav]
        y_mask = outputs["y_mask"]  # before the final waveform_decoder
        y_mask = y_mask.squeeze(1)  # [B, T_wav]
        y_lens = torch.sum(y_mask.to(torch.int32), dim=1).cpu()  # [B]
        wav_lens = self._func_wav_seq_lens_from_y_lens(y_lens)  # [B]
        wav_lens = wav_lens.to(torch.int32)

        wav_lens_ = rf.convert_to_tensor(wav_lens, dims=[batch_dim])
        wav_seq_dim = Dim(wav_lens_, name="T_wav")
        wav_ = rf.convert_to_tensor(wav, dims=[batch_dim, wav_seq_dim])

        if os.environ.get("RETURNN_DEBUG", None) == "1":
            for b in range(batch_size):
                save_audio(wav[b, : wav_lens[b]].cpu().numpy(), name=f"ttsGen-step{rf.get_run_ctx().step}-batch{b}")

        return wav_, wav_seq_dim

    @staticmethod
    def _get_func_wav_seq_lens_from_y_lens(tts_model: Any):
        # see i6_experiments.users.zeyer.external_models.coqui_ai_tts._demo for reference
        from TTS.tts.models.vits import Vits
        import torch
        from sympy.utilities.lambdify import lambdify
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        # noinspection PyProtectedMember
        from torch._subclasses.fake_tensor import FakeTensorMode

        # noinspection PyProtectedMember
        from torch._dynamo.source import ConstantSource

        # We assume that the model is a Vits model, and that we get the y_mask out,
        # and that we can calculate the final wave output length from that,
        # via the waveform_decoder.
        assert isinstance(tts_model, Vits)

        shape_env = ShapeEnv(duck_shape=False)
        with FakeTensorMode(allow_non_fake_inputs=True, shape_env=shape_env):
            # Assuming that waveform_decoder is from Vits.
            time_sym = shape_env.create_symbol(100, ConstantSource("T"))
            time_sym_ = shape_env.create_symintnode(time_sym, hint=None)
            fake_in = torch.empty(1, tts_model.waveform_decoder.conv_pre.in_channels, time_sym_)
            if getattr(tts_model.waveform_decoder, "cond_layer", None):
                fake_g_in = torch.empty(1, tts_model.waveform_decoder.cond_layer.in_channels, time_sym_)
            else:
                fake_g_in = None

            out = tts_model.waveform_decoder(fake_in, g=fake_g_in)
            print(f"{out.shape = }")
            out_size = out.shape[-1]
            assert isinstance(out_size, torch.SymInt)
            out_sym = out_size.node.expr
            out_sym_lambda = lambdify(time_sym, out_sym)

        return out_sym_lambda


# bind tts_repo_dir with functools.partial
def coqui_ai_tts_model_orth_post_process(text: str, *, tts_repo_dir: Union[str, Path]) -> str:
    import sys

    if isinstance(tts_repo_dir, Path):
        tts_repo_dir = tts_repo_dir.get_path()
    assert isinstance(tts_repo_dir, str)

    if tts_repo_dir not in sys.path:
        sys.path.insert(0, tts_repo_dir)

    from TTS.tts.utils.text.cleaners import multilingual_cleaners

    return multilingual_cleaners(text)
