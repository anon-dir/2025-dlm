"""
Auto-generated code via i6_experiments.common.helpers.dependency_boundary.
Do not modify by hand!
"""

import i6_experiments.users.zeyer.model_interfaces.model_with_checkpoints
obj = object.__new__(i6_experiments.users.zeyer.model_interfaces.model_with_checkpoints.ModelWithCheckpoint)
import i6_experiments.users.zeyer.model_interfaces.model
_model_def_with_cfg = object.__new__(i6_experiments.users.zeyer.model_interfaces.model.ModelDefWithCfg)
import i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc
_model_def_with_cfg.model_def = i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc.ctc_model_def
_model_def_with_cfg.behavior_version = 21
_model_def_with_cfg.backend = 'torch'
_model_def_with_cfg.batch_size_factor = 160
_dict2 = {
    'class': 'returnn.frontend.encoder.conformer.ConformerConvSubsample',
    'out_dims': [32, 64, 64],
    'filter_sizes': [(3, 3, ), (3, 3, ), (3, 3, )],
    'pool_sizes': [(1, 2, )],
    'strides': [(1, 1, ), (3, 1, ), (2, 1, )],
}
_dict5 = {
    'class': 'rf.relu_square',
}
_dict4 = {
    'class': 'returnn.frontend.encoder.conformer.ConformerPositionwiseFeedForward',
    'activation': _dict5,
    'with_bias': False,
}
_dict3 = {
    'class': 'returnn.frontend.encoder.conformer.ConformerEncoderLayer',
    'ff': _dict4,
    'num_heads': 8,
}
_dict1 = {
    'class': 'returnn.frontend.encoder.conformer.ConformerEncoder',
    'input_layer': _dict2,
    'num_layers': 16,
    'out_dim': 1024,
    'encoder_layer': _dict3,
}
_model_def_with_cfg.config = {
    'enc_build_dict': _dict1,
    'feature_batch_norm': True,
}
import i6_core.returnn.training
_pt_checkpoint = object.__new__(i6_core.returnn.training.PtCheckpoint)
from i6_experiments.common.utils.fake_job import make_fake_job
_returnn_training_job = make_fake_job(module='i6_core.returnn.training', name='ReturnnTrainingJob', sis_hash='phrWiFtUTwmK')
from sisyphus import tk
_pt_checkpoint.path = tk.Path('models/epoch.100.pt', creator=_returnn_training_job)
_dict = {
    'definition': _model_def_with_cfg,
    'checkpoint': _pt_checkpoint,
}
obj.__dict__.update(_dict)
