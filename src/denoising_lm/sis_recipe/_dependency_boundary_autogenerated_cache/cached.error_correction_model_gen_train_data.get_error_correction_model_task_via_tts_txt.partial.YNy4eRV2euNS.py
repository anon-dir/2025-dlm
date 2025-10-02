"""
Auto-generated code via i6_experiments.common.helpers.dependency_boundary.
Do not modify by hand!
"""

import i6_experiments.users.zeyer.datasets.task
obj = object.__new__(i6_experiments.users.zeyer.datasets.task.Task)
obj.name = 'librispeech'
import returnn_common.datasets_old_2022_10.interface
obj.train_dataset = object.__new__(returnn_common.datasets_old_2022_10.interface.DatasetConfigStatic)
obj.train_dataset.main_name = None
obj.train_dataset.main_dataset = None
import returnn.tensor
from i6_experiments.common.utils.fake_job import make_fake_job
_train_sentence_piece_job = make_fake_job(module='i6_core.text.label.sentencepiece.train', name='TrainSentencePieceJob', sis_hash='ofYcs4cMRS8T')
from sisyphus import tk
_dict1 = {
    'class': 'SentencePieces',
    'model_file': tk.Path('spm_out.model', creator=_train_sentence_piece_job),
}
_dict = {
    'dim_tags': [returnn.tensor.batch_dim, returnn.tensor.Dim(None, name='hyps_spatial')],
    'sparse_dim': returnn.tensor.Dim(10240, name='vocab'),
    'vocab': _dict1,
}
_dict2 = {
    'dim_tags': [returnn.tensor.batch_dim, returnn.tensor.Dim(None, name='real_spatial')],
    'sparse_dim': returnn.tensor.Dim(10240, name='vocab'),
    'vocab': _dict1,
}
obj.train_dataset.extern_data = {
    'hyps': _dict,
    'real': _dict2,
}
obj.train_dataset.default_input = 'hyps'
obj.train_dataset.default_target = 'real'
_returnn_dataset_to_text_lines_job = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='2RRHuDdaIz15')
_returnn_dataset_to_text_lines_job1 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='coi5auWoOkCS')
_returnn_dataset_to_text_lines_job2 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='nY5N6WL4Cn2w')
_returnn_dataset_to_text_lines_job3 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='CzVUyVaawPGB')
_returnn_dataset_to_text_lines_job4 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='qZpEmGsAXWdQ')
_returnn_dataset_to_text_lines_job5 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='lwHR5oOOn5S6')
_returnn_dataset_to_text_lines_job6 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='Q9SrHlQsokMi')
_returnn_dataset_to_text_lines_job7 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='GYi79c0ycibm')
_returnn_dataset_to_text_lines_job8 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='oCwJrv17JGPd')
_returnn_dataset_to_text_lines_job9 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='hfk9OLOqRcxM')
_concatenate_job = make_fake_job(module='i6_core.text.processing', name='ConcatenateJob', sis_hash='mu6WasCzoxcK')
_extract_vocab_labels_job = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.vocab', name='ExtractVocabLabelsJob', sis_hash='rPARYce8g0Tb')
_extract_vocab_special_labels_job = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.vocab', name='ExtractVocabSpecialLabelsJob', sis_hash='rPARYce8g0Tb')
_dict5 = {
    'class': 'Vocabulary',
    'vocab_file': tk.Path('vocab.txt.gz', creator=_extract_vocab_labels_job),
    'special_symbols_via_file': tk.Path('vocab_special_labels.py', creator=_extract_vocab_special_labels_job),
}
_dict4 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job1), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job2), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job3), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job4), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job5), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job6), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job7), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job8), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job9), tk.Path('out.gz', creator=_concatenate_job)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict5,
    'seq_end_symbol': None,
    'unknown_symbol': None,
}
_returnn_dataset_to_text_lines_job10 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='KWvwGEfdqeE6')
_returnn_dataset_to_text_lines_job11 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='CtwP4gGfqjb7')
_dict6 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job10), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job10), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job10), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job10), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job10), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job10), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job10), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job10), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job10), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job10), tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job11)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict1,
    'seq_end_symbol': None,
    'unknown_symbol': None,
    'partition_epoch': 20,
    'seq_ordering': 'laplace:.1000',
}
_dict3 = {
    'hyps': _dict4,
    'real': _dict6,
}
_dict7 = {
    'hyps': ('hyps', 'data', ),
    'real': ('real', 'data', ),
}
obj.train_dataset.train_dataset = {
    'class': 'MetaDataset',
    'datasets': _dict3,
    'data_map': _dict7,
    'seq_order_control_dataset': 'real',
}
_returnn_dataset_to_text_lines_job12 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='uSYxTqUq4fkd')
_dict10 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job12)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict5,
    'seq_end_symbol': None,
    'unknown_symbol': None,
}
_returnn_dataset_to_text_lines_job13 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='Zlch4d63SUBg')
_dict11 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job13)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict1,
    'seq_end_symbol': None,
    'unknown_symbol': None,
    'seq_ordering': 'sorted_reverse',
}
_dict9 = {
    'hyps': _dict10,
    'real': _dict11,
}
_dict12 = {
    'hyps': ('hyps', 'data', ),
    'real': ('real', 'data', ),
}
_dict8 = {
    'class': 'MetaDataset',
    'datasets': _dict9,
    'data_map': _dict12,
    'seq_order_control_dataset': 'real',
}
_returnn_dataset_to_text_lines_job14 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='egIRQt35L9Wo')
_dict15 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job14)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict5,
    'seq_end_symbol': None,
    'unknown_symbol': None,
}
_returnn_dataset_to_text_lines_job15 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='rP2hFYwDon3i')
_dict16 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job15)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict1,
    'seq_end_symbol': None,
    'unknown_symbol': None,
    'seq_ordering': 'sorted_reverse',
}
_dict14 = {
    'hyps': _dict15,
    'real': _dict16,
}
_dict17 = {
    'hyps': ('hyps', 'data', ),
    'real': ('real', 'data', ),
}
_dict13 = {
    'class': 'MetaDataset',
    'datasets': _dict14,
    'data_map': _dict17,
    'seq_order_control_dataset': 'real',
}
obj.train_dataset.eval_datasets = {
    'dev': _dict8,
    'devtrain': _dict13,
}
obj.train_dataset.use_deep_copy = True
obj.train_epoch_split = 20
obj.dev_dataset = object.__new__(returnn_common.datasets_old_2022_10.interface.DatasetConfigStatic)
obj.dev_dataset.main_name = 'dev-other'
_returnn_dataset_to_text_lines_job16 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='8P0aa4z5rwoH')
_extract_seq_list_job = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.extract_seq_list', name='ExtractSeqListJob', sis_hash='m979Wt8mxXCI')
_dict19 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job16)],
    'seq_list_file': [tk.Path('out_seq_list.txt', creator=_extract_seq_list_job)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict5,
    'seq_end_symbol': None,
    'unknown_symbol': None,
}
_returnn_dataset_to_text_lines_job17 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='zII3U9NeQu8m')
_dict20 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job17)],
    'seq_list_file': [tk.Path('out_seq_list.txt', creator=_extract_seq_list_job)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict1,
    'seq_end_symbol': None,
    'unknown_symbol': None,
    'seq_ordering': 'sorted_reverse',
}
_dict18 = {
    'hyps': _dict19,
    'real': _dict20,
}
_dict21 = {
    'hyps': ('hyps', 'data', ),
    'real': ('real', 'data', ),
}
obj.dev_dataset.main_dataset = {
    'class': 'MetaDataset',
    'datasets': _dict18,
    'data_map': _dict21,
    'seq_order_control_dataset': 'hyps',
}
obj.dev_dataset.extern_data = obj.train_dataset.extern_data
obj.dev_dataset.default_input = 'hyps'
obj.dev_dataset.default_target = 'real'
obj.dev_dataset.train_dataset = None
obj.dev_dataset.eval_datasets = None
obj.dev_dataset.use_deep_copy = True
_dataset_config_static = object.__new__(returnn_common.datasets_old_2022_10.interface.DatasetConfigStatic)
_dataset_config_static.main_name = 'dev-clean'
_returnn_dataset_to_text_lines_job18 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='yAMwwfz7MnYe')
_extract_seq_list_job1 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.extract_seq_list', name='ExtractSeqListJob', sis_hash='R7nn4IbmI4q5')
_dict23 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job18)],
    'seq_list_file': [tk.Path('out_seq_list.txt', creator=_extract_seq_list_job1)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict5,
    'seq_end_symbol': None,
    'unknown_symbol': None,
}
_returnn_dataset_to_text_lines_job19 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='iADwvSKcsl58')
_dict24 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job19)],
    'seq_list_file': [tk.Path('out_seq_list.txt', creator=_extract_seq_list_job1)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict1,
    'seq_end_symbol': None,
    'unknown_symbol': None,
    'seq_ordering': 'sorted_reverse',
}
_dict22 = {
    'hyps': _dict23,
    'real': _dict24,
}
_dict25 = {
    'hyps': ('hyps', 'data', ),
    'real': ('real', 'data', ),
}
_dataset_config_static.main_dataset = {
    'class': 'MetaDataset',
    'datasets': _dict22,
    'data_map': _dict25,
    'seq_order_control_dataset': 'hyps',
}
_dataset_config_static.extern_data = obj.train_dataset.extern_data
_dataset_config_static.default_input = 'hyps'
_dataset_config_static.default_target = 'real'
_dataset_config_static.train_dataset = None
_dataset_config_static.eval_datasets = None
_dataset_config_static.use_deep_copy = True
_dataset_config_static1 = object.__new__(returnn_common.datasets_old_2022_10.interface.DatasetConfigStatic)
_dataset_config_static1.main_name = 'test-clean'
_returnn_dataset_to_text_lines_job20 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='qCAUhKYsiKUK')
_extract_seq_list_job2 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.extract_seq_list', name='ExtractSeqListJob', sis_hash='ybgVidU9gnb8')
_dict27 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job20)],
    'seq_list_file': [tk.Path('out_seq_list.txt', creator=_extract_seq_list_job2)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict5,
    'seq_end_symbol': None,
    'unknown_symbol': None,
}
_returnn_dataset_to_text_lines_job21 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='QER6se1RPiCD')
_dict28 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job21)],
    'seq_list_file': [tk.Path('out_seq_list.txt', creator=_extract_seq_list_job2)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict1,
    'seq_end_symbol': None,
    'unknown_symbol': None,
    'seq_ordering': 'sorted_reverse',
}
_dict26 = {
    'hyps': _dict27,
    'real': _dict28,
}
_dict29 = {
    'hyps': ('hyps', 'data', ),
    'real': ('real', 'data', ),
}
_dataset_config_static1.main_dataset = {
    'class': 'MetaDataset',
    'datasets': _dict26,
    'data_map': _dict29,
    'seq_order_control_dataset': 'hyps',
}
_dataset_config_static1.extern_data = obj.train_dataset.extern_data
_dataset_config_static1.default_input = 'hyps'
_dataset_config_static1.default_target = 'real'
_dataset_config_static1.train_dataset = None
_dataset_config_static1.eval_datasets = None
_dataset_config_static1.use_deep_copy = True
_dataset_config_static2 = object.__new__(returnn_common.datasets_old_2022_10.interface.DatasetConfigStatic)
_dataset_config_static2.main_name = 'test-other'
_returnn_dataset_to_text_lines_job22 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='HeS3kdQuM3IH')
_extract_seq_list_job3 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.extract_seq_list', name='ExtractSeqListJob', sis_hash='si4nIXi8UC4t')
_dict31 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job22)],
    'seq_list_file': [tk.Path('out_seq_list.txt', creator=_extract_seq_list_job3)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict5,
    'seq_end_symbol': None,
    'unknown_symbol': None,
}
_returnn_dataset_to_text_lines_job23 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='gizfitNbnbrs')
_dict32 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job23)],
    'seq_list_file': [tk.Path('out_seq_list.txt', creator=_extract_seq_list_job3)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict1,
    'seq_end_symbol': None,
    'unknown_symbol': None,
    'seq_ordering': 'sorted_reverse',
}
_dict30 = {
    'hyps': _dict31,
    'real': _dict32,
}
_dict33 = {
    'hyps': ('hyps', 'data', ),
    'real': ('real', 'data', ),
}
_dataset_config_static2.main_dataset = {
    'class': 'MetaDataset',
    'datasets': _dict30,
    'data_map': _dict33,
    'seq_order_control_dataset': 'hyps',
}
_dataset_config_static2.extern_data = obj.train_dataset.extern_data
_dataset_config_static2.default_input = 'hyps'
_dataset_config_static2.default_target = 'real'
_dataset_config_static2.train_dataset = None
_dataset_config_static2.eval_datasets = None
_dataset_config_static2.use_deep_copy = True
_dataset_config_static3 = object.__new__(returnn_common.datasets_old_2022_10.interface.DatasetConfigStatic)
_dataset_config_static3.main_name = 'dev-clean'
_returnn_dataset_to_text_lines_job24 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='MSdgekJSw0L8')
_dict35 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job24)],
    'seq_list_file': [tk.Path('out_seq_list.txt', creator=_extract_seq_list_job1)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict5,
    'seq_end_symbol': None,
    'unknown_symbol': None,
}
_dict36 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job19)],
    'seq_list_file': [tk.Path('out_seq_list.txt', creator=_extract_seq_list_job1)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict1,
    'seq_end_symbol': None,
    'unknown_symbol': None,
    'seq_ordering': 'sorted_reverse',
}
_dict34 = {
    'hyps': _dict35,
    'real': _dict36,
}
_dict37 = {
    'hyps': ('hyps', 'data', ),
    'real': ('real', 'data', ),
}
_dataset_config_static3.main_dataset = {
    'class': 'MetaDataset',
    'datasets': _dict34,
    'data_map': _dict37,
    'seq_order_control_dataset': 'hyps',
}
_dataset_config_static3.extern_data = obj.train_dataset.extern_data
_dataset_config_static3.default_input = 'hyps'
_dataset_config_static3.default_target = 'real'
_dataset_config_static3.train_dataset = None
_dataset_config_static3.eval_datasets = None
_dataset_config_static3.use_deep_copy = True
_dataset_config_static4 = object.__new__(returnn_common.datasets_old_2022_10.interface.DatasetConfigStatic)
_dataset_config_static4.main_name = 'dev-other'
_returnn_dataset_to_text_lines_job25 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='AdHxn98YoDXp')
_dict39 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job25)],
    'seq_list_file': [tk.Path('out_seq_list.txt', creator=_extract_seq_list_job)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict5,
    'seq_end_symbol': None,
    'unknown_symbol': None,
}
_dict40 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job17)],
    'seq_list_file': [tk.Path('out_seq_list.txt', creator=_extract_seq_list_job)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict1,
    'seq_end_symbol': None,
    'unknown_symbol': None,
    'seq_ordering': 'sorted_reverse',
}
_dict38 = {
    'hyps': _dict39,
    'real': _dict40,
}
_dict41 = {
    'hyps': ('hyps', 'data', ),
    'real': ('real', 'data', ),
}
_dataset_config_static4.main_dataset = {
    'class': 'MetaDataset',
    'datasets': _dict38,
    'data_map': _dict41,
    'seq_order_control_dataset': 'hyps',
}
_dataset_config_static4.extern_data = obj.train_dataset.extern_data
_dataset_config_static4.default_input = 'hyps'
_dataset_config_static4.default_target = 'real'
_dataset_config_static4.train_dataset = None
_dataset_config_static4.eval_datasets = None
_dataset_config_static4.use_deep_copy = True
_dataset_config_static5 = object.__new__(returnn_common.datasets_old_2022_10.interface.DatasetConfigStatic)
_dataset_config_static5.main_name = 'lm-devtrain'
_returnn_dataset_to_text_lines_job26 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='3Di8CuVEbNzi')
_dict43 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job26)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict5,
    'seq_end_symbol': None,
    'unknown_symbol': None,
}
_returnn_dataset_to_text_lines_job27 = make_fake_job(module='i6_experiments.users.zeyer.datasets.utils.serialize', name='ReturnnDatasetToTextLinesJob', sis_hash='jebPIYMLF5sT')
_dict44 = {
    'class': 'LmDataset',
    'corpus_file': [tk.Path('out.txt.gz', creator=_returnn_dataset_to_text_lines_job27)],
    'use_cache_manager': True,
    'skip_empty_lines': False,
    'orth_vocab': _dict1,
    'seq_end_symbol': None,
    'unknown_symbol': None,
    'seq_ordering': 'sorted_reverse',
}
_dict42 = {
    'hyps': _dict43,
    'real': _dict44,
}
_dict45 = {
    'hyps': ('hyps', 'data', ),
    'real': ('real', 'data', ),
}
_dataset_config_static5.main_dataset = {
    'class': 'MetaDataset',
    'datasets': _dict42,
    'data_map': _dict45,
    'seq_order_control_dataset': 'hyps',
}
_dataset_config_static5.extern_data = obj.train_dataset.extern_data
_dataset_config_static5.default_input = 'hyps'
_dataset_config_static5.default_target = 'real'
_dataset_config_static5.train_dataset = None
_dataset_config_static5.eval_datasets = None
_dataset_config_static5.use_deep_copy = True
obj.eval_datasets = {
    'dev-clean': _dataset_config_static,
    'dev-other': obj.dev_dataset,
    'test-clean': _dataset_config_static1,
    'test-other': _dataset_config_static2,
    'trainlike-dev-clean': _dataset_config_static3,
    'trainlike-dev-other': _dataset_config_static4,
    'trainlike-lm-devtrain': _dataset_config_static5,
}
import i6_experiments.users.zeyer.datasets.score_results
obj.main_measure_type = object.__new__(i6_experiments.users.zeyer.datasets.score_results.MeasureType)
_dict46 = {
    'short_name': 'WER%',
    'lower_is_better': True,
}
obj.main_measure_type.__dict__.update(_dict46)
obj.main_measure_name = 'dev-other'
import functools
import i6_experiments.users.zeyer.datasets.utils.sclite_generic_score
import i6_experiments.users.zeyer.datasets.librispeech
obj.score_recog_output_func = functools.partial(i6_experiments.users.zeyer.datasets.utils.sclite_generic_score.generic_sclite_score_recog_out, post_proc_funcs=[i6_experiments.users.zeyer.datasets.librispeech._spm_to_words])
obj.recog_post_proc_funcs = [i6_experiments.users.zeyer.datasets.librispeech._spm_to_words]
import types
obj.collect_score_results_func = types.MethodType(i6_experiments.users.zeyer.datasets.task.Task.default_collect_score_results, obj)
