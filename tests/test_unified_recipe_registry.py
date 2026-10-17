"""Import real recipe dataclasses without opening data or constructing a model."""

import inspect

from openpi.models import pi0_mem
from openpi.training import config
from openpi.training.mem.recipes import shellgame_semantic_memory_pretrain


def test_production_sft_and_uot_have_distinct_recipe_identities():
    sft = config.get_config("pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta")
    uot = config.get_config(sft.name + "_uot")
    assert sft.uot is None
    assert sft.data.weights == [1.0, 20.0, 50.0]
    assert uot.uot.enabled and uot.uot.target_dataset_index == 2
    assert len(uot.data.datasets) == len(sft.data.datasets) == 3
    assert uot.uot.bc_dataset_probabilities == (0.45, 0.45, 0.1)
    assert config.get_config("pi05_umi_robustness").uot.enabled


def test_legacy_pf_and_both_memory_stages_remain_addressable():
    assert config.get_config("pi0_mem_compress_evan_shellgame_openpi_umi_success_260703").model.num_frames == 16
    action = config.get_config("pi0_mem_semantic_action_shellgame_eef7")
    assert action.uot is None
    assert shellgame_semantic_memory_pretrain.make_train_config().name == "shellgame_semantic_memory_pretrain"
    assert not hasattr(pi0_mem.Pi0Mem, "encode_visual_features")
    assert "rtc_actions" not in inspect.signature(pi0_mem.Pi0Mem.sample_actions).parameters
