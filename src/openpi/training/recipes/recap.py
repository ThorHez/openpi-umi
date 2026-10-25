"""ReCap value-model and advantage-conditioned policy recipes."""

from openpi.training.config import (  # noqa: F401
    AssetsConfig,
    CheckpointWeightLoaderWithValueHead,
    DataConfig,
    FakeDataConfig,
    LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP,
    LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP_Inference,
    LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value,
    MultiDataConfigFactory,
    TrainConfig,
    UmiDataConfig,
    WBCD_Bimamual_4_views_ImageHorizon1_ACP,
    WBCD_Bimamual_4_views_ImageHorizon1_ACP_Inference,
    WBCD_Bimamual_4_views_ImageHorizon1_Value,
    _optimizer,
    _pi0_value_config,
    _pi0_value_config_umi_bimanual,
    nnx,
    pi0_config,
    pi0_gripper,
    weight_loaders,
)


_CONFIGS = [
    #
    # Value-only training (train_value.py): Pi0 backbone + value head, fit normalized value targets.
    #
    TrainConfig(
        name="pi0_value_umi",
        model=_pi0_value_config,
        freeze_filter=_pi0_value_config.get_freeze_filter_value_head_only(),
        data=FakeDataConfig(repo_id="fake"),
        c_fail_coef=1.0,
        value_clip_min=-1.0,
        value_clip_max=0.0,
    ),
    #
    # Value training for UMI bimanual with head view + depth (HITL dataset).
    # Freezes the Pi0.5 backbone and only trains the value head (final_norm, value_fc1, value_fc2).
    # Uses episode_index / frame_index from dataset for normalized value target computation.
    #
    # Usage:
    #   python scripts/train_value.py pi0_value_umi_bimanual_headview_depth --exp_name=my_value_exp
    #
    # If ``episode_metadata_path`` is unset, episode/task metadata is read from each local dataset's
    # ``meta/episodes.jsonl`` (and ``meta/tasks.jsonl``) when present — same layout as LeRobot.
    # Optional override:
    #   --episode_metadata_path=/path/to/episode_metadata.json
    #
    # episode_metadata.json format:
    #   {
    #     "episodes": [{"episode_index": 0, "task_index": 0, "length": 500, "success": true}, ...],
    #     "task_max_lengths": {"0": 600}
    #   }
    #
    TrainConfig(
        name="pi0_value_umi_bimanual_headview_depth",
        model=_pi0_value_config_umi_bimanual,
        freeze_filter=_pi0_value_config_umi_bimanual.get_freeze_filter_value_head_only(),
        data=LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
            repo_id="/root/openpi-umi/data/fold_clothes_value_training/horizon_cloth_folding_value_training_20260401_20260417_ep177",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/fold_clothes_value_training/horizon_cloth_folding_value_training_20260401_20260417_ep177",
            ),
            base_config=UmiDataConfig(
                action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                robot_type="ARM=2 G=1 H=0",
            ),
        ),
        weight_loader=CheckpointWeightLoaderWithValueHead(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=1e-4,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=80_000,
        batch_size=72,
        num_workers=16,
        fsdp_devices=8,
        log_interval=10,
        save_interval=1000,
        keep_period=40_000,
        c_fail_coef=1.0,
        value_clip_min=-1.0,
        value_clip_max=0.0,
        episode_metadata_path=None,
        val_ratio=0.1
    ),
    TrainConfig(
        name="pi0_value_umi_bimanual_headview_depth_infer",
        model=_pi0_value_config_umi_bimanual,
        freeze_filter=_pi0_value_config_umi_bimanual.get_freeze_filter_value_head_only(),
        data=LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
            repo_id="/root/openpi-umi/data/horizon_cloth_folding_advantage_eval_failure_20260404_161149_to_20260404_161519_ep4",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/horizon_cloth_folding_advantage_eval_failure_20260404_161149_to_20260404_161519_ep4",
            ),
            base_config=UmiDataConfig(
                action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                robot_type="ARM=2 G=1 H=0",
            ),
        ),
        weight_loader=CheckpointWeightLoaderWithValueHead(
            "/root/openpi-umi/checkpoints/pi0_value_umi_bimanual_headview_depth/my_experiment_v2/79999/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=3_000,
            peak_lr=1e-4,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=60_000,
        batch_size=72,
        num_workers=16,
        fsdp_devices=8,
        log_interval=10,
        save_interval=1000,
        keep_period=30_000,
        c_fail_coef=1.0,
        value_clip_min=-1.0,
        value_clip_max=0.0,
        episode_metadata_path=None,
        val_ratio=0.1
    ),
    TrainConfig(
        name="pi0_value_umi_bimanual_headview_depth_multi_dataset",
        model=_pi0_value_config_umi_bimanual,
        freeze_filter=_pi0_value_config_umi_bimanual.get_freeze_filter_value_head_only(),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            datasets=[
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/horizon_cloth_folding_value_training_20260401_ep116_unified",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/horizon_cloth_folding_value_training_20260401_ep116_unified",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260320",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260320",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260323",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260323",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260316",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260316",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260317",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260317",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                )
            ],
        ),
        weight_loader=CheckpointWeightLoaderWithValueHead(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=70_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=80_000,
        batch_size=72,
        num_workers=12,
        fsdp_devices=8,
        log_interval=10,
        save_interval=2000,
        keep_period=40_000,
        c_fail_coef=1.0,
        value_clip_min=-1.0,
        value_clip_max=0.0,
        episode_metadata_path=None,
        val_ratio=0.1
    ),
    TrainConfig(
        name="pi0_value_umi_bimanual_wbcd_multi_dataset",
        model=_pi0_value_config_umi_bimanual,
        freeze_filter=_pi0_value_config_umi_bimanual.get_freeze_filter_value_head_only(),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            datasets=[
                WBCD_Bimamual_4_views_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_cold_light_20260725_173152_to_20260725_175908_ep46",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_cold_light_20260725_173152_to_20260725_175908_ep46",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_no_light_20260725_162118_to_20260725_162745_ep11",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_no_light_20260725_162118_to_20260725_162745_ep11",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_no_light_20260725_163021_to_20260725_164348_ep19",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_no_light_20260725_163021_to_20260725_164348_ep19",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_no_light_20260725_164427_to_20260725_170243_ep30",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_no_light_20260725_164427_to_20260725_170243_ep30",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_warm_light_20260725_175930_to_20260725_181533_ep34",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_warm_light_20260725_175930_to_20260725_181533_ep34",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_warm_light_20260725_181800_to_20260725_182257_ep7",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_warm_light_20260725_181800_to_20260725_182257_ep7",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1_Value(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_white_light_20260725_170306_to_20260725_172903_ep45",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_white_light_20260725_170306_to_20260725_172903_ep45",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                )
            ],
        ),
        weight_loader=CheckpointWeightLoaderWithValueHead(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=60_000,
        batch_size=72,
        num_workers=12,
        fsdp_devices=8,
        log_interval=10,
        save_interval=2000,
        keep_period=30_000,
        c_fail_coef=1.0,
        value_clip_min=-1.0,
        value_clip_max=0.0,
        episode_metadata_path=None,
        val_ratio=0.1
    ),
    #
    # ACP (Advantage-Conditioned Policy) training: Pi0.5 hybrid + Advantage tag in prompt.
    # Requires dataset with ``is_positive`` column (produced by ``scripts/lerobot_value_infer.py``).
    #
    # Usage:
    #   python scripts/train.py pi05_acp_umi_bimanual_headview_depth --exp_name=my_acp_exp
    #
    TrainConfig(
        name="pi05_acp_umi_bimanual_headview_depth",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_hitl_test",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_hitl_test",
            ),
            base_config=UmiDataConfig(
                action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                robot_type="ARM=2 G=1 H=0",
            ),
            acp_dropout=0.1,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=8e-5,
            decay_steps=40_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=50_000,
        batch_size=72,
        num_workers=12,
        fsdp_devices=2,
        log_interval=10,
        save_interval=2000,
        keep_period=10_000,
    ),
    TrainConfig(
        name="pi05_acp_umi_bimanual_headview_depth_multi_dataset",
        model=pi0_gripper.Pi0GripperConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
            gripper_binary_indices=(9, 19),
            gripper_binary_threshold=0.03,
            gripper_binary_loss_weight=0.5,
            gripper_binary_close_value=0.0,
            gripper_binary_open_value=0.085,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            # weights=[5.0, 1.0, 1.0, 0.5, 5.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_messy_demostration/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_messy_demostration/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260417_20260418_ep171",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260417_20260418_ep171",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260422_ep100",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260422_ep100",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_ep102",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_ep102",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/fold_clothes_value_training/horizon_cloth_folding_value_training_20260401_20260417_ep177",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_value_training/horizon_cloth_folding_value_training_20260401_20260417_ep177",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
            ]
        ),
        # 从标准 pi05 checkpoint 初始化时，新增的 gripper_binary_head 会保留随机初始化；
        # 后续训练保存出的 checkpoint 会自动包含这个 head。
        weight_loader=weight_loaders.CheckpointWeightLoaderWithGripperHead(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=80_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=40000,
    ),
    TrainConfig(
        name="pi05_acp_umi_bimanual_wbcd_multi_dataset",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            # Must match LeRobot actions shape [T, D]; datasets were converted with horizon=32.
            action_horizon=32,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            # weights=[5.0, 1.0, 1.0, 0.5, 5.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                WBCD_Bimamual_4_views_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_cold_light_20260725_173152_to_20260725_175908_ep46",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_cold_light_20260725_173152_to_20260725_175908_ep46",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                WBCD_Bimamual_4_views_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_no_light_20260725_162118_to_20260725_162745_ep11",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_no_light_20260725_162118_to_20260725_162745_ep11",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                WBCD_Bimamual_4_views_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_no_light_20260725_163021_to_20260725_164348_ep19",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_no_light_20260725_163021_to_20260725_164348_ep19",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                WBCD_Bimamual_4_views_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_no_light_20260725_164427_to_20260725_170243_ep30",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_no_light_20260725_164427_to_20260725_170243_ep30",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                WBCD_Bimamual_4_views_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_warm_light_20260725_175930_to_20260725_181533_ep34",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_warm_light_20260725_175930_to_20260725_181533_ep34",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                WBCD_Bimamual_4_views_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_warm_light_20260725_181800_to_20260725_182257_ep7",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_warm_light_20260725_181800_to_20260725_182257_ep7",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
                WBCD_Bimamual_4_views_ImageHorizon1_ACP(
                    repo_id="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_white_light_20260725_170306_to_20260725_172903_ep45",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/one_cloth/wbcd_one_cloth_white_light_20260725_170306_to_20260725_172903_ep45",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    acp_dropout=0.3,
                ),
            ]
        ),
        # 从标准 pi05 checkpoint 初始化时，新增的 gripper_binary_head 会保留随机初始化；
        # 后续训练保存出的 checkpoint 会自动包含这个 head。
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/openpi-umi/checkpoints/pi05_umi_wbcd_v3_260522_h32/260608/59999/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=15000,
    ),
    TrainConfig(
        name="pi05_acp_umi_bimanual_headview_depth_bimanual_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_ACP_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_bimanual_v1",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_bimanual_v1",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=())
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_bimanual_v1/params"),
    ),
    TrainConfig(
        name="pi05_acp_umi_bimanual_wbcd_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=WBCD_Bimamual_4_views_ImageHorizon1_ACP_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_bimanual_v1",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_bimanual_v1",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=())
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_bimanual_v1/params"),
    ),
]
