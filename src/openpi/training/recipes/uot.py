"""Unbalanced Optimal Transport (UOT) policy-training recipes.

These keep their explicit UOT objectives and use the UOT training entrypoint;
they are separate from ordinary supervised policy recipes and RLT.
"""

from openpi.training.config import (  # noqa: F401
    AssetsConfig,
    DataConfig,
    FakeDataConfig,
    LeRobotUmiDataConfigPadded_V4,
    LeRobotUmiDataConfigPadded_V4_ImageHorizon1,
    MultiDataConfigFactory,
    TrainConfig,
    UmiDataConfig,
    UOTConfig,
    WBCD_Bimamual_2_views_ImageHorizon1,
    WBCD_Bimamual_4_views_ImageHorizon1,
    _optimizer,
    nnx,
    pi0_config,
    weight_loaders,
)

_CONFIGS = [
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta_uot",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=32,
            # 每帧采样权重：[旧数据, 新数据]；按帧数补偿后，两个数据集期望各占 50%。
            weights=[1.0, 20.0, 50.0],
            datasets=[
                LeRobotUmiDataConfigPadded_V4(
                    repo_id="/data2/hzl_workspace_for_pi/openpi-umi/data/merged_pick_elec_meta_lerobot",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi/openpi-umi/data/merged_pick_elec_meta_lerobot",
                    ),
                    base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
                ),
                LeRobotUmiDataConfigPadded_V4_ImageHorizon1(
                    repo_id="/data2/hzl_workspace_for_pi/openpi-umi/data/collect_demo_zmq_10_13_2",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi/openpi-umi/data/collect_demo_zmq_10_13_2",
                    ),
                    base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
                ),
                LeRobotUmiDataConfigPadded_V4_ImageHorizon1(
                    repo_id="/data2/hzl_workspace_for_pi/openpi-umi/data/collect_demo_zmq_10_14",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi/openpi-umi/data/collect_demo_zmq_10_14",
                    ),
                    base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
                ),
            ]
        ),
        # === [论文 Ours/UOT] 前两个场景作为辅助 source，第三个场景作为 target；BC 总比例为 90%:10%。 ===
        uot=UOTConfig(
            enabled=True,
            pair_info_path="/data2/hzl_workspace_for_pi/openpi-umi-robustness/artifacts/dtw/elec_meta_pair_info.json.gz",
            target_dataset_index=2,
            bc_dataset_probabilities=(0.45, 0.45, 0.10),
            ot_batch_size=128,
            alignment_weight=0.1,
            entropy_reg=5e-4,
            marginal_relaxation=1e-3,
            temporal_window=10,
        ),
        # === [论文 Ours/UOT] 使用服务器上实际存在的 59999 步 π0.5 权重作为联合训练初始化。 ===
        weight_loader=weight_loaders.CheckpointWeightLoader("/data2/hzl_workspace_for_pi/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta/my_experiment_20261014/59999/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=3e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=60_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30000,
    ),
    TrainConfig(
        name="pi05_umi_robustness",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=360,
        ),
        # === [论文 Ours/UOT] 直接展开数据集与采样权重，便于在本训练配置内独立核对和修改。 ===
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 1.0, 3.0, 5.0],
            datasets=[
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0525_wbcd_hitl_shanghai",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0525_wbcd_hitl_shanghai",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_1",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_1",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_2",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_2",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_3",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_3",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0601_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0601_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_afternoon",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_afternoon",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0603_wbcd_hitl_with_wrist_1813_error",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0603_wbcd_hitl_with_wrist_1813_error",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                # target 数据集：UOTConfig.target_dataset_index 必须保持为 9。
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0802_wbcd_hitl_with_wrist_hefei",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0802_wbcd_hitl_with_wrist_hefei",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_afternoon",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_afternoon",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_evening",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_evening",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
            ],
        ),
        uot=UOTConfig(
            enabled=True,
            pair_info_path="/data2/hzl_workspace_for_pi/openpi-umi-robustness/artifacts/dtw/pi05_umi_robustness_pair_info.json.gz",
            target_dataset_index=9,
            # 与上面的 MultiDataConfigFactory.weights 保持一一对应；loader 会自动归一化。
            bc_dataset_probabilities=(
                0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 1.0, 3.0, 5.0
            ),
            # DTW 离线配对使用 candidate-pool=0 全量比较；视觉特征仍按模型接口做掩码均值池化。
            ot_batch_size=32,
            alignment_weight=0.1,
            entropy_reg=5e-4,
            marginal_relaxation=1e-3,
            temporal_window=10,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/data2/hzl_workspace_for_pi/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=60_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=70_000,
        batch_size=72,
        num_workers=16,
        fsdp_devices=8,
        log_interval=10,
        save_interval=1_000,
        keep_period=20_000,
    ),
]
