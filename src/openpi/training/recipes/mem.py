"""Memory-augmented policy recipes and their existing task-specific factories."""

from openpi.training.config import (  # noqa: F401
    AssetsConfig,
    FakeDataConfig,
    LeRobotUmiDataConfig_Bimamual_HeadView_Depth_Horizon1_Pi0Mem,
    LeRobotUmiDataConfig_Bimamual_Horizon1_Pi0Mem,
    LeRobotUmiDataConfig_Bimamual_WBCD_4Views_Horizon1_Pi0Mem,
    LeRobotUmiDataConfig_shellgame_Pi0Mem,
    LeRobotUmiDataConfig_shellgame_Pi0Mem_Inference,
    LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint,
    MultiDataConfigFactory,
    ShellgameCupEvalConfig,
    ShellgameMemoryClassifierConfig,
    TrainConfig,
    UmiDataConfig,
    _optimizer,
    _pi0_mem_pf_safe_shellgame_joint_model,
    _shellgame_history_classifier_probe_model,
    _shellgame_history_post_transformer_probe_model,
    _shellgame_joint_fixed_grid_query_action_model,
    _shellgame_real_memory_adapt_train_config,
    _shellgame_real_wrist_stage2_train_config,
    _shellgame_semantic_action_train_config,
    nnx,
    pi0_config,
    pi0_mem,
    pi0_mem_compress,
    pi0_mem_fixed_grid_query_action,
    pi0_mem_pf,
    weight_loaders,
)


_CONFIGS = [
    # =====================================================================
    # Pi0Mem wrist-only twin of the 260322 fold-clothes hybrid config.
    # Same 4 datasets / weights / schedule as the HeadView+Depth Pi0Mem entry
    # above, but every child uses the bimanual wrist-only factory
    # (LeRobotUmiDataConfig_Bimamual_Horizon1_Pi0Mem) — i.e. NO base_0_rgb
    # and NO base_0_depth streams. Only left_wrist_0_rgb + right_wrist_0_rgb
    # are loaded as T-frame video tensors. Pi0Mem.embed_prefix iterates over
    # whatever keys are in obs.images, so dropping the base streams is a
    # data-side concern only — no model code changes required.
    # Launch:
    #   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/mem/train_pi0_mem.py \
    #     pi0_mem_umi_32d_60k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322 \
    #     --exp-name=my_experiment [--overwrite | --resume]
    # =====================================================================
    TrainConfig(
        name="pi0_mem_umi_32d_60k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322",
        model=pi0_mem.Pi0MemConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
            num_frames=1,
            temporal_every=4,
            # siglip_remat_policy="dots_with_no_batch_dims_saveable",
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_Bimamual_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    num_frames=1,
                )
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
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
        num_train_steps=60_000,
        batch_size=72,
        num_workers=24,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30_000,
    ),
    # =====================================================================
    # Pi0MemCompress wrist-only twin of the Pi0Mem entry directly above.
    # Same dataset / weights / schedule / batch / LR, but the visual
    # backbone is openpi.models.siglip_mem_compress (compressed-history MEM)
    # instead of openpi.models.siglip_mem (per-block temporal attention).
    # Reuses the same Pi0Mem-aware DataConfig (LeRobotUmiDataConfig_Bimamual_
    # Horizon1_Pi0Mem) — the data pipeline is identical, only the model
    # changes. Launch with the compress training script:
    #   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/mem/train_pi0_mem_compress.py \
    #     pi0_mem_compress_umi_32d_60k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322 \
    #     --exp-name=my_experiment [--overwrite | --resume]
    # =====================================================================
    TrainConfig(
        name="pi0_mem_compress_umi_32d_60k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322",
        model=pi0_mem_compress.Pi0MemCompressConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
            num_frames=16,
            memory_every=4,
            history_memory_tokens=256,
            history_resampler_depth=1,
            history_use_current_condition=True,
            # history_gate_init=-4.6,
            #history_gate_init=0.0,
            # Fixed-gate experiment: uncomment one of these to force the memory
            # branch open instead of learning sigmoid(history_memory_gate_logit).
            # history_gate_fixed=0.5,
            history_gate_fixed=1.0,
            history_gate_lr_multiplier=20.0,
            # siglip_remat_policy="dots_with_no_batch_dims_saveable",
            diversity_weight=0.01,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_Bimamual_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    num_frames=16,
                    frame_stride=4
                )
            ],
        ),
        # Pi0MemCompress adds HistoryResampler_0/... and per-block
        # history_memory_gate_logit which are NOT present in pi05_base. Use
        # the memory-aware loader so those new params fall back to the
        # model's random init instead of failing pytree-structure validation.
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=25_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=72,
        num_workers=24,
        fsdp_devices=8,
        log_interval=10,
        keep_period=5_000,
    ),
    # =====================================================================
    # Pi0MemPF (Past-Future Temporal Bottleneck) twin of the fold-clothes
    # Pi0MemCompress config directly above.
    #
    # Same dataset / schedule / batch, but:
    #   - clips carry num_future_frames=8 future frames after the current one
    #     (layout [oldest_past ... current, future...]; data factory must use
    #     matching num_future_frames / future_frame_stride),
    #   - the visual backbone is openpi.models.siglip_pf (shared UTR compresses
    #     past -> Hmem and future -> Zpost; dual gated GTCA injection),
    #   - a Future Latent Prior Encoder predicts Zprior from current + Hmem +
    #     language + state, trained with prior/posterior dual-branch action
    #     losses plus latent alignment/regularization,
    #   - inference (sample_actions) keeps only the prior branch and needs NO
    #     future frames.
    # Launch:
    #   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/mem/train_pi0_mem_pf.py \
    #     pi0_mem_pf_umi_32d_30k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322 \
    #     --exp-name=my_experiment [--overwrite | --resume]
    # =====================================================================
    TrainConfig(
        name="pi0_mem_pf_umi_32d_30k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322",
        model=pi0_mem_pf.Pi0MemPFConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
            # Past side (identical to the compress twin).
            num_frames=32,
            frame_stride=5,

            memory_every=1,
            history_memory_tokens=256,
            history_resampler_depth=1,
            history_use_current_condition=True,
            history_gate_fixed=1.0,
            history_gate_lr_multiplier=20.0,
            diversity_weight=0.01,
            # Future side.
            num_future_frames=10,
            future_frame_stride=5,
            future_latent_tokens=64,
            future_gate_fixed=1.0,
            prior_encoder_depth=2,

            lambda_prior=1.0,
            lambda_post=1.0,
            lambda_align=1.0,
            lambda_reg=1e-4,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_Bimamual_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                )
            ],
        ),
        # PF-aware loader: falls back to init for UTR / Future* / FuturePrior /
        # gate params, and transparently remaps HistoryResampler_0 -> UTR_0
        # when pointed at a trained Pi0MemCompress checkpoint instead of
        # pi05_base (recommended: reuse your trained history compressor).
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryPF(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=25_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=72,
        num_workers=24,
        fsdp_devices=8,
        log_interval=10,
        keep_period=5_000,
    ),
    # =====================================================================
    # History-required variant of the fixed-gate Pi0MemCompress run above.
    #
    # Goal: make the policy stay accurate when current-frame evidence is
    # incomplete, so the clean history path has a direct reason to become useful.
    #
    # Added training-only mechanisms in scripts/mem/train_pi0_mem_compress.py:
    #   - current_frame_dropout_prob: sometimes blank the current frame while
    #     keeping history intact.
    #   - current_frame_mask_prob: softly mask current-frame pixels.
    #   - current_frame_corrupt_loss_weight: add a normalized second action
    #     loss on the corrupted-current / clean-history view.
    #
    # Launch:
    #   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/mem/train_pi0_mem_compress.py \
    #     pi0_mem_compress_umi_32d_30k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322_history_required \
    #     --exp-name=my_experiment --overwrite
    # =====================================================================
    TrainConfig(
        name="pi0_mem_compress_umi_32d_30k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322_history_required",
        model=pi0_mem_compress.Pi0MemCompressConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
            num_frames=16,
            memory_every=4,
            history_memory_tokens=256,
            history_resampler_depth=1,
            history_use_current_condition=True,
            history_gate_fixed=1.0,
            history_gate_lr_multiplier=20.0,
            diversity_weight=0.01,
            current_frame_dropout_prob=0.25,
            current_frame_mask_prob=0.15,
            current_frame_corrupt_loss_weight=1.0,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_Bimamual_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    num_frames=16,
                    frame_stride=4,
                )
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=25_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=72,
        num_workers=24,
        fsdp_devices=8,
        log_interval=10,
        keep_period=5_000,
    ),
    # =====================================================================
    # Light MEM pressure variant for the fixed-gate Pi0MemCompress run.
    #
    # Use this when time is limited and the goal is to keep clean END100
    # performance close to the original fixed-gate memory model while still
    # giving the history branch a mild training signal. Compared with
    # history_required above, this keeps clean action loss dominant:
    #
    #   normalized_action_loss = (clean + 0.25 * corrupt_current) / 1.25
    #
    # Launch:
    #   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/mem/train_pi0_mem_compress.py \
    #     pi0_mem_compress_umi_32d_30k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322_history_light \
    #     --exp-name=my_experiment --overwrite
    # =====================================================================
    TrainConfig(
        name="pi0_mem_compress_umi_32d_30k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322_history_light",
        model=pi0_mem_compress.Pi0MemCompressConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
            num_frames=16,
            memory_every=4,
            history_memory_tokens=256,
            history_resampler_depth=1,
            history_use_current_condition=True,
            history_gate_fixed=1.0,
            history_gate_lr_multiplier=20.0,
            diversity_weight=0.01,
            current_frame_dropout_prob=0.10,
            current_frame_mask_prob=0.05,
            current_frame_corrupt_loss_weight=0.25,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_Bimamual_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    num_frames=16,
                    frame_stride=4,
                )
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=25_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=72,
        num_workers=24,
        fsdp_devices=8,
        log_interval=10,
        keep_period=5_000,
    ),
    TrainConfig(
        name="pi0_mem_compress_umi_wbcd_history_light_v2_260623",
        model=pi0_mem_compress.Pi0MemCompressConfig(
            # pi05=True,
            # action_dim=32,
            # action_horizon=32,
            # action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            # max_token_len=360,
            # num_frames=8,
            # memory_every=4,
            # history_memory_tokens=256,
            # history_resampler_depth=1,
            # history_use_current_condition=True,
            # history_gate_fixed=1.0,
            # history_gate_lr_multiplier=20.0,
            # diversity_weight=0.01,
            # current_frame_dropout_prob=0.05,
            # current_frame_mask_prob=0.05,
            # current_frame_corrupt_loss_weight=0.25,

            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,

            num_frames=8,
            memory_every=4,
            history_memory_tokens=256,
            history_resampler_depth=1,
            history_use_current_condition=True,

            history_gate_fixed=1.0,
            history_gate_lr_multiplier=1.0,

            diversity_weight=0.01,

            # current_frame_index=-1,
            # current_frame_dropout_prob=0.05,
            # current_frame_mask_prob=0.0,
            # current_frame_corrupt_sample_prob=0.3,


            current_frame_index=-1,
            # current_frame_dropout_prob=0.1,
            # current_frame_mask_prob=0.0,
            # current_frame_corrupt_sample_prob=0.5,
            # current_frame_corrupt_loss_weight=0.0,


            current_frame_corrupt_sample_prob = 1.0,
            current_frame_dropout_prob = 0.3,
            current_frame_mask_prob = 0.0,
            current_frame_corrupt_loss_weight = 0.0
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            weights=[1.0, 1.0],
            datasets=[
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_Horizon1_Pi0Mem(
                    repo_id="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    num_frames=8,
                    frame_stride=4,
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_Horizon1_Pi0Mem(
                    repo_id="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    num_frames=8,
                    frame_stride=4,
                )
            ],
        ),
        # weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
        #     "gs://openpi-assets/checkpoints/pi05_base/params"
        # ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_mem_compress_umi_wbcd_history_light_v2_260623/260623/59999/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=72,
        num_workers=32,
        fsdp_devices=8,
        log_interval=10,
        keep_period=15_000,
    ),

    TrainConfig(
        name="pi0_mem_compress_evan_shellgame_openpi_260727",
        model=pi0_mem_compress.Pi0MemCompressConfig(
            # pi05=True,
            # action_dim=32,
            # action_horizon=32,
            # action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            # max_token_len=360,
            # num_frames=8,
            # memory_every=4,
            # history_memory_tokens=256,
            # history_resampler_depth=1,
            # history_use_current_condition=True,
            # history_gate_fixed=1.0,
            # history_gate_lr_multiplier=20.0,
            # diversity_weight=0.01,
            # current_frame_dropout_prob=0.05,
            # current_frame_mask_prob=0.05,
            # current_frame_corrupt_loss_weight=0.25,

            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            max_token_len=256,

            num_frames=32,
            memory_every=1,
            history_memory_tokens=256,
            history_resampler_depth=1,
            history_use_current_condition=True,

            history_gate_fixed=1.0,
            history_gate_lr_multiplier=1.0,

            diversity_weight=0.01,

            # current_frame_index=-1,
            # current_frame_dropout_prob=0.05,
            # current_frame_mask_prob=0.0,
            # current_frame_corrupt_sample_prob=0.3,


            current_frame_index=-1,
            # current_frame_dropout_prob=0.1,
            # current_frame_mask_prob=0.0,
            # current_frame_corrupt_sample_prob=0.5,
            # current_frame_corrupt_loss_weight=0.0,


            current_frame_corrupt_sample_prob = 1.0,
            current_frame_dropout_prob = 0.3,
            current_frame_mask_prob = 0.0,
            current_frame_corrupt_loss_weight = 0.0
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Pi0Mem(
                    repo_id="/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_current_frame_action",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_current_frame_action",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=32,
                    frame_stride=5,
                ),
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # weight_loader=weight_loaders.CheckpointWeightLoader(
        #     "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_mem_compress_umi_wbcd_history_light_v2_260623/260623/59999/params"
        # ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=72,
        num_workers=32,
        fsdp_devices=6,
        log_interval=10,
        keep_period=15_000,
    ),

    TrainConfig(
        name="pi0_mem_compress_evan_shellgame_openpi_joint_260727",
        model=pi0_mem_compress.Pi0MemCompressConfig(
            # pi05=True,
            # action_dim=32,
            # action_horizon=32,
            # action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            # max_token_len=360,
            # num_frames=8,
            # memory_every=4,
            # history_memory_tokens=256,
            # history_resampler_depth=1,
            # history_use_current_condition=True,
            # history_gate_fixed=1.0,
            # history_gate_lr_multiplier=20.0,
            # diversity_weight=0.01,
            # current_frame_dropout_prob=0.05,
            # current_frame_mask_prob=0.05,
            # current_frame_corrupt_loss_weight=0.25,

            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
            max_token_len=256,

            num_frames=30,
            memory_every=1,
            history_memory_tokens=256,
            history_resampler_depth=1,
            history_use_current_condition=True,

            history_gate_fixed=1.0,
            history_gate_lr_multiplier=1.0,

            diversity_weight=0.01,

            # current_frame_index=-1,
            # current_frame_dropout_prob=0.05,
            # current_frame_mask_prob=0.0,
            # current_frame_corrupt_sample_prob=0.3,


            current_frame_index=-1,
            # current_frame_dropout_prob=0.1,
            # current_frame_mask_prob=0.0,
            # current_frame_corrupt_sample_prob=0.5,
            # current_frame_corrupt_loss_weight=0.0,


            current_frame_corrupt_sample_prob = 1.0,
            current_frame_dropout_prob = 0.3,
            current_frame_mask_prob = 0.0,
            current_frame_corrupt_loss_weight = 0.0
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id="/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_joint",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_absolute_joint",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=30,
                    frame_stride=2,
                ),
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # weight_loader=weight_loaders.CheckpointWeightLoader(
        #     "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_mem_compress_umi_wbcd_history_light_v2_260623/260623/59999/params"
        # ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=54,
        num_workers=16,
        fsdp_devices=2,
        log_interval=10,
        keep_period=15_000,
        val_ratio=0.1,
        eval_interval=1_000,
        eval_batches=10,
        shellgame_cup_eval=ShellgameCupEvalConfig(
            enabled=True,
            raw_dataset_root=(
                "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                "shellgame_absolute_joint_dataset"
            ),
            robosuite_root="/data2/hzl_workspace_for_pi_mem/robosuite",
            interval=1_000,
            num_episodes=24,
            batch_size=6,
            num_sampling_steps=4,
            sample_seed=260806,
            selection_radius=0.06,
        ),
    ),

    _shellgame_semantic_action_train_config,
    _shellgame_real_wrist_stage2_train_config,
    _shellgame_real_memory_adapt_train_config,

    TrainConfig(
        name="pi0_mem_fixed_grid_query_action_shellgame_joint_260810",
        model=_shellgame_joint_fixed_grid_query_action_model,
        freeze_filter=(
            _shellgame_joint_fixed_grid_query_action_model.get_freeze_filter_action_memory_finetune()
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=(
                        "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                        "shellgame_lerobot_absolute_joint"
                    ),
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir=(
                            "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                            "shellgame_lerobot_absolute_joint"
                        ),
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=60,
                    frame_stride=1,
                ),
            ],
        ),
        # Warm-start the action expert and the proven action-memory reader from
        # the successful controlled experiment.  Fixed-grid history parameters
        # and width-dependent query input projections remain newly initialized.
        weight_loader=pi0_mem_fixed_grid_query_action.QueryActionCheckpointLoader(
            "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
            "pi0_shellgame_three_swap_query_crossattn_pi_joint_action_260810/"
            "query_crossattn_pi_flow_action300_b12_260810/299/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=3e-5,
            decay_steps=20_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=20_000,
        batch_size=12,
        num_workers=12,
        fsdp_devices=6,
        log_interval=10,
        save_interval=1_000,
        keep_period=5_000,
        val_ratio=0.1,
        eval_interval=500,
        eval_batches=20,
        wandb_enabled=False,
        shellgame_cup_eval=ShellgameCupEvalConfig(
            enabled=True,
            raw_dataset_root=(
                "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                "shellgame_absolute_joint_dataset"
            ),
            robosuite_root="/data2/hzl_workspace_for_pi_mem/robosuite",
            interval=500,
            num_episodes=60,
            batch_size=6,
            num_sampling_steps=4,
            sample_seed=260810,
            selection_radius=0.06,
        ),
    ),

    # Decision-focused continuation of the formal query-action run.  The
    # dedicated entry point oversamples frames 59..70 to 80% while retaining
    # all later grasp rows at 20%; validation remains uniformly post-swap.
    TrainConfig(
        name="pi0_mem_fixed_grid_query_action_decision_weighted_shellgame_joint_260810",
        model=_shellgame_joint_fixed_grid_query_action_model,
        freeze_filter=(
            _shellgame_joint_fixed_grid_query_action_model.get_freeze_filter_action_memory_finetune()
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=(
                        "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                        "shellgame_lerobot_absolute_joint"
                    ),
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir=(
                            "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                            "shellgame_lerobot_absolute_joint"
                        ),
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=60,
                    frame_stride=1,
                ),
            ],
        ),
        weight_loader=pi0_mem_fixed_grid_query_action.QueryActionCheckpointLoader(
            "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
            "pi0_mem_fixed_grid_query_action_shellgame_joint_260810/"
            "formal_postswap_60f_s1_query16_20k_6gpu_260810_v2/2000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-5,
            decay_steps=5_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=5_000,
        batch_size=12,
        num_workers=12,
        fsdp_devices=6,
        log_interval=10,
        save_interval=500,
        keep_period=1_000,
        val_ratio=0.1,
        eval_interval=250,
        eval_batches=20,
        wandb_enabled=False,
        shellgame_cup_eval=ShellgameCupEvalConfig(
            enabled=True,
            raw_dataset_root=(
                "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                "shellgame_absolute_joint_dataset"
            ),
            robosuite_root="/data2/hzl_workspace_for_pi_mem/robosuite",
            interval=250,
            num_episodes=60,
            batch_size=6,
            num_sampling_steps=4,
            sample_seed=260810,
            selection_radius=0.06,
        ),
    ),

    # Controlled causal test: use only target-decision rows and freeze the
    # complete Pi0 action expert.  Any action-loss improvement must pass
    # through the fixed-grid history encoder and direct memory interface.
    TrainConfig(
        name="pi0_mem_fixed_grid_query_action_memory_only_shellgame_joint_260810",
        model=_shellgame_joint_fixed_grid_query_action_model,
        freeze_filter=(
            _shellgame_joint_fixed_grid_query_action_model.get_freeze_filter_memory_path_only()
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=(
                        "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                        "shellgame_lerobot_absolute_joint"
                    ),
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir=(
                            "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                            "shellgame_lerobot_absolute_joint"
                        ),
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=60,
                    frame_stride=1,
                ),
            ],
        ),
        weight_loader=pi0_mem_fixed_grid_query_action.QueryActionCheckpointLoader(
            "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
            "pi0_mem_fixed_grid_query_action_decision_weighted_shellgame_joint_260810/"
            "decision80_postswap_from2000_5k_6gpu_260810/1000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=3e-5,
            decay_steps=2_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=2_000,
        batch_size=12,
        num_workers=12,
        fsdp_devices=6,
        log_interval=10,
        save_interval=500,
        keep_period=1_000,
        val_ratio=0.1,
        eval_interval=250,
        eval_batches=20,
        wandb_enabled=False,
        shellgame_cup_eval=ShellgameCupEvalConfig(
            enabled=True,
            raw_dataset_root=(
                "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                "shellgame_absolute_joint_dataset"
            ),
            robosuite_root="/data2/hzl_workspace_for_pi_mem/robosuite",
            interval=250,
            num_episodes=60,
            batch_size=6,
            num_sampling_steps=4,
            sample_seed=260810,
            selection_radius=0.06,
        ),
    ),

    # Diagnostic only: can the compressed memory learn the final ball slot
    # when directly supervised? This is classification-only and does not
    # produce a policy checkpoint intended for deployment.
    TrainConfig(
        name="pi0_mem_compress_shellgame_joint_history_classifier_probe_260807",
        model=_shellgame_history_classifier_probe_model,
        freeze_filter=_shellgame_history_classifier_probe_model.get_freeze_filter_history_classifier_probe(),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=(
                        "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                        "shellgame_lerobot_absolute_joint"
                    ),
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir=(
                            "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                            "shellgame_lerobot_absolute_joint"
                        ),
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=61,
                    frame_stride=1,
                ),
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
            "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
            "pi0_mem_compress_evan_shellgame_openpi_joint_260727/"
            "my_experiment_30f_s2_6gpu/23000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=1e-5,
            decay_steps=2_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=2_000,
        batch_size=72,
        num_workers=16,
        fsdp_devices=6,
        log_interval=10,
        save_interval=500,
        keep_period=500,
        val_ratio=0.1,
        eval_interval=200,
        # Five batches cover 360 held-out samples and keep the 61-frame
        # diagnostic evaluation short enough to monitor during training.
        eval_batches=5,
        shellgame_memory_classifier=ShellgameMemoryClassifierConfig(
            enabled=True,
            episodes_metadata_path=(
                "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                "shellgame_lerobot_absolute_joint/meta/episodes.jsonl"
            ),
            label_key="final_ball_cup",
            classes=("left", "middle", "right"),
            min_frame_index=60,
            max_frame_index=60,
            loss_weight=1.0,
            action_loss_weight=0.0,
        ),
    ),

    # Memorization sanity check for the diagnostic history classifier. It uses
    # the exact same 30 examples per class for training and validation; success
    # therefore means optimization/capacity works, not that the model generalizes.
    TrainConfig(
        name="pi0_mem_compress_shellgame_joint_history_classifier_overfit_260807",
        model=_shellgame_history_classifier_probe_model,
        freeze_filter=_shellgame_history_classifier_probe_model.get_freeze_filter_history_classifier_probe(),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=(
                        "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                        "shellgame_lerobot_absolute_joint"
                    ),
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir=(
                            "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                            "shellgame_lerobot_absolute_joint"
                        ),
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=61,
                    frame_stride=1,
                ),
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
            "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
            "pi0_mem_compress_evan_shellgame_openpi_joint_260727/"
            "my_experiment_30f_s2_6gpu/23000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=20,
            peak_lr=1e-5,
            decay_steps=500,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=10.0),
        ema_decay=None,
        num_train_steps=500,
        batch_size=30,
        num_workers=8,
        fsdp_devices=6,
        log_interval=10,
        save_interval=250,
        keep_period=250,
        # Kept positive to enable the validation path; overfit mode replaces
        # the ordinary split with the exact same balanced 90-sample subset.
        val_ratio=0.1,
        eval_interval=50,
        eval_batches=3,
        shellgame_memory_classifier=ShellgameMemoryClassifierConfig(
            enabled=True,
            episodes_metadata_path=(
                "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                "shellgame_lerobot_absolute_joint/meta/episodes.jsonl"
            ),
            label_key="final_ball_cup",
            classes=("left", "middle", "right"),
            min_frame_index=60,
            max_frame_index=60,
            loss_weight=1.0,
            action_loss_weight=0.0,
            overfit_samples_per_class=30,
            overfit_same_samples_for_validation=True,
            disable_train_augmentation=True,
        ),
    ),

    # Isolated post-compression-Transformer capacity test.  It intentionally
    # uses the same balanced 90 episodes as the preceding overfit probe, but
    # does not change the original Pi0MemCompress implementation or config.
    TrainConfig(
        name="pi0_mem_post_transformer_shellgame_joint_classifier_overfit_260807",
        model=_shellgame_history_post_transformer_probe_model,
        freeze_filter=(
            _shellgame_history_post_transformer_probe_model.get_freeze_filter_history_classifier_probe()
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=(
                        "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                        "shellgame_lerobot_absolute_joint"
                    ),
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir=(
                            "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                            "shellgame_lerobot_absolute_joint"
                        ),
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=61,
                    frame_stride=1,
                ),
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
            "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/"
            "pi0_mem_compress_evan_shellgame_openpi_joint_260727/"
            "my_experiment_30f_s2_6gpu/23000/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=20,
            peak_lr=1e-5,
            decay_steps=500,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=10.0),
        ema_decay=None,
        num_train_steps=500,
        # Joint sequence length is 512, so use one sample per each of six GPUs.
        batch_size=6,
        num_workers=8,
        fsdp_devices=6,
        log_interval=10,
        save_interval=250,
        keep_period=250,
        val_ratio=0.1,
        eval_interval=50,
        eval_batches=15,
        shellgame_memory_classifier=ShellgameMemoryClassifierConfig(
            enabled=True,
            episodes_metadata_path=(
                "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                "shellgame_lerobot_absolute_joint/meta/episodes.jsonl"
            ),
            label_key="final_ball_cup",
            classes=("left", "middle", "right"),
            min_frame_index=60,
            max_frame_index=60,
            loss_weight=1.0,
            action_loss_weight=0.0,
            overfit_samples_per_class=30,
            overfit_same_samples_for_validation=True,
            disable_train_augmentation=True,
        ),
    ),

    TrainConfig(
        name="pi0_mem_compress_evan_shellgame_openpi_260727_infer",
        model=pi0_mem_compress.Pi0MemCompressConfig(
            # pi05=True,
            # action_dim=32,
            # action_horizon=32,
            # action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            # max_token_len=360,
            # num_frames=8,
            # memory_every=4,
            # history_memory_tokens=256,
            # history_resampler_depth=1,
            # history_use_current_condition=True,
            # history_gate_fixed=1.0,
            # history_gate_lr_multiplier=20.0,
            # diversity_weight=0.01,
            # current_frame_dropout_prob=0.05,
            # current_frame_mask_prob=0.05,
            # current_frame_corrupt_loss_weight=0.25,

            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            max_token_len=256,

            num_frames=32,
            memory_every=1,
            history_memory_tokens=256,
            history_resampler_depth=1,
            history_use_current_condition=True,

            history_gate_fixed=1.0,
            history_gate_lr_multiplier=1.0,

            diversity_weight=0.01,

            # current_frame_index=-1,
            # current_frame_dropout_prob=0.05,
            # current_frame_mask_prob=0.0,
            # current_frame_corrupt_sample_prob=0.3,


            current_frame_index=-1,
            # current_frame_dropout_prob=0.1,
            # current_frame_mask_prob=0.0,
            # current_frame_corrupt_sample_prob=0.5,
            # current_frame_corrupt_loss_weight=0.0,


            current_frame_corrupt_sample_prob = 0.0,
            current_frame_dropout_prob = 0.0,
            current_frame_mask_prob = 0.0,
            current_frame_corrupt_loss_weight = 0.0
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Pi0Mem_Inference(
                    repo_id="/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_mem_compress_evan_shellgame_openpi_260727/my_experiment/8000_cp",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_mem_compress_evan_shellgame_openpi_260727/my_experiment/8000_cp",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=32,
                    frame_stride=5,
                ),
            ],
        ),
    ),

    # PF-safe v1: preserve the trained Pi0.5/history policy and adapt only the
    # isolated temporal bottleneck. Run with train_pi0_mem_pf_safe.py; the
    # original PF config and trainer above/below remain unchanged.
    TrainConfig(
        name="pi0_mem_pf_safe_evan_shellgame_joint_260806",
        model=_pi0_mem_pf_safe_shellgame_joint_model,
        freeze_filter=_pi0_mem_pf_safe_shellgame_joint_model.get_freeze_filter_temporal_only(),
        data=MultiDataConfigFactory(
            # FuturePrior/Ps is initialized for model.action_dim inputs. The
            # real joint state occupies the first 8 dimensions; pad it to 32,
            # not the legacy multi-dataset width of 96.
            state_pad_dim=32,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Pi0Mem_Joint(
                    repo_id=(
                        "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                        "shellgame_lerobot_absolute_joint"
                    ),
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir=(
                            "/data2/hzl_workspace_for_pi_mem/robosuite/outputs/"
                            "shellgame_lerobot_absolute_joint"
                        ),
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 8 + (0.0,) * 24,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=30,
                    frame_stride=2,
                    num_future_frames=10,
                    future_frame_stride=2,
                ),
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryPF(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=16,
        num_workers=16,
        fsdp_devices=1,
        log_interval=10,
        keep_period=15_000,
    ),

    TrainConfig(
        name="pi0_mem_pf_evan_shellgame_openpi_umi_success_pf_260709",
        model=pi0_mem_pf.Pi0MemPFConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
            max_token_len=512,

            num_frames=12,
            frame_stride=5,
            memory_every=1,
            history_memory_tokens=256,
            history_resampler_depth=1,
            history_use_current_condition=True,

            history_gate_fixed=1.0,
            history_gate_lr_multiplier=1.0,

            diversity_weight=0.01,

            current_frame_index=-1,
            # current_frame_corrupt_sample_prob=1.0,
            # current_frame_dropout_prob=0.3,
            current_frame_corrupt_sample_prob=0.0,
            current_frame_dropout_prob=0.0,
            current_frame_mask_prob=0.0,
            current_frame_corrupt_loss_weight=0.0,

            num_future_frames=12,
            future_frame_stride=5,
            future_latent_tokens=256,
            future_gate_fixed=1.0,
            prior_encoder_depth=2,
            lambda_prior=1.0,
            lambda_post=1.0,
            lambda_align=1.0,
            lambda_reg=1e-4,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=32,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Pi0Mem(
                    repo_id="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/shellgame_success_dataset_follow",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/shellgame_success_dataset_follow",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                ),
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryPF(
            "gs://openpi-assets/checkpoints/pi05_base/params"
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
        num_workers=32,
        fsdp_devices=8,
        log_interval=10,
        keep_period=15_000,
    ),
    TrainConfig(
        name="pi0_mem_pf_evan_shellgame_openpi_umi_success_pf_260709_infer",
        # Keep the complete PF architecture so trained checkpoints restore
        # exactly. sample_actions uses only the prior branch at inference.
        model=pi0_mem_pf.Pi0MemPFConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            max_token_len=512,

            num_frames=20,
            frame_stride=5,
            memory_every=1,
            history_memory_tokens=256,
            history_resampler_depth=1,
            history_use_current_condition=True,

            history_gate_fixed=1.0,
            history_gate_lr_multiplier=1.0,

            diversity_weight=0.01,

            current_frame_index=-1,
            current_frame_corrupt_sample_prob=0.0,
            current_frame_dropout_prob=0.0,
            current_frame_mask_prob=0.0,
            current_frame_corrupt_loss_weight=0.0,

            num_future_frames=20,
            future_frame_stride=5,
            future_latent_tokens=64,
            future_gate_fixed=1.0,
            prior_encoder_depth=2,
            lambda_prior=1.0,
            lambda_post=1.0,
            lambda_align=1.0,
            lambda_reg=1e-4,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=32,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Pi0Mem_Inference(
                    repo_id="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/shellgame_openpi_umi_success_follow_fps20",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/shellgame_openpi_umi_success_follow_fps20",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    # Inference has no access to future observations.
                    num_frames=20,
                    frame_stride=5,
                    num_future_frames=0,
                    future_frame_stride=5,
                ),
            ],
        ),
    ),

    # =====================================================================
    TrainConfig(
        name="pi0_mem_compress_umi_wbcd_history_light_v1_260605",
        model=pi0_mem_compress.Pi0MemCompressConfig(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=360,
            num_frames=8,
            memory_every=4,
            history_memory_tokens=256,
            history_resampler_depth=1,
            history_use_current_condition=True,
            history_gate_fixed=1.0,
            history_gate_lr_multiplier=20.0,
            diversity_weight=0.01,
            current_frame_dropout_prob=0.05,
            current_frame_mask_prob=0.05,
            current_frame_corrupt_loss_weight=0.25,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.2, 0.6, 0.6],
            datasets=[
                LeRobotUmiDataConfig_Bimamual_WBCD_4Views_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0525_wbcd_hitl_shanghai",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0525_wbcd_hitl_shanghai",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                    num_frames=8,
                    frame_stride=4,
                ),
                LeRobotUmiDataConfig_Bimamual_WBCD_4Views_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_1",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_1",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                    num_frames=8,
                    frame_stride=4,
                ),
                LeRobotUmiDataConfig_Bimamual_WBCD_4Views_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_2",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_2",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                    num_frames=8,
                    frame_stride=4,
                ),
                LeRobotUmiDataConfig_Bimamual_WBCD_4Views_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_3",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_3",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                    num_frames=8,
                    frame_stride=4,
                ),
                LeRobotUmiDataConfig_Bimamual_WBCD_4Views_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0601_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0601_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                    num_frames=8,
                    frame_stride=4,
                ),
                LeRobotUmiDataConfig_Bimamual_WBCD_4Views_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                    num_frames=8,
                    frame_stride=4,
                ),
                LeRobotUmiDataConfig_Bimamual_WBCD_4Views_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_afternoon",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_afternoon",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                    num_frames=8,
                    frame_stride=4,
                ),
                LeRobotUmiDataConfig_Bimamual_WBCD_4Views_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                    num_frames=8,
                    frame_stride=4,
                ),
                LeRobotUmiDataConfig_Bimamual_WBCD_4Views_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0603_wbcd_hitl_with_wrist_1813_error",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0603_wbcd_hitl_with_wrist_1813_error",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                    num_frames=8,
                    frame_stride=4,
                ),
                LeRobotUmiDataConfig_Bimamual_WBCD_4Views_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                    num_frames=8,
                    frame_stride=4,
                ),
                LeRobotUmiDataConfig_Bimamual_WBCD_4Views_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                    num_frames=8,
                    frame_stride=4,
                )
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
            "/data1/hzl_workspace_for_pi/openpi-umi/checkpoints/260602/69999/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=60_000,
        batch_size=72,
        num_workers=24,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30_000,
    ),

    # =====================================================================
    # Pure current-frame baseline for the Pi0MemCompress experiment above.
    #
    # Motivation: ablation diagnostics on the fixed-gate memory run showed that
    # the history branch changes predictions, but the largest changes tend to
    # *increase* GT MSE. This config keeps the exact same UMI data, action
    # layout, optimizer, schedule, batch, and training script, while disabling
    # the compressed-history path:
    #
    #   - memory_every=0: no Transformer block reads history memory.
    #   - history_memory_tokens=0: no HistoryResampler is instantiated.
    #   - diversity_weight=0.0: no auxiliary memory-token loss.
    #
    # The visual encoder still receives the same video tensor shape from the
    # Pi0Mem-aware data loader, but only the configured current frame contributes
    # to the policy. Use this as the fair "no-history" baseline against the
    # fixed-gate compressed-memory checkpoint.
    #
    # Launch:
    #   XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/mem/train_pi0_mem_compress.py \
    #     pi0_mem_compress_umi_32d_30k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322_current_only \
    #     --exp-name=my_experiment --overwrite
    # =====================================================================
    TrainConfig(
        name="pi0_mem_compress_umi_32d_30k_batch_72_v4_bimanual_wrist_only_horizon1_fold_clothes_260322_current_only",
        model=pi0_mem_compress.Pi0MemCompressConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
            num_frames=16,
            memory_every=0,
            history_memory_tokens=0,
            history_resampler_depth=1,
            history_use_current_condition=False,
            history_gate_fixed=0.0,
            diversity_weight=0.0,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_Bimamual_Horizon1_Pi0Mem(
                    repo_id="/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data1/hzl_workspace_for_pi/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_120313_to_20260427_124451_ep51",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                    num_frames=16,
                    frame_stride=4,
                )
            ],
        ),
        # Still use the memory-compress-aware loader because the target visual
        # encoder creates history gate params even when memory_every=0. Missing
        # keys from pi05_base are therefore expected and should keep random init.
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=25_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=72,
        num_workers=24,
        fsdp_devices=8,
        log_interval=10,
        keep_period=5_000,
    ),

    # Legacy PF recipe retained under its original name and configuration.
    TrainConfig(
        name="pi0_mem_compress_evan_shellgame_openpi_umi_success_260703",
        model=pi0_mem_compress.Pi0MemCompressConfig(
            # pi05=True,
            # action_dim=32,
            # action_horizon=32,
            # action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            # max_token_len=360,
            # num_frames=8,
            # memory_every=4,
            # history_memory_tokens=256,
            # history_resampler_depth=1,
            # history_use_current_condition=True,
            # history_gate_fixed=1.0,
            # history_gate_lr_multiplier=20.0,
            # diversity_weight=0.01,
            # current_frame_dropout_prob=0.05,
            # current_frame_mask_prob=0.05,
            # current_frame_corrupt_loss_weight=0.25,

            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            max_token_len=512,

            num_frames=16,
            memory_every=4,
            history_memory_tokens=256,
            history_resampler_depth=1,
            history_use_current_condition=True,

            history_gate_fixed=1.0,
            history_gate_lr_multiplier=1.0,

            diversity_weight=0.01,

            # current_frame_index=-1,
            # current_frame_dropout_prob=0.05,
            # current_frame_mask_prob=0.0,
            # current_frame_corrupt_sample_prob=0.3,


            current_frame_index=-1,
            # current_frame_dropout_prob=0.1,
            # current_frame_mask_prob=0.0,
            # current_frame_corrupt_sample_prob=0.5,
            # current_frame_corrupt_loss_weight=0.0,


            current_frame_corrupt_sample_prob = 1.0,
            current_frame_dropout_prob = 0.3,
            current_frame_mask_prob = 0.0,
            current_frame_corrupt_loss_weight = 0.0
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Pi0Mem(
                    repo_id="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/fold_clothes_messy_demostration/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                    num_frames=16,
                    frame_stride=10,
                ),
            ],
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithMemoryCompress(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        # weight_loader=weight_loaders.CheckpointWeightLoader(
        #     "/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_mem_compress_umi_wbcd_history_light_v2_260623/260623/59999/params"
        # ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=50_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=72,
        num_workers=32,
        fsdp_devices=8,
        log_interval=10,
        keep_period=15_000,
    ),
]
