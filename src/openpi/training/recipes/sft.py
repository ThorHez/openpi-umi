"""Supervised policy recipes, including inference, normalization and UOT variants."""

from openpi.training.config import (  # noqa: F401
    AssetsConfig,
    CheckpointWeightLoaderWithDiscreteHead,
    DataConfig,
    FakeDataConfig,
    LeRobotAlohaDataConfig,
    LeRobotDROIDDataConfig,
    LeRobotLiberoDataConfig,
    LeRobotUmiDataConfig,
    LeRobotUmiDataConfigPadded,
    LeRobotUmiDataConfigPadded_V2,
    LeRobotUmiDataConfigPadded_V3,
    LeRobotUmiDataConfigPadded_V4,
    LeRobotUmiDataConfigPadded_V4_Bimanual,
    LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon1,
    LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon2,
    LeRobotUmiDataConfigPadded_V4_Bimanual_Inference,
    LeRobotUmiDataConfigPadded_V4_Hybrid,
    LeRobotUmiDataConfigPadded_V4_ImageHorizon1,
    LeRobotUmiDataConfigPadded_V4_Inference,
    LeRobotUmiDataConfigPadded_V4_Norm_stats,
    LeRobotUmiDataConfigPadded_V5,
    LeRobotUmiDataConfig_Bimamual_DeskHeight_ImageHorizon1_Hybrid,
    LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid,
    LeRobotUmiDataConfig_Bimamual_HeadView_DeskHeight_ImageHorizon2_Hybrid,
    LeRobotUmiDataConfig_Bimamual_HeadView_ImageHorizon1_Hybrid,
    LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid,
    LeRobotUmiDataConfig_Bimanual_Inference,
    LeRobotUmiDataConfig_Bimanual_Inference_With_Depth,
    LeRobotUmiDataConfig_Bimanual_Inference_With_Depth_Ray,
    LeRobotUmiDataConfig_Bimanual_Inference_v2,
    LeRobotUmiDataConfig_Hybrid,
    LeRobotUmiDataConfig_Inference,
    LeRobotUmiDataConfig_shellgame_Base,
    LeRobotUmiDataConfig_shellgame_Base_Inference,
    ModelType,
    MultiDataConfigFactory,
    RLDSDroidDataConfig,
    Shellgame_Compute_Norm_Stats,
    SimpleDataConfig,
    TrainConfig,
    UOTConfig,
    UmiArxInferenceDataConfig,
    UmiDataConfig,
    WBCD_Bimamual_2_views_ImageHorizon1,
    WBCD_Bimamual_4_views_ImageHorizon1,
    WBCD_Bimamual_ImageHorizon1,
    WBCD_V1_Bimanual_Horizon1_Compute_Norm_Stats,
    _optimizer,
    _transforms,
    droid_policy,
    droid_rlds_dataset,
    nnx,
    pi0_config,
    pi0_discrete,
    pi0_fast,
    pi0_gripper,
    roboarena_config,
    weight_loaders,
)


_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Fine-tuning UMI configs.
    #
    # These configs are for training on UMI robot dataset.
    # UMI uses end-effector control with 7D actions (position, rotation, gripper).
    TrainConfig(
        name="pi0_umi",
        # Using pi0 model for full fine-tuning
        model=pi0_config.Pi0Config(
            action_dim=7,  # UMI has 7D actions
            action_horizon=10,  # Typical action chunk size
        ),
        # Configure the UMI dataset
        data=LeRobotUmiDataConfig(
            repo_id="your_hf_username/umi_dataset",  # Replace with your HuggingFace dataset repo
            base_config=DataConfig(
                prompt_from_task=True,  # Load task instructions from the dataset
            ),
            use_delta_actions=True,  # Convert absolute actions to delta actions
        ),
        # Load from pi0 base checkpoint for transfer learning
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi0_umi_low_mem_finetune",
        # Using pi0 with LoRA for memory-efficient fine-tuning
        model=pi0_config.Pi0Config(
            action_dim=7,
            action_horizon=10,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotUmiDataConfig(
            repo_id="your_hf_username/umi_dataset",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        batch_size=32,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_umi",
        # Using pi0-FAST model for faster inference
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,  # Single-arm robot, so ~180 tokens is sufficient
        ),
        data=LeRobotUmiDataConfig(
            repo_id="your_hf_username/umi_dataset",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        batch_size=32,
    ),
    TrainConfig(
        name="pi0_fast_umi_low_mem_finetune",
        # Using pi0-FAST with LoRA for memory-efficient fine-tuning
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7,
            action_horizon=10,
            max_token_len=180,
            paligemma_variant="gemma_2b_lora",
        ),
        data=LeRobotUmiDataConfig(
            repo_id="your_hf_username/umi_dataset",
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        batch_size=32,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_umi",
        # Using pi0.5 model - more capable than pi0, with better performance
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=7,
            action_horizon=10,
            discrete_state_input=False,  # UMI uses continuous state
        ),
        data=LeRobotUmiDataConfig(
            repo_id="/root/openpi/umi_lerobot_dataset",
            assets=AssetsConfig(
                assets_dir="/root/openpi/umi_lerobot_dataset",
                asset_id=".",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Note: pi05_base has action_dim=32 (DROID), incompatible with UMI's action_dim=7
        # Training from scratch instead
        weight_loader=weight_loaders.NoOpWeightLoader(),
        # PI0.5 typically uses cosine decay schedule with lower learning rate
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=1,  # Must be divisible by number of GPUs (4)
        num_workers=0,  # Set to 0 to avoid /dev/shm space issues
        # fsdp_devices=4,  # Temporarily disabled - may be incompatible with Pi0.5
    ),
    #
    # Online UMI configs (merged from config.py.online).
    #
    TrainConfig(
        name="pi05_umi_10d_80k_95_real_umi_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=10,
            action_horizon=10,
            discrete_state_input=False,
        ),
        data=LeRobotUmiDataConfigPadded_V3(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v6_train",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v6_train",
            ),
            base_config=DataConfig(prompt_from_task=True),
            training_mode=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/openpi-umi/checkpoints/pi05_umi_10d_80k_95_real_umi_v2/my_experiment/79999/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=80_000,
        batch_size=32,
        num_workers=8,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V3(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v6_train",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v6_train",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=False),
            training_mode=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=5e-5,
            decay_steps=90_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=100_000,
        batch_size=64,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_compute_norm_stats",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Norm_stats(
            repo_id="/data2/hzl_workspace_for_pi/openpi-umi/data/collect_demo_zmq_10_14",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/data2/hzl_workspace_for_pi/openpi-umi/data/collect_demo_zmq_10_14",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_bimanual_compute_norm_stats",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Bimanual(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_with_desk_height",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_with_desk_height",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_bimanual_horizon1_compute_norm_stats",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon1(
            repo_id="/root/openpi-umi/data/fold_clothes_value_training/horizon_cloth_folding_value_training_20260401_20260417_ep177",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/fold_clothes_value_training/horizon_cloth_folding_value_training_20260401_20260417_ep177",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_wbcd_bimanual_horizon1_compute_norm_stats",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=WBCD_V1_Bimanual_Horizon1_Compute_Norm_Stats(
            repo_id="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_evening",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_evening",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_shellgame_norm_stats",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=Shellgame_Compute_Norm_Stats(
            repo_id="/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_current_frame_action",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/data2/hzl_workspace_for_pi_mem/robosuite/outputs/shellgame_lerobot_current_frame_action",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_bimanual_horizon1_compute_norm_stats_fsdp_2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon1(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_hitl_test_with_value",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_hitl_test_with_value",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=2,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_bimanual_horizon2_compute_norm_stats",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Bimanual_Horizon2(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_desk_height_20260216_all_clean",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_desk_height_20260216_all_clean",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_bimanual_headview_horizon1_compute_norm_stats",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
        ),
        data=LeRobotUmiDataConfig_Bimamual_HeadView_ImageHorizon1_Hybrid(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_error_with_depth_20260305_clip",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_error_with_depth_20260305_clip",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V3(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v7.1_test",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v7.1_test",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True),
            training_mode=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=95_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=100_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
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
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
        ),
        data=LeRobotUmiDataConfigPadded_V4_Hybrid(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=CheckpointWeightLoaderWithDiscreteHead("gs://openpi-assets/checkpoints/pi05_base/params"),
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
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_multi_task",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
            max_token_len=756,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            weights=[1.0, 4.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v7.3_merge_20251216",
                    assets=AssetsConfig(
                        asset_id="v73_merge",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v7.3_merge_20251216",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
                        # 双臂: "ARM=2,G=0,H=0"；单臂: "ARM=1,G=0,H=0"（G=全局视角, H=高度，按需改为 1）
                        robot_type="ARM=1 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
                    assets=AssetsConfig(
                        asset_id="pick_elec",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_pick_elec_meta_12_15_all_clean",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
                        # 双臂: "ARM=2,G=0,H=0"；单臂: "ARM=1,G=0,H=0"（G=全局视角, H=高度，按需改为 1）
                        robot_type="ARM=1 G=0 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=CheckpointWeightLoaderWithDiscreteHead("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
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
        keep_period=40000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_multi_dataset_fold_clothes",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
            max_token_len=756,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            weights=[4.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_20260126",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_20260126",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_with_desk_height",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_with_desk_height",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=CheckpointWeightLoaderWithDiscreteHead("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
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
        keep_period=40000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_two_stage_augment",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
            max_token_len=756,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            weights=[1.0, 10.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_20260126",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_20260126",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_two_stage_28_02_clip",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_two_stage_28_02_clip",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=CheckpointWeightLoaderWithDiscreteHead("/root/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_multi_dataset_fold_clothes/my_experiment/79999/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=35_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=40_000,
        batch_size=72,
        num_workers=12,
        fsdp_devices=8,
        log_interval=10,
        keep_period=40000,
    ),

    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_all_data",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
            max_token_len=756,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            weights=[1.0, 1.0, 1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260316",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260316",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260317",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260317",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_depth_20260303_20260304_clean",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_depth_20260303_20260304_clean",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_depth_20260312_clean",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_depth_20260312_clean",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_depth_20260305_clean",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_depth_20260305_clean",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_v2_clean",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_v2_clean",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_merge_fold_red_clothes_cyrus_20251217_clean",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_merge_fold_red_clothes_cyrus_20251217_clean",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=CheckpointWeightLoaderWithDiscreteHead("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=90_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=100_000,
        batch_size=72,
        num_workers=12,
        fsdp_devices=8,
        log_interval=10,
        keep_period=50_000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_horizon_folding_260322",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            # weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
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
                # LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                #     repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260317",
                #     assets=AssetsConfig(
                #         asset_id=".",
                #         assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_aligned_depth_qf06_260317",
                #     ),
                #     base_config=UmiDataConfig(
                #         action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                #         robot_type="ARM=2 G=1 H=0",
                #     ),
                # ),
                # LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                #     repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260320",
                #     assets=AssetsConfig(
                #         asset_id=".",
                #         assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260320",
                #     ),
                #     base_config=UmiDataConfig(
                #         action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                #         robot_type="ARM=2 G=1 H=0",
                #     ),
                # ),
                # LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                #     repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260323",
                #     assets=AssetsConfig(
                #         asset_id=".",
                #         assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_red_1815_right_horizon_260323",
                #     ),
                #     base_config=UmiDataConfig(
                #         action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                #         robot_type="ARM=2 G=1 H=0",
                #     ),
                # ),
            ]
        ),
        weight_loader=CheckpointWeightLoaderWithDiscreteHead("/root/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_horizon_folding_260322/my_experiment/19000/params"),
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
        num_workers=16,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30_000,
    ),

    # Clean, memory-free pi0.5 baseline for ShellGame. This intentionally uses
    # only the current wrist and third-person images: it is the control against
    # which recurrent/semantic-memory policies should be compared.
    TrainConfig(
        name="pi05_shellgame_baseline_v1",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            max_token_len=256,
            # The dataset contains raw robosuite OSC actions (6 arm + gripper).
            # Keep the 32-D pi0.5 head for checkpoint compatibility, but never
            # train the 25 zero-padding dimensions.
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfig_shellgame_Base_Inference(
            repo_id=(
                "/data2/hzl_workspace_for_pi_mem/openpi-umi/"
                "data/shellgame_static_phase_instruction_dataset2"
            ),
            assets=AssetsConfig(
                asset_id=".",
                assets_dir=(
                    "/data2/hzl_workspace_for_pi_mem/openpi-umi/"
                    "data/shellgame_static_phase_instruction_dataset2"
                ),
            ),
            base_config=UmiDataConfig(
                action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
                robot_type="ARM=1 G=0 H=0",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=32,
        num_workers=16,
        fsdp_devices=2,
        log_interval=10,
        save_interval=1_000,
        # A full checkpoint is about 42 GiB on this checkout. Retain only the
        # latest one so the baseline remains safe on the shared data volume.
        keep_period=None,
        wandb_enabled=False,
    ),

    TrainConfig(
        name="pi0_base_evan_shellgame_openpi_260727",
        model=pi0_config.Pi0Config(
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

            # num_frames=32,
            # memory_every=1,
            # history_memory_tokens=256,
            # history_resampler_depth=1,
            # history_use_current_condition=True,

            # history_gate_fixed=1.0,
            # history_gate_lr_multiplier=1.0,

            # diversity_weight=0.01,

            # current_frame_index=-1,
            # current_frame_dropout_prob=0.05,
            # current_frame_mask_prob=0.0,
            # current_frame_corrupt_sample_prob=0.3,


            # current_frame_index=-1,
            # current_frame_dropout_prob=0.1,
            # current_frame_mask_prob=0.0,
            # current_frame_corrupt_sample_prob=0.5,
            # current_frame_corrupt_loss_weight=0.0,


            # current_frame_corrupt_sample_prob = 1.0,
            # current_frame_dropout_prob = 0.3,
            # current_frame_mask_prob = 0.0,
            # current_frame_corrupt_loss_weight = 0.0
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Base(
                    repo_id="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/shellgame_static_phase_instruction_dataset",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/shellgame_static_phase_instruction_dataset",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_shellgame_Base(
                    repo_id="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/shellgame_static_phase_instruction_dataset2",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi_mem/openpi-umi/data/shellgame_static_phase_instruction_dataset2",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
                        robot_type="ARM=1 G=0 H=0",
                    ),
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
        name="pi0_base_evan_shellgame_openpi_260727_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            max_token_len=256,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            weights=[1.0],
            datasets=[
                LeRobotUmiDataConfig_shellgame_Base_Inference(
                    repo_id="/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_base_evan_shellgame_openpi_260727/my_experiment/15000",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/data2/hzl_workspace_for_pi_mem/openpi-umi/checkpoints/pi0_base_evan_shellgame_openpi_260727/my_experiment/15000/assets",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
                        robot_type="ARM=1 G=0 H=0",
                    ),
                ),
            ],
        ),
    ),
    TrainConfig(
        name="pi05_umi_wbcd_v3_260522_h32",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=360,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            # weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            weights=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 1.0, 3.0, 5.0],
            datasets=[
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0525_wbcd_hitl_shanghai",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0525_wbcd_hitl_shanghai",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_1",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_1",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_2",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_2",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_3",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_3",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0601_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0601_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_afternoon",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_afternoon",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0603_wbcd_hitl_with_wrist_1813_error",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0603_wbcd_hitl_with_wrist_1813_error",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0802_wbcd_hitl_with_wrist_hefei",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0802_wbcd_hitl_with_wrist_hefei",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_afternoon",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_afternoon",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_4_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_evening",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_evening",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/openpi-umi/checkpoints/pi05_umi_wbcd_v3_260522_h32/260803_hefei/44000/params"),
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
        keep_period=30_000,
    ),

    TrainConfig(
        name="pi05_umi_wbcd_v3_260522_h32_2views_hefei",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=360,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            # weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            weights=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.2, 0.3, 0.3, 0.3, 0.3, 0.3, 1.0, 3.0, 5.0],
            datasets=[
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0525_wbcd_hitl_shanghai",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0525_wbcd_hitl_shanghai",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_1",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_1",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_2",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_2",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_3",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_3",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0601_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0601_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_afternoon",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_afternoon",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0603_wbcd_hitl_with_wrist_1813_error",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0603_wbcd_hitl_with_wrist_1813_error",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0802_wbcd_hitl_with_wrist_hefei",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0802_wbcd_hitl_with_wrist_hefei",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_afternoon",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_afternoon",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_evening",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0803_wbcd_hitl_with_wrist_hefei_evening",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/openpi-umi/checkpoints/pi05_umi_wbcd_v3_260522_h32/260608/59999/params"),
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
        fsdp_devices=6,
        log_interval=10,
        keep_period=30_000,
    ),
     TrainConfig(
        name="pi05_umi_wbcd_v3_260522_h32_2views",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=32,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=360,
        ),
                data=MultiDataConfigFactory(
            state_pad_dim=96,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            # weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            weights=[0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3, 0.3, 0.2, 0.3, 0.3, 0.3, 0.3],
            datasets=[
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0525_wbcd_hitl_shanghai",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0525_wbcd_hitl_shanghai",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_1",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_1",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_2",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_2",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_3",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0526_wbcd_hitl_shanghai_3",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0601_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0601_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_afternoon",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_afternoon",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0602_wbcd_hitl_with_wrist_1813_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0603_wbcd_hitl_with_wrist_1813_error",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0603_wbcd_hitl_with_wrist_1813_error",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0604_wbcd_hitl_with_wrist_1813_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
                WBCD_Bimamual_2_views_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai_night",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/0608_wbcd_hitl_with_wrist_shanghai_night",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=2 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/openpi-umi/checkpoints/pi05_umi_wbcd_v3_260522_h32/260604/59999/params"),
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
        num_workers=16,
        fsdp_devices=6,
        log_interval=10,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_umi_wbcd_v1_260511",
        model=pi0_gripper.Pi0GripperConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=360,
            gripper_binary_indices=(9, 19),
            gripper_binary_threshold=0.03,
            gripper_binary_loss_weight=0.5,
            gripper_binary_close_value=0.0,
            gripper_binary_open_value=0.085,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=96,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            # weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            weights=[1.0, 1.0, 1.0],
            datasets=[
                WBCD_Bimamual_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/wbcd_260511_ori",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/wbcd_260511_ori",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                WBCD_Bimamual_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/umi/wbcd_umi_260510_260511",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/umi/wbcd_umi_260510_260511",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
                WBCD_Bimamual_ImageHorizon1(
                    repo_id="/root/openpi-umi/data/wbcd/wbcd_260513",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/wbcd/wbcd_260513",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=0 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderWithGripperHead("/root/openpi-umi/checkpoints/pi05_umi_wbcd_v1_260511/my_experiment/79999/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=70_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=80_000,
        batch_size=72,
        num_workers=16,
        fsdp_devices=8,
        log_interval=10,
        keep_period=40_000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_messy_folding_gripper_binary_260422",
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
            weights=[1.0, 1.0, 2.0, 7.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                # LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                #     repo_id="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                #     assets=AssetsConfig(
                #         asset_id=".",
                #         assets_dir="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                #     ),
                #     base_config=UmiDataConfig(
                #         action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                #         robot_type="ARM=2 G=1 H=0",
                #     ),
                # ),
                # LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                #     repo_id="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260417_20260418_ep171",
                #     assets=AssetsConfig(
                #         asset_id=".",
                #         assets_dir="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260417_20260418_ep171",
                #     ),
                #     base_config=UmiDataConfig(
                #         action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                #         robot_type="ARM=2 G=1 H=0",
                #     ),
                # ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260422_ep100",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_advantage_error_hitl_20260422_ep100",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_ep102",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/fold_clothes_action_finetuning/horizon_cloth_folding_test_20260427_ep102",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
            ]
        ),
        # 从标准 pi05 checkpoint 初始化时，新增的 gripper_binary_head 会保留随机初始化；
        # 后续训练保存出的 checkpoint 会自动包含这个 head。
        weight_loader=weight_loaders.CheckpointWeightLoaderWithGripperHead(
            "/root/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_messy_folding_gripper_binary_260422/my_experiment_v3/19999/params"
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
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=20000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_messy_folding_260410",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            weights=[1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
            ]
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_horizon_folding_260322/my_experiment_v2/59999/params"),
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
        num_workers=12,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30_000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v5",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V5(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v7.3_merge_20251216",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v7.3_merge_20251216",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
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
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_freeze_vlm_only",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v7.3_merge_20251216",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v7.3_merge_20251216",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ).get_freeze_filter_freeze_vlm_only(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=40_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=20000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_bimanual",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Bimanual(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_merge_mason_ray_cyrus_20251224_20260107_exclude_25_all_clean",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_merge_mason_ray_cyrus_20251224_20260107_exclude_25_all_clean",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=2_000,
            peak_lr=8e-5,
            decay_steps=70_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=80_000,
        batch_size=72,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=20000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_10d_relative",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V2(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v4_train",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v4_train",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
            use_relative_state=True,
            use_10d_pose=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=80_000,
        batch_size=32,
        num_workers=8,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfigPadded(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v3_train",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v3_train",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=80_000,
        batch_size=32,
        num_workers=8,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_umi_32d_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=UmiArxInferenceDataConfig(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v6_train",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v6_train",
            ),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_v2/my_experiment/79999/params"),
    ),
    TrainConfig(
        name="pi05_umi_32d_retrain",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfigPadded(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v3",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v3",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/root/openpi-umi/checkpoints/pi05_umi_32d/my_experiment_v2/79999/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=80_000,
        batch_size=32,
        num_workers=8,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
            prompt="pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216/params"),
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_pick_elec_hybrid_20260126",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_pick_elec_hybrid_20260126",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
            prompt="pick up electronic components and place them into the correct boxes",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoaderIgnoreDiscreteHead(
            "/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_pick_elec_hybrid_20260126/params"),
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v5_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_v5_20260122",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_v5_20260122",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_v5_20260122/params"),
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_freeze_vlm_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216_freeze_vlm_only",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216_freeze_vlm_only",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216_freeze_vlm_only/params"),
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_bimanual_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Bimanual_Inference(
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
        name="pi05_umi_32d",
        # Using 32-dim actions (padded from 7-dim) to be compatible with pi05_base pretrained model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # Padded to match pretrained model
            action_horizon=10,
            discrete_state_input=False,
            # Only compute loss on first 7 dimensions (real UMI actions), ignore padded dims
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfigPadded(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v3",  # New dataset with 32-dim actions
            assets=AssetsConfig(
                # Will load norm_stats from assets/pi05_umi_32d/umi_lerobot_dataset_32d/
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v3",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Now we can load pretrained weights!
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=32,
        num_workers=8,
        fsdp_devices=8,
    ),
    TrainConfig(
        name="pi05_umi_32d_mp",
        # Using 32-dim actions (padded from 7-dim) to be compatible with pi05_base pretrained model
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # Padded to match pretrained model
            action_horizon=10,
            discrete_state_input=False,
            # Only compute loss on first 7 dimensions (real UMI actions), ignore padded dims
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfigPadded(
            repo_id="/root/openpi/umi_lerobot_dataset_7d",  # New dataset with 32-dim actions
            assets=AssetsConfig(
                # Will load norm_stats from assets/pi05_umi_32d/umi_lerobot_dataset_32d/
                asset_id=".",
                assets_dir="/root/openpi/umi_lerobot_dataset_7d",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Now we can load pretrained weights!
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=4,
        num_workers=2,
        fsdp_devices=2,
    ),
    TrainConfig(
        name="pi05_umi_multi_dataset",
        # Multi-dataset training: concatenate multiple UMI datasets with optional per-dataset weights.
        # Use scripts/train_multi_dataset.py (not train_hybrid.py) for this config.
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
        ),
        data=MultiDataConfigFactory(
            datasets=[
                LeRobotUmiDataConfigPadded(
                    repo_id="/root/openpi-umi/data/umi_lerobot_dataset_v3",
                    assets=AssetsConfig(asset_id=".", assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_v3"),
                    base_config=DataConfig(prompt_from_task=True),
                    use_delta_actions=True,
                ),
                # Add more datasets here; each can have different repo_id and norm_stats.
                # LeRobotUmiDataConfigPadded(
                #     repo_id="/path/to/second_dataset",
                #     assets=AssetsConfig(asset_id=".", assets_dir="/path/to/second_dataset"),
                #     ...
                # ),
            ],
            weights=None,  # Optional: list of float per dataset for sampling; None = uniform
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=30_000,
        batch_size=32,
        num_workers=4,
    ),
    TrainConfig(
        name="pi05_umi_32d_lora",
        # LoRA version with 32-dim actions
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # Only compute loss on first 7 dimensions (real UMI actions), ignore padded dims
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfig(
            repo_id="/root/openpi/umi_lerobot_dataset_32d",
            assets=AssetsConfig(
                assets_dir="/root/openpi/umi_lerobot_dataset_32d",
                asset_id=".",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Load pretrained weights before LoRA fine-tuning
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=4,  # Reduced batch size per GPU
        num_workers=0,  # Set to 0 to avoid multiprocessing overhead
        fsdp_devices=1,  # Disable FSDP - use data parallel instead
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
    ),
    TrainConfig(
        name="pi05_umi_32d_lora_v3.1",
        seed=43,
        # LoRA version with 32-dim actions
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            # Only compute loss on first 7 dimensions (real UMI actions), ignore padded dims
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfigPadded(
            repo_id="/root/openpi/umi_lerobot_dataset_7d",
            assets=AssetsConfig(
                assets_dir="/root/openpi/umi_lerobot_dataset_7d",
                asset_id=".",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Load pretrained weights before LoRA fine-tuning
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=32,  # Reduced batch size per GPU
        num_workers=0,  # Set to 0 to avoid multiprocessing overhead
        fsdp_devices=1,  # Disable FSDP - use data parallel instead
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]}
    ),
    TrainConfig(
        name="pi05_umi_lora",
        # LoRA fine-tuning for low memory usage
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=7,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotUmiDataConfig(
            repo_id="/root/openpi/umi_lerobot_dataset_7d",
            assets=AssetsConfig(
                assets_dir="/root/openpi/umi_lerobot_dataset_7d",
                asset_id=".",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Training from scratch with LoRA architecture (can't load pi05_base due to action_dim mismatch)
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=80_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # Turn off EMA for LoRA training to save memory
        ema_decay=None,
        num_train_steps=30_000,
        batch_size=8,  # Can use larger batch with LoRA
        num_workers=0,
        # Freeze non-LoRA parameters (though we're training from scratch, this defines the architecture)
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
    ),
    TrainConfig(
        name="pi05_umi_from_scratch",
        # Training pi0.5 from scratch (not recommended unless you have large dataset)
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=7,
            action_horizon=10,
            discrete_state_input=False,
        ),
        data=LeRobotUmiDataConfig(
            repo_id="/root/openpi/umi_lerobot_dataset",
            assets=AssetsConfig(
                assets_dir="/root/openpi/umi_lerobot_dataset",
                asset_id=".",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # No weight loader - training from scratch
        weight_loader=weight_loaders.NoOpWeightLoader(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=1e-4,
            decay_steps=100_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=100_000,
        batch_size=4,
        num_workers=0,  # Set to 0 to avoid /dev/shm space issues
        fsdp_devices=4,  # Enable FSDP to shard model across 4 GPUs to reduce memory per GPU
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    TrainConfig(
        name="pi05_umi_32d_clean",
        # UMI with 32-dim padding (clean version - padding stays as 0, not -1.0)
        # Uses 7-dim norm_stats + pads after normalization
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_loss_mask=(1.0,) * 7 + (0.0,) * 25,
        ),
        data=LeRobotUmiDataConfigPadded(
            repo_id="/root/openpi/umi_lerobot_dataset_7d",
            assets=AssetsConfig(
                assets_dir="/root/openpi/umi_lerobot_dataset_7d",
                asset_id=".",
            ),
            base_config=DataConfig(prompt_from_task=True),
            use_delta_actions=True,
        ),
        # Load pretrained weights
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=5_000,
            peak_lr=5e-5,
            decay_steps=30_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=30_000,
        batch_size=4,
        num_workers=0,
        fsdp_devices=1,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
    ),
    #
    # RoboArena configs.
    #
    # Public recipe names retained from the earlier multi_dataset/value_infer branches.
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_bimanual_headview_deskheight_horizon2_compute_norm_stats",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
        ),
        data=LeRobotUmiDataConfig_Bimamual_HeadView_DeskHeight_ImageHorizon2_Hybrid(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_desk_height_20260216_all_clean",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_fold_clothes_with_desk_height_20260216_all_clean",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_bimanual_deskheight_horizon1_compute_norm_stats",
        model=pi0_discrete.Pi0DiscreteConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            enable_discrete_head=True,
            fast_model_tokenizer_kwargs={
                "fast_tokenizer_path": "/root/fast_tokenizer"
            },
        ),
        data=LeRobotUmiDataConfig_Bimamual_DeskHeight_ImageHorizon1_Hybrid(
            repo_id="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_with_desk_height_clean",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/root/openpi-umi/data/umi_lerobot_dataset_merge_fold_clothes_mason_ray_cyrus_24_07_exclude_25_all_with_desk_height_clean",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
        ),
        batch_size=512,
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_messy_folding_gripper_binary_260413",
        model=pi0_gripper.Pi0GripperConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
            gripper_binary_indices=(9, 19),
            gripper_binary_threshold=0.02,
            gripper_binary_loss_weight=0.5,
            gripper_binary_close_value=0.0,
            gripper_binary_open_value=0.085,
        ),
        data=MultiDataConfigFactory(
            state_pad_dim=128,
            # 采样权重，与下面 datasets 一一对应；None 表示均匀采样
            weights=[1.0, 1.0, 1.0],  # [v7.3_merge, pick_elec, fold_merge_exclude25, fold_desk_height_head]
            datasets=[
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/cloth_folding_hitl_replay_buffer_0407_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260408_ep85_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
                LeRobotUmiDataConfig_Bimamual_HeadView_Depth_ImageHorizon1_Hybrid(
                    repo_id="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                    assets=AssetsConfig(
                        asset_id=".",
                        assets_dir="/root/openpi-umi/data/horizon_cloth_folding_advantage_messy_demostration_20260409_ep83_gripper_bin",
                    ),
                    base_config=UmiDataConfig(
                        action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
                        robot_type="ARM=2 G=1 H=0",
                    ),
                ),
            ]
        ),
        # 从标准 pi05 checkpoint 初始化时，新增的 gripper_binary_head 会保留随机初始化；
        # 后续训练保存出的 checkpoint 会自动包含这个 head。
        weight_loader=weight_loaders.CheckpointWeightLoaderWithGripperHead(
            "/root/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_horizon_folding_260322/my_experiment_v2/59999/params"
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
        num_workers=8,
        fsdp_devices=8,
        log_interval=10,
        keep_period=30000,
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_infer_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216",
            ),
            base_config=DataConfig(prompt_from_task=True, use_quantile_norm=True, action_sequence_keys=()),
            prompt="pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/29999_merge_20251216/params"),
    ),
    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_bimanual_infer_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfigPadded_V4_Bimanual_Inference(
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
        name="pi05_umi_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 10 + (0.0,) * 22,
            max_token_len=256,
        ),
        data=LeRobotUmiDataConfig_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/79999_multi_task_20260226",
            assets=AssetsConfig(
                asset_id="v73_merge",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/79999_multi_task_20260226/assets",
            ),
            base_config=UmiDataConfig(robot_type="ARM=1 G=0 H=0",),
            prompt="pick up and place the orange cube in the orange box, then pick up and place the black cube in the black box",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/79999_multi_task_20260226/params"),
    ),
    TrainConfig(
        name="pi05_umi_bimanual_infer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfig_Bimanual_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/33000_fold_clothes_clean_data_all_20260311",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/33000_fold_clothes_clean_data_all_20260311/assets",
            ),
            base_config=UmiDataConfig(robot_type="ARM=2 G=0 H=0",),
            prompt="fold the clothes",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/33000_fold_clothes_clean_data_all_20260311/params"),
    ),
    TrainConfig(
        name="pi05_umi_bimanual_infer_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfig_Bimanual_Inference_v2(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/38000_fold_clothes_clean_data_all_20260318",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/38000_fold_clothes_clean_data_all_20260318/assets",
            ),
            base_config=UmiDataConfig(robot_type="ARM=2 G=0 H=0",),
            prompt="fold the clothes",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/38000_fold_clothes_clean_data_all_20260318/params"),
    ),
    TrainConfig(
        name="pi05_umi_bimanual_infer_with_depth_38000",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfig_Bimanual_Inference_With_Depth(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/38000_fold_clothes_clean_data_all_20260318",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/38000_fold_clothes_clean_data_all_20260318/assets",
            ),
            base_config=UmiDataConfig(robot_type="ARM=2 G=1 H=0",),
            prompt="fold the clothes",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/38000_fold_clothes_clean_data_all_20260318/params"),
    ),
    TrainConfig(
        name="pi05_umi_bimanual_infer_with_depth",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfig_Bimanual_Inference_With_Depth(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/79999_fold_clothes_depth_20260319",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/79999_fold_clothes_depth_20260319/assets",
            ),
            base_config=UmiDataConfig(robot_type="ARM=2 G=1 H=0",),
            prompt="fold the clothes",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/79999_fold_clothes_depth_20260319/params"),
    ),
    TrainConfig(
        name="pi05_umi_bimanual_infer_with_depth_horizon",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfig_Bimanual_Inference_With_Depth(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_horizon_folding_20260327",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_horizon_folding_20260327/assets",
            ),
            base_config=UmiDataConfig(robot_type="ARM=2 G=1 H=0",),
            prompt="fold the clothes",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_horizon_folding_20260327/params"),
    ),
    TrainConfig(
        name="pi05_umi_bimanual_infer_with_depth_messy",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfig_Bimanual_Inference_With_Depth(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_messy_folding_260410",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_messy_folding_260410/assets",
            ),
            base_config=UmiDataConfig(robot_type="ARM=2 G=1 H=0",),
            prompt="fold the clothes",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_messy_folding_260410/params"),
    ),
    TrainConfig(
        name="pi05_umi_bimanual_infer_with_depth_gripper_binary",
        model=pi0_gripper.Pi0GripperConfig(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
            gripper_binary_indices=(9, 19),
            gripper_binary_threshold=0.02,
            gripper_binary_loss_weight=0.5,
            gripper_binary_close_value=0.0,
            gripper_binary_open_value=0.085,
        ),
         data=LeRobotUmiDataConfig_Bimanual_Inference_With_Depth(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_messy_folding_260409",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_messy_folding_260409/assets",
            ),
            base_config=UmiDataConfig(robot_type="ARM=2 G=1 H=0",),
            prompt="fold the clothes",
        ),
        # 从标准 pi05 checkpoint 初始化时，新增的 gripper_binary_head 会保留随机初始化；
        # 后续训练保存出的 checkpoint 会自动包含这个 head。
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/root/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_batch_72_v4_hybrid_fold_clothes_horizon_folding_260322/my_experiment_v2/59999/params"
        )
    ),
    TrainConfig(
        name="pi05_umi_bimanual_infer_with_depth_ray",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfig_Bimanual_Inference_With_Depth_Ray(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_messy_folding_260409",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_messy_folding_260409/assets",
            ),
            base_config=UmiDataConfig(robot_type="ARM=2 G=1 H=0",),
            prompt="fold the clothes",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/59999_messy_folding_260409/params"),
    ),
    TrainConfig(
        name="pi05_umi_bimanual_infe_no_robot_type",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
            action_loss_mask=(1.0,) * 20 + (0.0,) * 12,
            max_token_len=512,
        ),
        data=LeRobotUmiDataConfig_Bimanual_Inference(
            repo_id="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/79999_v4_bimanual_20260111",
            assets=AssetsConfig(
                asset_id=".",
                assets_dir="/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/79999_v4_bimanual_20260111/assets",
            ),
            base_config=UmiDataConfig(robot_type=None,),
            prompt="fold the clothes",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/media/admin123/E/hzl_workspace_for_pi/openpi-umi/checkpoints/79999_v4_bimanual_20260111/params"),
    ),
    *roboarena_config.get_roboarena_configs(),

    TrainConfig(
        name="pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta",
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
        weight_loader=weight_loaders.CheckpointWeightLoader("/data2/hzl_workspace_for_pi/openpi-umi/checkpoints/pi05_umi_32d_80k_95_real_umi_batch_72_v4_elec_meta/my_experiment/59999/params"),
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
]
