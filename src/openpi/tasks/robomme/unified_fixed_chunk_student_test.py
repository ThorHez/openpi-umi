import jax
import jax.numpy as jnp

from openpi.tasks.robomme import unified_fixed_chunk_student as student_lib


def _small_model(
    *,
    use_write_gate=False,
    use_event_gate=False,
    event_gate_modulation_strength=0.0,
    use_event_update_routing=False,
    event_update_routing_temperature=1.0,
    use_event_correction=False,
    use_oracle_event_correction=False,
    use_causal_evidence_state=False,
    use_recurrent_memory=True,
):
    return student_lib.UnifiedFixedChunkRecurrentStudent(
        max_steps=5,
        frames=2,
        spatial_tokens=4,
        input_width=16,
        width=16,
        num_memory_tokens=24,
        encoder_width=16,
        encoder_depth=1,
        encoder_heads=4,
        memory_depth=1,
        memory_heads=4,
        dtype_mm="float32",
        use_write_gate=use_write_gate,
        use_event_gate=use_event_gate,
        event_gate_modulation_strength=event_gate_modulation_strength,
        use_event_update_routing=use_event_update_routing,
        event_update_routing_temperature=event_update_routing_temperature,
        use_event_correction=use_event_correction,
        use_oracle_event_correction=use_oracle_event_correction,
        use_causal_evidence_state=use_causal_evidence_state,
        use_recurrent_memory=use_recurrent_memory,
    )


def _inputs():
    return {
        "patch_tokens": jax.random.normal(jax.random.key(1), (2, 5, 2, 4, 16)),
        "task_ids": jnp.asarray([0, 3], dtype=jnp.int32),
        "goal_color_ids": jnp.asarray([[1, 0], [2, 0]], dtype=jnp.int32),
        "required_counts": jnp.asarray([0, 3], dtype=jnp.int32),
        "queried_ordinals": jnp.asarray([0, 0], dtype=jnp.int32),
        "num_regions": jnp.asarray([3, 4], dtype=jnp.int32),
        "sequence_mask": jnp.asarray(
            [[True, True, True, False, False], [True, True, False, False, False]]
        ),
    }


def test_scan_student_preserves_memory_only_for_sequence_padding():
    model = _small_model()
    inputs = _inputs()
    variables = model.init(jax.random.key(0), **inputs)
    output = model.apply(variables, **inputs)
    assert output["all_memories"].shape == (2, 6, 24, 16)
    assert jnp.array_equal(output["chunk_memories"][0, 3], output["chunk_memories"][0, 2])
    assert jnp.array_equal(output["chunk_memories"][0, 4], output["chunk_memories"][0, 3])
    assert jnp.array_equal(output["chunk_memories"][1, 2], output["chunk_memories"][1, 1])


def test_fixed_chunk_student_has_no_event_interface_or_parameters():
    model = _small_model()
    inputs = _inputs()
    variables = model.init(jax.random.key(0), **inputs)
    flat_names = "/".join(jax.tree_util.tree_leaves(
        jax.tree_util.tree_map_with_path(lambda path, _: "/".join(str(item) for item in path), variables["params"])
    ))
    assert "event" not in flat_names.lower()
    assert set(inputs) == {
        "patch_tokens",
        "task_ids",
        "goal_color_ids",
        "required_counts",
        "queried_ordinals",
        "num_regions",
        "sequence_mask",
    }


def test_soft_write_gate_starts_carry_biased_and_masks_padding():
    model = _small_model(use_write_gate=True)
    inputs = _inputs()
    variables = model.init(jax.random.key(0), **inputs)
    output = model.apply(variables, **inputs)
    gates = output["write_gates"]
    expected = jax.nn.sigmoid(jnp.asarray(-2.0))
    assert gates.shape == (2, 5)
    assert jnp.allclose(gates[inputs["sequence_mask"]], expected)
    assert jnp.array_equal(gates[~inputs["sequence_mask"]], jnp.zeros((5,)))
    assert "gate_out" in str(variables["params"])


def test_causal_evidence_state_is_shared_and_masks_padding():
    model = _small_model(use_causal_evidence_state=True)
    inputs = _inputs()
    variables = model.init(jax.random.key(0), **inputs)
    output = model.apply(variables, **inputs)
    tokens = output["causal_evidence_tokens"]
    assert tokens.shape == (2, 5, 16)
    assert jnp.array_equal(tokens[~inputs["sequence_mask"]], jnp.zeros((5, 16)))
    assert "shared_causal_evidence_state" in str(variables["params"])


def test_no_recurrent_carry_resets_each_valid_chunk_but_keeps_padding_frozen():
    model = _small_model(use_recurrent_memory=False)
    inputs = _inputs()
    variables = model.init(jax.random.key(0), **inputs)
    output = model.apply(variables, **inputs)
    assert output["all_memories"].shape == (2, 6, 24, 16)
    assert jnp.array_equal(output["chunk_memories"][0, 3], output["chunk_memories"][0, 2])
    assert jnp.array_equal(output["chunk_memories"][1, 2], output["chunk_memories"][1, 1])


def test_independent_event_gate_is_identity_when_modulation_is_disabled():
    model = _small_model(use_write_gate=True, use_event_gate=True)
    inputs = _inputs()
    variables = model.init(jax.random.key(0), **inputs)
    output = model.apply(variables, **inputs)
    valid = inputs["sequence_mask"]
    assert jnp.allclose(output["event_gates"][valid], 0.5)
    assert jnp.array_equal(output["event_gates"][~valid], jnp.zeros((5,)))
    assert jnp.allclose(output["gate_modulations"], 1.0)
    assert jnp.allclose(output["effective_write_gates"], output["write_gates"])
    flat_names = str(variables["params"])
    assert "event_out" in flat_names
    assert "gate_out" in flat_names


def test_event_gate_modulation_is_bounded_and_keeps_padding_frozen():
    model = _small_model(
        use_write_gate=True,
        use_event_gate=True,
        event_gate_modulation_strength=1.0,
    )
    inputs = _inputs()
    variables = model.init(jax.random.key(0), **inputs)
    output = model.apply(variables, **inputs)
    valid = inputs["sequence_mask"]
    assert jnp.all(output["gate_modulations"][valid] <= 1.25)
    assert jnp.all(output["gate_modulations"][valid] >= 0.75)
    assert jnp.array_equal(output["effective_write_gates"][~valid], jnp.zeros((5,)))


def test_event_update_routing_is_exact_identity_at_initialization():
    routed_model = _small_model(
        use_write_gate=True,
        use_event_gate=True,
        use_event_update_routing=True,
    )
    base_model = _small_model(use_write_gate=True, use_event_gate=True)
    inputs = _inputs()
    routed_variables = routed_model.init(jax.random.key(0), **inputs)
    routed_output = routed_model.apply(routed_variables, **inputs)
    base_output = base_model.apply(routed_variables, **inputs)
    assert jnp.array_equal(routed_output["all_memories"], base_output["all_memories"])
    assert jnp.array_equal(
        routed_output["event_update_residual_norms"], jnp.zeros((2, 5))
    )
    assert jnp.array_equal(
        routed_output["hold_update_residual_norms"], jnp.zeros((2, 5))
    )
    assert "route_event_out" in str(routed_variables["params"])
    assert "route_hold_out" in str(routed_variables["params"])
    valid = inputs["sequence_mask"]
    assert jnp.allclose(
        routed_output["event_update_routing_probabilities"][valid],
        routed_output["event_gates"][valid],
    )


def test_event_update_routing_temperature_sharpens_probability_without_initial_drift():
    model = _small_model(
        use_write_gate=True,
        use_event_gate=True,
        use_event_update_routing=True,
        event_update_routing_temperature=0.25,
    )
    inputs = _inputs()
    variables = model.init(jax.random.key(0), **inputs)
    output = model.apply(variables, **inputs)
    valid = inputs["sequence_mask"]
    assert jnp.allclose(output["event_update_routing_probabilities"][valid], 0.5)
    assert jnp.array_equal(
        output["event_update_routing_probabilities"][~valid], jnp.zeros((5,))
    )
    assert jnp.array_equal(output["routed_update_residual_norms"], jnp.zeros((2, 5)))


def test_oracle_event_correction_is_exact_identity_at_zero_initialization():
    correction_model = _small_model(
        use_write_gate=True,
        use_event_correction=True,
        use_oracle_event_correction=True,
    )
    base_model = _small_model(use_write_gate=True)
    inputs = _inputs()
    oracle = jnp.asarray(
        [[False, True, False, False, False], [True, False, False, False, False]]
    )
    correction_variables = correction_model.init(
        jax.random.key(0), **inputs, oracle_event_mask=oracle
    )
    correction_output = correction_model.apply(
        correction_variables, **inputs, oracle_event_mask=oracle
    )
    base_output = base_model.apply(correction_variables, **inputs)
    assert jnp.array_equal(correction_output["all_memories"], base_output["all_memories"])
    assert jnp.array_equal(correction_output["event_correction_norms"], jnp.zeros((2, 5)))
    assert jnp.array_equal(
        correction_output["event_correction_gates"],
        oracle.astype(jnp.float32) * inputs["sequence_mask"],
    )
    assert "correction_out" in str(correction_variables["params"])


def test_oracle_event_correction_delta_loss_ignores_hold_and_padding():
    base = jnp.zeros((1, 3, 2, 2))
    teacher = base.at[:, 0].set(2.0).at[:, 1].set(7.0).at[:, 2].set(9.0)
    correction = jnp.zeros_like(base).at[:, 0].set(2.0)
    sequence_mask = jnp.asarray([[True, True, False]])
    change_mask = jnp.asarray([[True, False, True]])
    loss, metrics = student_lib.oracle_event_correction_delta_loss(
        correction, base, teacher, sequence_mask, change_mask
    )
    assert loss == 0.0
    assert metrics["event_correction_target_rms"] == 2.0


def test_weighted_losses_are_zero_for_exact_teacher_memory():
    memory = jax.random.normal(jax.random.key(3), (2, 4, 24, 16))
    weights = jnp.asarray([[1.0, 6.0, 1.0, 0.0], [1.0, 1.0, 6.0, 1.0]])
    loss, metrics = student_lib.weighted_memory_distillation_loss(memory, memory, weights)
    assert loss < 1e-6
    assert metrics["memory_mse_loss"] == 0


def test_no_change_consistency_ignores_transitions_and_padding():
    memory = jnp.zeros((1, 5, 3, 2))
    memory = memory.at[:, 1].set(1.0)
    memory = memory.at[:, 2].set(2.0)
    memory = memory.at[:, 3].set(2.0)
    memory = memory.at[:, 4].set(7.0)
    sequence_mask = jnp.asarray([[True, True, True, False]])
    state_change_mask = jnp.asarray([[True, True, False, False]])
    loss = student_lib.no_change_memory_consistency_loss(
        memory, sequence_mask, state_change_mask
    )
    assert loss == 0.0
