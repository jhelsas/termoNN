# Project Symbol Map (AST Summary)

## `./generate_ast_map.py`
### Functions
- `parse_file()`
- `generate_project_map()`
- `format_map()`

## `./main.py`
### Functions
- `bc_harmonic()`
- `bc_nested()`
- `solve_koch_snowflake_example()`
- `solve_nested_snowflakes_example()`
- `compare_pinn_fem()`

## `./scripts/calculate_w.py`
## `./scripts/run_test_energy.py`
## `./scripts/test_energy_functional.py`
### Classes
- **Model**
  - `__init__()`
  - `forward()`

## `./scripts/test_energy_issue.py`
### Classes
- **Model**
  - `__init__()`
  - `forward()`

### Functions
- `f_fn()`

## `./scripts/test_solve_energy.py`
### Functions
- `f_fn()`
- `bc_fn()`

## `./src/__init__.py`
## `./src/core/data.py`
### Functions
- `set_seed()`
- `get_device()`
- `generate_domain_data()`
- `generate_adaptive_domain_data()`
- `generate_boundary_data()`

## `./src/core/geometry.py`
### Classes
- **PolygonDomain**
  - `__init__()`
  - `_calculate_area()`
  - `sample_curvature_biased_interior()`
  - `sample_curvature_biased_boundary()`
  - `is_inside()`
  - `sample_interior()`
  - `sample_boundary()`
  - `exact_distance()`

### Functions
- `generate_koch_snowflake()`

## `./src/core/viz.py`
### Functions
- `plot_results()`
- `plot_history()`
- `plot_comparison()`

## `./src/fem/__init__.py`
## `./src/fem/solver.py`
### Functions
- `solve_fem()`

## `./src/pinn/model.py`
### Classes
- **Sine**
  - `__init__()`
  - `forward()`
- **PINN**
  - `__init__()`
  - `_init_siren()`
  - `forward()`
  - `device()`
- **ExactBoundaryAnsatz**
  - `__init__()`
  - `forward()`
  - `device()`

## `./src/pinn/physics.py`
### Functions
- `poisson_loss()`
- `sobolev_laplace_loss()`
- `laplace_loss()`
- `boundary_loss()`
- `boundary_gradient_loss()`
- `range_loss()`
- `energy_loss()`

## `./src/pinn/solver.py`
### Functions
- `train()`

## `./tests/__init__.py`
## `./tests/base_test.py`
### Classes
- **PINNTestCase**
  - `setUp()`
  - `assertTensorsEqual()`
  - `assertTensorFinite()`

## `./tests/test_advanced_robustness.py`
### Classes
- **TestAdvancedRobustness**
  - `test_multiscale_spectral_capacity()`
  - `test_adaptive_weights_sobolev_stability()`
  - `test_distance_gradient_at_sharp_vertex()`
  - `test_sobolev_memory_pressure()`

## `./tests/test_ansatz.py`
### Classes
- **TestAnsatz**
  - `setUp()`
  - `test_ansatz_outer_boundary()`
  - `test_ansatz_inner_boundary()`
  - `test_ansatz_interior_gradients()`
  - `test_ansatz_invalid_mode()`

## `./tests/test_consistency.py`
### Classes
- **TestConsistency**
  - `test_reproducibility()`
  - `test_solver_no_data_crash()`
  - `test_spectral_omega_scalar_vs_tuple()`
  - `test_adaptive_weights_sanity()`
  - `test_multi_hole_geometry_overlap()`

## `./tests/test_coverage_sweep.py`
### Classes
- **TestCoverageSweep**
  - `test_solver_full_adaptive_path()`
  - `test_geometry_remainder_logic()`
  - `test_physics_poisson_none_source()`
  - `test_data_device_fallbacks()`

## `./tests/test_data.py`
### Classes
- **TestUtils**
  - `test_domain_sampling()`
  - `test_domain_sampling_count()`
  - `test_boundary_sampling_count()`
  - `test_boundary_sampling()`
  - `test_reproducibility()`
  - `test_boundary_values_range()`
  - `test_boundary_distribution()`
  - `test_reproducibility_with_polygon()`
  - `test_domain_default_fallback()`
  - `test_adaptive_sampling_rar()`
  - `test_adaptive_sampling_determinism()`
  - `test_adaptive_sampling_error_targeting()`
  - `test_adaptive_sampling_range_targeting()`

## `./tests/test_energy_physics.py`
### Classes
- **TestEnergyFormulation**
  - `test_energy_loss_const()`
  - `test_energy_loss_linear()`
  - `test_area_calculation()`

## `./tests/test_fem.py`
### Classes
- **TestFEMComparison**
  - `test_fem_square_laplace()`
  - `test_fem_snowflake_harmonic()`

## `./tests/test_fem_unit.py`
### Classes
- **TestFEM**
  - `test_fem_laplace_linear()`
  - `test_fem_poisson_constant_source()`
  - `test_fem_hole_masking()`
  - `test_fem_output_format()`

## `./tests/test_geometry.py`
### Classes
- **TestGeometry**
  - `test_polygon_domain_is_inside()`
  - `test_polygon_domain_sampling()`
  - `test_sample_boundary_vertex_pinning()`
  - `test_non_convex_sampling_success()`
  - `test_custom_bc_fn_mapping()`
  - `test_koch_snowflake_generation()`
  - `test_koch_snowflake_domain()`
  - `test_multi_hole_id_assignment()`
  - `test_polygon_device_consistency()`
  - `test_bc_fn_signature_detection()`

## `./tests/test_integration.py`
### Classes
- **TestIntegration**
  - `test_gradient_flow()`
  - `test_full_training_cycle_adam()`
  - `test_lbfgs_closure_integration()`
  - `test_plotting_workflow()`
  - `test_range_loss_integration()`
  - `test_polygon_integration_training()`

## `./tests/test_model.py`
### Classes
- **TestModel**
  - `test_model_output_shape()`
  - `test_activation_differentiability()`
  - `test_model_serialization()`
  - `test_initialization_sanity()`
  - `test_gradient_finiteness()`
  - `test_parameter_scaling()`
  - `test_model_device_movement()`
  - `test_forward_pass_different_batch_sizes()`
  - `test_eval_mode_consistency()`
  - `test_siren_activation_support()`
  - `test_siren_initialization_scaling()`
  - `test_activation_selection()`
  - `test_siren_first_layer_init()`
  - `test_multi_frequency_distribution()`
  - `test_omega_buffer_device_transfer()`
  - `test_siren_gradient_stability_extreme_omega()`
  - `test_adaptive_activations_optimization()`
  - `test_fourier_feature_mapping_projection()`
  - `test_residual_skip_connections()`

## `./tests/test_model_advanced.py`
### Classes
- **TestModelAdvanced**
  - `test_sine_activation_adaptive()`
  - `test_pinn_multi_frequency_init()`
  - `test_pinn_siren_initialization_scaling()`
  - `test_pinn_tanh_fallback()`
  - `test_fourier_features()`
  - `test_output_transform()`
  - `test_residual_connections()`

## `./tests/test_modern_integration.py`
### Classes
- **TestModernFeaturesIntegration**
  - `test_train_with_sobolev_and_multiscale()`
  - `test_sobolev_config_toggle()`

## `./tests/test_multiscale_fourier.py`
### Classes
- **TestMultiScaleFourier**
  - `test_multiscale_initialization()`
  - `test_mapping_dimension_logic()`
  - `test_forward_pass_multiscale()`
  - `test_fourier_features_disabled()`

## `./tests/test_physics.py`
### Classes
- **TestPhysics**
  - `test_laplace_loss_zeros()`
  - `test_laplace_loss_linear_sloped()`
  - `test_laplace_loss_quad_identity()`
  - `test_laplace_loss_saddle()`
  - `test_boundary_loss_perfect_fit()`
  - `test_laplace_harmonic_function()`
  - `test_hessian_independence()`
  - `test_physics_batch_consistency()`
  - `test_numerical_stability_small_values()`
  - `test_laplace_loss_large_coordinates()`
  - `test_poisson_loss_with_source()`
  - `test_poisson_sine_source()`
  - `test_poisson_variable_source()`
  - `test_range_loss_zeros()`
  - `test_range_loss_active()`
  - `test_combined_poisson_and_range_loss()`
  - `test_range_loss_gradients()`
  - `test_poisson_source_shape_robustness()`
  - `test_poisson_with_multi_frequency_model()`

## `./tests/test_physics_advanced.py`
### Classes
- **TestPhysicsAdvanced**
  - `test_range_loss_behavior()`
  - `test_poisson_loss_known_solution()`
  - `test_grad_nan_safety()`
  - `test_boundary_gradient_loss()`

## `./tests/test_physics_stability.py`
### Classes
- **TestPhysicsStability**
  - `test_range_loss_gradient_direction()`
  - `test_poisson_residue_high_frequency_source()`
  - `test_adaptive_weight_warmup_bounds()`
  - `test_gradient_consistency_with_no_source()`

## `./tests/test_sobolev_physics.py`
### Classes
- **TestSobolevPhysics**
  - `test_sobolev_loss_harmonic()`
  - `test_sobolev_gradient_flow()`
  - `test_sobolev_vs_standard_laplace()`

## `./tests/test_solver.py`
### Classes
- **TestSolver**
  - `test_adaptive_weight_clamping()`
  - `test_train_config_patience()`
  - `test_adaptive_sampling_rar()`

## `./tests/test_solver_advanced.py`
### Classes
- **TestSolverAdvanced**
  - `test_history_tracking_adam()`
  - `test_history_tracking_lbfgs()`
  - `test_lbfgs_point_regeneration()`
  - `test_adaptive_weight_clamping()`

## `./tests/test_viz.py`
### Classes
- **TestViz**
  - `test_plot_results_smoke()`
  - `test_plot_results_masking_robustness()`

## `src/__init__.py`
## `src/core/data.py`
### Functions
- `set_seed()`
- `get_device()`
- `generate_domain_data()`
- `generate_adaptive_domain_data()`
- `generate_boundary_data()`

## `src/core/geometry.py`
### Classes
- **PolygonDomain**
  - `__init__()`
  - `_calculate_area()`
  - `sample_curvature_biased_interior()`
  - `sample_curvature_biased_boundary()`
  - `is_inside()`
  - `sample_interior()`
  - `sample_boundary()`
  - `exact_distance()`

### Functions
- `generate_koch_snowflake()`

## `src/core/viz.py`
### Functions
- `plot_results()`
- `plot_history()`
- `plot_comparison()`

## `src/fem/__init__.py`
## `src/fem/solver.py`
### Functions
- `solve_fem()`

## `src/pinn/model.py`
### Classes
- **Sine**
  - `__init__()`
  - `forward()`
- **PINN**
  - `__init__()`
  - `_init_siren()`
  - `forward()`
  - `device()`
- **ExactBoundaryAnsatz**
  - `__init__()`
  - `forward()`
  - `device()`

## `src/pinn/physics.py`
### Functions
- `poisson_loss()`
- `sobolev_laplace_loss()`
- `laplace_loss()`
- `boundary_loss()`
- `boundary_gradient_loss()`
- `range_loss()`
- `energy_loss()`

## `src/pinn/solver.py`
### Functions
- `train()`

## `tests/__init__.py`
## `tests/base_test.py`
### Classes
- **PINNTestCase**
  - `setUp()`
  - `assertTensorsEqual()`
  - `assertTensorFinite()`

## `tests/test_advanced_robustness.py`
### Classes
- **TestAdvancedRobustness**
  - `test_multiscale_spectral_capacity()`
  - `test_adaptive_weights_sobolev_stability()`
  - `test_distance_gradient_at_sharp_vertex()`
  - `test_sobolev_memory_pressure()`

## `tests/test_ansatz.py`
### Classes
- **TestAnsatz**
  - `setUp()`
  - `test_ansatz_outer_boundary()`
  - `test_ansatz_inner_boundary()`
  - `test_ansatz_interior_gradients()`
  - `test_ansatz_invalid_mode()`

## `tests/test_consistency.py`
### Classes
- **TestConsistency**
  - `test_reproducibility()`
  - `test_solver_no_data_crash()`
  - `test_spectral_omega_scalar_vs_tuple()`
  - `test_adaptive_weights_sanity()`
  - `test_multi_hole_geometry_overlap()`

## `tests/test_coverage_sweep.py`
### Classes
- **TestCoverageSweep**
  - `test_solver_full_adaptive_path()`
  - `test_geometry_remainder_logic()`
  - `test_physics_poisson_none_source()`
  - `test_data_device_fallbacks()`

## `tests/test_data.py`
### Classes
- **TestUtils**
  - `test_domain_sampling()`
  - `test_domain_sampling_count()`
  - `test_boundary_sampling_count()`
  - `test_boundary_sampling()`
  - `test_reproducibility()`
  - `test_boundary_values_range()`
  - `test_boundary_distribution()`
  - `test_reproducibility_with_polygon()`
  - `test_domain_default_fallback()`
  - `test_adaptive_sampling_rar()`
  - `test_adaptive_sampling_determinism()`
  - `test_adaptive_sampling_error_targeting()`
  - `test_adaptive_sampling_range_targeting()`

## `tests/test_energy_physics.py`
### Classes
- **TestEnergyFormulation**
  - `test_energy_loss_const()`
  - `test_energy_loss_linear()`
  - `test_area_calculation()`

## `tests/test_fem.py`
### Classes
- **TestFEMComparison**
  - `test_fem_square_laplace()`
  - `test_fem_snowflake_harmonic()`

## `tests/test_fem_unit.py`
### Classes
- **TestFEM**
  - `test_fem_laplace_linear()`
  - `test_fem_poisson_constant_source()`
  - `test_fem_hole_masking()`
  - `test_fem_output_format()`

## `tests/test_geometry.py`
### Classes
- **TestGeometry**
  - `test_polygon_domain_is_inside()`
  - `test_polygon_domain_sampling()`
  - `test_sample_boundary_vertex_pinning()`
  - `test_non_convex_sampling_success()`
  - `test_custom_bc_fn_mapping()`
  - `test_koch_snowflake_generation()`
  - `test_koch_snowflake_domain()`
  - `test_multi_hole_id_assignment()`
  - `test_polygon_device_consistency()`
  - `test_bc_fn_signature_detection()`

## `tests/test_integration.py`
### Classes
- **TestIntegration**
  - `test_gradient_flow()`
  - `test_full_training_cycle_adam()`
  - `test_lbfgs_closure_integration()`
  - `test_plotting_workflow()`
  - `test_range_loss_integration()`
  - `test_polygon_integration_training()`

## `tests/test_model.py`
### Classes
- **TestModel**
  - `test_model_output_shape()`
  - `test_activation_differentiability()`
  - `test_model_serialization()`
  - `test_initialization_sanity()`
  - `test_gradient_finiteness()`
  - `test_parameter_scaling()`
  - `test_model_device_movement()`
  - `test_forward_pass_different_batch_sizes()`
  - `test_eval_mode_consistency()`
  - `test_siren_activation_support()`
  - `test_siren_initialization_scaling()`
  - `test_activation_selection()`
  - `test_siren_first_layer_init()`
  - `test_multi_frequency_distribution()`
  - `test_omega_buffer_device_transfer()`
  - `test_siren_gradient_stability_extreme_omega()`
  - `test_adaptive_activations_optimization()`
  - `test_fourier_feature_mapping_projection()`
  - `test_residual_skip_connections()`

## `tests/test_model_advanced.py`
### Classes
- **TestModelAdvanced**
  - `test_sine_activation_adaptive()`
  - `test_pinn_multi_frequency_init()`
  - `test_pinn_siren_initialization_scaling()`
  - `test_pinn_tanh_fallback()`
  - `test_fourier_features()`
  - `test_output_transform()`
  - `test_residual_connections()`

## `tests/test_modern_integration.py`
### Classes
- **TestModernFeaturesIntegration**
  - `test_train_with_sobolev_and_multiscale()`
  - `test_sobolev_config_toggle()`

## `tests/test_multiscale_fourier.py`
### Classes
- **TestMultiScaleFourier**
  - `test_multiscale_initialization()`
  - `test_mapping_dimension_logic()`
  - `test_forward_pass_multiscale()`
  - `test_fourier_features_disabled()`

## `tests/test_physics.py`
### Classes
- **TestPhysics**
  - `test_laplace_loss_zeros()`
  - `test_laplace_loss_linear_sloped()`
  - `test_laplace_loss_quad_identity()`
  - `test_laplace_loss_saddle()`
  - `test_boundary_loss_perfect_fit()`
  - `test_laplace_harmonic_function()`
  - `test_hessian_independence()`
  - `test_physics_batch_consistency()`
  - `test_numerical_stability_small_values()`
  - `test_laplace_loss_large_coordinates()`
  - `test_poisson_loss_with_source()`
  - `test_poisson_sine_source()`
  - `test_poisson_variable_source()`
  - `test_range_loss_zeros()`
  - `test_range_loss_active()`
  - `test_combined_poisson_and_range_loss()`
  - `test_range_loss_gradients()`
  - `test_poisson_source_shape_robustness()`
  - `test_poisson_with_multi_frequency_model()`

## `tests/test_physics_advanced.py`
### Classes
- **TestPhysicsAdvanced**
  - `test_range_loss_behavior()`
  - `test_poisson_loss_known_solution()`
  - `test_grad_nan_safety()`
  - `test_boundary_gradient_loss()`

## `tests/test_physics_stability.py`
### Classes
- **TestPhysicsStability**
  - `test_range_loss_gradient_direction()`
  - `test_poisson_residue_high_frequency_source()`
  - `test_adaptive_weight_warmup_bounds()`
  - `test_gradient_consistency_with_no_source()`

## `tests/test_sobolev_physics.py`
### Classes
- **TestSobolevPhysics**
  - `test_sobolev_loss_harmonic()`
  - `test_sobolev_gradient_flow()`
  - `test_sobolev_vs_standard_laplace()`

## `tests/test_solver.py`
### Classes
- **TestSolver**
  - `test_adaptive_weight_clamping()`
  - `test_train_config_patience()`
  - `test_adaptive_sampling_rar()`

## `tests/test_solver_advanced.py`
### Classes
- **TestSolverAdvanced**
  - `test_history_tracking_adam()`
  - `test_history_tracking_lbfgs()`
  - `test_lbfgs_point_regeneration()`
  - `test_adaptive_weight_clamping()`

## `tests/test_viz.py`
### Classes
- **TestViz**
  - `test_plot_results_smoke()`
  - `test_plot_results_masking_robustness()`

