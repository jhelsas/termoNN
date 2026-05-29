# Project Symbol Map (AST Summary)

## `./comparison_results.py`
### Classes
- **FEMWrapper**
  - `__init__()`
  - `forward()`
  - `device()`

### Functions
- `get_nested_domain()`
- `bc_nested()`
- `generate_fem_solution()`
- `generate_pinn_solution()`
- `compare_solutions()`

## `./generate_ast_map.py`
### Functions
- `parse_file()`
- `generate_project_map()`
- `format_map()`

## `./main.py`
### Functions
- `solve_koch_snowflake_example()`
- `solve_nested_snowflakes_example()`

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
  - `is_inside()`
  - `sample_interior()`
  - `sample_boundary()`

### Functions
- `generate_koch_snowflake()`

## `./src/core/viz.py`
### Functions
- `plot_results()`

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

## `./src/pinn/physics.py`
### Functions
- `poisson_loss()`
- `laplace_loss()`
- `boundary_loss()`
- `boundary_gradient_loss()`
- `range_loss()`

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

## `./tests/test_consistency.py`
### Classes
- **TestConsistency**
  - `test_reproducibility()`
  - `test_solver_no_data_crash()`
  - `test_spectral_omega_scalar_vs_tuple()`
  - `test_adaptive_weights_sanity()`
  - `test_multi_hole_geometry_overlap()`

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
  - `test_polygon_boundary_sampling()`
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
  - `is_inside()`
  - `sample_interior()`
  - `sample_boundary()`

### Functions
- `generate_koch_snowflake()`

## `src/core/viz.py`
### Functions
- `plot_results()`

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

## `src/pinn/physics.py`
### Functions
- `poisson_loss()`
- `laplace_loss()`
- `boundary_loss()`
- `boundary_gradient_loss()`
- `range_loss()`

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

## `tests/test_consistency.py`
### Classes
- **TestConsistency**
  - `test_reproducibility()`
  - `test_solver_no_data_crash()`
  - `test_spectral_omega_scalar_vs_tuple()`
  - `test_adaptive_weights_sanity()`
  - `test_multi_hole_geometry_overlap()`

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
  - `test_polygon_boundary_sampling()`
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

