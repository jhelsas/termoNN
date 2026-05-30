import torch
import torch.optim as optim
from src.pinn.model import PINN, ExactBoundaryAnsatz
from src.pinn.physics import boundary_loss, poisson_loss, range_loss, boundary_gradient_loss, sobolev_laplace_loss, energy_loss
from src.core.data import generate_domain_data, generate_boundary_data, generate_adaptive_domain_data, set_seed, get_device

def train(domain=None, bc_fn=None, f_fn=None, config=None) -> tuple:
    """
    Trains the PINN model using a two-stage approach (Adam + L-BFGS).
    Includes self-adaptive weighting, weight warmup, and adaptive sampling.
    Returns: (model, history)
    """
    # Default configuration
    default_config = {
        "num_layers": 4,
        "hidden_dim": 20,
        "activation": "sine",
        "omega": 30.0,
        "adaptive_activations": False,
        "adam_epochs": 2000,
        "lbfgs_epochs": 500,
        "adam_lr": 0.001,
        "adam_points_domain": 2000,
        "adam_points_bc": 400,
        "lbfgs_points_domain": 3000,
        "lbfgs_points_bc": 600,
        "lambda_bc": 100.0,
        "lambda_grad_bc": 0.0,
        "lambda_range": 0.0,
        "use_adaptive_sampling": False,
        "adaptive_every": 100,
        "use_self_adaptive_weights": False,
        "adaptive_weight_every": 100,
        "seed": 42
    }
    
    if config:
        default_config.update(config)
    cfg = default_config

    # Sanitation
    required_keys = ["hidden_dim", "num_layers", "activation", "lambda_bc"]
    for key in required_keys:
        if key not in cfg: raise KeyError(f"Missing required config key: {key}")
    
    print(f"--- PINN Configuration ---")
    print(f"  - Model: {cfg['num_layers']} layers, {cfg['hidden_dim']} units, {cfg['activation']} activation")
    print(f"  - Omega: {cfg.get('omega', 30.0)}")
    print(f"  - Weighting: lambda_bc={cfg['lambda_bc']}, lambda_range={cfg.get('lambda_range', 0.0)}")

    set_seed(cfg["seed"])
    device = get_device()
    
    # History tracking
    history = {
        "adam": {"loss": [], "loss_pde": [], "loss_bc": [], "lambda_bc": [], "lambda_range": []},
        "lbfgs": {"loss": [], "loss_pde": [], "loss_bc": [], "loss_range": []}
    }
    
    model = PINN(
        hidden_dim=cfg["hidden_dim"], 
        num_layers=cfg["num_layers"],
        activation=cfg["activation"],
        omega=cfg["omega"],
        adaptive_activations=cfg.get("adaptive_activations", False),
        use_fourier_features=cfg.get("use_fourier_features", False),
        fourier_scale=cfg.get("fourier_scale", 10.0),
        output_transform=cfg.get("output_transform", None)
    ).to(device)
    
    # Wrap with Ansatz if requested
    if cfg.get("use_ansatz", False) and domain is not None:
        print("  - Ansatz: ENABLED (Exact Boundary Enforcement)")
        model = ExactBoundaryAnsatz(model, domain, bc_mode='nested').to(device)
        # If using Ansatz, boundary conditions are satisfied by construction.
        # We can set lambda_bc to 0 to remove it from the loss entirely.
        current_lambda_bc = 0.0
        cfg["lambda_bc"] = 0.0
    else:
        current_lambda_bc = cfg["lambda_bc"]
        
    current_lambda_range = cfg["lambda_range"]

    # Stage 1: Adam
    optimizer_adam = optim.Adam(model.parameters(), lr=cfg["adam_lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_adam, 'min', patience=200, factor=0.5)
    
    n_points_d = cfg.get("lbfgs_points_domain", 3000)
    n_points_b = cfg.get("lbfgs_points_bc", 600)
    
    x_domain, y_domain = generate_domain_data(n_points_d, device=device, domain=domain)

    print(f"--- Stage 1: Adam Optimization ({cfg['adam_epochs']} epochs) ---")
    for epoch in range(cfg["adam_epochs"]):
        # Self-Adaptive Weighting (Only if not using Ansatz)
        if cfg["use_self_adaptive_weights"] and not cfg.get("use_ansatz", False) and epoch > 0 and epoch % cfg["adaptive_weight_every"] == 0:
            with torch.set_grad_enabled(True):
                model.zero_grad()
                l_pde = poisson_loss(model, x_domain, y_domain, f_fn=f_fn)
                l_pde.backward(retain_graph=True)
                grad_pde = [p.grad.abs().max() for p in model.parameters() if p.grad is not None]
                max_grad_pde = torch.stack(grad_pde).max() if grad_pde else torch.tensor(1.0)

                model.zero_grad()
                x_bc, y_bc, u_bc, n_bc = generate_boundary_data(n_points_b, device=device, domain=domain, bc_fn=bc_fn)
                l_bc = boundary_loss(model, x_bc, y_bc, u_bc)
                l_bc.backward(retain_graph=True)
                grad_bc = [p.grad.abs().mean() for p in model.parameters() if p.grad is not None]
                mean_grad_bc = torch.stack(grad_bc).mean() if grad_bc else torch.tensor(1.0)
                
                target_lambda_bc = torch.clamp(max_grad_pde / (mean_grad_bc + 1e-8), max=2000.0)
                current_lambda_bc = 0.9 * current_lambda_bc + 0.1 * target_lambda_bc
                
                if current_lambda_range > 0:
                    model.zero_grad()
                    coords_d = torch.stack([x_domain, y_domain], dim=1)
                    l_r = range_loss(model(coords_d))
                    l_r.backward(retain_graph=True)
                    grad_r = [p.grad.abs().mean() for p in model.parameters() if p.grad is not None]
                    mean_grad_r = torch.stack(grad_r).mean() if grad_r else torch.tensor(1.0)
                    target_lambda_r = torch.clamp(max_grad_pde / (mean_grad_r + 1e-8), max=2000.0)
                    current_lambda_range = 0.9 * current_lambda_range + 0.1 * target_lambda_r

        # Resampling
        if cfg["use_adaptive_sampling"] and epoch % cfg["adaptive_every"] == 0:
            x_domain, y_domain = generate_adaptive_domain_data(model, n_points_d, device=device, domain=domain, f_fn=f_fn, config=cfg)
        elif not cfg["use_adaptive_sampling"] and epoch % 100 == 0:
            x_domain, y_domain = generate_domain_data(n_points_d, device=device, domain=domain)
            
        x_bc, y_bc, u_bc, n_bc = generate_boundary_data(n_points_b, device=device, domain=domain, bc_fn=bc_fn)
        
        optimizer_adam.zero_grad()
        if cfg.get("use_energy", False):
            # Energy formulation (Deep Ritz)
            loss_pde = energy_loss(model, x_domain, y_domain, area=domain.area if domain else 1.0, f_fn=f_fn)
        elif f_fn is None and cfg.get("use_sobolev", False):
            # Sobolev formulation
            h1_weight = cfg.get("sobolev_h1_weight", 1e-4)
            loss_pde = sobolev_laplace_loss(model, x_domain, y_domain, h1_weight=h1_weight)
        else:
            # Standard strong form
            loss_pde = poisson_loss(model, x_domain, y_domain, f_fn=f_fn)
        
        loss_bc = boundary_loss(model, x_bc, y_bc, u_bc)
        total_loss = loss_pde + current_lambda_bc * loss_bc
        
        if cfg.get("lambda_grad_bc", 0) > 0:
            total_loss += cfg["lambda_grad_bc"] * boundary_gradient_loss(model, x_bc, y_bc, n_bc[:, 0], n_bc[:, 1])

        if current_lambda_range > 0:
            coords_d = torch.stack([x_domain, y_domain], dim=1)
            total_loss += current_lambda_range * range_loss(model(coords_d))
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer_adam.step()
        scheduler.step(total_loss.detach())
        
        # Track history
        history["adam"]["loss"].append(total_loss.item())
        history["adam"]["loss_pde"].append(loss_pde.item())
        history["adam"]["loss_bc"].append(loss_bc.item())
        history["adam"]["lambda_bc"].append(float(current_lambda_bc))
        history["adam"]["lambda_range"].append(float(current_lambda_range))
        
        if epoch % 500 == 0:
            print(f"Adam Epoch {epoch:4d} | Loss: {total_loss.item():.6f} (PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f})")

    # Stage 2: L-BFGS
    print(f"--- Stage 2: L-BFGS Fine-tuning ---")
    
    # Regenerate collocation points for the high-persistence stage
    n_points_d_lbfgs = cfg.get("lbfgs_points_domain", 3000)
    n_points_b_lbfgs = cfg.get("lbfgs_points_bc", 600)
    
    x_domain, y_domain = generate_domain_data(n_points_d_lbfgs, device=device, domain=domain)
    x_bc, y_bc, u_bc, n_bc = generate_boundary_data(n_points_b_lbfgs, device=device, domain=domain, bc_fn=bc_fn)
    
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1, max_iter=40, tolerance_grad=1e-9, history_size=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer_lbfgs.zero_grad()
        if cfg.get("use_energy", False):
            l_pde = energy_loss(model, x_domain, y_domain, area=domain.area if domain else 1.0, f_fn=f_fn)
        elif f_fn is None and cfg.get("use_sobolev", False):
            h1_weight = cfg.get("sobolev_h1_weight", 1e-4)
            l_pde = sobolev_laplace_loss(model, x_domain, y_domain, h1_weight=h1_weight)
        else:
            l_pde = poisson_loss(model, x_domain, y_domain, f_fn=f_fn)
        l_bc = boundary_loss(model, x_bc, y_bc, u_bc)
        total_loss = l_pde + current_lambda_bc * l_bc
        
        if cfg.get("lambda_grad_bc", 0) > 0:
            total_loss += cfg["lambda_grad_bc"] * boundary_gradient_loss(model, x_bc, y_bc, n_bc[:, 0], n_bc[:, 1])

        l_r_val = 0.0
        if current_lambda_range > 0:
            coords_d, coords_b = torch.stack([x_domain, y_domain], dim=1), torch.stack([x_bc, y_bc], dim=1)
            l_r = range_loss(torch.cat([model(coords_d), model(coords_b)], dim=0))
            total_loss += current_lambda_range * l_r
            l_r_val = l_r.item()
        total_loss.backward()
        total_loss.pde, total_loss.bc, total_loss.range = l_pde.item(), l_bc.item(), l_r_val
        return total_loss

    for epoch in range(cfg["lbfgs_epochs"]):
        loss = optimizer_lbfgs.step(closure)
        
        # Track history
        history["lbfgs"]["loss"].append(loss.item())
        history["lbfgs"]["loss_pde"].append(loss.pde)
        history["lbfgs"]["loss_bc"].append(loss.bc)
        history["lbfgs"]["loss_range"].append(loss.range)
        
        if epoch % 100 == 0:
            print(f"L-BFGS Epoch {epoch:3d} | Loss: {loss.item():.8f} (PDE: {loss.pde:.8f}, BC: {loss.bc:.8f}, R: {loss.range:.8f})")
            
    return model, history
