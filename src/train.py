"""
Training Pipeline for BEAM-Net
=================================
Handles the full training loop with:
  - MLflow experiment tracking (metrics, params, artifacts)
  - Learning rate scheduling (cosine with warmup)
  - Validation monitoring with early stopping
  - Uncertainty calibration tracking

The training objective minimizes the BEAM-Net energy functional (Eq. 19)
decomposed into task loss + KL complexity + sparsity penalty.

STDP-compatible weight updates (§6.3, Eq. 26) are applied alongside
gradient-based optimization as a biological regularizer.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional

from src.config import load_config
from src.beam_net import BEAMNet
from src.data_loader import get_dataloaders, preprocess_for_beam_net
from src.utils import set_seed, get_device, compute_ece
from src.parquet_logger import BeamNetParquetLogger


def train(config_path: str = "configs/experiment.yaml"):
    """
    Full training pipeline for BEAM-Net.
    
    Steps:
      1. Load config and initialize MLflow
      2. Create dataloaders
      3. Build model
      4. Train with validation monitoring
      5. Log final metrics and save model
    """
    cfg = load_config(config_path)
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["experiment"]["device"])
    
    # ---- MLflow setup ----
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment(cfg["experiment"]["name"])
        mlflow.start_run()
        mlflow.log_params(_flatten_dict(cfg))
        use_mlflow = True
    except Exception as e:
        print(f"[Train] MLflow unavailable ({e}), logging to console only")
        use_mlflow = False
    
    # ---- Parquet logger setup (lakehouse pattern) ----
    try:
        # Use MLflow run_id as experiment_id for cross-system joins
        mlflow_run_id = mlflow.active_run().info.run_id if use_mlflow else "standalone"
        # Hash to int for Parquet partition (Parquet prefers numeric partition keys)
        experiment_id = abs(hash(mlflow_run_id)) % (10 ** 8)
        
        parquet_logger = BeamNetParquetLogger(
            experiment_id=experiment_id,
            dataset=cfg["data"]["dataset"],
            model_variant="beam_net",
        )
        use_parquet = True
        print(f"[Train] Parquet logger initialized, experiment_id={experiment_id}")
    except Exception as e:
        print(f"[Train] Parquet logger unavailable ({e}), skipping high-volume logging")
        use_parquet = False
        parquet_logger = None
    
    # ---- Data ----
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    
    # Determine input dimensionality from first batch
    sample_batch = next(iter(train_loader))
    sample_x = preprocess_for_beam_net(sample_batch[0], cfg)
    d_input = sample_x.shape[-1]
    n_classes = len(set(sample_batch[1].numpy())) if hasattr(sample_batch[1], 'numpy') else 10
    
    print(f"[Train] d_input={d_input}, n_classes={n_classes}, device={device}")
    
    # ---- Model ----
    model = BEAMNet(
        d_input=d_input,
        n_classes=n_classes,
        cfg=cfg,
        mode="classification",
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[Train] Model parameters: {param_count:,}")
    
    # ---- Optimizer & Scheduler ----
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"] - cfg["training"]["warmup_epochs"],
    )
    
    # ---- Training loop ----
    best_val_acc = 0.0
    best_model_path = os.path.join(cfg["report"]["output_dir"], "best_model.pt")
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_ece": [],
               "sparsity": [], "epistemic_unc": [], "aleatoric_unc": []}
    
    for epoch in range(cfg["training"]["epochs"]):
        t0 = time.time()
        
        # ---- Train epoch ----
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        n_batches = 0
        
        for batch_idx, (x_raw, target) in enumerate(tqdm(
            train_loader, desc=f"Epoch {epoch+1}", leave=False
        )):
            x = preprocess_for_beam_net(x_raw, cfg).to(device)
            target = target.to(device)
            
            # Forward
            result, info = model(x)
            loss, loss_dict = model.compute_loss(result, target, info)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            epoch_loss += loss_dict["total_loss"]
            for k, v in loss_dict.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            n_batches += 1
        
        # Average metrics
        epoch_loss /= n_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= n_batches
        
        # Warmup scheduler
        if epoch >= cfg["training"]["warmup_epochs"]:
            scheduler.step()
        
        # ---- Validation ----
        val_loss, val_acc, val_ece, val_sparsity, val_epi, val_ale = evaluate_epoch(
            model, val_loader, cfg, device
        )
        
        elapsed = time.time() - t0
        
        # ---- Logging ----
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} | "
              f"Train loss: {epoch_loss:.4f} | Val loss: {val_loss:.4f} | "
              f"Val acc: {val_acc:.4f} | Val ECE: {val_ece:.4f} | "
              f"Sparsity: {val_sparsity:.4f} | Time: {elapsed:.1f}s")
        
        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_ece"].append(val_ece)
        history["sparsity"].append(val_sparsity)
        history["epistemic_unc"].append(val_epi)
        history["aleatoric_unc"].append(val_ale)
        
        if use_mlflow:
            mlflow.log_metrics({
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_ece": val_ece,
                "val_sparsity": val_sparsity,
                "epistemic_uncertainty": val_epi,
                "aleatoric_uncertainty": val_ale,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=epoch)
        
        # ---- Save best model ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "cfg": cfg,
            }, best_model_path)
    
    # ---- Final test evaluation ----
    # ---- Final test evaluation ----
    model.load_state_dict(torch.load(best_model_path)["model_state_dict"])
    test_loss, test_acc, test_ece, test_sparsity, test_epi, test_ale = evaluate_epoch(
        model, test_loader, cfg, device,
        parquet_logger=parquet_logger if use_parquet else None,  # ← ADD THIS
    )
    
    print(f"\n{'='*60}")
    print(f"Final Test Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  ECE:       {test_ece:.4f}")
    print(f"  Sparsity:  {test_sparsity:.4f}")
    print(f"  Epistemic: {test_epi:.4f}")
    print(f"  Aleatoric: {test_ale:.4f}")
    print(f"{'='*60}\n")
    
    if use_mlflow:
        mlflow.log_metrics({
            "test_accuracy": test_acc,
            "test_ece": test_ece,
            "test_sparsity": test_sparsity,
        })
        mlflow.log_artifact(best_model_path)
        mlflow.end_run()
    
    # Save history for report generation
    import json
    hist_path = os.path.join(cfg["report"]["output_dir"], "training_history.json")
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    with open(hist_path, "w") as f:
        json.dump(history, f)
    
    return history, {
        "test_accuracy": test_acc,
        "test_ece": test_ece,
        "test_sparsity": test_sparsity,
        "test_epistemic": test_epi,
        "test_aleatoric": test_ale,
    }


def evaluate_epoch(model, loader, cfg, device, parquet_logger=None):
    """Run evaluation on a dataloader, computing all BEAM-Net metrics.
    
    If parquet_logger is provided, per-sample predictions are written
    to Parquet for post-hoc analysis (reliability diagrams, uncertainty
    correlation studies, etc.).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_targets = []
    
    # Per-sample tracking for Parquet logging
    all_sample_ids = []
    all_true_labels = []
    all_pred_labels = []
    all_max_probs = []
    all_epistemic = []
    all_aleatoric = []
    
    sparsity_acc = 0.0
    epi_acc = 0.0
    ale_acc = 0.0
    n_batches = 0
    sample_counter = 0
    
    with torch.no_grad():
        for x_raw, target in loader:
            x = preprocess_for_beam_net(x_raw, cfg).to(device)
            target = target.to(device)
            
            result, info = model(x)
            loss, loss_dict = model.compute_loss(result, target, info)
            total_loss += loss_dict["total_loss"]
            
            # Accuracy
            preds = result.argmax(dim=-1)
            correct += (preds == target).sum().item()
            total += target.shape[0]
            
            # For ECE
            probs = torch.softmax(result, dim=-1)
            all_probs.append(probs.cpu())
            all_targets.append(target.cpu())
            
            # Per-sample data for Parquet
            batch_size = target.shape[0]
            all_sample_ids.extend(range(sample_counter, sample_counter + batch_size))
            all_true_labels.extend(target.cpu().numpy().tolist())
            all_pred_labels.extend(preds.cpu().numpy().tolist())
            all_max_probs.extend(probs.max(dim=-1)[0].cpu().numpy().tolist())
            all_epistemic.extend(info["epistemic_uncertainty"].cpu().numpy().tolist())
            all_aleatoric.extend(info["aleatoric_uncertainty"].cpu().numpy().tolist())
            sample_counter += batch_size
            
            sparsity_acc += info["sparsity"].mean().item()
            epi_acc += info["epistemic_uncertainty"].mean().item()
            ale_acc += info["aleatoric_uncertainty"].mean().item()
            n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    accuracy = correct / max(total, 1)
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    ece = compute_ece(all_probs, all_targets, n_bins=15)
    sparsity = sparsity_acc / max(n_batches, 1)
    epi = epi_acc / max(n_batches, 1)
    ale = ale_acc / max(n_batches, 1)
    
    # Write per-sample predictions to Parquet (if logger available)
    if parquet_logger is not None:
        try:
            import numpy as np
            parquet_logger.log_test_predictions(
                sample_ids=np.array(all_sample_ids),
                true_labels=np.array(all_true_labels),
                predicted_labels=np.array(all_pred_labels),
                max_probs=np.array(all_max_probs),
                epistemic_unc=np.array(all_epistemic),
                aleatoric_unc=np.array(all_aleatoric),
            )
        except Exception as e:
            print(f"[Train] Parquet write failed (non-fatal): {e}")
    
    return avg_loss, accuracy, ece, sparsity, epi, ale


def _flatten_dict(d, parent_key="", sep="."):
    """Flatten nested dict for MLflow param logging."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":
    train()