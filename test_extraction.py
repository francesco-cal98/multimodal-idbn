#!/usr/bin/env python3
"""
Test script to verify the iDBN and iMDBN extraction is complete and functional.
"""

import sys
import torch
from pathlib import Path

# Add imdbn to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all module imports."""
    print("=" * 70)
    print("TEST 1: Module Imports")
    print("=" * 70)
    
    try:
        from imdbn.models import RBM, iDBN, iMDBN
        print("✓ Successfully imported RBM, iDBN, iMDBN from imdbn.models")
    except Exception as e:
        print(f"✗ Failed to import models: {e}")
        return False
    
    try:
        from imdbn.utils.wandb_utils import plot_2d_embedding_and_correlations
        print("✓ Successfully imported wandb_utils")
    except Exception as e:
        print(f"✗ Failed to import wandb_utils: {e}")
        return False
    
    try:
        from imdbn.utils.probe_utils import log_linear_probe
        print("✓ Successfully imported probe_utils")
    except Exception as e:
        print(f"✗ Failed to import probe_utils: {e}")
        return False
    
    print()
    return True


def test_rbm_instantiation():
    """Test RBM instantiation."""
    print("=" * 70)
    print("TEST 2: RBM Instantiation")
    print("=" * 70)
    
    try:
        from imdbn.models import RBM
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rbm = RBM(
            num_visible=100,
            num_hidden=50,
            learning_rate=0.1,
            weight_decay=0.0001,
            momentum=0.5,
        )
        rbm = rbm.to(device)
        
        # Test forward pass
        x = torch.randn(8, 100, device=device)
        h = rbm.forward(x)
        
        print(f"✓ RBM instantiated successfully")
        print(f"  - Visible units: {rbm.num_visible}")
        print(f"  - Hidden units: {rbm.num_hidden}")
        print(f"  - Weight shape: {rbm.W.shape}")
        print(f"  - Forward pass output shape: {h.shape}")
        print()
        return True
    except Exception as e:
        print(f"✗ RBM instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_idbn_instantiation():
    """Test iDBN instantiation (without data loaders)."""
    print("=" * 70)
    print("TEST 3: iDBN Instantiation")
    print("=" * 70)
    
    try:
        from imdbn.models import iDBN
        from torch.utils.data import DataLoader, TensorDataset
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create dummy data
        X = torch.randn(128, 784)
        y = torch.randint(0, 10, (128,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16)
        val_loader = DataLoader(dataset, batch_size=16)
        
        params = {
            "LEARNING_RATE": 0.1,
            "WEIGHT_PENALTY": 0.0001,
            "INIT_MOMENTUM": 0.5,
            "FINAL_MOMENTUM": 0.95,
            "LEARNING_RATE_DYNAMIC": True,
            "CD": 1,
            "SPARSITY": False,
            "SPARSITY_FACTOR": 0.1,
        }
        
        idbn = iDBN(
            layer_sizes=[784, 200, 100],
            params=params,
            dataloader=dataloader,
            val_loader=val_loader,
            device=device,
        )
        
        print(f"✓ iDBN instantiated successfully")
        print(f"  - Architecture: {idbn.arch_str}")
        print(f"  - Number of layers: {len(idbn.layers)}")
        print(f"  - Layer dimensions: {[idbn.layers[i].num_hidden for i in range(len(idbn.layers))]}")
        print()
        return True
    except Exception as e:
        print(f"✗ iDBN instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_imdbn_instantiation():
    """Test iMDBN instantiation (without data loaders)."""
    print("=" * 70)
    print("TEST 4: iMDBN Instantiation")
    print("=" * 70)
    
    try:
        from imdbn.models import iMDBN
        from torch.utils.data import DataLoader, TensorDataset
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create dummy data
        X = torch.randn(128, 784)
        y = torch.zeros(128, 10)
        y[torch.arange(128), torch.randint(0, 10, (128,))] = 1.0  # one-hot
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16)
        val_loader = DataLoader(dataset, batch_size=16)
        
        params = {
            "LEARNING_RATE": 0.1,
            "WEIGHT_PENALTY": 0.0001,
            "INIT_MOMENTUM": 0.5,
            "FINAL_MOMENTUM": 0.95,
            "LEARNING_RATE_DYNAMIC": True,
            "CD": 1,
            "JOINT_CD": 1,
            "JOINT_LEARNING_RATE": 0.1,
            "CROSS_GIBBS_STEPS": 10,
            "JOINT_AUX_COND_STEPS": 5,
        }
        
        imdbn = iMDBN(
            layer_sizes_img=[784, 200, 100],
            joint_layer_size=64,
            params=params,
            dataloader=dataloader,
            val_loader=val_loader,
            device=device,
            num_labels=10,
        )
        
        print(f"✓ iMDBN instantiated successfully")
        print(f"  - Architecture: {imdbn.arch_str}")
        print(f"  - Image layers: {len(imdbn.image_idbn.layers)}")
        print(f"  - Joint RBM visible units: {imdbn.joint_rbm.num_visible}")
        print(f"  - Joint RBM hidden units: {imdbn.joint_rbm.num_hidden}")
        print(f"  - Number of classes: {imdbn.num_labels}")
        print(f"  - Image latent dim: {imdbn.Dz_img}")
        print()
        return True
    except Exception as e:
        print(f"✗ iMDBN instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_methods():
    """Test key methods."""
    print("=" * 70)
    print("TEST 5: Key Methods")
    print("=" * 70)
    
    try:
        from imdbn.models import iDBN
        from torch.utils.data import DataLoader, TensorDataset
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create dummy data
        X = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=16)
        val_loader = DataLoader(dataset, batch_size=16)
        
        params = {
            "LEARNING_RATE": 0.1,
            "WEIGHT_PENALTY": 0.0001,
            "INIT_MOMENTUM": 0.5,
            "FINAL_MOMENTUM": 0.95,
            "LEARNING_RATE_DYNAMIC": True,
            "CD": 1,
        }
        
        idbn = iDBN(
            layer_sizes=[784, 200, 100],
            params=params,
            dataloader=dataloader,
            val_loader=val_loader,
            device=device,
        )
        
        # Test represent
        x_test = torch.randn(8, 784, device=device)
        z = idbn.represent(x_test)
        print(f"✓ represent() works: input {x_test.shape} -> output {z.shape}")
        
        # Test reconstruct
        x_rec = idbn.reconstruct(x_test)
        print(f"✓ reconstruct() works: input {x_test.shape} -> output {x_rec.shape}")
        
        # Test decode
        z_test = torch.randn(8, 100, device=device)
        x_dec = idbn.decode(z_test)
        print(f"✓ decode() works: input {z_test.shape} -> output {x_dec.shape}")
        
        print()
        return True
    except Exception as e:
        print(f"✗ Method testing failed: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("*" * 70)
    print("iDBN and iMDBN Extraction Verification Tests")
    print("*" * 70)
    print()
    
    results = []
    results.append(("Module Imports", test_imports()))
    results.append(("RBM Instantiation", test_rbm_instantiation()))
    results.append(("iDBN Instantiation", test_idbn_instantiation()))
    results.append(("iMDBN Instantiation", test_imdbn_instantiation()))
    results.append(("Key Methods", test_methods()))
    
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    print()
    if all_passed:
        print("✓ All tests passed! Extraction is complete and functional.")
    else:
        print("✗ Some tests failed. Please review the output above.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
