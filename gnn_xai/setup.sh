#!/bin/bash
# setup.sh — GNN-XAI Framework (X-Node Edition)
set -e
echo "======================================================"
echo "  GNN-XAI Framework (X-Node Edition) — Setup"
echo "======================================================"
python3 --version
echo "Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo "Installing PyTorch Geometric..."
pip install torch-geometric
echo "Installing other packages..."
pip install numpy pandas matplotlib seaborn scikit-learn networkx scipy tqdm Pillow
echo ""
echo "Verifying..."
python3 -c "
import torch, torch_geometric, networkx, sklearn, matplotlib
print(f'  torch:           {torch.__version__}')
print(f'  torch_geometric: {torch_geometric.__version__}')
print(f'  networkx:        {networkx.__version__}')
print('  All OK!')
"
echo ""
echo "======================================================"
echo "  Quick test (2-3 min, X-Node only):"
echo "  python main.py --arch xnode --dataset ba2motifs \\"
echo "                 --epochs 50 --explain_n 10 --skip_posthoc"
echo ""
echo "  Full benchmark (25 min, all methods):"
echo "  python main.py --arch xnode --dataset ba2motifs \\"
echo "                 --epochs 100 --explain_n 20"
echo ""
echo "  MRI brain tumour demo:"
echo "  python main.py --arch xnode --dataset mri \\"
echo "                 --epochs 80 --explain_n 15 --skip_subgraphx"
echo "======================================================"
