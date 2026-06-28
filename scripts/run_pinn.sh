#!/bin/bash
source venv/bin/activate
python3 comparison_results.py --step pinn > pinn_log.txt 2>&1
echo "PINN run finished. Check pinn_log.txt"
