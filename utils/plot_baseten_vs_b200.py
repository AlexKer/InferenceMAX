#!/usr/bin/env python3
"""
Compare Baseten benchmark results with InferenceMAX B200 TensorRT results.
Plots: Token Throughput per GPU vs End-to-end Latency

Usage: 
    python3 utils/plot_baseten_vs_b200.py <baseten_results_dir> <inferencemax_results_json>
    
Example:
    python3 utils/plot_baseten_vs_b200.py baseten_comparison_results_20241104_120000 agg_gptoss_1k1k.json
"""

import sys
import json
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from collections import defaultdict


def load_baseten_results(results_dir):
    """Load Baseten benchmark results from a directory."""
    results = []
    for json_file in Path(results_dir).glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
                # Extract concurrency from metadata or filename
                conc = result.get('max_concurrency', None)
                if conc is None:
                    # Try to extract from filename
                    import re
                    match = re.search(r'conc(\d+)', json_file.stem)
                    if match:
                        conc = int(match.group(1))
                
                if conc is not None:
                    result['concurrency'] = conc
                    results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    
    # Sort by concurrency
    results.sort(key=lambda x: x.get('concurrency', 0))
    return results


def load_b200_trt_results(json_file):
    """Load B200 TensorRT results from InferenceMAX aggregated JSON."""
    with open(json_file, 'r') as f:
        all_data = json.load(f)
    
    # Filter for B200 TensorRT (b200-trt, b200-nvs, or b200 with trt framework)
    # Note: hw comes from RUNNER_TYPE env var, which is 'b200-trt' for B200 TRT runs
    b200_trt = [
        r for r in all_data 
        if (r.get('hw') == 'b200-trt' or r.get('hw') == 'b200-nvs' or 
            (r.get('hw') == 'b200' and r.get('framework') == 'trt'))
    ]
    
    print(f"Loaded {len(b200_trt)} B200 TensorRT data points")
    
    # Group by concurrency to find TP8 configuration per concurrency (matching Baseten's 8 GPUs)
    # If TP8 doesn't exist for a concurrency, fall back to best TP available
    best_by_conc = {}
    for r in b200_trt:
        conc = r.get('conc', 0)
        tp = r.get('tp', 0)
        tput_per_gpu = r.get('tput_per_gpu', 0)
        
        # Prefer TP8 to match Baseten's 8 GPU setup
        if conc not in best_by_conc:
            best_by_conc[conc] = r
        elif tp == 8:
            # Always prefer TP8 if available
            best_by_conc[conc] = r
        elif best_by_conc[conc].get('tp', 0) != 8 and tput_per_gpu > best_by_conc[conc]['tput_per_gpu']:
            # Fall back to best TP if TP8 not available
            best_by_conc[conc] = r
    
    # Also collect all data points for reference
    return sorted(best_by_conc.values(), key=lambda x: x.get('conc', 0)), b200_trt


def plot_comparison(baseten_results, b200_best, b200_all, output_file):
    """
    Plot Token Throughput per GPU vs End-to-end Latency.
    
    For B200 TRT: Uses tput_per_gpu (already normalized by TP size)
    For Baseten: Normalizes total throughput by 8 GPUs (matching TP8 setup)
                 to compare per-GPU performance fairly.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract B200 TRT data (best configuration per concurrency)
    b200_concs = []
    b200_tputs = []
    b200_e2els = []
    b200_labels = []
    
    for r in b200_best:
        conc = r.get('conc', 0)
        tput_per_gpu = r.get('tput_per_gpu', 0)
        e2el = r.get('median_e2el', 0)
        tp = r.get('tp', 0)
        
        # For fair comparison: use per-GPU when TP=8 (matches Baseten), 
        # use total throughput when TP != 8 (different GPU counts)
        if tp == 8:
            # TP8: compare per-GPU (both use 8 GPUs)
            tput_value = tput_per_gpu
            label_suffix = f'TP{tp} (per GPU)'
        else:
            # TP != 8: compare total throughput (different GPU counts)
            tput_value = tput_per_gpu * tp
            label_suffix = f'TP{tp} (total)'
        
        b200_concs.append(conc)
        b200_tputs.append(tput_value)
        b200_e2els.append(e2el)
        b200_labels.append(label_suffix)
    
    # Extract Baseten data
    baseten_concs = []
    baseten_tputs = []
    baseten_e2els = []
    
    for result in baseten_results:
        conc = result.get('concurrency', 0)
        if conc == 0:
            continue
        
        # Filter out unreliable results (too few completed requests)
        completed = result.get('completed', 0)
        total_requests = result.get('total_requests', result.get('num_prompts', completed))
        
        # Require at least 50% completion rate and at least 10 completed requests
        if completed < 10 or (total_requests > 0 and completed / total_requests < 0.5):
            print(f"  Skipping CONC={conc}: only {completed}/{total_requests} requests completed (unreliable)")
            continue
        
        # Baseten throughput: use per-GPU for TP8 comparison, total for other TP
        BASETEN_GPU_COUNT = 8
        total_tput = result.get('total_token_throughput', 0)
        e2el = result.get('median_e2el_ms', 0) / 1000.0  # Convert to seconds
        
        # Check if B200 TRT at this concurrency uses TP8
        b200_tp_at_conc = None
        for r in b200_best:
            if r.get('conc', 0) == conc:
                b200_tp_at_conc = r.get('tp', 0)
                break
        
        # Use per-GPU if B200 uses TP8, otherwise use total throughput
        if b200_tp_at_conc == 8:
            tput_value = total_tput / BASETEN_GPU_COUNT  # per-GPU for TP8 comparison
        else:
            tput_value = total_tput  # total throughput for different TP comparison
        
        baseten_concs.append(conc)
        baseten_tputs.append(tput_value)
        baseten_e2els.append(e2el)
    
    # Plot B200 TRT (per-GPU for TP8, total for other TP)
    if b200_tputs:
        ax.scatter(b200_e2els, b200_tputs, s=150, alpha=0.8, color='#2E7D32', 
                  marker='o', label='B200 TensorRT', edgecolors='black', linewidths=1.5)
        
        # Annotate B200 points with concurrency and TP
        for i, (e2el, tput, conc, label) in enumerate(zip(b200_e2els, b200_tputs, b200_concs, b200_labels)):
            ax.annotate(f'C{conc}\n{label}', (e2el, tput), 
                       xytext=(8, 8), textcoords='offset points', 
                       fontsize=9, ha='left', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='#2E7D32'))
    
    # Plot Baseten (per-GPU for TP8 comparison, total for other TP)
    if baseten_tputs:
        ax.scatter(baseten_e2els, baseten_tputs, s=150, alpha=0.8, color='#1976D2', 
                  marker='s', label='Baseten Optimized (8 GPUs)', edgecolors='black', linewidths=1.5)
        
        # Annotate Baseten points with concurrency
        for i, (e2el, tput, conc) in enumerate(zip(baseten_e2els, baseten_tputs, baseten_concs)):
            ax.annotate(f'C{conc}', (e2el, tput), 
                       xytext=(8, -15), textcoords='offset points', 
                       fontsize=9, ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='#1976D2'))
    
    # Connect points with lines for each series (sorted by concurrency, not latency)
    # This shows the progression as concurrency increases
    if len(b200_e2els) > 1:
        # Sort by concurrency to show progression
        sorted_b200 = sorted(zip(b200_concs, b200_e2els, b200_tputs))
        b200_conc_sorted, b200_e2el_sorted, b200_tput_sorted = zip(*sorted_b200)
        ax.plot(b200_e2el_sorted, b200_tput_sorted, '--', color='#2E7D32', alpha=0.3, linewidth=1.5)
    
    if len(baseten_e2els) > 1:
        # Sort by concurrency to show progression (not by latency)
        sorted_baseten = sorted(zip(baseten_concs, baseten_e2els, baseten_tputs))
        baseten_conc_sorted, baseten_e2el_sorted, baseten_tput_sorted = zip(*sorted_baseten)
        ax.plot(baseten_e2el_sorted, baseten_tput_sorted, '--', color='#1976D2', alpha=0.3, linewidth=1.5)
    
    ax.set_xlabel('End-to-End Latency (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Token Throughput (tok/s)', fontsize=12, fontweight='bold')
    ax.set_title('Token Throughput vs End-to-End Latency\nB200 TensorRT vs Baseten Optimized (GPT-OSS-120B, 1k/1k)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    # Add note about normalization and latency behavior
    note_text = ('Note: Fair comparison - per-GPU throughput for TP8 (CONC=4,8),\n'
                 'total throughput for different TP configurations (CONC=16,32,64).\n'
                 'Baseten uses 8 GPUs. B200 TRT uses TP8 where available, TP2/TP1 otherwise.')
    ax.text(0.02, 0.98, note_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set reasonable axis limits
    if b200_e2els or baseten_e2els:
        all_e2els = b200_e2els + baseten_e2els
        all_tputs = b200_tputs + baseten_tputs
        ax.set_xlim(left=0, right=max(all_e2els) * 1.1)
        ax.set_ylim(bottom=0, top=max(all_tputs) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_file}")
    plt.close()


def plot_comparison_with_all_b200_configs(baseten_results, b200_best, b200_all, output_file):
    """
    Plot comparison showing all B200 TRT configurations (not just best per concurrency).
    This gives more context about B200 performance across different TP configurations.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Extract all B200 TRT data points
    b200_all_e2els = []
    b200_all_tputs = []
    b200_all_concs = []
    b200_all_tps = []
    
    for r in b200_all:
        e2el = r.get('median_e2el', 0)
        tput_per_gpu = r.get('tput_per_gpu', 0)
        conc = r.get('conc', 0)
        tp = r.get('tp', 0)
        
        b200_all_e2els.append(e2el)
        b200_all_tputs.append(tput_per_gpu)
        b200_all_concs.append(conc)
        b200_all_tps.append(tp)
    
    # Extract Baseten data
    baseten_concs = []
    baseten_tputs = []
    baseten_e2els = []
    
    for result in baseten_results:
        conc = result.get('concurrency', 0)
        if conc == 0:
            continue
        
        # Filter out unreliable results (too few completed requests)
        completed = result.get('completed', 0)
        total_requests = result.get('total_requests', result.get('num_prompts', completed))
        
        # Require at least 50% completion rate and at least 10 completed requests
        if completed < 10 or (total_requests > 0 and completed / total_requests < 0.5):
            print(f"  Skipping CONC={conc}: only {completed}/{total_requests} requests completed (unreliable)")
            continue
        
        # Baseten throughput: use per-GPU for TP8 comparison, total for other TP
        BASETEN_GPU_COUNT = 8
        total_tput = result.get('total_token_throughput', 0)
        e2el = result.get('median_e2el_ms', 0) / 1000.0
        
        # Check if B200 TRT at this concurrency uses TP8
        b200_tp_at_conc = None
        for r in b200_best:
            if r.get('conc', 0) == conc:
                b200_tp_at_conc = r.get('tp', 0)
                break
        
        # Use per-GPU if B200 uses TP8, otherwise use total throughput
        if b200_tp_at_conc == 8:
            tput_value = total_tput / BASETEN_GPU_COUNT  # per-GPU for TP8 comparison
        else:
            tput_value = total_tput  # total throughput for different TP comparison
        
        baseten_concs.append(conc)
        baseten_tputs.append(tput_value)
        baseten_e2els.append(e2el)
    
    # Plot all B200 TRT configurations (smaller, lighter)
    if b200_all_tputs:
        # Group by TP for different colors/markers
        tp_markers = {1: 'o', 2: '^', 4: 's', 8: 'D'}
        tp_colors = {1: '#81C784', 2: '#66BB6A', 4: '#4CAF50', 8: '#2E7D32'}
        
        for tp in sorted(set(b200_all_tps)):
            tp_e2els = [e for e, t in zip(b200_all_e2els, b200_all_tps) if t == tp]
            tp_tputs = [t for t, tp_val in zip(b200_all_tputs, b200_all_tps) if tp_val == tp]
            
            ax.scatter(tp_e2els, tp_tputs, s=80, alpha=0.5, 
                      color=tp_colors.get(tp, '#2E7D32'),
                      marker=tp_markers.get(tp, 'o'),
                      label=f'B200 TRT TP{tp} (all)', edgecolors='black', linewidths=0.5)
        
        # Highlight best config per concurrency
        b200_best_e2els = [r.get('median_e2el', 0) for r in b200_best]
        b200_best_tputs = [r.get('tput_per_gpu', 0) for r in b200_best]
        b200_best_concs = [r.get('conc', 0) for r in b200_best]
        b200_best_tps = [r.get('tp', 0) for r in b200_best]
        
        ax.scatter(b200_best_e2els, b200_best_tputs, s=200, alpha=0.9, 
                  color='#1B5E20', marker='*', 
                  label='B200 TRT (best per CONC)', 
                  edgecolors='black', linewidths=2, zorder=10)
        
        # Annotate best points
        for e2el, tput, conc, tp in zip(b200_best_e2els, b200_best_tputs, b200_best_concs, b200_best_tps):
            ax.annotate(f'C{conc}\nTP{tp}', (e2el, tput), 
                       xytext=(10, 10), textcoords='offset points', 
                       fontsize=9, ha='left', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#1B5E20', linewidth=2))
    
    # Plot Baseten
    if baseten_tputs:
        ax.scatter(baseten_e2els, baseten_tputs, s=200, alpha=0.9, color='#1976D2', 
                  marker='s', label='Baseten Optimized', 
                  edgecolors='black', linewidths=2, zorder=10)
        
        # Annotate Baseten points
        for e2el, tput, conc in zip(baseten_e2els, baseten_tputs, baseten_concs):
            ax.annotate(f'C{conc}', (e2el, tput), 
                       xytext=(10, -20), textcoords='offset points', 
                       fontsize=10, ha='left', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='#1976D2', linewidth=2))
        
        # Connect Baseten points (sorted by concurrency)
        if len(baseten_e2els) > 1:
            # Sort by concurrency, not latency
            sorted_baseten = sorted(zip(baseten_concs, baseten_e2els, baseten_tputs))
            baseten_conc_sorted, baseten_e2el_sorted, baseten_tput_sorted = zip(*sorted_baseten)
            ax.plot(baseten_e2el_sorted, baseten_tput_sorted, '--', color='#1976D2', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('End-to-End Latency (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Token Throughput (tok/s)', fontsize=13, fontweight='bold')
    ax.set_title('Token Throughput vs End-to-End Latency\nB200 TensorRT (all configs) vs Baseten Optimized (GPT-OSS-120B, 1k/1k)', 
                 fontsize=15, fontweight='bold', pad=20)
    
    note_text = ('Note: Fair comparison - per-GPU throughput for TP8 (CONC=4,8),\n'
                 'total throughput for different TP configurations (CONC=16,32,64).\n'
                 'Baseten uses 8 GPUs. Best B200 config per concurrency marked with ★.')
    ax.text(0.02, 0.98, note_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top', 
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    
    ax.legend(loc='best', fontsize=10, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if b200_all_e2els or baseten_e2els:
        all_e2els = b200_all_e2els + baseten_e2els
        all_tputs = b200_all_tputs + baseten_tputs
        ax.set_xlim(left=0, right=max(all_e2els) * 1.1)
        ax.set_ylim(bottom=0, top=max(all_tputs) * 1.1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved detailed comparison plot: {output_file}")
    plt.close()


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 utils/plot_baseten_vs_b200.py <baseten_results_dir> <inferencemax_results_json>")
        print("\nExample:")
        print("  python3 utils/plot_baseten_vs_b200.py baseten_comparison_results_20241104_120000 agg_gptoss_1k1k.json")
        sys.exit(1)
    
    baseten_dir = Path(sys.argv[1])
    inferencemax_json = Path(sys.argv[2])
    
    if not baseten_dir.exists():
        print(f"Error: Baseten results directory not found: {baseten_dir}")
        sys.exit(1)
    
    if not inferencemax_json.exists():
        print(f"Error: InferenceMAX results file not found: {inferencemax_json}")
        sys.exit(1)
    
    # Load results
    print(f"Loading Baseten results from: {baseten_dir}")
    baseten_results = load_baseten_results(baseten_dir)
    print(f"  Found {len(baseten_results)} Baseten results")
    
    print(f"\nLoading B200 TRT results from: {inferencemax_json}")
    b200_best, b200_all = load_b200_trt_results(inferencemax_json)
    print(f"  Found {len(b200_best)} best configurations (one per concurrency)")
    print(f"  Found {len(b200_all)} total B200 TRT data points")
    
    if not baseten_results:
        print("Error: No Baseten results found!")
        sys.exit(1)
    
    if not b200_best:
        print("Error: No B200 TRT results found!")
        sys.exit(1)
    
    # Generate comparison plots
    output_dir = baseten_dir
    plot_comparison(baseten_results, b200_best, b200_all, 
                   output_dir / "baseten_vs_b200_comparison.png")
    plot_comparison_with_all_b200_configs(baseten_results, b200_best, b200_all,
                                          output_dir / "baseten_vs_b200_detailed.png")
    
    print(f"\n✓ Comparison plots generated successfully!")
    print(f"  - baseten_vs_b200_comparison.png (clean comparison)")
    print(f"  - baseten_vs_b200_detailed.png (with all B200 configs)")


if __name__ == '__main__':
    main()

