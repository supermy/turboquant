#!/usr/bin/env python3
"""
PolarQuant Sequence Diagram Generator

Generate visual sequence diagrams using matplotlib
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_sequence_diagram():
    """Draw PolarQuant compression/decompression sequence diagram"""
    
    # Use English fonts only to avoid Chinese character issues
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Color definitions
    colors = {
        'user': '#E8F5E9',
        'config': '#E3F2FD',
        'pq': '#FFF3E0',
        'utils': '#F3E5F5',
        'cv': '#E0F2F1',
        'arrow': '#1976D2',
        'text': '#212121',
        'note': '#FFF9C4'
    }
    
    # Participant positions
    participants = {
        'User': (1, 'User/Caller'),
        'Config': (3, 'PolarQuantConfig\nConfig Class'),
        'PQ': (5.5, 'PolarQuant\nMain Class'),
        'Utils': (8, 'Utils Module\n(utils.py)'),
        'CV': (10.5, 'CompressedVector\nData Class')
    }
    
    # Draw participant boxes
    for name, (x, label) in participants.items():
        box = FancyBboxPatch((x-0.5, 12.5), 1.0, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=colors.get(name.lower(), '#FFFFFF'),
                            edgecolor='#333333',
                            linewidth=2)
        ax.add_patch(box)
        ax.text(x, 12.9, label, ha='center', va='center',
               fontsize=9, fontweight='bold', color=colors['text'])
        
        # Draw lifeline
        ax.plot([x, x], [0.5, 12.5], 'k--', alpha=0.3, linewidth=1)
    
    # Title
    ax.text(6, 13.8, 'PolarQuant Compression/Decompression Sequence Diagram',
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # ===== Initialization Phase =====
    y = 11.5
    ax.text(0.3, y, 'Phase 1: Initialization', fontsize=12, fontweight='bold', color='#1565C0')
    
    # 1. Create config
    y -= 0.6
    ax.annotate('', xy=(3, y), xytext=(1, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.text(2, y+0.15, 'Create Config', ha='center', fontsize=9)
    ax.text(2, y-0.18, 'dimension, bits...', ha='center', fontsize=8, style='italic')
    
    # 2. Initialize PolarQuant
    y -= 0.6
    ax.annotate('', xy=(5.5, y), xytext=(1, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.text(3.25, y+0.15, 'PolarQuant(config)', ha='center', fontsize=9)
    
    # 3. Generate rotation matrix
    y -= 0.6
    ax.annotate('', xy=(8, y), xytext=(5.5, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.text(6.75, y+0.15, 'random_orthogonal_matrix()', ha='center', fontsize=9)
    
    y -= 0.45
    ax.annotate('', xy=(5.5, y), xytext=(8, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5, linestyle='--'))
    ax.text(6.75, y+0.15, 'rotation_matrix Q (d x d)', ha='center', fontsize=9)
    
    # 4. Compute centroids
    y -= 0.6
    ax.annotate('', xy=(8, y), xytext=(5.5, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.text(6.75, y+0.15, 'compute_lloyd_max_centroids()', ha='center', fontsize=9)
    
    y -= 0.45
    ax.annotate('', xy=(5.5, y), xytext=(8, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5, linestyle='--'))
    ax.text(6.75, y+0.15, 'angle_centroids (2^bits)', ha='center', fontsize=9)
    
    # Return quantizer
    y -= 0.6
    ax.annotate('', xy=(1, y), xytext=(5.5, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5, linestyle='--'))
    ax.text(3.25, y+0.15, 'quantizer ready', ha='center', fontsize=9)
    
    # ===== Compression Phase =====
    y -= 0.9
    ax.text(0.3, y, 'Phase 2: Compression', fontsize=12, fontweight='bold', color='#1565C0')
    
    # 1. Call compress
    y -= 0.6
    ax.annotate('', xy=(5.5, y), xytext=(1, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.text(3.25, y+0.15, 'compress(x)', ha='center', fontsize=9)
    
    # 2. Random rotation
    y -= 0.6
    ax.annotate('', xy=(8, y), xytext=(5.5, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.text(6.75, y+0.15, '_rotate(x)', ha='center', fontsize=9)
    
    y -= 0.45
    ax.annotate('', xy=(5.5, y), xytext=(8, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5, linestyle='--'))
    ax.text(6.75, y+0.15, 'x_rotated = Q @ x', ha='center', fontsize=9)
    
    # 3. Polar transform
    y -= 0.6
    ax.annotate('', xy=(8, y), xytext=(5.5, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.text(6.75, y+0.15, 'cartesian_to_polar()', ha='center', fontsize=9)
    
    y -= 0.45
    ax.annotate('', xy=(5.5, y), xytext=(8, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5, linestyle='--'))
    ax.text(6.75, y+0.15, '(r, angles)', ha='center', fontsize=9)
    
    # 4. Radius quantization
    y -= 0.6
    ax.plot([5.5, 5.9], [y, y], color=colors['arrow'], lw=1.5)
    ax.text(6.2, y+0.15, '_quantize_radius() [log scale]', ha='center', fontsize=9)
    
    # 5. Angle quantization
    y -= 0.6
    ax.annotate('', xy=(8, y), xytext=(5.5, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.text(6.75, y+0.15, 'lloyd_max_quantize()', ha='center', fontsize=9)
    
    y -= 0.45
    ax.annotate('', xy=(5.5, y), xytext=(8, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5, linestyle='--'))
    ax.text(6.75, y+0.15, 'angle_indices', ha='center', fontsize=9)
    
    # 6. Create compressed vector
    y -= 0.6
    ax.annotate('', xy=(10.5, y), xytext=(5.5, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.text(8, y+0.15, 'CompressedVector()', ha='center', fontsize=9)
    
    y -= 0.45
    ax.annotate('', xy=(5.5, y), xytext=(10.5, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5, linestyle='--'))
    ax.text(8, y+0.15, 'compressed', ha='center', fontsize=9)
    
    # Return result
    y -= 0.6
    ax.annotate('', xy=(1, y), xytext=(5.5, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5, linestyle='--'))
    ax.text(3.25, y+0.15, 'compressed vector', ha='center', fontsize=9)
    
    # ===== Decompression Phase =====
    y -= 0.9
    ax.text(0.3, y, 'Phase 3: Decompression', fontsize=12, fontweight='bold', color='#1565C0')
    
    # 1. Call decompress
    y -= 0.6
    ax.annotate('', xy=(5.5, y), xytext=(1, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.text(3.25, y+0.15, 'decompress(compressed)', ha='center', fontsize=9)
    
    # 2. Dequantize radius
    y -= 0.6
    ax.plot([5.5, 5.9], [y, y], color=colors['arrow'], lw=1.5)
    ax.text(6.2, y+0.15, '_dequantize_radius()', ha='center', fontsize=9)
    
    # 3. Dequantize angles
    y -= 0.6
    ax.annotate('', xy=(8, y), xytext=(5.5, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.text(6.75, y+0.15, 'lloyd_max_dequantize()', ha='center', fontsize=9)
    
    y -= 0.45
    ax.annotate('', xy=(5.5, y), xytext=(8, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5, linestyle='--'))
    ax.text(6.75, y+0.15, 'angles', ha='center', fontsize=9)
    
    # 4. Polar to cartesian
    y -= 0.6
    ax.annotate('', xy=(8, y), xytext=(5.5, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.text(6.75, y+0.15, 'polar_to_cartesian()', ha='center', fontsize=9)
    
    y -= 0.45
    ax.annotate('', xy=(5.5, y), xytext=(8, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5, linestyle='--'))
    ax.text(6.75, y+0.15, 'x_rotated', ha='center', fontsize=9)
    
    # 5. Inverse rotation
    y -= 0.6
    ax.annotate('', xy=(8, y), xytext=(5.5, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5))
    ax.text(6.75, y+0.15, '_inverse_rotate()', ha='center', fontsize=9)
    
    y -= 0.45
    ax.annotate('', xy=(5.5, y), xytext=(8, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5, linestyle='--'))
    ax.text(6.75, y+0.15, 'x = Q^T @ x_rotated', ha='center', fontsize=9)
    
    # Return result
    y -= 0.6
    ax.annotate('', xy=(1, y), xytext=(5.5, y),
               arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=1.5, linestyle='--'))
    ax.text(3.25, y+0.15, 'x_reconstructed', ha='center', fontsize=9)
    
    # Add legend
    legend_y = 0.4
    ax.plot([1, 1.6], [legend_y, legend_y], '->', color=colors['arrow'], lw=1.5)
    ax.text(1.9, legend_y, 'Call', va='center', fontsize=9)
    
    ax.plot([3.5, 4.1], [legend_y, legend_y], '->', color=colors['arrow'], lw=1.5, linestyle='--')
    ax.text(4.4, legend_y, 'Return', va='center', fontsize=9)
    
    # Add Chinese annotations as separate text boxes
    annotation_text = """
    Chinese Annotations / 中文标注:
    
    Step 1 (步骤1): Random Rotation / 随机旋转
    - Purpose: Normalize distribution to Beta(d/2, d/2)
    - 目的: 将分布归一化为 Beta 分布
    
    Step 2 (步骤2): Polar Transform / 极坐标变换  
    - Convert Cartesian to (radius, angles)
    - 笛卡尔坐标转 (半径, 角度)
    
    Step 3a (步骤3a): Radius Quantization / 半径量化
    - Log-scale uniform quantization
    - 对数尺度均匀量化
    
    Step 3b (步骤3b): Angle Quantization / 角度量化
    - Lloyd-Max optimal quantization
    - Lloyd-Max 最优量化
    
    Compression Ratio / 压缩比:
    - Original: d x 32 bits (float32)
    - Compressed: r_bits + (d-1) x a_bits
    - Ratio: ~8x for typical settings
    """
    
    ax.text(0.3, -1.5, annotation_text, fontsize=8, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.8, pad=0.5))
    
    plt.tight_layout()
    plt.savefig('/Users/moyong/project/ai/turboquant/kimi25/polarquant/docs/sequence_diagram.png',
               dpi=150, bbox_inches='tight', facecolor='white')
    print("Sequence diagram saved to: docs/sequence_diagram.png")
    plt.show()


def draw_algorithm_flow():
    """Draw algorithm flowchart"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(6, 11.5, 'PolarQuant Algorithm Flowchart', ha='center', fontsize=18, fontweight='bold')
    ax.text(6, 11.0, '(PolarQuant 算法流程图)', ha='center', fontsize=12)
    
    # Box style
    box_style = dict(boxstyle='round,pad=0.6', facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2)
    arrow_style = dict(arrowstyle='->', color='#1976D2', lw=2.5)
    
    # Input
    ax.text(6, 10.0, 'Input Vector x in R^d\n(输入向量)', ha='center', fontsize=12, bbox=box_style)
    
    # Step 1: Random rotation
    ax.annotate('', xy=(6, 9.0), xytext=(6, 9.5), arrowprops=arrow_style)
    ax.text(6, 8.5, 'Step 1: Random Rotation\n(步骤1: 随机旋转)\ny = Q * x', 
           ha='center', fontsize=11, bbox=box_style)
    
    # Step 2: Polar transform
    ax.annotate('', xy=(6, 7.5), xytext=(6, 8.0), arrowprops=arrow_style)
    ax.text(6, 6.8, 'Step 2: Polar Transform\n(步骤2: 极坐标变换)\n(r, theta) = cart2pol(y)', 
           ha='center', fontsize=11, bbox=box_style)
    
    # Branch
    ax.annotate('', xy=(6, 5.8), xytext=(6, 6.2), arrowprops=arrow_style)
    
    # Left: Radius quantization
    ax.annotate('', xy=(3, 5.0), xytext=(6, 5.5), arrowprops=arrow_style)
    ax.text(3, 4.3, 'Step 3a: Radius Quantization\n(步骤3a: 半径量化)\nLog-scale\nr -> idx_r', 
           ha='center', fontsize=10, bbox=box_style)
    
    # Right: Angle quantization
    ax.annotate('', xy=(9, 5.0), xytext=(6, 5.5), arrowprops=arrow_style)
    ax.text(9, 4.3, 'Step 3b: Angle Quantization\n(步骤3b: 角度量化)\nLloyd-Max\ntheta -> idx_theta', 
           ha='center', fontsize=10, bbox=box_style)
    
    # Merge: Store indices
    ax.annotate('', xy=(6, 3.5), xytext=(3, 4.0), arrowprops=arrow_style)
    ax.annotate('', xy=(6, 3.5), xytext=(9, 4.0), arrowprops=arrow_style)
    ax.text(6, 2.8, 'Step 4: Store Indices\n(步骤4: 存储索引)\n(idx_r, idx_theta)', 
           ha='center', fontsize=11, bbox=dict(
        boxstyle='round,pad=0.6', facecolor='#E8F5E9', edgecolor='#388E3C', linewidth=2))
    
    # Output
    ax.annotate('', xy=(6, 2.0), xytext=(6, 2.4), arrowprops=arrow_style)
    ax.text(6, 1.3, 'Compressed Representation\n(压缩表示)\nHigh Compression Ratio (~8x)', 
           ha='center', fontsize=11, bbox=dict(
        boxstyle='round,pad=0.6', facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=2))
    
    # Add description box
    desc_text = """Key Concepts / 关键概念:

Q: Random Orthogonal Matrix (随机正交矩阵)
   - Generated via QR decomposition
   - 通过 QR 分解生成

Lloyd-Max: Optimal Scalar Quantization (最优标量量化)
   - Minimizes Mean Squared Error
   - 最小化均方误差

Compression Ratio Formula (压缩比公式):
   Original: d x 32 bits
   Compressed: r_bits + (d-1) x a_bits
   Example: d=256, r_bits=8, a_bits=4 -> Ratio = 7.97x"""
    
    ax.text(0.3, 5.5, desc_text, fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='#FFF9C4', alpha=0.9, pad=0.8))
    
    plt.tight_layout()
    plt.savefig('/Users/moyong/project/ai/turboquant/kimi25/polarquant/docs/algorithm_flow.png',
               dpi=150, bbox_inches='tight', facecolor='white')
    print("Algorithm flowchart saved to: docs/algorithm_flow.png")
    plt.show()


if __name__ == '__main__':
    print("Generating PolarQuant sequence diagrams...")
    print("(Generating in English to avoid font issues)")
    print()
    draw_sequence_diagram()
    print()
    draw_algorithm_flow()
    print()
    print("All diagrams generated successfully!")
    print("Files saved in: docs/")
