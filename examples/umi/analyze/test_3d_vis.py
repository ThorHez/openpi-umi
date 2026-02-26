#!/usr/bin/env python3
"""
测试3D可视化 - 生成一个简单的测试HTML来验证plotly是否正常工作
"""

import numpy as np
import plotly.graph_objects as go

# 创建简单的螺旋线测试数据
t = np.linspace(0, 10, 100)
x = np.cos(t)
y = np.sin(t)
z = t / 3

# 创建3D图表
fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='lines+markers',
    line=dict(color='blue', width=4),
    marker=dict(size=3, color='red')
)])

fig.update_layout(
    title='Simple 3D Test - Spiral',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    width=800,
    height=600
)

# 保存HTML
fig.write_html('test_3d_simple.html', include_plotlyjs='cdn')
print("✓ 测试HTML已生成: test_3d_simple.html")
print("  如果这个文件能正常显示，说明plotly工作正常")
print("  如果这个文件也是空白，可能是浏览器或网络问题")































