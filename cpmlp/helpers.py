"""Activations and plot colours shared by the main-text experiments."""
import numpy as np


def silu(z):
    return z / (1.0 + np.exp(-z))


def relu(z):
    return np.maximum(0.0, z)


def relu2(z):
    return np.where(z > 0, z * z, 0.0)


FEATURE_COLORS = {
    "tied-quadratic":   "#B71C1C",
    "tied-ReLU2":       "#E65100",
    "tied-SwiGLU":      "#F9A825",
    "untied-quadratic": "#0D47A1",
    "untied-SwiGLU":    "#2E7D32",
    "untied-ReGLU":     "#00838F",
}
