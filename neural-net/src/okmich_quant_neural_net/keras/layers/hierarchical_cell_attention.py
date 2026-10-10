"""
Hierarchical Cell Attention
===========================

Structured attention over a tabular (rows x columns) input, as introduced by the Technical Analyst Attention Network
(TAAN). Unlike token attention, the layer keeps the matrix structure: every cell is scored by a trainable weight, the
scores are aggregated per row, and a temperature softmax over the rows yields one attention distribution per head. The
output is the attention-weighted row for each head, so ``num_heads`` acts as a soft top-k over rows.

Formula (X in R^{R x F}):
    S_cell  = X ⊙ W_cell                      W_cell in R^{R x F}
    s_row   = sum_F S_cell                    (R,)
    S_head  = s_row[:, None] * W_row          W_row in R^{R x H}
    alpha_h = softmax_R(S_head[:, h] / tau)   one distribution over rows per head
    out_h   = sum_R alpha_h[r] * X[r, :]      (H, F)

Reference:
    Tavassolian, Yousefloei, Abdoos, Vahidi Asl (2026). "Technical Analyst Attention Network (TAAN): An Interpretable
    Deep Learning Model for Algorithmic Trading in Crypto and Forex Markets." IEEE IICAI.
    DOI 10.1109/IICAI70155.2026.11620664
"""

from keras import layers, ops
from keras.saving import register_keras_serializable


@register_keras_serializable()
class HierarchicalCellAttention(layers.Layer):
    """Row-selecting attention over a (B, R, F) matrix: returns ``(out, attn)``.

    Args:
        num_heads: Number of attention heads; each head is a soft selection of one row.
        temperature: Softmax temperature. Below 1 sharpens the selection, above 1 flattens it.
        **kwargs: Passed to the base Layer.

    Shapes:
        input  (B, R, F)   R and F must be static (the weights are sized (R, F) and (R, H))
        out    (B, H, F)   attention-weighted rows, one per head
        attn   (B, R, H)   attention weights; sum over R equals 1 for every head

    Examples:
        hca = HierarchicalCellAttention(num_heads=4, name="rsi_attention")
        out, attn = hca(inputs)  # (B, 14, 10) -> (B, 4, 10), (B, 14, 4)
    """

    def __init__(self, num_heads: int = 4, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.num_heads = int(num_heads)
        self.temperature = float(temperature)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(f"expected input of rank 3 (batch, rows, cols), got shape {input_shape}")
        rows, cols = input_shape[1], input_shape[2]
        if rows is None or cols is None:
            raise ValueError(f"rows and cols must be static (weights are sized by them), got shape {input_shape}")
        rows, cols = int(rows), int(cols)
        self.w_cell = self.add_weight(name="w_cell", shape=(rows, cols), initializer="glorot_uniform", trainable=True)
        self.w_row = self.add_weight(name="w_row", shape=(rows, self.num_heads), initializer="glorot_uniform",
                                     trainable=True)
        super().build(input_shape)

    def call(self, x):
        cell_scores = x * self.w_cell  # (B, R, F)
        row_scores = ops.sum(cell_scores, axis=-1)  # (B, R)
        head_scores = ops.expand_dims(row_scores, -1) * self.w_row  # (B, R, H)
        attn = ops.softmax(head_scores / self.temperature, axis=1)  # softmax over rows, per head
        out = ops.einsum("brh,brf->bhf", attn, x)  # (B, H, F)
        return out, attn

    def compute_output_shape(self, input_shape):
        batch, rows, cols = input_shape
        return (batch, self.num_heads, cols), (batch, rows, self.num_heads)

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "temperature": self.temperature})
        return config
