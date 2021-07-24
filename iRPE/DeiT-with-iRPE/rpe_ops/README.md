# 2D RPE Operators

## rpe\_index
- Inputs
    input: float32 (B, H, L_query, num_buckets)
    index: int64 (L_query, L_key) 
- Outputs
    Y: float32 (B, H, L_query, L_key)
