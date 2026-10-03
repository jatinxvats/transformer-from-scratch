def warmup_lr_lambda(step_num, d_model, warmup_steps):
    """
    Transformer paper's warmup + inverse-sqrt decay LR schedule (Section 5.3).
    Linearly increases LR for the first `warmup_steps`, then decays
    proportionally to 1/sqrt(step_num).
    """
    step_num = max(step_num, 1)  # avoid step 0 (0^-0.5 = infinity)
    return (d_model ** -0.5) * min(step_num ** -0.5, step_num * warmup_steps ** -1.5)