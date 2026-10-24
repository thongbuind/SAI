from foundation_lab.core.log import log_progress

def freeze_blocks(model, block_indices):
    """Đóng băng các decoder block theo chỉ số (``freeze_blocks`` trong params của stage)."""
    for idx in block_indices:
        for param in model.blocks[idx].parameters():
            param.requires_grad = False

def log_trainable_summary(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log_progress(f"Tham số trainable: {trainable:,} / {total:,} (frozen: {total - trainable:,})")
