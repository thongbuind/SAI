import numpy as np

def split_train_val_test(X, Y, loss_masks, lengths, train_ratio, val_ratio, seed=54):
    """Split kiểu dev_2: hoán vị theo VỊ TRÍ mẫu rồi cắt theo tỉ lệ.

    Lưu ý: split phụ thuộc thứ tự mẫu đầu vào, không theo ID mẫu. Nếu tập đầu vào thay đổi
    (vd. trộn lại với seed khác) thì cùng một mẫu có thể rơi vào split khác.
    """
    total_sample = len(X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_sample)

    train_end = int(total_sample * train_ratio)
    val_end = int(total_sample * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, Y_train, lengths_train = X[train_idx], Y[train_idx], lengths[train_idx]
    X_val, Y_val, lengths_val = X[val_idx], Y[val_idx], lengths[val_idx]
    X_test, Y_test, lengths_test = X[test_idx], Y[test_idx], lengths[test_idx]

    if loss_masks is not None:
        mask_train = loss_masks[train_idx]
        mask_val = loss_masks[val_idx]
        mask_test = loss_masks[test_idx]

        return (X_train, Y_train, mask_train, lengths_train,
                X_val, Y_val, mask_val, lengths_val,
                X_test, Y_test, mask_test, lengths_test)
    else:
        return (X_train, Y_train, None, lengths_train,
                X_val, Y_val, None, lengths_val,
                X_test, Y_test, None, lengths_test)
