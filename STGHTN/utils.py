import numpy as np
import torch

def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        wape = np.divide(np.sum(mae), np.sum(label))
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape, wape


def huber_loss(data, label, rho=1):
    '''
    Parameters
    ----------
    data: mx.sym.var, shape is (B, T', N)

    label: mx.sym.var, shape is (B, T', N)

    rho: float

    Returns
    ----------
    loss: mx.sym
    '''

    loss = torch.abs(data - label)
    loss = torch.where(loss > rho, loss - 0.5 * rho,
                        (0.5 / rho) * torch.square(loss))

    return torch.sum(loss)