from data_prep.loadData import Dataset
from common import utils

def collect(cfg, shuffle):
    dataset = Dataset(**cfg.data)
    train, val, test = dataset()
    train, val, test = utils.get_loaders([train, val, test], cfg.common.batch_size, shuffle)
    return [train, val, test]
