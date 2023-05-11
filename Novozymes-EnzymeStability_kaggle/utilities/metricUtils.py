import numpy as np
from scipy import stats
from sklearn import metrics as skmetrics

class AccuracyComputer():
    def __init__(self):
        self.tgt = []
        self.prd = []

    def reset(self):
        self.__init__()

    def add_entry(self, prd, tgt):
        self.prd.extend(prd.cpu().detach().numpy())
        self.tgt.extend(tgt.cpu())

    def get_spearmanr(self):
        acc = stats.spearmanr(self.tgt, self.prd, axis=None).correlation
        return acc

    def get_r2score(self):
        acc = skmetrics.r2_score(self.tgt, self.prd)
        return acc

    def get_mae(self):
        ae = np.absolute(np.asarray(self.tgt, dtype="float") - np.asarray(self.prd, dtype="float"))
        mae = np.mean(ae)
        sdv = np.std(ae)
        iqr = np.percentile(ae, [25 ,75])
        return mae, sdv, iqr

    def get_mse(self):
        se = np.square(np.asarray(self.tgt, dtype="float") - np.asarray(self.prd, dtype="float"))
        mse = np.mean(se)
        sdv = np.std(se)
        iqr  = np.percentile(se, [25 ,75])
        return mse, sdv, iqr

    def print_summary(self, ret=False):
        text = 'Mean Absolute Error :: Avg:{} :: StDiv={} :: IQR:{}\n'.format(*self.get_mae())  \
                +'Mean Squared Error :: Avg:{} :: StDiv={} :: IQR:{}\n'.format(*self.get_mse())  \
                +'Spearman rcoeff :: {}\n'.format(self.get_spearmanr()) \
                +'R2 Score dcoeff :: {}\n'.format(self.get_r2score())
        if ret == True:
            return text
        else:
            print(text)


if __name__ == "__main__":

    obj = AccuracyComputer()
    obj.tgt = [53, 29, 74, 92, 43, 30, 17, 6, 7, 84]
    obj.prd = [9, 41, 96, 97, 7, 73, 29, 92, 84, 31]

    obj.print_summary()
