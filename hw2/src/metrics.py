import torch
torch.cuda.manual_seed_all(9487)

class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Accuracy(Metrics):
    """
    Args:
         ats (int): @ to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self, at=1):
        self.at = at
        self.n = 0
        self.n_corrects = 0
        self.name = 'Accuracy@{}'.format(at) if at >1 else 'Accuracy'

    def reset(self):
        self.n = 0
        self.n_corrects = 0

    def update(self, logits, label):

        total_num = logits.size(0)
        correct = (logits.argmax(1) == label).int().sum().cpu().item()

        self.n += total_num
        self.n_corrects += correct

        return self.get_score()

    def get_score(self):
        return self.n_corrects / self.n

    def print_score_msg(self):
        return f'{self.name}={self.get_score()}'
