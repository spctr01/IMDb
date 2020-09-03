import torch

class IMDBDataset:
    def __init__(self, reviews, targets):
        self.reviews = reviews
        self.target = targets


    def __len__(self):
        return len (self.reviews)

    def __getitem__(self, item):
        review = self.reviews[item, :]
        target = self.target[item]

        return{'review': torch.tensor(review, dtype = torch.long),
                'target': torch.tensor(target, dtype= torch.float)}
                