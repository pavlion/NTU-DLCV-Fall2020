from .lovasz_loss import lovasz_hinge, lovasz_softmax, StableBCELoss




if __name__ == "__main__":
    
    import torch
    probas = torch.randn(32, 3, 224, 224)
    labels = torch.randn(32, 224, 224)
    print(lovasz_softmax(probas, labels))
