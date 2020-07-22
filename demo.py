#-*-coding:utf-8-*-
import config
import matplotlib.pyplot as plt
def adjust_learning_rate(epoch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < config.warm_up_epoch:
        lr = 1e-6 + (config.lr - 1e-6) * epoch / (config.warm_up_epoch)
    else:
        lr = config.lr * (config.lr_gamma ** (epoch / config.lr_decay_step[0]))

    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr

    return lr
lrs=[]
for epoch in range(0,1000):
    lr=adjust_learning_rate(epoch)
    lrs.append(lr)
plt.plot(lrs)
plt.show()