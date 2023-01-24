"""
hetero 20 epoch 5 lr
max_layer 1
permuted nodes  0
permuted nodes  0
permuted nodes  0
permuted nodes  3
Loss merged: 7.081671146431354
Loss average: 7.081748076159545
Loss 1: 7.031536400920213
Loss 2: 7.655910862816705
Loss permuted: 7.655910862816705


hetero 20 epoch 1 lr
max_layer 1
permuted nodes 0
permuted nodes 3
permuted nodes 0
permuted nodes 13
Loss merged: 6.656022842484291
Loss average: 6.649247034631594
Loss 1: 6.375983021476052
Loss 2: 7.199618792293047
Loss permuted: 7.199618792293047


hetero 20 epoch 0.5 lr
max_layer 1
permuted nodes 0
permuted nodes 0
permuted nodes 4
permuted nodes 11
Loss merged: 6.814038353736954
Loss average: 6.797719695351341
Loss 1: 6.423141859998607
Loss 2: 7.298387122876717
Loss permuted: 7.29838678571913
"""

# generate a pyplot graph where the x axis is the learning rate and the y axis is the number of nodes permuted for each row
# based on the results from the comment above
import matplotlib.pyplot as plt


def learning_rate():
    x = ["0.5", "1", "5", "15"]
    y0 = [0, 0, 0, 0]
    y1 = [0, 0, 3, 0]
    y2 = [4, 0, 0, 0]
    y3 = [11, 13, 3, 14]
    # put embedding plot on top
    plt.plot(x, y1, label='QKV')
    plt.plot(x, y2, label='Projection')
    plt.plot(x, y3, label='MLP')
    plt.plot(x, y0, label='Embedding')
    plt.xlabel('Learning Rate')
    plt.ylabel('Number of nodes permuted')
    plt.legend()
    plt.savefig('graph_learning_rate.pdf')


def loss_diff_graph():
    x = ["5", "20", "50"]
    y_small = [9.976011e-05, 0.0001097255, 0.0007338379]
    y_large = [0.0006620137, -0.007864682, -0.0001157317]
    # put embedding plot on top
    plt.plot(x, y_small, label='Small Model')
    plt.plot(x, y_large, label='Big Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Difference')
    plt.legend()
    plt.savefig('graph_loss_diff.pdf')


if __name__ == '__main__':
    learning_rate()
