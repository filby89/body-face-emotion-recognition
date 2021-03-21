import torch
import numpy as np

def accuracy(output, target, topk=(1,), weighted = False):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # print(pred)
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def visualize_skeleton(sequence, joints):
    import numpy as np
    import cv2
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.animation as animation
    import matplotlib
    matplotlib.use('Agg')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.add_subplot(111)

    def update_plot(i, data, scat):
        scat.set_offsets(data[i].reshape(-1,3))
        return scat

    # for frame in range(0,len(sequence)):
    frame = 100
    skeleton = sequence[frame].reshape(-1,3)
    scat = ax.scatter(skeleton[:,0], skeleton[:,1], skeleton[:,2])
    for edge in joints:
        ax.plot((skeleton[edge[0], 0],skeleton[edge[1], 0]),
                (skeleton[edge[0], 1], skeleton[edge[1], 1]),
                (skeleton[edge[0], 2], skeleton[edge[1], 2]))
    # ani =   animation.FuncAnimation(fig, update_plot, frames=range(len(sequence)),
    #                               fargs=(sequence, scat))

    plt.savefig("out.png")

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [16, 9]

def visualize_skeleton_openpose(joints, hand_left, hand_right, filename="fig.png"):
    joints_edges = [[15, 17], [15, 0], [16, 0], [16, 18], [1, 0], [1, 2],
                  [3, 2], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10],
                  [10, 11], [11, 24], [23, 22], [8, 12], [13, 12], [13, 14], [14, 21], [19, 21],
                  [19, 20]]

    hands_edges = [[0, 1], [1, 2], [2, 3], [3, 4],
                       [0, 5], [5, 6], [6, 7], [7, 8],
                       [0, 9], [9, 10], [10, 11], [11, 12],
                       [0, 13], [13, 14], [14, 15], [15, 16],
                       [0, 17], [17, 18], [18, 19], [19, 20]]

    import matplotlib.animation as animation
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # def update_plot(i, data, scat):
    #     scat.set_offsets(data[i].reshape(-1,2))
    #     return scat
    joints[joints[:,2]<0.01] = np.nan
    joints[np.isnan(joints[:,2])] = np.nan

    hand_right[hand_right[:,2]<0.01] = np.nan
    hand_right[np.isnan(hand_right[:,2])] = np.nan

    hand_left[hand_left[:,2]<0.01] = np.nan
    hand_left[np.isnan(hand_left[:,2])] = np.nan

    # hand_right[hand_right<0.3] = 'nan'
    # hand_left[hand_left[:,2]<0.3] = 'nan'
    # skeleton = sequence[frame].reshape(-1, 2)
    # joints[:,0] = 1-joints[:,0]
    scat = ax.scatter(joints[:, 0], joints[:, 1])
    for edge in joints_edges:
        ax.plot((joints[edge[0], 0], joints[edge[1], 0]),
                (joints[edge[0], 1], joints[edge[1], 1]))

    joints = hand_right
    # joints[:,0] = 1-joints[:,0]
    scat = ax.scatter(joints[:, 0], joints[:, 1])
    for edge in hands_edges:
        ax.plot((joints[edge[0], 0], joints[edge[1], 0]),
                (joints[edge[0], 1], joints[edge[1], 1]))


    joints = hand_left
    # joints[:,0] = 1-joints[:,0]
    scat = ax.scatter(joints[:, 0], joints[:, 1])
    for edge in hands_edges:
        ax.plot((joints[edge[0], 0], joints[edge[1], 0]),
                (joints[edge[0], 1], joints[edge[1], 1]))

    # ax.set_xlim(right=1, left=0)
    # ax.set_ylim(top=1, bottom=0)
    plt.gca().invert_yaxis()


    plt.savefig(filename)
    plt.close()

def plot_pose(pose):
    """Plot the 3D pose showing the joint connections."""
    import mpl_toolkits.mplot3d.axes3d as p3

    _CONNECTION = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
        [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
        [15, 16]]

    def joint_color(j):
        """
        TODO: 'j' shadows name 'j' from outer scope
        """

        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in range(1, 4):
            _c = 1
        if j in range(4, 7):
            _c = 2
        if j in range(9, 11):
            _c = 3
        if j in range(11, 14):
            _c = 4
        if j in range(14, 17):
            _c = 5
        return colors[_c]

    assert (pose.ndim == 2)
    assert (pose.shape[0] == 3)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for c in _CONNECTION:
        col = '#%02x%02x%02x' % joint_color(c[0])
        ax.plot([pose[0, c[0]], pose[0, c[1]]],
                [pose[1, c[0]], pose[1, c[1]]],
                [pose[2, c[0]], pose[2, c[1]]], c=col)
    for j in range(pose.shape[1]):
        col = '#%02x%02x%02x' % joint_color(j)
        ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                   c=col, marker='o', edgecolor=col)
    smallest = pose.min()
    largest = pose.max()
    ax.set_xlim3d(smallest, largest)
    ax.set_ylim3d(smallest, largest)
    ax.set_zlim3d(smallest, largest)

    return fig

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_weighted_loss_weights(dataset, num_classes):
    print("Calculating sampler weights...")
    # labels_array = [x['emotion'] for x in dataset.data]
    labels_array = dataset#.Y_body

    from sklearn.utils import class_weight
    import numpy as np
    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels_array), labels_array)
    assert(class_weights.size == num_classes)
    # class_weights = 1/class_weights
    print("Class Weights: ", class_weights)
    return class_weights

# calculates the weights for doing balanced sampling
def get_sampler_weights(dataset, num_classes):
    print("Calculating sampler weights...")
    # labels_array = [x['emotion'] for x in dataset.data]
    labels_array = dataset#.Y_body

    from sklearn.utils import class_weight
    import numpy as np
    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels_array), labels_array)
    assert(class_weights.size == num_classes)

    sampler_weights = torch.zeros(len(labels_array))
    i=0
    for label in labels_array:
        sampler_weights[i] = class_weights[int(label)]
        # print(i)
        i+=1

    return sampler_weights

import torch.nn as nn
import torch.nn.functional as F

class SequentialLoss(nn.Module):
    def __init__(self):
        super(SequentialLoss, self).__init__()

    def forward(self, output, target, lengths):
        total_loss = 0
        # print(output.size(),target.size())
        for batch_idx in range(output.size(0)):
            weights = torch.arange(lengths[batch_idx]).float().cuda()/lengths[batch_idx].float()
            for sequence_idx in range(lengths[batch_idx]):
                out = output[batch_idx,sequence_idx,:].unsqueeze(0)
                tar = target[batch_idx].unsqueeze(0)
                # print(out.size(), tar.size())
                # print(out,target)
                total_loss += weights[sequence_idx] * F.cross_entropy(out,tar)
        return total_loss/output.size(0)



map_to_emo_family = {
    0: 0,
    1: 0,
    2: 0,
    3: 1,
    4: 2,
    5: 1,
    6: 2,
    7: 1,
    8: 2,
    9: 3,
    10: 3,
    11: 3
}

def load_checkpoint(checkpoint_file):
	return torch.load(checkpoint_file)


def save_checkpoint(state, filename):
	filename = 'checkpoints/%s'%filename
	torch.save(state, filename)



class GroupCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(GroupCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        output1 = output.clone()
        output1[:,0] = output[:,0]+output[:,1]+output[:,2]
        output1[:,1] = output[:,3]+output[:,5]+output[:,7]
        output1[:,2] = output[:,4]+output[:,6]+output[:,8]
        output1[:,3] = output[:,9]+output[:,10]+output[:,11]
        output1 = output[:,:4]
        return F.cross_entropy(output1,target)


def pad_sequence(sequences, batch_first=False, padding_value=0, max_len=100):
    r"""Pad a list of variable length Tensors with zero

    ``pad_sequence`` stacks a list of Tensors along a new dimension,
    and pads them to equal length. For example, if the input is list of
    sequences with size ``L x *`` and if batch_first is False, and ``T x B x *``
    otherwise.

    `B` is batch size. It is equal to the number of elements in ``sequences``.
    `T` is length of the longest sequence.
    `L` is length of the sequence.
    `*` is any number of trailing dimensions, including none.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *`` where `T` is the
            length of the longest sequence.
        Function assumes trailing dimensions and type of all the Tensors
            in sequences are same.

    Arguments:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): output will be in ``B x T x *`` if True, or in
            ``T x B x *`` otherwise
        padding_value (float, optional): value for padded elements. Default: 0.

    Returns:
        Tensor of size ``T x B x *`` if batch_first is False
        Tensor of size ``B x T x *`` otherwise
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    # max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor



#
# class MiniLSTM(nn.Module):
#     def __init__(self, num_input, num_hidden, num_input_lstm, encode=True, add_spatial_attention=False, confidence=False, num_layers=2  , bidirectional=True):
#         super(MiniLSTM, self).__init__()
#         self.encode = encode
#         self.confidence = confidence
#
#         self.num_layers = num_layers
#         self.bidrectional = bidirectional
#
#         if add_spatial_attention or confidence:
#             tmp = int(num_input * 2 / 3)
#         else:
#             tmp = num_input
#
#         self.encoder = nn.Sequential(
#             nn.Linear(tmp, num_input_lstm),
#             nn.Dropout(0.2),
#             nn.PReLU(),
#             # nn.BatchNorm1d(200)
#         )
#         self.lstm = nn.LSTM(num_input_lstm, num_hidden, batch_first=True, num_layers=num_layers, bidirectional=bidirectional, dropout=0.2)
#         self.num_hidden = num_hidden
#         init_lstm(self.lstm)
#
#         self.add_spatial_attention = add_spatial_attention
#
#         if True:#add_spatial_attention:
#             self.spatial_attention_lstm = nn.LSTM(num_input_lstm, 100, batch_first=True, num_layers=1)
#
#             self.spatial_attention = nn.Sequential(
#                 nn.Linear(100, int(num_input/3))
#             )
#             # self.spatial_attention._modules['0'].weight.data.fill_(1)
#             # self.spatial_attention._modules['0'].bias.data.fill_(1)
#
#     def forward(self, features, lengths):
#         # if self.add_spatial_attention:
#         #     scores = self.spatial_attention(features)
#         #     features = features.view(features.size(0),features.size(1),-1,3)
#         #     confidences = features[:,:,:,2]
#         #     features_positions_x = features[:,:,:,0].clone()
#         #     features_positions_y = features[:,:,:,1].clone()
#         #
#         #     features_positions = torch.stack((features_positions_x*scores,features_positions_y*scores),dim=3)
#         #
#         #     features = features_positions.view(features_positions.size(0), features_positions.size(1), -1)
#
#         if self.confidence:
#             features = features.view(features.size(0),features.size(1),-1,3)
#             confidences = features[:,:,:,2]
#             features_positions_x = features[:,:,:,0].clone()
#             features_positions_y = features[:,:,:,1].clone()
#
#             features_positions = torch.stack((features_positions_x*confidences,features_positions_y*confidences),dim=3)
#
#             features = features_positions.view(features_positions.size(0), features_positions.size(1), -1)
#
#         if self.encode:
#             encoded_features = self.encoder(features)
#         else:
#             encoded_features = features
#
#         if self.bidrectional:
#             num_directions = 2
#         else:
#             num_directions = 1
#
#
#         timesteps = True
#         if timesteps:
#             h_att = torch.zeros(1, features.size(0),
#                              100).cuda()  # 2 for bidirection
#             c_att = torch.zeros(1, features.size(0), 100).cuda()
#
#
#             h0 = torch.zeros(self.num_layers * num_directions, features.size(0),
#                              self.num_hidden).cuda()  # 2 for bidirection
#             c0 = torch.zeros(self.num_layers * num_directions, features.size(0), self.num_hidden).cuda()
#
#             output = torch.zeros(features.size(0),encoded_features.size(1),self.num_hidden*2).cuda()
#
#
#             for i in range(1,encoded_features.size(1)): # batch
#                 if i > 0:
#                     l, (h_att, c_att) = self.spatial_attention_lstm(encoded_features[:,i-1:i,:], (h_att, c_att))
#                     scores = self.spatial_attention(l).squeeze()
#                 a = encoded_features[:,i:i+1,:]
#                 output[:,i:i+1,:], (h0,c0) = self.lstm(scores*encoded_features[:,i:i+1,:],(h0,c0))
#
#             for i in range(0,encoded_features.size(0)):
#                 output[:,lengths[i]:,:] = 0
#         else:
#
#             h0 = torch.zeros(self.num_layers * num_directions, encoded_features.size(0),
#                              self.num_hidden).cuda()  # 2 for bidirection
#             c0 = torch.zeros(self.num_layers * num_directions, encoded_features.size(0), self.num_hidden).cuda()
#
#             packed = pack_padded_sequence(encoded_features, lengths, batch_first=True)
#
#             output, _ = self.lstm(packed,(h0,c0))
#
#             output, _ = pad_packed_sequence(output,batch_first=True)
#
#
#         # output = output[torch.arange(output.size(0)), lengths-1, :]
#
#         output = torch.sum(output,dim=1)/lengths.unsqueeze(1).float()
#
#         return output
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('%s.png'%filename)

def visualize_with_tsne(data, labels):
    from sklearn.manifold import TSNE
    import seaborn as sns
    import matplotlib.pyplot as plt

    X_embedded = TSNE(n_components=2).fit_transform(data)

    df = pd.DataFrame()
    df['x'] = X_embedded[:, 0]
    df['y'] = X_embedded[:, 1]
    df['label'] = labels

    sns.lmplot(x='x', y='y', fit_reg=False, data=df, hue='label')
    plt.show()



def calc_gradients(params):
    grad_array = []
    _mean = []
    _max = []
    for param in params:
        grad_array.append(param.grad.data)
        _mean.append(torch.mean(param.grad.data))
        _max.append(torch.max(param.grad.data))
    print(np.mean(_mean))
    print(np.max(_max))


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

import errno
import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def random_search():
    import random
    hidden_size = random.choice([50, 100, 150, 200, 250, 300])
    spatial_net_features = random.choice([50, 100, 150, 200, 250, 300])
    spatial_net_one_feature = random.choice([50, 100, 150, 200, 250, 300])
    num_input_lstm = random.choice([32, 64, 100, 128, 200, 256, 512])
    num_layers = random.choice([1, 2, 3, 4])
    bidirectional = random.choice([True, False])
    lr = random.choice([1e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 9e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3])
    step_size = random.choice([30, 60, 90, 120])
    epochs = random.choice([50, 100, 150, 200, 250, 300])
    weight_decay = random.choice([1e-4, 2e-4, 3e-4, 7e-4, 5e-4, 1e-3, 4e-3, 7e-3])
    dropout = random.choice([0, 0.2, 0.4, 0.5, 0.6, 0.8])
    batch_size = random.choice([16, 32, 64, 120])

    num_channels = random.choice([16, 32, 64, 128])
    kernel_size = random.choice([2, 3, 5, 7, 9, 11])
    num_tcn_layers = random.choice([2, 3, 4, 6, 8, 10])

    return {"hidden_size": hidden_size, "num_layers": num_layers, "bidirectional": bidirectional, "epochs": epochs,
            "step_size": step_size, "lr": lr,
            "weight_decay": weight_decay, "dropout": dropout, "batch_size": batch_size, "grad_clip": 0.1,
            "multiply_with_confidence": 0.3, "num_input_lstm": num_input_lstm,
            "num_channels": num_channels, "kernel_size": kernel_size, "num_tcn_layers": num_tcn_layers,
            "spatial_net_features": spatial_net_features,
            "spatial_net_one_feature": spatial_net_one_feature}

