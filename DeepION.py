from byol_pytorch import BYOL
from torchvision import models
from sklearn.preprocessing import MinMaxScaler
from Augmentation import *
import umap

def img2tensor(img):

    img = img.transpose((2, 3, 0, 1))
    img = img.astype(np.float32)
    img = torch.from_numpy(img).cuda()
    return img

def DeepION_training(input_filename, image_size, mode):

    print('Step 1: Start DeepION training ...')

    oridata =np.loadtxt(input_filename)

    oridata = oridata / np.sum(oridata, axis=0).reshape(1, -1)[0]
    data = MinMaxScaler().fit_transform(oridata)

    resnet = models.resnet18(pretrained=True).cuda()

    if mode == 'COL':

        argument_fn = torch.nn.Sequential(
            ColorJitter(0.8, 0.8, 0, p=1),
            RandomBoxBlur((5, 5), p=0.5),
            RandomMissing(p=1)
        )

        argument_fn2 = torch.nn.Sequential(
            ColorJitter(0.8, 0.8, 0, p=1),
            RandomBoxBlur((5, 5), p=0.5),
            RandomMissing(p=1)
        )

    if mode == 'ISO':

        argument_fn = torch.nn.Sequential(
            ColorJitter(0.8, 0.8, 0, p=1),
            RandomBoxBlur((5, 5), p=0.5),
            IntensityDependentMissing(p=1),
            RandomMissing(p=1)
        )

        argument_fn2 = torch.nn.Sequential(
            ColorJitter(0.8, 0.8, 0, p=1),
            RandomBoxBlur((5, 5), p=0.5),
            IntensityDependentMissing(p=1),
            RandomMissing(p=1)
        )

    learner = BYOL(resnet, image_size=image_size, hidden_layer='avgpool',
                   augment_fn=argument_fn, augment_fn2=argument_fn2)

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

    mini_batch = 100
    num = len(data[0])

    for epoch in range(200):

        index = np.arange(len(data[0]))
        np.random.shuffle(index)
        data = data[:, index]
        total_loss = 0

        for batch in range(num // mini_batch):
            # print(data.shape)
            image_array = data[:, batch * mini_batch: (batch + 1) * mini_batch]
            image_array = image_array.reshape(image_size[0], image_size[1], mini_batch, 1)
            image_array = np.concatenate([image_array, image_array, image_array], axis=3)
            image_tensor = img2tensor(image_array)
            loss = learner(image_tensor)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()

            print('epoch %d : batch : %d loss: %.4f' % (epoch, batch, loss.item()))

            total_loss += loss.item()

        image_array = np.concatenate((image_array[-mini_batch:], image_array[-mini_batch:]), axis=0)
        image_tensor = img2tensor(image_array)
        images = image_tensor.cuda()
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()
        total_loss += loss.item()

    torch.save(resnet.state_dict(), mode + '_ResNet18_params.pth')


def DeepION_predicting(input_filename, image_size, mode):

    print('Step 2: Start DeepION predicting ...')

    oridata =np.loadtxt(input_filename)

    oridata = oridata / np.sum(oridata, axis=0).reshape(1, -1)[0]
    data = MinMaxScaler().fit_transform(oridata)

    resnet = models.resnet18(pretrained=True).cuda()

    resnet.load_state_dict(torch.load(mode + '_ResNet18_params.pth'))
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

    mini_batch = 100
    num = len(data[0])

    features = np.zeros((num, 512))

    for batch in range(num // mini_batch):
        # print(data.shape)
        with torch.no_grad():
            image_array = data[:, batch * mini_batch: (batch + 1) * mini_batch]
            image_array = image_array.reshape(image_size[0], image_size[1], mini_batch, 1)
            image_array = np.concatenate([image_array, image_array, image_array], axis=3)
            image_tensor = img2tensor(image_array)
            embedding = resnet(image_tensor)
            embedding = embedding[:, :, 0, 0]
            embedding = embedding.detach().cpu().numpy()
            features[batch * mini_batch:(batch + 1) * mini_batch] = embedding

    with torch.no_grad():
        image_array = data[:, (batch + 1) * mini_batch:]
        image_array = image_array.reshape(image_size[0], image_size[1], len(image_array[0]), 1)
        image_array = np.concatenate([image_array, image_array, image_array], axis=3)
        image_tensor = img2tensor(image_array)

        feature = resnet(image_tensor)
        feature = feature[:, :, 0, 0]
        feature = feature.detach().cpu().numpy()
        features[(batch + 1) * mini_batch:] = feature

    return features

def DimensionalityReduction(features):

    print('Step 3: Start Dimensionality Reduction ...')

    return_feature = umap.UMAP(n_components=20, metric='cosine', random_state=0).fit_transform(features)

    return_feature = MinMaxScaler().fit_transform(return_feature)

    return return_feature
