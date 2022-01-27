import torchvision.datasets


def mnist_subset(root, train, download, transform, subset=(0, 1), shift=0):
    mnist = torchvision.datasets.MNIST(root=root, train=train, download=download, transform=transform)
    index = None
    for sub in subset:
        if index is None:
            index = (mnist.targets == sub)
        else:
            index += (mnist.targets == sub)
    mnist.data, mnist.targets = mnist.data[index], mnist.targets[index]
    if shift != 0:
        mnist.data = (1 - shift) * mnist.data + shift
    return mnist


