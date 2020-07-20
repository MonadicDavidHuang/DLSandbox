from model import DCGAN

image_size = (128, 128)
batch_size = 32


def train():
    dcgan = DCGAN(image_size)

    dcgan.train(1000)


train()
