from seqMNIST import *
from tqdm import tqdm
from pytorch_lightning import Trainer
from lstm_model import *
from test_tube import Experiment
tqdm.monitor_interval = 0


def main():

    input_dim = 28  # row-by-row sequential MNIST
    assert 784 % input_dim == 0
    hidden_dim = 128
    output_dim = 10
    learning_rate = 0.0005
    batch_size = 32
    gradient_clip = 2.0

    model = LstmModel(input_dim, hidden_dim, output_dim)
    lightning_module = SeqMNIST(model, learning_rate, batch_size)
    exp = Experiment(save_dir='experiments')
    trainer = Trainer(experiment=exp, gradient_clip=gradient_clip)
    trainer.fit(lightning_module)


if __name__ == '__main__':
    main()
