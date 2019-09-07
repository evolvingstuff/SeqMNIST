from seqMNIST import *
from tqdm import tqdm
from pytorch_lightning import Trainer
from lstm_model import *
from test_tube import Experiment
tqdm.monitor_interval = 0


def main():

    # input_scan_dim=28 is row-by-row sequential MNIST
    # input_scan_dim=1 to make it pixel-by-pixel
    input_scan_dim = 28
    hidden_dim = 128
    output_dim = 10
    learning_rate = 0.0005
    batch_size = 2 #
    gradient_clip = 2.0
    is_permuted = False

    gpus = None
    if torch.cuda.is_available():
        gpus = [0]

    model = LstmModel(input_scan_dim, hidden_dim, output_dim)
    lightning_module = SeqMNIST(model, learning_rate, batch_size, is_permuted)
    exp = Experiment(save_dir='experiments')
    trainer = Trainer(experiment=exp, gradient_clip=gradient_clip, gpus=gpus)
    trainer.fit(lightning_module)


if __name__ == '__main__':
    main()
