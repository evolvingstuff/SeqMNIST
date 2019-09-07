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
    batch_size = 32
    gradient_clip = 2.0
    is_permuted = False
    max_epochs = 100
    percent_validation = 0.2

    gpus = None
    if torch.cuda.is_available():
        gpus = [0]

    model = LstmModel(input_scan_dim, hidden_dim, output_dim)
    lightning_module = SeqMNIST(model, learning_rate, batch_size, is_permuted, percent_validation)
    exp = Experiment(save_dir='experiments')
    trainer = Trainer(experiment=exp, track_grad_norm=-1, print_nan_grads=False,
                      gradient_clip=gradient_clip, gpus=gpus, max_nb_epochs=max_epochs)
    trainer.fit(lightning_module)
    # trainer.test() # TODO this appears not to work


if __name__ == '__main__':
    main()
