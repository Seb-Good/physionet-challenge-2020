import sys
import logging as log
import click
import os

# import modules
from model.config_ecgnet import hparams

from pipeline import Pipeline


@click.command()
@click.option('--fold', default=0, help='fold to train')
@click.option('--batch', default=hparams['batch_size'], help='batch size')
@click.option('--lr', default=hparams['lr'], help='learning rate')
@click.option('--epochs', default=hparams['n_epochs'], help='number of epoches to run')
@click.option('--ssl', default=False, help='semi-supervised learning')
def main(fold, batch, lr, epochs, ssl):

    cross_val = Pipeline(start_fold=fold, batch_size=batch, lr=lr, epochs=epochs)
    score = cross_val.train()
    log.info(f'Model F1 macro = {score}')
    log.info(f'Model fold = {fold}')


if __name__ == "__main__":
    main()
