import sys
import logging as log
import click
import os

# import modules
from model.config_wavenet import hparams

from pipeline import Pipeline
from data_generator import DataGenerator

@click.command()
@click.option('--fold', default=0, help='fold to train')
@click.option('--batch', default=hparams['batch_size'], help='batch size')
@click.option('--lr', default=hparams['lr'], help='learning rate')
@click.option('--epochs', default=hparams['n_epochs'], help='number of epoches to run')
@click.option('--ssl', default=False, help='semi-supervised learning')
def main(fold, batch, lr, epochs,ssl):

    get_data = DataGenerator(ssl=ssl)
    cross_val = Pipeline(get_data=get_data,start_fold=fold, batch_size=batch, lr=lr, epochs=epochs)
    score = cross_val.train()
    log.info(f'Model F1 macro = {score}')
    log.info(f'Model fold = {fold}')


if __name__ == "__main__":
    main()
