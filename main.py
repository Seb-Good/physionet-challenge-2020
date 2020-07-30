import logging as log
import click

# import modules
from cv_pipeline import CVPipeline
from pre_processing import PrepareData
from config import *


@click.command()
@click.option('--start_fold', default=hparams['start_fold'], help='fold to train')
@click.option('--batch_size', default=hparams['batch_size'], help='batch size')
@click.option('--lr', default=hparams['lr'], help='learning rate')
@click.option('--n_epochs', default=hparams['n_epochs'], help='number of epoches to run')
def main(start_fold, batch_size, lr, n_epochs):

    # update hparams
    hparams['lr'] = lr
    hparams['batch_size'] = batch_size
    hparams['start_fold'] = start_fold
    hparams['n_epochs'] = n_epochs

    # if p_proc:
    #     pre_processing = PrepareData()
    #     pre_processing.run()

    cross_val = CVPipeline(
        hparams=hparams,
        split_table_path=SPLIT_TABLE_PATH,
        split_table_name=SPLIT_TABLE_NAME,
        pic_folder=PIC_FOLDER,
        debug_folder=DEBUG_FOLDER,
    )

    score = cross_val.train()

    # TODO: add a script for oof predictions

    log.info(f'Model metric = {score}')
    log.info(f'Model fold = {fold}')


if __name__ == "__main__":
    main()
