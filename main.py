import logging as log
import click

# import modules
from cv_pipeline import CVPipeline
from pre_processing import PrepareData
from config import hparams,DATA_PATH,SPLIT_TABLE_PATH,SPLIT_TABLE_NAME,DEBUG_FOLDER,Model


@click.command()
@click.option('--start_fold', default=hparams['start_fold'], help='fold to train')
@click.option('--batch_size', default=hparams['batch_size'], help='batch size')
@click.option('--lr', default=hparams['lr'], help='learning rate')
@click.option('--n_epochs', default=hparams['n_epochs'], help='number of epoches to run')
@click.option('--p_proc', default=False, help='does it need to run preprocessing?')
@click.option('--train', default=True, help='does it need to train the model?')
def main(start_fold, batch_size, lr, n_epochs,p_proc,train):

    # update hparams
    hparams['lr'] = lr
    hparams['batch_size'] = batch_size
    hparams['start_fold'] = start_fold
    hparams['n_epochs'] = n_epochs

    if p_proc:
        pre_processing = PrepareData(input_folders=DATA_PATH, split_folder=SPLIT_TABLE_PATH,split_table_name=SPLIT_TABLE_NAME)
        pre_processing.run()

    if train:
        cross_val = CVPipeline(
            hparams=hparams,
            split_table_path=SPLIT_TABLE_PATH,
            split_table_name=SPLIT_TABLE_NAME,
            debug_folder=DEBUG_FOLDER,
            model = Model
        )

        score,fold = cross_val.train()

        log.info(f'Model metric = {score}')
        log.info(f'Model fold = {fold}')


if __name__ == "__main__":
    main()
