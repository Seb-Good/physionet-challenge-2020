# Classification of 12-lead ECGs: the PhysioNet/Computing in Cardiology Challenge 2020
Laussen Labs submission for the [2020 Physionet Challenge](https://physionetchallenges.github.io/2020/).

## Data Setup Instructions
1. Download Physionet2020 tar.gz file from:
https://storage.cloud.google.com/physionet-challenge-2020-12-lead-ecg-public/PhysioNetChallenge2020_Training_CPSC.tar.gz

2. Place ```PhysioNetChallenge2020_Training_CPSC.tar.gz``` in path ```physionet-challenge-2020/data/raw```

3. Run Notebooks 1 ```physionet-challenge-2020/notebooks/1_format_data.ipynb```. This will extract the tar file and the 
place the original ```.mat``` and ```.hea``` files in ```physionet-challenge-2020/data/raw/Training_WFDB```. The 
```.mat``` file contains an array of size ```[12, num_samples]```. Where the number of samples varies from 6 seconds 
to 60 seconds at 500 Hz. The ```.hea``` contains information about the 12-lead signals, the patient (Age and Sex), and 
the rhythm labels (```#Dx```). Below is an exampe for file ```A0001.hea```.   

    ```text
    A0001 12 500 7500 05-Feb-2020 11:39:16
    A0001.mat 16+24 1000/mV 16 0 28 -1716 0 I
    A0001.mat 16+24 1000/mV 16 0 7 2029 0 II
    A0001.mat 16+24 1000/mV 16 0 -21 3745 0 III
    A0001.mat 16+24 1000/mV 16 0 -17 3680 0 aVR
    A0001.mat 16+24 1000/mV 16 0 24 -2664 0 aVL
    A0001.mat 16+24 1000/mV 16 0 -7 -1499 0 aVF
    A0001.mat 16+24 1000/mV 16 0 -290 390 0 V1
    A0001.mat 16+24 1000/mV 16 0 -204 157 0 V2
    A0001.mat 16+24 1000/mV 16 0 -96 -2555 0 V3
    A0001.mat 16+24 1000/mV 16 0 -112 49 0 V4
    A0001.mat 16+24 1000/mV 16 0 -596 -321 0 V5
    A0001.mat 16+24 1000/mV 16 0 -16 -3112 0 V6
    #Age: 74
    #Sex: Male
    #Dx: RBBB
    #Rx: Unknown
    #Hx: Unknown
    #Sx: Unknows
    ``` 

    An  additional formatted dataset is also extracted and saved to ```physionet-challenge-2020/data/formatted```. Here, the
    12-lead signals have been saved as ```.npy``` files with the same dimensions as the ```.mat``` file. A ```.json``` file
    is saved for every waveform file where all relevant information has been extracted from the ```.hea``` file and 
    formatted for training. An example is presented below for ```A0001```.

    ```json
    {
      "age": "74",
      "channel_order": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6" ],
      "filename": "A0001",
      "label_train": [ 0, 0, 0, 0, 0, 0, 1, 0, 0],
      "labels": [ "RBBB"],
      "labels_full": ["Right bundle branch block"],
      "labels_int": [6],
      "sex": "Male",
      "shape": [12, 7500]
    }
    ``` 

4. Run Notebooks 2 ```2_visualize_formatted_data.ipynb```. This notebook pulls data from 
```physionet-challenge-2020/data/formatted``` and presents an interactive visualization of what waveforms (see example 
below). 

    ![Example ECG Waveform](/documents/README/example.png) 

5. Run Notebooks 3 ```3_explore_formatted_data.ipynb```. This notebook pulls data from 
```physionet-challenge-2020/data/formatted``` and presents some summary plots of the training population.

    ![Patient Age Distribution](/documents/README/age_distribution.png) 

6. Good luck!

## Model Submission
The ```driver.py``` file was provided by the contest organizers and shows how a submission model is run. They will not 
use any ```driver.py``` that we submit so no modifications should be made to this file. The code below is a copy of the 
```___main___``` function from the ```driver.py``` file. 

```python
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier

if __name__ == '__main__':

    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    # Find files.
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    classes = get_classes(input_directory, input_files)

    # Load model.
    print('Loading 12ECG model...')
    model = load_12ECG_model()

    # Iterate over files.
    print('Extracting 12ECG features...')
    num_files = len(input_files)

    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i+1, num_files))
        tmp_input_file = os.path.join(input_directory, f)
        data, header_data = load_challenge_data(tmp_input_file)
        current_label, current_score = run_12ECG_classifier(data, header_data, classes, model)
        # Save results.
        
        save_challenge_predictions(output_directory, f, current_score, current_label, classes)

    print('Done.')
```

There are two functions loaded at the start of ```driver.py``` 
that we have control over. They are called as follows:

```python
# Load your model and return a model object.
model = load_12ECG_model()
``` 
and 
```python
# Process data from one sample (.mat and .hea) and return predicted labels and probabilities.
current_label, current_score = run_12ECG_classifier(data, header_data, classes, model)
```

Output should be in the following format.

```python
# Class labels
current_label = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0])

# Class probabilities
current_score = np.array([3.89575280e-06, 1.04941707e-02, 1.73853521e-04, 1.84738655e-02, 1.03849954e-04, 
                          1.97162270e-04, 7.78743392e-03, 9.46390033e-01, 2.45864340e-03], dtype=float32)
```

An example of the LaussenLabs Physionet Challenge 2017 model being called is presented below.

```python
# 3rd party imports
import os
import pickle

# Local imports
from kardioml import WORKING_PATH


def run_12ECG_classifier(data, header_data, classes, model):
    """Get predictions."""
    current_label, current_score = model.challenge_prediction(data=data, header_data=header_data)

    return current_label, current_score


def load_12ECG_model():
    """Load Physionet2017 Model"""
    # Unpickle data model
    with open(os.path.join(WORKING_PATH, 'models', 'physionet2017', 'physionet2017.model'), "rb") as input_file:
        phyionet2017_model = pickle.load(input_file)

    return phyionet2017_model
```

## Helper Functions
Scoring functions can be found at ```physionet-challenge-2020/kardioml/scoring/scoring_metrics.py```. The model is 
evaluated basked on the ```f_measure``` so use this for model tuning.

```python
accuracy, f_measure, f_beta, g_beta = compute_beta_score(labels, output, beta, num_classes, check_errors=True)

auroc, auprc = compute_auc(labels, probabilities, num_classes, check_errors=True)
```

Some additional functions for parsing the header and loading ```.mat``` and ```.hea``` files is available at 
```physionet-challenge-2020/kardioml/data/data_loader.py```. 

## Model Building
An example of the LaussenLabs Physionet Challenge 2017 model can be seen at 
```physionet-challenge-2020/kardioml/models/physionet2017```. This model uses extracted features from lead I and trains
an XGBoost classifier.

## License
[BSD 2-Clause License](LICENSE.txt)