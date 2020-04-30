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

6. Good luck!

## Helper Functions

## License
[BSD 2-Clause License](LICENSE.txt)