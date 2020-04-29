# Import 3rd party libraries
import os

# Set working directory
WORKING_PATH = (
    os.path.dirname(
        os.path.dirname(
            os.path.realpath(__file__)
        )
    )
)

# Set data directory
DATA_PATH = os.path.join(WORKING_PATH, 'data')

# Dataset file name
DATA_FILE_NAME = 'PhysioNetChallenge2020_Training_CPSC.tar.gz'

# Extracted folder name
EXTRACTED_FOLDER_NAME = 'Training_WFDB'

# ECG ample rate (hz)
FS = 500

# Number of labels
LABELS_COUNT = 9

# Labels lookup
LABELS_LOOKUP = {'AF': {'label_int': 0, 'label': 'AF', 'label_full': 'Atrial fibrillation'},
                 'I-AVB': {'label_int': 1, 'label': 'I-AVB', 'label_full': 'First-degree atrioventricular block'},
                 'LBBB': {'label_int': 2, 'label': 'LBBB', 'label_full': 'Left bundle branch block'},
                 'Normal': {'label_int': 3, 'label': 'Normal', 'label_full': 'Normal sinus rhythm'},
                 'PAC': {'label_int': 4, 'label': 'PAC', 'label_full': 'Premature atrial complex'},
                 'PVC': {'label_int': 5, 'label': 'PVC', 'label_full': 'Premature ventricular complex'},
                 'RBBB': {'label_int': 6, 'label': 'RBBB', 'label_full': 'Right bundle branch block'},
                 'STD': {'label_int': 7, 'label': 'STD', 'label_full': 'ST-segment depression'},
                 'STE': {'label_int': 8, 'label': 'STE', 'label_full': 'ST-segment elevation'}}

# amplitude values per mV
AMP_CONVERSION = 1000

# ECG leads
ECG_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Filter band limits
FILTER_BAND_LIMITS = [3, 45]
