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

# Set output directory
OUTPUT_PATH = os.path.join(WORKING_PATH, 'output')

# Dataset file name
DATA_FILE_NAMES = {'A': 'PhysioNetChallenge2020_Training_CPSC.tar.gz',
                   'B': 'PhysioNetChallenge2020_Training_2.tar.gz',
                   'C': 'PhysioNetChallenge2020_Training_StPetersburg.tar.gz',
                   'D': 'PhysioNetChallenge2020_Training_PTB.tar.gz',
                   'E': 'PhysioNetChallenge2020_PTB-XL.tar.gz',
                   'F': 'PhysioNetChallenge2020_Training_E.tar.gz'}

# Extracted folder name
EXTRACTED_FOLDER_NAMES = {'A': 'Training_WFDB',
                          'B': 'Training_2',
                          'C': 'Training_StPetersburg',
                          'D': 'Training_PTB',
                          'E': 'WFDB',
                          'F': 'WFDB'}

# ECG leads
ECG_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Filter band limits
FILTER_BAND_LIMITS = [3, 45]

# Number of leads
NUM_LEADS = 12

# SNOMED-CT lookup
SNOMEDCT_LOOKUP = {
    270492004: {'label': 'IAVB', 'label_full': '1st degree av block'},
    195042002: {'label': 'IIAVB', 'label_full': '2nd degree av block'},
    164951009: {'label': 'abQRS', 'label_full': 'abnormal QRS'},
    61277005: {'label': 'AIVR', 'label_full': 'accelerated idioventricular rhythm'},
    426664006: {'label': 'AJR', 'label_full': 'accelerated junctional rhythm'},
    413444003: {'label': 'AMIs', 'label_full': 'acute myocardial ischemia'},
    426434006: {'label': 'AnMIs', 'label_full': 'anterior ischemia'},
    54329005: {'label': 'AnMI', 'label_full': 'anterior myocardial infarction'},
    164889003: {'label': 'AF', 'label_full': 'atrial fibrillation'},
    195080001: {'label': 'AFAFL', 'label_full': 'atrial fibrillation and flutter'},
    164890007: {'label': 'AFL', 'label_full': 'atrial flutter'},
    195126007: {'label': 'AH', 'label_full': 'atrial hypertrophy'},
    251268003: {'label': 'AP', 'label_full': 'atrial pacing pattern'},
    713422000: {'label': 'ATach', 'label_full': 'atrial tachycardia'},
    233917008: {'label': 'AVB', 'label_full': 'av block'},
    251170000: {'label': 'BPAC', 'label_full': 'blocked premature atrial contraction'},
    74615001: {'label': 'BTS', 'label_full': 'brady tachy syndrome'},
    426627000: {'label': 'Brady', 'label_full': 'bradycardia'},
    418818005: {'label': 'Brug', 'label_full': 'brugada syndrome'},
    6374002: {'label': 'BBB', 'label_full': 'bundle branch block'},
    426749004: {'label': 'CAF', 'label_full': 'chronic atrial fibrillation'},
    413844008: {'label': 'CMIs', 'label_full': 'chronic myocardial ischemia'},
    78069008: {'label': 'CRPC', 'label_full': 'chronic rheumatic pericarditis'},
    27885002: {'label': 'CHB', 'label_full': 'complete heart block'},
    713427006: {'label': 'CRBBB', 'label_full': 'complete right bundle branch block'},
    77867006: {'label': 'SQT', 'label_full': 'decreased qt interval'},
    82226007: {'label': 'DIVB', 'label_full': 'diffuse intraventricular block'},
    428417006: {'label': 'ERe', 'label_full': 'early repolarization'},
    251143007: {'label': 'ART', 'label_full': 'ecg artefacts'},
    29320008: {'label': 'ER', 'label_full': 'ectopic rhythm'},
    423863005: {'label': 'EA', 'label_full': 'electrical alternans'},
    251259000: {'label': 'HTV', 'label_full': 'high t voltage'},
    251120003: {'label': 'ILBBB', 'label_full': 'incomplete left bundle branch block'},
    713426002: {'label': 'IRBBB', 'label_full': 'incomplete right bundle branch block'},
    251200008: {'label': 'ICA', 'label_full': 'indeterminate cardiac axis'},
    425419005: {'label': 'IIs', 'label_full': 'inferior ischaemia'},
    704997005: {'label': 'ISTD', 'label_full': 'inferior st segment depression'},
    50799005: {'label': 'IAVD', 'label_full': 'isorhythmic dissociation'},
    426995002: {'label': 'JE', 'label_full': 'junctional escape'},
    251164006: {'label': 'JPC', 'label_full': 'junctional premature complex'},
    426648003: {'label': 'JTach', 'label_full': 'junctional tachycardia'},
    425623009: {'label': 'LIs', 'label_full': 'lateral ischaemia'},
    445118002: {'label': 'LAnFB', 'label_full': 'left anterior fascicular block'},
    253352002: {'label': 'LAA', 'label_full': 'left atrial abnormality'},
    67741000119109: {'label': 'LAE', 'label_full': 'left atrial enlargement'},
    446813000: {'label': 'LAH', 'label_full': 'left atrial hypertrophy'},
    39732003: {'label': 'LAD', 'label_full': 'left axis deviation'},
    164909002: {'label': 'LBBB', 'label_full': 'left bundle branch block'},
    445211001: {'label': 'LPFB', 'label_full': 'left posterior fascicular block'},
    164873001: {'label': 'LVH', 'label_full': 'left ventricular hypertrophy'},
    370365005: {'label': 'LVS', 'label_full': 'left ventricular strain'},
    251146004: {'label': 'LQRSV', 'label_full': 'low qrs voltages'},
    251147008: {'label': 'LQRSVLL', 'label_full': 'low qrs voltages in the limb leads'},
    251148003: {'label': 'LQRSP', 'label_full': 'low qrs voltages in the precordial leads'},
    28189009: {'label': 'MoII', 'label_full': 'mobitz type 2 second degree atrioventricular block'},
    54016002: {'label': 'MoI', 'label_full': 'mobitz type i wenckebach atrioventricular block'},
    713423005: {'label': 'MATach', 'label_full': 'multifocal atrial tachycardia'},
    164865005: {'label': 'MI', 'label_full': 'myocardial infarction'},
    164861001: {'label': 'MIs', 'label_full': 'myocardial ischemia'},
    65778007: {'label': 'NSIACB', 'label_full': 'non-specific interatrial conduction block'},
    698252002: {'label': 'NSIVCB', 'label_full': 'nonspecific intraventricular conduction disorder'},
    428750005: {'label': 'NSSTTA', 'label_full': 'nonspecific st t abnormality'},
    164867002: {'label': 'OldMI', 'label_full': 'old myocardial infarction'},
    10370003: {'label': 'PR', 'label_full': 'pacing rhythm'},
    67198005: {'label': 'PSVT', 'label_full': 'paroxysmal supraventricular tachycardia'},
    164903001: {'label': 'PAVB21', 'label_full': 'partial atrioventricular block 2:1'},
    284470004: {'label': 'PAC', 'label_full': 'premature atrial contraction'},
    164884008: {'label': 'PVC', 'label_full': 'premature ventricular complexes'},
    427172004: {'label': 'PVC', 'label_full': 'premature ventricular contractions'},
    111975006: {'label': 'LQT', 'label_full': 'prolonged qt interval'},
    164947007: {'label': 'LPR', 'label_full': 'prolonged pr interval'},
    164917005: {'label': 'QAb', 'label_full': 'qwave abnormal'},
    164921003: {'label': 'RAb', 'label_full': 'r wave abnormal'},
    253339007: {'label': 'RAAb', 'label_full': 'right atrial abnormality'},
    446358003: {'label': 'RAH', 'label_full': 'right atrial hypertrophy'},
    47665007: {'label': 'RAD', 'label_full': 'right axis deviation'},
    59118001: {'label': 'RBBB', 'label_full': 'right bundle branch block'},
    89792004: {'label': 'RVH', 'label_full': 'right ventricular hypertrophy'},
    55930002: {'label': 'STC', 'label_full': 's t changes'},
    49578007: {'label': 'SPRI', 'label_full': 'shortened pr interval'},
    427393009: {'label': 'SA', 'label_full': 'sinus arrhythmia'},
    426177001: {'label': 'SB', 'label_full': 'sinus bradycardia'},
    426783006: {'label': 'SNR', 'label_full': 'sinus rhythm'},
    427084000: {'label': 'STach', 'label_full': 'sinus tachycardia'},
    429622005: {'label': 'STD', 'label_full': 'st depression'},
    164931005: {'label': 'STE', 'label_full': 'st elevation'},
    164930006: {'label': 'STIAb', 'label_full': 'st interval abnormal'},
    63593006: {'label': 'SVPB', 'label_full': 'supraventricular premature beats'},
    426761007: {'label': 'SVT', 'label_full': 'supraventricular tachycardia'},
    251139008: {'label': 'ALR', 'label_full': 'suspect arm ecg leads reversed'},
    164934002: {'label': 'TAb', 'label_full': 't wave abnormal'},
    59931005: {'label': 'TInv', 'label_full': 't wave inversion'},
    251242005: {'label': 'UTall', 'label_full': 'tall u wave'},
    164937009: {'label': 'UAb', 'label_full': 'u wave abnormal'},
    11157007: {'label': 'VBig', 'label_full': 'ventricular bigeminy'},
    17338001: {'label': 'VEB', 'label_full': 'ventricular ectopic beats'},
    75532003: {'label': 'VEsB', 'label_full': 'ventricular escape beat'},
    81898007: {'label': 'VEsR', 'label_full': 'ventricular escape rhythm'},
    164896001: {'label': 'VF', 'label_full': 'ventricular fibrillation'},
    111288001: {'label': 'VFL', 'label_full': 'ventricular flutter'},
    266249003: {'label': 'VH', 'label_full': 'ventricular hypertrophy'},
    251266004: {'label': 'VPP', 'label_full': 'ventricular pacing pattern'},
    195060002: {'label': 'VPEx', 'label_full': 'ventricular pre excitation'},
    164895002: {'label': 'VTach', 'label_full': 'ventricular tachycardia'},
    251180001: {'label': 'VTrig', 'label_full': 'ventricular trigeminy'},
    195101003: {'label': 'WAP', 'label_full': 'wandering atrial pacemaker'},
    74390002: {'label': 'WPW', 'label_full': 'wolff parkinson white pattern'}
}
