# Import 3rd party libraries
import os

# Set working directory
WORKING_DIR = (
    os.path.dirname(
        os.path.dirname(
            os.path.realpath(__file__)
        )
    )
)

# Set data directory
DATA_DIR = os.path.join(WORKING_DIR, 'data')

# Set log directory
LOG_DIR = os.path.join(WORKING_DIR, 'logs')
