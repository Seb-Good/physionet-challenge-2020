"""
utils.py
--------
Utility functions for formatting ECG datasets.
By: Sebastian D. Goodfellow, Ph.D., 2020
"""

# 3rd party imports
import json
from urllib.request import urlopen


def get_snomedct_concept_by_id(id_number):
    """SNOMED-CT request."""
    base_url = 'https://browser.ihtsdotools.org/snowstorm/snomed-ct'
    edition = 'MAIN'
    version = '2019-07-31'
    url = base_url + '/browser/' + edition + '/' + version + '/concepts/' + id_number
    response = urlopen(url).read()
    data = json.loads(response.decode('utf-8'))
    return data['fsn']['term']
