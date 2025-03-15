import os
import socket
from pathlib import Path
from munch import munchify

hostname = socket.gethostname()
DEFAULTS = dict()

# =============================================================================
# Data Paths - Please set the correct paths for your system here
# =============================================================================

# Path that is treated as the input directory for the Gaussian Garments repo.
# The ActorsHQ dataset will be reorangized to match the input requirements for
# the Gaussian Garments pipeline and the symlinks will be created to this path.
# Data will NOT be duplicated here, only the generated garment masks, and 
# unpacked smplx model for each frame will be written to this path.
# This should be the same as the data_root set in Gaussian Garments defaults file.
DEFAULTS['data_root'] = Path(f'')


# Path that is treated as the output directory for the Gaussian Garments repo.
# This is required so that stage1 outputs can be written to this path in a format
# that can then be read by subsequent stages of Gaussian-Garments pipeline.
# This should be the same as the output_root set in Gaussian Garments defaults file.
DEFAULTS['output_root'] = Path(f'')


# Path to where the ActorsHQ model files are stored. This directory should 
# contain the 'smplx' subdirectory, which has male, female, neutral .npz files.
DEFAULTS['aux_root'] = Path(f'')


# Path to where the ActorsHQ dataset is stored, only used for reading data.
DEFAULTS['AHQ_data_root'] = Path(f'')


if 'borong-System-Product-Name' in hostname:
    DEFAULTS['data_root'] = Path(f'/home/hramzan/Desktop/semester-project/Gaussian-Garments/data/input')

    DEFAULTS['output_root'] = Path(f'/home/hramzan/Desktop/semester-project/Gaussian-Garments/data/outputs')

    DEFAULTS['aux_root'] = Path(f'/home/hramzan/Desktop/semester-project/Gaussian-Garments/data/input')

    DEFAULTS['AHQ_data_root'] = Path(f'/run/user/'+str(os.getuid())+f'/gvfs/smb-share:server=mocap-stor-02.inf.ethz.ch,share=work/ait_datasets/zext_HumanRF_4x')

elif 'ohws68' in hostname:
    DEFAULTS['GG_root'] = Path(f'')
    DEFAULTS['data_root'] = Path(f'')

elif hostname == 'ait-server-04.inf.ethz.ch':
    DEFAULTS['AHQ_data_root'] = Path(f'/mnt/work/ait_datasets/zext_HumanRF_4x/')
    DEFAULTS['data_root'] = Path(f'/data/agrigorev/02_Projects/opengaga/Inputs/')
    DEFAULTS['output_root'] = Path(f'/data/agrigorev/02_Projects/opengaga/Outputs/')
    DEFAULTS['aux_root'] = Path(f'')


# =============================================================================
# Mask generation parameters
# =============================================================================

# Thresholds for the bounding box predictions from GroundingDINO
DEFAULTS['box_threshold'] = 0.35
DEFAULTS['text_threshold'] = 0.25

# List of cameras that are portrait cameras in the ActorsHQ dataset
DEFAULTS['portrait_cams'] = ['Cam005', 'Cam006', 'Cam007', 'Cam008', 'Cam021', 'Cam022', 'Cam023', 'Cam024', 'Cam037', 'Cam038', 'Cam039', 'Cam040', 'Cam053', 'Cam054', 'Cam055', 'Cam056', 'Cam077', 'Cam078', 'Cam079', 'Cam080', 'Cam093', 'Cam094', 'Cam095', 'Cam096', 'Cam109', 'Cam110', 'Cam111', 'Cam112', 'Cam121', 'Cam122', 'Cam123', 'Cam124', 'Cam125', 'Cam126', 'Cam127', 'Cam128', 'Cam131', 'Cam132', 'Cam133', 'Cam134', 'Cam135', 'Cam136', 'Cam139', 'Cam140', 'Cam141', 'Cam142', 'Cam143', 'Cam144', 'Cam147', 'Cam148', 'Cam149', 'Cam150', 'Cam151', 'Cam152', 'Cam155', 'Cam156', 'Cam157', 'Cam158', 'Cam159', 'Cam160']


# turns the dictionary into a Munch object (so you can use e.g. DEFAULTS.data_root)
DEFAULTS = munchify(DEFAULTS)