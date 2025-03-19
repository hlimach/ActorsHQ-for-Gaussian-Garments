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
    DEFAULTS['aux_root'] = Path(f'/data/agrigorev/02_Projects/opengaga/aux_data/')


# =============================================================================
# Mask generation parameters
# =============================================================================

# Set this to True if you want to generate masks only for vertical cameras
only_portrait = True

portrait_cams = ['Cam005', 'Cam006', 'Cam007', 'Cam008', 'Cam021', 'Cam022', 'Cam023', 'Cam024', 'Cam037', 'Cam038', 'Cam039', 'Cam040', 'Cam053', 'Cam054', 'Cam055', 'Cam056', 'Cam077', 'Cam078', 'Cam079', 'Cam080', 'Cam093', 'Cam094', 'Cam095', 'Cam096', 'Cam109', 'Cam110', 'Cam111', 'Cam112', 'Cam121', 'Cam122', 'Cam123', 'Cam124', 'Cam125', 'Cam126', 'Cam127', 'Cam128', 'Cam131', 'Cam132', 'Cam133', 'Cam134', 'Cam135', 'Cam136', 'Cam139', 'Cam140', 'Cam141', 'Cam142', 'Cam143', 'Cam144', 'Cam147', 'Cam148', 'Cam149', 'Cam150', 'Cam151', 'Cam152', 'Cam155', 'Cam156', 'Cam157', 'Cam158', 'Cam159', 'Cam160']

all_garments = ['Cam002', 'Cam003', 'Cam005', 'Cam006', 'Cam007', 'Cam008', 'Cam010', 'Cam011', 'Cam014', 'Cam015', 'Cam018', 'Cam019', 'Cam021', 'Cam022', 'Cam023', 'Cam024', 'Cam026', 'Cam027', 'Cam030', 'Cam031', 'Cam034', 'Cam035', 'Cam037', 'Cam038', 'Cam039', 'Cam040', 'Cam042', 'Cam043', 'Cam046', 'Cam047', 'Cam050', 'Cam051', 'Cam053', 'Cam054', 'Cam055', 'Cam056', 'Cam058', 'Cam059', 'Cam062', 'Cam063', 'Cam066', 'Cam067', 'Cam070', 'Cam071', 'Cam074', 'Cam075', 'Cam077', 'Cam078', 'Cam079', 'Cam080', 'Cam082', 'Cam083', 'Cam086', 'Cam087', 'Cam090', 'Cam091', 'Cam093', 'Cam094', 'Cam095', 'Cam096', 'Cam098', 'Cam099', 'Cam102', 'Cam103', 'Cam106', 'Cam107', 'Cam109', 'Cam110', 'Cam111', 'Cam112', 'Cam114', 'Cam115', 'Cam118', 'Cam119', 'Cam126', 'Cam127', 'Cam128', 'Cam131', 'Cam132', 'Cam135', 'Cam136', 'Cam139', 'Cam140', 'Cam143', 'Cam144', 'Cam147', 'Cam148', 'Cam151', 'Cam152', 'Cam155', 'Cam156', 'Cam159', 'Cam160']

only_upper = ['Cam004', 'Cam012', 'Cam016', 'Cam020', 'Cam028', 'Cam032', 'Cam036', 'Cam044', 'Cam048', 'Cam052', 'Cam060', 'Cam064', 'Cam068', 'Cam072', 'Cam076', 'Cam084', 'Cam088', 'Cam092', 'Cam100', 'Cam104', 'Cam108', 'Cam116', 'Cam120', 'Cam121', 'Cam122', 'Cam123', 'Cam124', 'Cam125', 'Cam133', 'Cam134', 'Cam141', 'Cam142', 'Cam149', 'Cam150', 'Cam157', 'Cam158']

only_lower = ['Cam001', 'Cam009', 'Cam013', 'Cam017', 'Cam025', 'Cam029', 'Cam033', 'Cam041', 'Cam045', 'Cam049', 'Cam057', 'Cam061', 'Cam065', 'Cam069', 'Cam073', 'Cam081', 'Cam085', 'Cam089', 'Cam097', 'Cam101', 'Cam105', 'Cam113', 'Cam117', 'Cam129', 'Cam130', 'Cam137', 'Cam138', 'Cam145', 'Cam146', 'Cam153', 'Cam154']

DEFAULTS['upper'] = only_upper + all_garments
DEFAULTS['lower'] = only_lower + all_garments
DEFAULTS['dress'] = only_upper + only_lower + all_garments

if only_portrait:
    DEFAULTS['upper'][:] = list(set(DEFAULTS['upper']) & set(portrait_cams))
    DEFAULTS['lower'][:] = list(set(DEFAULTS['lower']) & set(portrait_cams))
    DEFAULTS['dress'][:] = list(set(DEFAULTS['dress']) & set(portrait_cams))

# Thresholds for the bounding box predictions from GroundingDINO
DEFAULTS['box_threshold'] = 0.35
DEFAULTS['text_threshold'] = 0.25

# turns the dictionary into a Munch object (so you can use e.g. DEFAULTS.data_root)
DEFAULTS = munchify(DEFAULTS)