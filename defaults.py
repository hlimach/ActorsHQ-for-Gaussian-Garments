import os
import socket
from pathlib import Path
from munch import munchify

hostname = socket.gethostname()
DEFAULTS = dict()

# =============================================================================
#  Data paths
# =============================================================================

# DO NOT EDIT

if 'borong-System-Product-Name' in hostname:
    DEFAULTS['GG_root'] = Path(f'/home/hramzan/Desktop/semester-project/Gaussian-Garments')
    DEFAULTS['data_root'] = Path(f'/run/user/'+str(os.getuid())+f'/gvfs/smb-share:server=mocap-stor-02.inf.ethz.ch,share=work/ait_datasets/zext_HumanRF_4x')
elif 'ohws68' in hostname:
    DEFAULTS['GG_root'] = Path(f'')
    DEFAULTS['data_root'] = Path(f'')
if hostname == 'ait-server-04.inf.ethz.ch':
    DEFAULTS['GG_root'] = Path(f'/local/home/agrigorev/Workdir/Gaussian-Garments')
    DEFAULTS['data_root'] = Path(f'/run/user/'+str(os.getuid())+f'/mnt/work/ait_datasets/zext_HumanRF_4x/')
    DEFAULTS['output_root'] = Path(f'/data/agrigorev/02_Projects/opengaga/Outputs/')
else:
    DEFAULTS['GG_root'] = Path(f'../Gaussian-Garments')
    DEFAULTS['data_root'] = Path('../Gaussian-Garments/data/input')
    DEFAULTS['output_root'] = DEFAULTS['GG_root'] / 'data' / 'outputs'
    

# DEFAULTS['output_root'] = DEFAULTS['GG_root'] / 'data' / 'outputs'

# =============================================================================
#  Mask generation parameters
# =============================================================================
DEFAULTS['box_threshold'] = 0.35
DEFAULTS['text_threshold'] = 0.25
DEFAULTS['portrait_cams'] = ['Cam005', 'Cam006', 'Cam007', 'Cam008', 'Cam021', 'Cam022', 'Cam023', 'Cam024', 'Cam037', 'Cam038', 'Cam039', 'Cam040', 'Cam053', 'Cam054', 'Cam055', 'Cam056', 'Cam077', 'Cam078', 'Cam079', 'Cam080', 'Cam093', 'Cam094', 'Cam095', 'Cam096', 'Cam109', 'Cam110', 'Cam111', 'Cam112', 'Cam121', 'Cam122', 'Cam123', 'Cam124', 'Cam125', 'Cam126', 'Cam127', 'Cam128', 'Cam131', 'Cam132', 'Cam133', 'Cam134', 'Cam135', 'Cam136', 'Cam139', 'Cam140', 'Cam141', 'Cam142', 'Cam143', 'Cam144', 'Cam147', 'Cam148', 'Cam149', 'Cam150', 'Cam151', 'Cam152', 'Cam155', 'Cam156', 'Cam157', 'Cam158', 'Cam159', 'Cam160']


# turns the dictionary into a Munch object (so you can use e.g. DEFAULTS.data_root)
DEFAULTS = munchify(DEFAULTS)