import os, glob, pickle
import numpy as np

# ======================================================================
# METHODS FOR MANAGING and LOADING DATA 
# ======================================================================
# 
# This script contains methods for:
# 
#   (1) Saving data summary into a data.pickle file which contains a 
#       dictionary containing study IDs, pp statistics and whether or not
#       that study is in the training or validation cohort. This dict
#       should be updated every time new data is added, prior to using 
#       the data.load() method.
# 
#   (2) Load random slice(s) of data from random studies in either training
#       or validation cohort.
# 
# ======================================================================

def create_summary(root='../data', valid_ratio=0.2):
    """
    Method to create a summary *.pickle file in the current directory
    containing summary dictionary for the dataset of form:

    summary = {
        'train': [
            {'studyid': 'study_00', 'mean': 0, 'sd': 0},
            {'studyid': 'study_01', 'mean': 0, 'sd': 0}, ...
        ],
        'valid': [
            {'studyid': 'study_02', 'mean': 0, 'sd': 0},
            {'studyid': 'study_03', 'mean': 0, 'sd': 0}, ...
        ]
    }

    """
    dfiles = glob.glob('%s/*/dat.npy' % root)
    summary = {'train': [], 'valid': []} 

    for n, dfile in enumerate(dfiles): 
        print('Saving summary %03i: %s' % (n + 1, dfile))
        dat = np.memmap(dfile, dtype='int16', mode='r')

        sid = os.path.basename(os.path.dirname(dfile))
        group = 'train' if np.random.rand() > valid_ratio else 'valid'
        summary[group].append({
            'studyid': sid,
            'mean': np.mean(dat[dat > 0]),
            'sd': np.std(dat[dat > 0])})

    pickle.dump(summary, open('data.pickle', 'wb'))

def load(mode='train', n=1, root='../data'):
    """
    Method to open n random slices of data and corresponding labels 

    :params

      (str) mode : 'train' or 'valid'
      (int) n : number of examples to open
      (str) root : root directory containing data

    :return

      (np.array) dat : N x I x J x 4 input (dtype = 'float32')
      (np.array) lbl : N x I x J x 1 label (dtype = 'uint8')

    """
    global summary

    indices = np.random.randint(0, len(summary[mode]), n)
    dats = []
    lbls = []

    for ind in indices:
        stats = summary[mode][ind]
        
        # --- Load random data slice
        fname = '%s/%s/dat.npy' % (root, stats['studyid'])
        dat = np.memmap(fname, dtype='int16', mode='r')
        dat = dat.reshape(-1, 240, 240, 4)

        z = np.random.randint(dat.shape[0])
        dats.append((dat[z] - stats['mean']) / stats['sd'])
        
        # --- Load corresponding label slice
        fname = '%s/%s/lbl.npy' % (root, stats['studyid'])
        lbl = np.memmap(fname, dtype='uint8', mode='r')
        lbl = lbl.reshape(-1, 240, 240, 1)
        lbls.append(lbl[z])

    dats = np.stack(dats, axis=0)
    lbls = np.stack(lbls, axis=0)

    return dats, lbls

# --- Load summary pickle load into memory if present
if os.path.exists('data.pickle'):
    summary = pickle.load(open('data.pickle', 'rb'))

if __name__ == '__main__':

    # create_summary()
    pass
