# Overview 

During the tutorial session, all code written by the participants will be completed using the Jupyter notebook, an iPython kernel / server that will be running from your own personal AWS instance and accessed through a web-browser. Through this web-based interface one will be able to write, edit and run code in an easy way without needing to use the Linux command line. For more advanced users, this entire Github repository is available for access from the command line at `~/dl_tutorial`.

To setup and connect to the Jupyter see instructions on home page: https://github.com/peterchang77/dl_tutorial 

### Outline

The workshop session will be organized into a five-part series. The majority of background information is present in the first lecture. The remaining four notebooks cover implementation of training and inference of a CNN. For best results, consider advancing through the following topics in order:

&nbsp;&nbsp;&nbsp;&nbsp; **01 - Introduction to Data, Tensorflow and Deep Learning** <br/>
&nbsp;&nbsp;&nbsp;&nbsp; **02 - Training a Classifier** <br/>
&nbsp;&nbsp;&nbsp;&nbsp; **03 - Inference with a Classifier** <br/>
&nbsp;&nbsp;&nbsp;&nbsp; **04 - Training a U-Net** <br/>
&nbsp;&nbsp;&nbsp;&nbsp; **05 - Inference with a U-Net**

Without an active AWS instance it is possible to simply launch the `*.ipynb` files directly here in Github to preview content.

# Data I/O

The `data.py` module abracts a pipeline for loading random slices of preprocessed data. The syntax is as follows:
```
import data

dat, lbl = data.load(mode='train', n=16)

```

From the corresponding docstring within the method:
```
def load(mode='train', n=1, sid=None, z=None, return_mask=False):
    """
    Method to open n random slices of data and corresponding labels. Note that this
    method will load data in a stratified manner such that approximately 50% of all 
    returned data will contain tumor.

    :params

      (str) mode : 'train' or 'valid'
      (int) n : number of examples to open
      (str) sid : if provided, will load specific study ID
      (int) z : if provided, will load specifc slice
      (bool) return_mask : if True, will also return mask containing brain parenchyma

    :return

      (np.array) dat : N x I x J x 4 input (dtype = 'float32')
      (np.array) lbl : N x I x J x 1 label (dtype = 'uint8')
      (np.array) msk : N x I x J x 1 lmask (dtype = 'float32'), (optional)

    """
```

### Data source directory

By default, upon import, the `data.py` module will search for a directory on your local machine located at `/data/brats/npy`; this is the location of the full dataset if you clone the AWS AMI provided as part of this tutorial and are following along currently on an EC2 instance. If found, this directory will be set as the root for loading data. If absent, then the toy dataset present as part of this repository located at `../data` will be used.

Note that the data source directory can be manually set any time after module import with the `data.set_root()` method.
