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

dat, lbl = data.load(mode='train', n=16, root='../data')

```

From the corresponding docstring within the method:
```
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
```

Not here that the root for testing purposes can point to dl_tutorial/data for testing purposes. During the workshop session, the full dataset will be mapped in the EC2 instances in `/data/brats/npy`.
