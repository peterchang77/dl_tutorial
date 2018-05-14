import matplotlib.pyplot as plt
import numpy as np, cv2

def imshow(dat, lbl=None, index=0, radius=1, vm=None, title=None):
    """
    Method to display dat with lbl overlay if provided.

    :params

      (int) index : index of channel to display; set to None if RGB image
      (int) radius : thickness of outline for overlays 

    """
    im = np.squeeze(dat)
    if im.ndim == 3:
        im = im[..., index]

    if im.ndim > 2:
        print('Error input dat is not H x W x N in dimensions')
        return

    # --- Overlay if lbl also provided
    if lbl is not None:

        m = np.squeeze(lbl)
        if m.ndim > 2:
            print('Error input lbl is not H x W x N in dimension')
            return

        masks = []
        for ch in range(np.max(m)):
            masks.append(perim_2d(m == ch, radius=radius))
        im = imoverlay(im, np.stack(masks, axis=2)) 

    # --- Display image
    cmap = None if im.ndim == 3 or im.dtype == 'uint8' else plt.cm.gist_gray
    if vm is not None:
        plt.imshow(im, cmap=cmap, vmin=vm[0], vmax=vm[1])
    else:
        plt.imshow(im, cmap=cmap)

    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.show()

def imoverlay(dat, lbl, vm=None):
    """
    Method to superimpose lbls on 2D image

    :params

      (np.array) dat : 2D image of format H x W or H x W x C

        if C is empty (grayscale), image will be converted to 3-channel grayscale
        if C == 1 (grayscale), image will be squeezed then converted to 3-channel grayscale
        if C == 3 (rgb), image will not be converted

      (np.array) lbl : 2D lbl(s) of format H x W or H x W x N

    """
    # --- Adjust shapes of dat
    if dat.ndim  == 3 and dat.shape[-1] == 1:
        dat = np.squeeze(dat)

    if dat.ndim  == 2:
        dat = gray2rgb(dat, vm=vm)

    # --- Adjust shapes of lbl
    if lbl.ndim  == 2:
        lbl = np.expand_dims(lbl, axis=2)
    lbl = lbl.astype('bool')

    # --- Overlay lbl(s)
    if dat.ndim == 3 and lbl.ndim == 3 and dat.shape[2] == 3:

        rgb = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
        overlay = []

        for channel in range(3):
            layer = dat[..., channel]
            for i in range(lbl.shape[2]):
                layer[lbl[..., i]] = rgb[i % 6][channel]
            overlay.append(layer)

        return np.stack(overlay, axis=2)

def gray2rgb(dat, maximum_val=1, percentile=0, vm=None):
    """
    Method to convert H x W grayscale tensor to H x W x 3 RGB grayscale

    :params

    (np.array) dat : input H x W tensor
    (int) maximum_val : maximum value in output
      if maximum_val == 1, output is assumed to be float32
      if maximum_val == 255, output is assumed to be uint8 (standard 256 x 256 x 256 RGB image)
    (int) percentile : lower bound to set to 0

    """
    if vm is None:
        dat_min, dat_max = np.percentile(dat, percentile), np.percentile(dat, 100 - percentile)
    else:
        dat_min, dat_max = vm[0], vm[1]

    den = dat_max - dat_min
    den = 1 if den == 0 else den
    dat = (dat - dat_min) / den
    dat[dat > 1] = 1
    dat[dat < 0] = 0
    dat = dat * maximum_val
    dat = np.expand_dims(dat, 2)
    dat = np.tile(dat, [1, 1, 3])

    dtype = 'float32' if maximum_val == 1 else 'uint8'

    return dat.astype(dtype)

def perim_2d(lbl, radius=1):
    """
    Method to create perimeter of 2D binary mask through morphologic erosion.

    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(radius * 2 + 1, radius * 2 + 1))
    m = lbl > 0 

    return m ^ cv2.erode(src=m.astype('uint8'), kernel=kernel, iterations=1)

