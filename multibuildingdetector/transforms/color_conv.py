import cupy as cp


def hsv_to_rgb(hsv):
    """HSV to RGB color space conversion.
    Parameters
    ----------
    hsv : numpy array
        The image in HSV format, in a 3-D array of shape ``(.., .., 3)``.
    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    Conversion between RGB and HSV color spaces results in some loss of
    precision, due to integer arithmetic and rounding [1]_.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/HSL_and_HSV
    """
    hsv = cp.array(hsv)
    hi = cp.floor(hsv[:, :, 0] * 6)
    f = hsv[:, :, 0] * 6 - hi
    p = hsv[:, :, 2] * (1 - hsv[:, :, 1])
    q = hsv[:, :, 2] * (1 - f * hsv[:, :, 1])
    t = hsv[:, :, 2] * (1 - (1 - f) * hsv[:, :, 1])
    v = hsv[:, :, 2]

    hi = cp.dstack([hi, hi, hi]).astype(cp.uint8) % 6
    out = cp.choose(hi, cp.stack([cp.dstack((v, t, p)),
                                  cp.dstack((q, v, p)),
                                  cp.dstack((p, v, t)),
                                  cp.dstack((p, q, v)),
                                  cp.dstack((t, p, v)),
                                  cp.dstack((v, p, q))]))

    return cp.asnumpy(out)


def rgb_to_hsv(rgb):
    """RGB to HSV color space conversion.
    Parameters
    ----------
    rgb : numpy array
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    Returns
    -------
    out : ndarray
        The image in HSV format, in a 3-D array of shape ``(.., .., 3)``.
    Conversion between RGB and HSV color spaces results in some loss of
    precision, due to integer arithmetic and rounding [1]_.
    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/HSL_and_HSV
    """
    try:
        delta = cp.array(rgb.ptp(-1))
        rgb = cp.array(rgb)
        out = cp.empty_like(rgb)
    except:
        print('Error in init')

    # -- V channel
    out_v = rgb.max(-1)

    # -- S channel
    # Ignore warning for zero divided by zero
    try:
        out_s = delta / out_v
        out_s[delta == 0.] = 0.
    except:
        print('Error in ^s')

    # -- H channel
    # red is max
    try:
        idx = (rgb[:, :, 0] == out_v)
        out[idx][:, 0] = (rgb[idx][:, 1] - rgb[idx][:, 2]) / delta[idx]
    except:
        print('Error in red')

    # green is max
    try:
        idx = (rgb[:, :, 1] == out_v)
        out[idx][:, 0] = 2. + (rgb[idx][:, 2] - rgb[idx][:, 0]) / delta[idx]
    except:
        print('Error in green')

    # blue is max
    try:
        idx = (rgb[:, :, 2] == out_v)
        out[idx][:, 0] = 4. + (rgb[idx][:, 0] - rgb[idx][:, 1]) / delta[idx]
        out_h = (out[:, :, 0] / 6.) % 1.
        out_h[delta == 0.] = 0.
    except:
        print('Error in blue')

    # -- output
    try:
        out[:, :, 0] = out_h
        out[:, :, 1] = out_s
        out[:, :, 2] = out_v
    except:
        print('Error in out')

    # remove NaN
    try:
        out[cp.isnan(out)] = 0
    except:
        print('Error in nan')

    return cp.asnumpy(out)
