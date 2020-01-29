import os
import typing

import cv2 as cv
import lz4.frame
import numpy as np
import tensorflow as tf


def compress_array(
        array: np.ndarray,
        compr_type: typing.Optional[str] = "auto",
) -> bytes:
    """Compresses the provided numpy array according to a predetermined strategy.

    If ``compr_type`` is 'auto', the best strategy will be automatically selected based on the input
    array type. If ``compr_type`` is an empty string (or ``None``), no compression will be applied.
    """
    assert compr_type is None or compr_type in ["lz4", "float16+lz4", "uint8+jpg",
                                                "uint8+jp2", "uint16+jp2", "auto", ""], \
        f"unrecognized compression strategy '{compr_type}'"
    if compr_type is None or not compr_type:
        return array.tobytes()
    if compr_type == "lz4":
        return lz4.frame.compress(array.tobytes())
    if compr_type == "float16+lz4":
        assert np.issubdtype(array.dtype, np.floating), "no reason to cast to float16 is not float32/64"
        return lz4.frame.compress(array.astype(np.float16).tobytes())
    if compr_type == "uint8+jpg":
        assert array.ndim == 2 or (array.ndim == 3 and (array.shape[2] == 1 or array.shape[2] == 3)), \
            "jpg compression via tensorflow requires 2D or 3D image with 1/3 channels in last dim"
        if array.ndim == 2:
            array = np.expand_dims(array, axis=2)
        assert array.dtype == np.uint8, "jpg compression requires uint8 array"
        return tf.io.encode_jpeg(array).numpy()
    if compr_type == "uint8+jp2" or compr_type == "uint16+jp2":
        assert array.ndim == 2 or (array.ndim == 3 and (array.shape[2] == 1 or array.shape[2] == 3)), \
            "jp2 compression via opencv requires 2D or 3D image with 1/3 channels in last dim"
        if array.ndim == 2:
            array = np.expand_dims(array, axis=2)
        assert array.dtype == np.uint8 or array.dtype == np.uint16, "jp2 compression requires uint8/16 array"
        if os.getenv("OPENCV_IO_ENABLE_JASPER") is None:
            # for local/trusted use only; see issue here: https://github.com/opencv/opencv/issues/14058
            os.environ["OPENCV_IO_ENABLE_JASPER"] = "1"
        retval, buffer = cv.imencode(".jp2", array)
        assert retval, "JPEG2000 encoding failed"
        return buffer.tobytes()
    # could also add uint16 png/tiff via opencv...
    if compr_type == "auto":
        # we cheat for auto-decompression by prefixing the strategy in the bytecode
        if array.ndim == 2 or (array.ndim == 3 and (array.shape[2] == 1 or array.shape[2] == 3)):
            if array.dtype == np.uint8:
                return b"uint8+jpg" + compress_array(array, compr_type="uint8+jpg")
            if array.dtype == np.uint16:
                return b"uint16+jp2" + compress_array(array, compr_type="uint16+jp2")
        return b"lz4" + compress_array(array, compr_type="lz4")


def decompress_array(
        buffer: typing.Union[bytes, np.ndarray],
        compr_type: typing.Optional[str] = "auto",
        dtype: typing.Optional[typing.Any] = None,
        shape: typing.Optional[typing.Union[typing.List, typing.Tuple]] = None,
) -> np.ndarray:
    """Decompresses the provided numpy array according to a predetermined strategy.

    If ``compr_type`` is 'auto', the correct strategy will be automatically selected based on the array's
    bytecode prefix. If ``compr_type`` is an empty string (or ``None``), no decompression will be applied.

    This function can optionally convert and reshape the decompressed array, if needed.
    """
    compr_types = ["lz4", "float16+lz4", "uint8+jpg", "uint8+jp2", "uint16+jp2"]
    assert compr_type is None or compr_type in compr_types or compr_type in ["", "auto"], \
        f"unrecognized compression strategy '{compr_type}'"
    assert isinstance(buffer, bytes) or buffer.dtype == np.uint8, "invalid raw data buffer type"
    if isinstance(buffer, np.ndarray):
        buffer = buffer.tobytes()
    if compr_type == "lz4" or compr_type == "float16+lz4":
        buffer = lz4.frame.decompress(buffer)
    if compr_type == "uint8+jpg":
        # tf.io.decode_jpeg often segfaults when initializing parallel pipelines, let's avoid it...
        # buffer = tf.io.decode_jpeg(buffer).numpy()
        buffer = cv.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv.IMREAD_UNCHANGED)
    if compr_type.endswith("+jp2"):
        if os.getenv("OPENCV_IO_ENABLE_JASPER") is None:
            # for local/trusted use only; see issue here: https://github.com/opencv/opencv/issues/14058
            os.environ["OPENCV_IO_ENABLE_JASPER"] = "1"
        buffer = cv.imdecode(np.frombuffer(buffer, dtype=np.uint8), flags=cv.IMREAD_UNCHANGED)
    if compr_type == "auto":
        decompr_buffer = None
        for compr_code in compr_types:
            if buffer.startswith(compr_code.encode("ascii")):
                decompr_buffer = decompress_array(buffer[len(compr_code):], compr_type=compr_code,
                                                  dtype=dtype, shape=shape)
                break
        assert decompr_buffer is not None, "missing auto-decompression code in buffer"
        buffer = decompr_buffer
    array = np.frombuffer(buffer, dtype=dtype)
    if shape is not None:
        array = array.reshape(shape)
    return array


