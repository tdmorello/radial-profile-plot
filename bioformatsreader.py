"""
A class to read image files with BioFormats
"""

import logging
import pathlib

import jpype
import jpype.imports
import numpy as np
from jpype.types import JObject

if not jpype.isJVMStarted():
    # jpype.startJVM(classpath=['jars/*'])
    jpype.startJVM(classpath=['/Users/tim/Research/Data-Analysis/radial-profile-plot/jars/*'])

from loci.common import DebugTools  # type: ignore
from loci.common.services import ServiceFactory  # type: ignore
from loci.formats import ImageReader  # type: ignore
from loci.formats.services import OMEXMLService  # type: ignore
from ome.xml.meta import MetadataRetrieve  # type: ignore

DebugTools.enableLogging("ERROR")

logger = logging.getLogger(__name__)


class BioFormatsReader:
    """
    Bio-Formats ImageReader class with some added functionality to work
    with data in native Python
    """

    _pixel_dtypes = {
        'int8': np.dtype(np.uint8),
        'uint8': np.dtype(np.uint8),
        'uint16': np.dtype(np.uint16),
    }

    def __init__(self, filepath):
        '''Initialize the Reader object.

        Parameters
        ----------
        filepath : string or path-like object
            Path to an image file.
        '''
        # change _rdr to rdr
        # change read_image to read?
        # change _filepath to filepath (make unsettable)

        self._filepath = pathlib.Path(filepath)
        self.rdr = None
        self._metadata = None

        self._init_reader()

    def _init_reader(self) -> None:
        factory = ServiceFactory()
        service = JObject(factory.getInstance(OMEXMLService), OMEXMLService)
        metadata = service.createOMEXMLMetadata()

        self.rdr = ImageReader()
        self.rdr.setMetadataStore(metadata)

        logger.debug("Opening '%s'", str(self._filepath))
        self.rdr.setId(str(self._filepath))
        self._metadata = JObject(metadata, MetadataRetrieve)

    @property
    def pixel_dtype(self):
        return self._pixel_dtypes[self._metadata.getPixelsType(0).toString()]

    @property
    def size_X(self):
        return self.rdr.getSizeX()

    @property
    def size_Y(self):
        return self.rdr.getSizeY()

    @property
    def size_C(self):
        return self.rdr.getSizeC()

    def read(self, c: int=None, series: int=None, z: int=0, t: int=0, XYWH=None):  # noqa
        '''Read bytes to numpy array

        Parameters
        ----------
        c : int
            The channel index, by default all channels.
        series : int, optional
            The series index, by default 0.
        z : int, optional
            The Z index, by default 0.
        t : int, optional
            The T index, by default 0.
        XYWH :
            Read a section of the image

        Returns
        -------
        ndarray
            A numpy array representation of a single image.
        '''

        dtype = self.pixel_dtype

        if series is not None:
            self.rdr.setSeries(series)

        if XYWH is not None:
            def openBytes_func(idx: int):
                return self.rdr.openBytes(idx, XYWH[0], XYWH[1], XYWH[2], XYWH[3])
            width, height = XYWH[2], XYWH[3]
        else:
            openBytes_func = self.rdr.openBytes
            width, height = self.rdr.getSizeX(), self.rdr.getSizeY()

        if c is None:
            imgs = []
            for i in range(self.rdr.getSizeC()):
                index = self.rdr.getIndex(z, i, t)
                byte_array = openBytes_func(index)
                # Convert to array first to copy byte array (8 bit), then properly read as data type
                img = np.frombuffer(np.array(byte_array), dtype).reshape(height, width)
                imgs.append(img)
            img = np.dstack(imgs)
        else:
            index = self.rdr.getIndex(z, c, t)
            byte_array = openBytes_func(index)
            # Convert to array first to copy byte array (8 bit), then properly read as data type
            img = np.frombuffer(np.array(byte_array), dtype).reshape(height, width)

        return img


def main():
    filepath = 'data/b41f-sag_L-neun_pv-w09_2-cla-63x.czi'
    reader = BioFormatsReader(filepath)
    XYWH = (5000, 5000, 150, 200)
    img = reader.read(XYWH=XYWH)
    print(img.shape)


if __name__ == '__main__':
    main()
