# Standard library imports
import logging
import pathlib

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt

# Start logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.WARNING)

# Java imports
import jpype
import jpype.imports
from jpype.types import JObject
if not jpype.isJVMStarted():
    jpype.startJVM(classpath = ['jars/*'])

from loci.formats import IFormatReader, ChannelSeparator, ImageReader  # type: ignore
from ome.xml.meta import MetadataRetrieve  # type: ignore
from loci.common.services import ServiceFactory  # type: ignore
from loci.formats.services import OMEXMLService  # type: ignore
from loci.formats.in_ import DynamicMetadataOptions  # type: ignore
from loci.common import DebugTools  # type: ignore
DebugTools.enableLogging("Error")


class BioFormatsReader:
    '''Bio-Formats ImageReader class with some added functionality to work
    with data in native Python
    '''

    _pixel_dtypes = {
        'int8': np.dtype(np.uint8),
        'uint8': np.dtype(np.uint8),
        'uint16': np.dtype(np.uint16),
    }

    def __init__(self, filepath, autostitch=True):
        '''Initialize the Reader object.

        Parameters
        ----------
        filepath : string or path-like object
            Path to an image file.
        '''
        self._filepath = pathlib.Path(filepath)
        self._rdr = None
        self._metadata = None

        self._init_reader()

    def _init_reader(self) -> None:
        factory = ServiceFactory()
        service = JObject(factory.getInstance(OMEXMLService), OMEXMLService)
        metadata = service.createOMEXMLMetadata()

        self._rdr = ImageReader()
        self._rdr.setMetadataStore(metadata)

        logger.debug("Opening '%s'", str(self._filepath))
        self._rdr.setId(str(self._filepath))
        self._metadata = JObject(metadata, MetadataRetrieve)

    @property
    def pixel_dtype(self):
        return self._pixel_dtypes[self._metadata.getPixelsType(0).toString()]
    
    @property
    def size_X(self):
        return self._rdr.getSizeX()

    @property
    def size_Y(self):
        return self._rdr.getSizeY()

    def read_image(self, c: int, series: int=0, z: int=0, t: int=0, XYWH=None):  # noqa
        '''Read bytes to numpy array

        Parameters
        ----------
        c : int
            The channel index.
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

        self._rdr.setSeries(series)
        index = self._rdr.getIndex(z, c, t)
        dtype = self.pixel_dtype

        if XYWH is not None:
            byte_array = self._rdr.openBytes(index, *XYWH)
            shape = XYWH[3], XYWH[2]
            img = np.array(byte_array, dtype=dtype).reshape(shape)
        else:
            byte_array = self._rdr.openBytes(index)
            shape = self._rdr.getSizeY(), self._rdr.getSizeX()
            img = np.array(byte_array, dtype=dtype).reshape(shape)

        return img

print('done')


def main():
    filepath = 'data/b32f-sag_L-lxn_pv-w03_2-cla-63x-proc.czi'
    reader = BioFormatsReader(filepath)
    XYWH = (5000, 5000, 150, 200)
    img = reader.read_image(0, XYWH=XYWH)
    plt.imshow(img)


if __name__ == "__main__":
    main()
