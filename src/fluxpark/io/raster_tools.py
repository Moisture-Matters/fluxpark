from pathlib import Path
import numpy as np
from osgeo import gdal, osr
import warnings
import re


class GeoTiffReader:
    def __init__(self, source_path, band=1, nodata_value=float(-9999)):
        """
        Initialize the reader.

        Parameters
        ----------
        source_path : str
            Path to the source GeoTIFF.
        band : int, optional
            Band number to read (1-based index). Default is 1.
        nodata_value : float, optional
            Value to use for NoData pixels. Default is -9999.
        """
        self.source_path = source_path
        self.band = band
        self.nodata_value = nodata_value

    def read_and_reproject(
        self,
        dst_epsg,
        bounds,
        cellsize,
        resample_alg=gdal.GRA_NearestNeighbour,
        cutline_path=None,
        fillnodata=False,
        tempfile_dir=None,
        source_extra=0,
        **kwargs,
    ):
        """
        Read and reproject a GeoTIFF. Reprojection is just done to be sure that
        the fluxpark inputfiles lineup and fine gird input maps can be used
        directly for course calculations.

        Parameters
        ----------
        dst_epsg : int
            EPSG code of target projection.
        bounds : tuple
            Bounding box as (min_x, max_x, min_y, max_y) in target CRS.
        cellsize : float
            Target cell size (assumed square).
        resample_alg : int, optional
            GDAL resampling algorithm. Default is NearestNeighbour.
        cutline_path : str, optional
            Path to a cutline dataset (e.g., shapefile).
        fillnodata : bool, optional
            If True, fill NoData values using GDAL's FillNodata. Default=False.
        tempfile_dir : str, optional
            Directory to write temporary file if fillnodata is True.
        source_extra : int, optional
            Pixel buffer for warp (e.g., to avoid edge artifacts). Default=0.
        **kwargs : dummy
            This is used to permid using ad grid_params dict holding all info

        Returns
        -------
        np.ndarray
            Reprojected raster data as NumPy array.
        """
        src_path_str = str(Path(self.source_path)).replace("\\", "/")
        ds_in = gdal.Open(src_path_str, gdal.GA_ReadOnly)

        if fillnodata:
            if not tempfile_dir:
                raise ValueError(
                    "You must specify `tempfile_dir` when `fillnodata=True`."
                )
            driver = gdal.GetDriverByName("GTiff")
            temp_path = Path(tempfile_dir) / "temp_fill.tif"
            ds_temp = driver.CreateCopy(temp_path, ds_in, 0)
            ds_temp = None  # close

            ds_temp = gdal.Open(temp_path, gdal.GA_Update)
            band_obj = ds_temp.GetRasterBand(self.band)
            gdal.FillNodata(
                targetBand=band_obj,
                maskBand=None,
                maxSearchDist=5,
                smoothingIterations=0,
            )
            ds_temp = None
            ds_in = gdal.Open(temp_path, gdal.GA_ReadOnly)

        # Unpack bounds
        x_min, x_max, y_min, y_max = bounds

        # Warp (reproject + resample)
        warp_opts = {
            "dstSRS": f"EPSG:{dst_epsg}",
            "resampleAlg": resample_alg,
            "xRes": cellsize,
            "yRes": -cellsize,
            "format": "VRT",
            "dstNodata": self.nodata_value,
            "targetAlignedPixels": True,
            "overviewLevel": 0,
            "outputBounds": [x_min, y_min, x_max, y_max],
            "outputBoundsSRS": f"EPSG:{dst_epsg}",
        }

        if source_extra > 0:
            warp_opts["warpOptions"] = [f"SOURCE_EXTRA={source_extra}"]

        if cutline_path:
            warp_opts["cutlineDSName"] = cutline_path

        # warp the original tiff to the correct coordinates and cellsize
        ds_warp = gdal.Warp("", ds_in, **warp_opts)

        if source_extra > 0:
            # Crop back to the bounding box if source_extra is used
            ds_warp = gdal.Translate(
                "",
                ds_warp,
                projWin=[x_min, y_max, x_max, y_min],
                projWinSRS=f"EPSG:{dst_epsg}",
                format="VRT",
            )

        arr = ds_warp.GetRasterBand(self.band).ReadAsArray().astype(np.float32)
        arr[np.isnan(arr)] = self.nodata_value

        # Clean up
        ds_in = None
        ds_warp = None
        if fillnodata and temp_path.exists():
            temp_path.unlink()

        return arr


class NetCDFReader:
    def __init__(self, source_path, variable="prediction", nodata_value=float(-9999)):
        self.source_path = source_path
        self.variable = variable
        self.nodata_value = nodata_value

    def read_and_reproject(
        self,
        dst_epsg,
        bounds,
        cellsize,
        resample_alg=gdal.GRA_NearestNeighbour,
        cutline_path=None,
        fillnodata=False,
        tempfile_dir=None,
        source_extra=0,
        **kwargs,
    ):
        """
        Read and reproject a GeoTIFF. Reprojection is just done to be sure that
        the fluxpark inputfiles lineup and fine gird input maps can be used
        directly for course calculations.

        Parameters
        ----------
        dst_epsg : int
            EPSG code of target projection.
        bounds : tuple
            Bounding box as (min_x, max_x, min_y, max_y) in target CRS.
        cellsize : float
            Target cell size (assumed square).
        resample_alg : int, optional
            GDAL resampling algorithm. Default is NearestNeighbour.
        cutline_path : str, optional
            Path to a cutline dataset (e.g., shapefile).
        fillnodata : bool, optional
            If True, fill NoData values using GDAL's FillNodata. Default=False.
        tempfile_dir : str, optional
            Directory to write temporary file if fillnodata is True.
        source_extra : int, optional
            Pixel buffer for warp (e.g., to avoid edge artifacts). Default=0.
        **kwargs : dummy
            This is used to permid using ad grid_params dict holding all info

        Returns
        -------
        np.ndarray
            Reprojected raster data as NumPy array.
        """
        src_path_str = str(Path(self.source_path)).replace("\\", "/")
        x_min, x_max, y_min, y_max = bounds

        # Open NetCDF and find subdatasets
        ds_nc = gdal.Open(src_path_str)

        subdatasets = ds_nc.GetSubDatasets()
        if not subdatasets:
            raise ValueError("No subdatasets found in NetCDF file.")

        # search subdataset with 'prediction' in name
        prediction_path = None
        for name, desc in subdatasets:
            if self.variable.lower() in name.lower():
                prediction_path = name
                break

        if not prediction_path:
            raise ValueError("No {self.variable} subdataset gevonden.")

        ds_nc = gdal.Open(prediction_path)

        # fetch *all* metadata from the NetCDF driver
        md = ds_nc.GetMetadata()

        # find the one key that ends in “proj4_params” (case‑insensitive)
        proj4 = None
        for k, v in md.items():
            if k.lower().endswith("proj4_params"):
                proj4 = v
                break
        if proj4 is None:
            raise RuntimeError("No proj4_params found in metadata!")

        # clean it up:
        # 1) strip whitespace/newlines
        # 2) remove any surrounding quotes
        # 3) drop stray '+<>'
        proj4 = proj4.strip()
        proj4 = re.sub(r'^"+|"+$', '', proj4)   # strip leading/trailing quotes
        proj4 = re.sub(r'\+\<[^>]*\>', '', proj4).strip()

        # Finally import into OGRSpatialReference
        sr = osr.SpatialReference()
        sr.ImportFromProj4(proj4)
        wkt = sr.ExportToWkt()

        # 2) Open x en y as rasters (1 pixel high, width = dim‑length)
        ds_x = gdal.Open(f'NETCDF:"{src_path_str}":x', gdal.GA_ReadOnly)
        ds_y = gdal.Open(f'NETCDF:"{src_path_str}":y', gdal.GA_ReadOnly)

        w_x = ds_x.RasterXSize
        w_y = ds_y.RasterXSize

        # Read full row for both
        x_vals = ds_x.GetRasterBand(1).ReadAsArray(0, 0, w_x, 1)[0]
        y_vals = ds_y.GetRasterBand(1).ReadAsArray(0, 0, w_y, 1)[0]

        # Build geotransform:
        #    - originX = first x-value
        #    - pixelX size = diff between x[1] and x[0]
        #    - originY = last y-value (When y increases from South->North)
        #    - pixelY size = negative (y[1] - y[0]) if y_vals increase
        origin_x = float(x_vals[0])
        pixel_x_sz = float(x_vals[1] - x_vals[0])
        if y_vals[1] > y_vals[0]:
            origin_y = float(y_vals[-1])
            pixel_y_sz = -abs(float(y_vals[1] - y_vals[0]))
        else:
            origin_y = float(y_vals[0])
            pixel_y_sz = float(y_vals[1] - y_vals[0])
        geo_transform = (origin_x, pixel_x_sz, 0.0, origin_y, 0.0, pixel_y_sz)

        # translate the nc file to be able to update geo_transform
        ds_in = gdal.Translate(
            '',
            ds_nc,
            format='MEM',
            outputType=gdal.GDT_Float32,
            # creationOptions=['SRC_METHOD=BACKMAP']
        )
        ds_in.SetGeoTransform(geo_transform)
        ds_in.SetProjection(wkt)

        ds_nc.GetGeoTransform()
        ds_in.GetGeoTransform()
        ds_nc.GetProjection()
        ds_in.GetProjection()

        if fillnodata:
            if not tempfile_dir:
                raise ValueError(
                    "You must specify `tempfile_dir` when `fillnodata=True`."
                )
            driver = gdal.GetDriverByName("GTiff")
            temp_path = Path(tempfile_dir) / "temp_fill.tif"
            ds_temp = driver.CreateCopy(temp_path, ds_in, 0)
            ds_temp = None  # close

            ds_temp = gdal.Open(temp_path, gdal.GA_Update)
            band_obj = ds_temp.GetRasterBand(1)
            gdal.FillNodata(
                targetBand=band_obj,
                maskBand=None,
                maxSearchDist=5,
                smoothingIterations=0,
            )
            ds_temp = None
            ds_in = gdal.Open(temp_path, gdal.GA_ReadOnly)

        # Reproject and clip using GDAL Warp (to VRT)
        warp_opts = {
            "dstSRS": f"EPSG:{dst_epsg}",
            "resampleAlg": resample_alg,
            "xRes": cellsize,
            "yRes": -cellsize,
            "format": "MEM",
            "dstNodata": self.nodata_value,
            "targetAlignedPixels": True,
            "overviewLevel": 0,
            "outputBounds": [x_min, y_min, x_max, y_max],
            "outputBoundsSRS": f"EPSG:{dst_epsg}",
        }

        if source_extra > 0:
            warp_opts["warpOptions"] = [f"SOURCE_EXTRA={source_extra}"]

        if cutline_path:
            warp_opts["cutlineDSName"] = cutline_path

        # warp the raster to the correct coordinates and cellsize
        ds_warp = gdal.Warp("", ds_in, **warp_opts)

        if source_extra > 0:
            # Crop back to the bounding box if source_extra is used
            ds_warp = gdal.Translate(
                "",
                ds_warp,
                projWin=[x_min, y_max, x_max, y_min],
                projWinSRS=f"EPSG:{dst_epsg}",
                format="VRT",
            )

        arr = ds_warp.ReadAsArray().astype(np.float32)
        arr[arr == self.nodata_value] = np.nan

        # Clean up
        ds_nc = None
        ds_in = None
        ds_warp = None
        if fillnodata and temp_path.exists():
            temp_path.unlink()

        return arr


def write_geotiff(
    out_dir,
    out_filename,
    out_array,
    x_min,
    y_max,
    cellsize,
    epsg_code,
    nodata_value=float(-9999),
    dtype=gdal.GDT_Float32,
    compress="LZW",
):
    """
    Write a 2D NumPy array to a GeoTIFF, MEM, or VSIMEM raster.

    Parameters
    ----------
    out_dir : str
        Output directory: "" for MEM, "/vsimem/" for in-memory file, or a disk
        path.
    out_filename : str
        Filename (used only if out_dir is not "").
    out_array : np.ndarray
        2D raster data array.
    x_min : float
        Upper-left X coordinate (corner of top-left pixel).
    y_max : float
        Upper-left Y coordinate (corner of top-left pixel).
    cellsize : float
        Pixel resolution (assumed square).
    epsg_code : int
        EPSG code for spatial reference.
    nodata_value : float or int
        NoData value to write.
    dtype : int
        GDAL data type (e.g., gdal.GDT_Float32).
    compress : str, optional
        Compression type (default "LZW").

    Returns
    -------
    gdal.Dataset or None
        If written in-memory (MEM), returns the dataset. Otherwise None.

    Notes
    -----
    Data is written using the GDAL MEM driver if `out_dir` is an empty string.
    In that case, `out_filename` should also be set to "".

    Writing to /vsimem/ is only used if `out_dir` explicitly starts with
    "/vsimem/". The dataset is then retrievable via the given path, but not
    returned by this function.

    This behavior avoids using the GTiff driver with /vsimem/, which is less
    efficient than the MEM driver for pure in-memory operations.
    """
    if out_array.ndim != 2:
        raise ValueError("Only 2D arrays are supported.")

    rows, cols = out_array.shape

    # Determine output mode
    is_mem = out_dir == ""

    normalized_dir = str(Path(out_dir)).replace("\\", "/").lower()
    is_vsimem = "vsimem" in normalized_dir

    uses_filename = not is_mem

    # Build output path if needed
    if uses_filename:
        out_path = Path(out_dir) / out_filename
        out_path_str = str(out_path).replace("\\", "/")
    else:
        out_path_str = ""  # MEM mode or anonymous GTiff

    # Choose driver
    if is_mem:
        driver = gdal.GetDriverByName("MEM")
        if out_filename.lower().endswith((".tif", ".vrt")):
            warnings.warn(
                f"You specified a filename ending in "
                f"{Path(out_filename).suffix}. This is only supported if you "
                f"also specify a path. The data set will not be retrievable."
            )
    else:
        driver = gdal.GetDriverByName("GTiff")
        # Warn if .vrt is used with GTiff
        if out_filename.lower().endswith(".vrt"):
            warnings.warn(
                "You specified a .vrt file, but creating raster data sets "
                "from np.array's does not produce valid VRT files. Use '.tif'"
                "instead, or use empty string for 'out_dir' and 'out_filename'"
                " for true in-memory processing."
            )

    # Set creation options
    options = [f"COMPRESS={compress}"] if not is_mem else []

    # Create output dataset
    outdata = driver.Create(out_path_str, cols, rows, 1, dtype, options=options)

    # Georeference and projection
    geotransform = (x_min, cellsize, 0, y_max, 0, -cellsize)
    outdata.SetGeoTransform(geotransform)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg_code)
    outdata.SetProjection(srs.ExportToWkt())

    # Write data
    outdata.GetRasterBand(1).WriteArray(out_array)
    outdata.GetRasterBand(1).SetNoDataValue(nodata_value)

    if is_vsimem and outdata is not None and out_filename == "":
        warnings.warn(
            "You are writing to /vsimem/ without a file name. "
            "This file cannot be accessed later without a file name."
        )

    # Return dataset if in memory, otherwise flush
    if is_mem:
        return outdata
    elif is_vsimem:
        return None  # find the file via path.
    else:
        outdata = None  # flush to disk
        return None
