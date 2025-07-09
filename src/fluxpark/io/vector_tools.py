from typing import Optional, Union
from pathlib import Path
from osgeo import ogr, osr, gdal

_EXT2DRIVER = {
    ".gpkg": "GPKG",
    ".shp": "ESRI Shapefile",
    ".geojson": "GeoJSON",
    ".json": "GeoJSON",
    ".csv": "CSV",
    ".dxf": "DXF",
    ".gml": "GML",
    ".kml": "KML",
    ".gpx": "GPX",
    ".fgb": "FlatGeobuf",
    ".sqlite": "SQLite",
}


def write_point_layer(
    x, y, values, dst_epsg: int, output_path: Optional[Union[str, Path]] = None
) -> ogr.Layer:
    """
    Write a point layer to memory or disk using GDAL/OGR.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the points.
    values : array-like
        Z-values to assign to each point.
    dst_epsg : int
        EPSG code for the spatial reference system.
    output_path : str or Path, optional
        If given, saves to disk using appropriate GDAL driver.
        If None, returns an in-memory OGR layer.

    Returns
    -------
    gdal.Dataset
        In-memory vector dataset containing the point layer.
    """
    # Create empty in-memory vector dataset
    driver_mem = gdal.GetDriverByName("Memory")
    ds_mem = driver_mem.Create("", 0, 0, 0, gdal.GDT_Unknown)

    # Define spatial reference system
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(dst_epsg)

    # Create layer and add field
    layer = ds_mem.CreateLayer("points", srs, ogr.wkbPoint)
    field_def = ogr.FieldDefn("z", ogr.OFTReal)
    layer.CreateField(field_def)

    # Add features
    for xi, yi, zi in zip(x, y, values):
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("z", float(zi))

        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(float(xi), float(yi))
        feature.SetGeometry(point)

        layer.CreateFeature(feature)
        feature = None  # free memory

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()
        driver = _EXT2DRIVER.get(ext)

        if not driver:
            raise ValueError(f"No suitable driver for extension '{ext}'")

        dst_path_str = str(path).replace("\\", "/")
        gdal.VectorTranslate(dst_path_str, ds_mem, format=driver)

    return ds_mem
