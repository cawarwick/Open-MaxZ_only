import xml.etree.ElementTree as ET
from pathlib import Path
import tifffile as tif
import numpy as np
import os
import math
import h5py
import csv
import os
from scipy.signal import resample
from scipy import signal
import scipy.signal as signal
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi
import subprocess

# ---------------- USER PARAMETERS ---------------- #

Imaging_Folder = r"E:\PrRFMap\#550_009"

group_size = 5  # Number of frames to average together

# -------------------------------------------------- #

#Main body of code#
 
# -------------------------------------------------- #


Imaging_Folder = (Imaging_Folder.replace(os.sep, "/")+"/") 
split= Imaging_Folder.split('/')
num_folders = len(split)
Parent_Folder_Name=split[(num_folders-2)]

def extract_TS_metadata():
    ##this function extracts metadata from the ThorImageLS XML file and returns it as a dictionary for use in re-arrangin the OME TIFFs
    ##The experiment.xml must be in the same folder as the OME TIFFs for this to work.
    location = Imaging_Folder+"Experiment.xml"
    tree = ET.parse(location)
    root = tree.getroot()

    Meta_Data = {
        "steps": None,
        "pixelX": None,
        "pixelY": None,
        "frameRate": None,
        "flybackCycles": None,##This is actually # Z plus # flyback, so total frame cycles
        "Timelapse timepoints": None,
        "pixelSizeUM": None,
        "ChannelEnable Set": None,
    }

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        # ----- Standard tag text extraction -----
        if elem.text and elem.text.strip():
            if tag in Meta_Data:
                Meta_Data[tag] = elem.text.strip()

        # ----- Attribute-based extraction -----
        for attr, value in elem.attrib.items():

            # Timelapse timepoints
            if tag == "Timelapse" and attr == "timepoints":
                value = int(value)
                Meta_Data["Timelapse timepoints"] = value
                
            # ChannelEnable Set
            if tag == "ChannelEnable" and attr == "Set":
                Meta_Data["ChannelEnable Set"] = value
                value = int(value)
                if value == 1:##Note I can't figure out how to detect channels based only on the metadata. This needs to be either fixed or removed entirely.
                    Meta_Data["ChannelEnable Set"] = 1
                if value == 3:
                    Meta_Data["ChannelEnable Set"] = 2
                if value == 11:
                    Meta_Data["ChannelEnable Set"] = 3
                else:
                    print("Channels were not detected as 1,2, or 3 double check input data")

            # Generic attribute fallback
            if attr in Meta_Data:
                Meta_Data[attr] = value

    return Meta_Data

def Load_OMEtiff(Meta_Data):
    ###this function loads the OME TIFFs and re-arranges them into the right order for typical analysis 
    # It also saves a green channel maximum intensity projection in a subfolder.
    ## Note that this assumes there are two channels since I couldn't find the parameter in the metadata that tells you how many channels there are. 
    print("\nTIFF Metadata:")
    print("-" * 40)
    for k, v in Meta_Data.items():
        print(f"{k:25}: {v}")
    Z = Meta_Data.get("steps")

    X = int(Meta_Data['pixelX'])   # pixels
    Y = int(Meta_Data['pixelY'])
    Z = int(Meta_Data['steps'])
    T = int(Meta_Data['Timelapse timepoints'])
    C = 2 #Meta_Data['pixelX']

    print("Z steps", Z)
    print("flyback cycles", Meta_Data['flybackCycles'])

    MAX_FILE_SIZE = 3.8 * 1024**3  # 4 GB
    
    # --------------------------------------------------
    # LOAD TIFF (CHECK FOR MULTIPLE REGIONS)
    # --------------------------------------------------
    input_tif = Imaging_Folder+"Image_scan_1_region_0_0.tif" ##this is the current namaing for ThorImageLS

    # Check for second file
    base, ext = os.path.splitext(input_tif)

    second_tif = Imaging_Folder+"Image_scan_1_region_0_1.tif" ##this is the current namaing for ThorImageLS

    if os.path.exists(second_tif):

        print(f"Multiple TIFF files detected:")
        print(f"  {input_tif}")
        print(f"  {second_tif}")

        with tif.TiffFile(input_tif) as tf:
            stack1 = tf.asarray()

        with tif.TiffFile(second_tif) as tf:
            stack2 = tf.asarray()

        # Concatenate along the image-plane dimension
        stack = np.concatenate((stack1, stack2), axis=0)

        print(f"Combined stack shape: {stack.shape}")

    else:

        print("No second TIFF found. Loading single file.")

        with tif.TiffFile(input_tif) as tf:
            stack = tf.asarray()

    #print("TIFF loading complete. Continuing processing...")
    
    # Save the name of the parent folder to save the tiffs with the right name
    split= Imaging_Folder.split('/')
    num_folders = len(split)
    Parent_Folder_Name=split[(num_folders-2)]
    #print(Parent_Folder_Name)
    output_folder = Imaging_Folder + "Rearranged_Output/" 
    os.makedirs(output_folder, exist_ok=True)
    output_name = output_folder + Parent_Folder_Name

    total_planes = stack.shape[0]
    expected_planes = Z * T * C

    if total_planes != expected_planes:
        raise ValueError(
            f"Plane count mismatch: found {total_planes}, expected {expected_planes}"
        )

    #print("Reshaping to XYZTC...")
    # Original assumed order: T, C, Z, Y, X
    stack = stack.reshape(C, T, Z, Y, X)

    # --------------------------------------------------
    # CREATE GREEN CHANNEL MAXIMUM INTENSITY PROJECTION
    # --------------------------------------------------

    #print("Generating green channel maximum projection...")

    #print(stack.shape)

    # Stack dimensions: C, T, Z, Y, X
    green_stack = stack[0, :, :, :, :]

    # Maximum intensity projection across Z
    green_mip = np.max(green_stack, axis=1)

    #print(green_mip.shape)
    
    mip_output = f"{output_name}_green_MIP.tif"

    green_mip_mean = np.mean(green_mip, axis=(1, 2))

    #print(f"Saving {green_mip.shape[0]} individual MIP pages...")

    tif.imwrite(
        mip_output,
        green_mip,                      # shape (T, Y, X)
        imagej=True,
        metadata={"axes": "TYX"},
        photometric="minisblack",
        compression=None,
    )
    try:
        imagej_exe = r"C:\Users\warwickc\Desktop\Fiji new\Fiji.app\ImageJ-win64.exe"  # Update this path to your ImageJ executable
        open_in_imagej(mip_output, imagej_exe) ##optional but make sure to specify the path to ImageJ on the local machine
    except Exception as e:
        print(f"Could not open ImageJ: {e}")
    
    avg_path = f"{output_name}_green_MIP_AVG.tif"
    tif.imwrite(avg_path, group_average(green_mip, group_size), imagej=True,
        metadata={"axes": "TYX"}, photometric="minisblack", compression=None)

    try:
        imagej_exe = r"C:\Users\warwickc\Desktop\Fiji new\Fiji.app\ImageJ-win64.exe"  # Update this path to your ImageJ executable
        open_in_imagej(avg_path, imagej_exe) ##optional but make sure to specify the path to ImageJ on the local machine
    except Exception as e:
        print(f"Could not open ImageJ: {e}")   

    #print(f"Green MIP saved: {mip_output}")

    """
    # --------------------------------------------------
    ##For saving the full XYZTC stacks everything below this is only useful for saving the full re-arranged data. 
    # --------------------------------------------------

    print("Reordering to XYCZT (page order C fastest)...")
    # Desired iteration order:
    # T -> Z -> C
    reordered = []
    for t in range(T):
        for z in range(Z):
            for c in range(C):
                reordered.append(stack[c, t, z])

    reordered = np.asarray(reordered)


    bytes_per_plane = reordered[0].nbytes
    planes_per_file_max = int(MAX_FILE_SIZE // bytes_per_plane)

    # Must be divisible by Z*C
    planes_per_block = int(Z * C) 
    planes_per_file = int((planes_per_file_max // planes_per_block) * planes_per_block)

    if planes_per_file == 0:
        raise ValueError("Single Z×C block exceeds 4 GB")

    n_files = math.ceil(len(reordered) / planes_per_file)

    print(f"Writing {n_files} output files...")

    for i in range(n_files):
        start = int(i * planes_per_file)
        end = int(min((i + 1) * planes_per_file, len(reordered)))

        out_name = f"{output_name}_{i+1:03d}.tif"

        print(f"Writing {out_name} ({end - start} planes)")

        tif.imwrite(
            out_name,
            reordered[start:end],
            compression=None,
            bigtiff=False
        )
    """

    return(green_mip_mean,green_mip)

def open_in_imagej(tiff_path, imagej_exe):
    tiff_path = str(Path(tiff_path).resolve())
    subprocess.Popen([imagej_exe, tiff_path])   # Popen = don't block Python

def group_average(stack, group_size):
    """Average every `group_size` consecutive frames along axis 0.
    Leftover frames that don't fill a full group are dropped.

    stack: (T, Y, X) array
    Returns (T // group_size, Y, X) in the original dtype.
    """
    n_groups = stack.shape[0] // group_size
    trimmed = stack[: n_groups * group_size]
    grouped = trimmed.reshape(n_groups, group_size, *stack.shape[1:])
    averaged = grouped.mean(axis=1, dtype=np.float32)

    if np.issubdtype(stack.dtype, np.integer):
        info = np.iinfo(stack.dtype)
        averaged = np.clip(np.rint(averaged), info.min, info.max)
    return averaged.astype(stack.dtype)

##Full Main program
Meta_Data = extract_TS_metadata() ##extract metadata of image files
Load_OMEtiff(Meta_Data) #use metadata to load the OME Tiffs and rearrange the images and save a MaxZ for use in analysis
