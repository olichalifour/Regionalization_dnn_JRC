# -*- coding: utf-8 -*-
import os
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
from fnmatch import fnmatch

# -------------------------------------------------------------------
# USER PATHS
# -------------------------------------------------------------------
CSV_PATH = "optimized_params.csv"

SOURCE_ROOT = "/BGFS/DISASTER/russcar/cal_workflow_2025/catchments"
OUTPUT_BASE = "/home/chaliol/Lisflood_param_test_glofas5"

# -------------------------------------------------------------------
# PARAMETER NAME MAPPING
# CSV name -> XML name
# -------------------------------------------------------------------
PARAM_NAME_MAP = {
    "CalChanMan1": "CalChanMan",
}

# -------------------------------------------------------------------
# XML HELPERS
# -------------------------------------------------------------------
def update_xml_parameter(root, name, value):
    for node in root.iter("textvar"):
        if node.attrib.get("name") == name:
            node.set("value", str(value))
            return True
    return False


def update_path(root, name, value):
    for node in root.iter("textvar"):
        if node.attrib.get("name") == name:
            node.set("value", value)
            return True
    return False


def insert_netcdf_options(root):
    for group in root.iter("group"):
        children = list(group)

        maps_idx = None
        pathroot_idx = None

        for i, node in enumerate(children):
            if node.tag == "textvar" and node.attrib.get("name") == "MapsCaching":
                maps_idx = i
            if node.tag == "textvar" and node.attrib.get("name") == "PathRoot":
                pathroot_idx = i

        if maps_idx is None or pathroot_idx is None:
            continue

        insert_at = pathroot_idx

        elements = [
            ET.Element("comment", text="\nOutputMapsChunks option\n"),
            ET.Element("textvar", {"name": "OutputMapsChunks", "value": "1"}),
            ET.Element("comment", text="\nOutputMapsDataType option\n"),
            ET.Element("textvar", {"name": "OutputMapsDataType", "value": "float64"}),
        ]

        for i, el in enumerate(elements):
            group.insert(insert_at + i, el)
            el.tail = "\n"

        return True

    print("? NetCDF options not inserted (PathRoot not found)")
    return False


# -------------------------------------------------------------------
# DISCOVER CATCHMENT DIRECTORIES
# -------------------------------------------------------------------
def find_catchment_dir(catch_id):
    """
    Find directory whose name == catch_id exactly.
    Ignore directories like 95_V, 95_test, etc.
    """
    for root, dirs, _ in os.walk(SOURCE_ROOT):
        for d in dirs:
            if d == catch_id:
                return os.path.join(root, d)
    return None


def find_settings_files(settings_dir):
    """
    Automatically detect PreRun and Run XML files.
    """
    prerun = None
    run = None

    for f in os.listdir(settings_dir):
        if fnmatch(f, "*PreRun*.xml"):
            prerun = f
        elif fnmatch(f, "*Run*.xml"):
            run = f

    return prerun, run


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
param_cols = [c for c in df.columns if c not in ["catchment_id", "best_kge"]]

os.makedirs(OUTPUT_BASE, exist_ok=True)

for _, row in df.iterrows():

    catch_id = str(int(row["catchment_id"]))
    print(f"\n==============================")
    print(f"Processing catchment {catch_id}")
    print(f"==============================")

    src_dir = find_catchment_dir(catch_id)

    if src_dir is None:
        print(f"? Catchment {catch_id} not found in source tree")
        continue

    src_settings = os.path.join(src_dir, "settings")
    if not os.path.isdir(src_settings):
        print(f"? No settings directory for {catch_id}")
        continue

    prerun_name, run_name = find_settings_files(src_settings)

    if prerun_name is None or run_name is None:
        print(f"? Missing PreRun or Run XML for {catch_id}")
        continue

    # ----------------------------------------------------------------
    # OUTPUT STRUCTURE
    # ----------------------------------------------------------------
    catch_out = os.path.join(OUTPUT_BASE, catch_id)
    settings_out = os.path.join(catch_out, "settings")
    out_dir = os.path.join(catch_out, "out")

    if os.path.exists(catch_out):
        shutil.rmtree(catch_out)

    os.makedirs(settings_out)
    os.makedirs(out_dir)

    # ----------------------------------------------------------------
    # PROCESS XML FILES
    # ----------------------------------------------------------------
    for xml_name in [prerun_name, run_name]:

        src_file = os.path.join(src_settings, xml_name)
        dst_file = os.path.join(settings_out, xml_name)

        shutil.copy(src_file, dst_file)

        tree = ET.parse(dst_file)
        root = tree.getroot()

        # --- Parameters ---
        for p in param_cols:
            xml_param = PARAM_NAME_MAP.get(p, p)
            update_xml_parameter(root, xml_param, row[p])

        # --- Paths ---
        update_path(root, "PathRoot", src_dir)
        update_path(root, "PathOut", out_dir)
        update_path(root, "PathInit", out_dir)

        # --- NetCDF options ---
        insert_netcdf_options(root)

        tree.write(dst_file, encoding="utf-8", xml_declaration=True)

    print(f"? Catchment {catch_id} done")

print("\n   DONE - All catchments processed successfully.")