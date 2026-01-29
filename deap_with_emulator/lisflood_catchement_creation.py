# -*- coding: utf-8 -*-
import os
import shutil
import pandas as pd
import xml.etree.ElementTree as ET

# -------------------------------------------------------------------
# USER PATHS
# -------------------------------------------------------------------
CSV_PATH = "optimized_params.csv"

OUTPUT_BASE = "/home/chaliol/Lisflood_param_test_glofas5"

SOURCE1 = "/BGFS/DISASTER/russcar/GloFASNext_calibration/catchments"
SOURCE2 = "/BGFS/DISASTER/russcar/GloFASNext_calibration_july2022/catchments"

PRERUN_NAME = "OSLisfloodGloFASv5calibration_v1_template_PreRunlong_term_run.xml"
RUN_NAME    = "OSLisfloodGloFASv5calibration_v1_template_Runlong_term_run.xml"


# -------------------------------------------------------------------
# Helper: update parameter value
# -------------------------------------------------------------------
def update_xml_parameter(root, name, value):
    for node in root.iter("textvar"):
        if node.attrib.get("name") == name:
            node.set("value", str(value))
            return True
    return False


# -------------------------------------------------------------------
# Helper: update ONLY PathRoot
# -------------------------------------------------------------------
def update_pathroot(root, map_dir):
    for node in root.iter("textvar"):
        if node.attrib.get("name") == "PathRoot":
            node.set("value", map_dir)
            return True
    return False


# -------------------------------------------------------------------
# Helper: update PathOut
# -------------------------------------------------------------------
def update_output_path(root, out_dir):
    for node in root.iter("textvar"):
        if node.attrib.get("name") == "PathOut":
            node.set("value", out_dir)
            return True
    return False


# -------------------------------------------------------------------
# Helper: update PathInit = PathOut
# -------------------------------------------------------------------
def update_pathinit(root, out_dir):
    for node in root.iter("textvar"):
        if node.attrib.get("name") == "PathInit":
            node.set("value", out_dir)
            return True
    return False


# -------------------------------------------------------------------
# Helper: Insert NetCDF options before PathRoot and add reference textvars
#           after MapsCaching inside <lfbinding> (falls back to proj4_params)
#           and insert Theta2AvUpsTS after Theta1AvUpsTS and Theta3 after Theta2
# -------------------------------------------------------------------
def insert_netcdf_options(root):
    inserted_before_pathroot = False
    inserted_after_lfbinding = False
    # --- PART A: insert comment + textvars before PathRoot ---
    for group in root.iter("group"):
        children = list(group)

        maps_caching_index = None
        pathroot_index = None
        for i, node in enumerate(children):
            if node.tag == "textvar" and node.attrib.get("name") == "MapsCaching":
                maps_caching_index = i
            if node.tag == "textvar" and node.attrib.get("name") == "PathRoot":
                pathroot_index = i

        if maps_caching_index is None or pathroot_index is None:
            continue

        insert_at = pathroot_index  # insert right before PathRoot

        # Prepare nodes (textvar elements MUST have 'value' attribute for lisflood)
        comment_chunks = ET.Element("comment")
        comment_chunks.text = (
            "\nThe option \"OutputMapsChunks\" may take the following values:\n"
            "    - \"[positive integer number]\"  : Dump outputs to disk every X steps (default 1)\n"
        )

        chunks_textvar = ET.Element("textvar", attrib={"name": "OutputMapsChunks", "value": "1"})

        comment_dtype = ET.Element("comment")
        comment_dtype.text = (
            "\nThe option \"OutputMapsDataType\" sets the output data type and may take the following values:\n"
            "    - \"float64\"\n"
            "    - \"float32\"\n"
        )

        dtype_textvar = ET.Element("textvar", attrib={"name": "OutputMapsDataType", "value": "float64"})

        # Build insertion list and insert with newline tails for readability
        lst = [
            comment_chunks,
            chunks_textvar,
            comment_dtype,
            dtype_textvar,
        ]

        for idx, element in enumerate(lst):
            group.insert(insert_at + idx, element)
            element.tail = "\n"

        inserted_before_pathroot = True
        break  # done for this part

    if not inserted_before_pathroot:
        print(" ?? Warning: MapsCaching or PathRoot not found ? NetCDF options before PathRoot not inserted.")

    # # --- PART B: attempt to insert reference textvars inside <lfbinding> after MapsCaching ---
    # # Check if the reference textvars already exist (as references with value starting with $())
    # has_ref_chunks = any(
    #     (node.tag == "textvar" and node.attrib.get("name") == "OutputMapsChunks" and node.attrib.get("value", "").startswith("$("))
    #     for node in root.iter()
    # )
    # has_ref_dtype = any(
    #     (node.tag == "textvar" and node.attrib.get("name") == "OutputMapsDataType" and node.attrib.get("value", "").startswith("$("))
    #     for node in root.iter()
    # )
    #
    # if has_ref_chunks and has_ref_dtype:
    #     # already present; we still go to Part C (theta insertion) below
    #     pass
    # else:
    #     # Find <lfbinding>
    #     lfbinding = None
    #     for node in root.iter():
    #         if node.tag == "lfbinding":
    #             lfbinding = node
    #             break
    #
    #     # If we found lfbinding, insert after MapsCaching inside it
    #     if lfbinding is not None:
    #         children = list(lfbinding)
    #         maps_idx = None
    #         for i, node in enumerate(children):
    #             if node.tag == "textvar" and node.attrib.get("name") == "MapsCaching":
    #                 maps_idx = i
    #                 break
    #
    #         if maps_idx is not None:
    #             insert_at = maps_idx + 1  # right after MapsCaching
    #
    #             inserts = []
    #             if not has_ref_chunks:
    #                 tv_chunks_ref = ET.Element("textvar", attrib={"name": "OutputMapsChunks", "value": "$(OutputMapsChunks)"})
    #                 inserts.append(tv_chunks_ref)
    #             if not has_ref_dtype:
    #                 tv_dtype_ref = ET.Element("textvar", attrib={"name": "OutputMapsDataType", "value": "$(OutputMapsDataType)"})
    #                 inserts.append(tv_dtype_ref)
    #
    #             for idx, element in enumerate(inserts):
    #                 lfbinding.insert(insert_at + idx, element)
    #                 element.tail = "\n"
    #
    #             inserted_after_lfbinding = True
    #
    #     # Fallback: if lfbinding or MapsCaching in lfbinding not found, fall back to proj4_params insertion
    #     if not inserted_after_lfbinding:
    #         for group in root.iter("group"):
    #             children = list(group)
    #             for i, node in enumerate(children):
    #                 if node.tag == "textvar" and node.attrib.get("name") == "proj4_params":
    #                     insert_at = i + 1  # insert right after proj4_params
    #
    #                     inserts = []
    #                     if not has_ref_chunks:
    #                         tv_chunks_ref = ET.Element("textvar", attrib={"name": "OutputMapsChunks", "value": "$(OutputMapsChunks)"})
    #                         inserts.append(tv_chunks_ref)
    #                     if not has_ref_dtype:
    #                         tv_dtype_ref = ET.Element("textvar", attrib={"name": "OutputMapsDataType", "value": "$(OutputMapsDataType)"})
    #                         inserts.append(tv_dtype_ref)
    #
    #                     for idx, element in enumerate(inserts):
    #                         group.insert(insert_at + idx, element)
    #                         element.tail = "\n"
    #
    #                     inserted_after_lfbinding = True
    #                     break
    #             if inserted_after_lfbinding:
    #                 break
    #
    #     if not inserted_after_lfbinding and not (has_ref_chunks and has_ref_dtype):
    #         print(" ?? Warning: Could not find <lfbinding> MapsCaching or proj4_params - reference textvars not inserted.")

    # # --- PART C: insert Theta2AvUpsTS after Theta1AvUpsTS and Theta3 after Theta2 if not present ---
    # # Check if Theta2 and Theta3 already exist anywhere
    # has_theta2 = any((node.tag == "textvar" and node.attrib.get("name") == "Theta2AvUpsTS") for node in root.iter())
    # has_theta3 = any((node.tag == "textvar" and node.attrib.get("name") == "Theta3AvUpsTS") for node in root.iter())

    # # If Theta2 not present, try inserting it after Theta1
    # if not has_theta2:
    #     inserted_theta2_local = False
    #     for group in root.iter("group"):
    #         children = list(group)
    #         for i, node in enumerate(children):
    #             if node.tag == "textvar" and node.attrib.get("name") == "Theta1AvUpsTS":
    #                 insert_at = i + 1  # right after Theta1AvUpsTS
    #
    #                 # Create Theta2 textvar with comment child
    #                 theta2_tv = ET.Element("textvar", attrib={"name": "Theta2AvUpsTS", "value": "$(PathOut)/thTopUpsX.tss"})
    #                 theta2_comment = ET.Element("comment")
    #                 theta2_comment.text = "\n    Soil moisture upper layer [cu mm / cu mm]\n"
    #                 theta2_tv.append(theta2_comment)
    #
    #                 group.insert(insert_at, theta2_tv)
    #                 theta2_tv.tail = "\n"
    #
    #                 inserted_theta2_local = True
    #                 break
    #         if inserted_theta2_local:
    #             has_theta2 = True
    #             break
    #     if not inserted_theta2_local:
    #         print(" ?? Warning: Theta1AvUpsTS not found - Theta2AvUpsTS not inserted.")

    # # Now ensure Theta3 exists immediately after Theta2 (either pre-existing or newly inserted)
    # if not has_theta3:
    #     theta2_located = False
    #     for group in root.iter("group"):
    #         children = list(group)
    #         for i, node in enumerate(children):
    #             if node.tag == "textvar" and node.attrib.get("name") == "Theta2AvUpsTS":
    #                 insert_at = i + 1  # right after Theta2AvUpsTS
    #
    #                 # Create Theta3 textvar with comment child
    #                 theta3_tv = ET.Element("textvar", attrib={"name": "Theta3AvUpsTS", "value": "$(PathOut)/th2AvUps.tss"})
    #                 theta3_comment = ET.Element("comment")
    #                 theta3_comment.text = "\n    Soil moisture layer 2 [cu mm / cu mm]\n"
    #                 theta3_tv.append(theta3_comment)
    #
    #                 group.insert(insert_at, theta3_tv)
    #                 theta3_tv.tail = "\n"
    #
    #                 theta2_located = True
    #                 has_theta3 = True
    #                 break
    #         if theta2_located:
    #             break
    #     if not theta2_located:
    #         # Theta2 wasn't found, so Theta3 couldn't be inserted
    #         print(" ?? Warning: Theta2AvUpsTS not found - Theta3AvUpsTS not inserted.")
    # # else Theta3 already present -> nothing to do

    return True

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

    src_dir = os.path.join(SOURCE1, catch_id)
    if not os.path.exists(src_dir):
        src_dir = os.path.join(SOURCE2, catch_id)
        if not os.path.exists(src_dir):
            print(f" ? No calibration directory found for {catch_id}")
            continue

    src_settings = os.path.join(src_dir, "settings")

    if not os.path.exists(src_settings):
        print(f" ? No settings folder found for {catch_id}")
        continue

    catch_dir = os.path.join(OUTPUT_BASE, catch_id)

    if os.path.exists(catch_dir):
        shutil.rmtree(catch_dir)

    os.makedirs(catch_dir)
    settings_dir = os.path.join(catch_dir, "settings")
    out_dir = os.path.join(catch_dir, "out")

    os.makedirs(settings_dir)
    os.makedirs(out_dir)

    for filename in [PRERUN_NAME, RUN_NAME]:
        src_file = os.path.join(src_settings, filename)
        dst_file = os.path.join(settings_dir, filename)

        if not os.path.exists(src_file):
            print(f" ?? Missing file: {src_file}")
            continue

        shutil.copy(src_file, dst_file)

        tree = ET.parse(dst_file)
        root = tree.getroot()

        for p in param_cols:
            val = row[p]
            update_xml_parameter(root, p, val)

        update_pathroot(root, src_dir)
        update_output_path(root, out_dir)
        update_pathinit(root, out_dir)

        # NEW: insert NetCDF options
        insert_netcdf_options(root)

        tree.write(dst_file, encoding="utf-8", xml_declaration=True)

print("\nDONE - All catchments processed.")