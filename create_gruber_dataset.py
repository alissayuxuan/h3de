"""Run this before training a model to prepare the data."""

import argparse
import json
import os
from functools import partial
from os import PathLike
from typing import Callable

import numpy as np
import pandas as pd

from TPTBox import NII, BIDS_Global_info 
from TPTBox.core.poi import POI
from TPTBox import Subject_Container
from pqdm.processes import pqdm


def get_gruber_poi(container) -> POI:
    poi_query = container.new_query(flatten=True)
    poi_query.filter_format("poi")

    if not poi_query.candidates:
        print("ERROR: No POI candidates found!")
        return None
    
    poi_candidate = poi_query.candidates[0]
    print(f"Loading POI from: {poi_candidate}")

    try:
        poi = POI.load(poi_candidate.file["json"])
        return poi
    except Exception as e:
        print(f"Error loading POI: {str(e)}")
        return None


def get_ct(container) -> NII:
    ct_query = container.new_query(flatten=True)
    ct_query.filter_format("ct")
    ct_query.filter_filetype("nii.gz")  # only nifti files
    ct_candidate = ct_query.candidates[0]

    try:
        ct = ct_candidate.open_nii()
        return ct
    except Exception as e:
        print(f"Error opening CT: {str(e)}")
        return None


def get_files(
    container,
    get_poi: Callable,
    get_ct_fn: Callable
) -> tuple[POI, NII]:
    return (
        get_poi(container),
        get_ct_fn(container),
    )


def process_container(
    subject,
    container,
    save_path: PathLike,
    get_files_fn: Callable[[Subject_Container], tuple[POI, NII]],
):
    splits = {
        "train": [
            "WS-17", "WS-18", "WS-53", "WS-09", "WS-25", "WS-08", "WS-22", 
            "WS-63", "WS-52", "WS-54", "WS-36", "WS-47", "WS-34", "WS-23", 
            "WS-13", "WS-38", "WS-43", "WS-07", "WS-56", "WS-31", "WS-05", 
            "WS-30", "WS-50", "WS-19", "WS-40", "WS-29", "WS-00"
        ],
        "val": [
            "WS-16", "WS-62", "WS-45", "WS-26"
        ],
        "test": [
            "WS-06", "WS-55", "WS-15", "WS-46", "WS-48"
        ]
    }

    print(f"Processing Subject: {subject}")
    poi, ct = get_files_fn(container)

    if subject in splits["train"]:
        phase = "train"
    elif subject in splits["val"]:
        phase = "val"
    elif subject in splits["test"]:
        phase = "test"

    poi_path = os.path.join(save_path, phase, subject)
    ct_path = os.path.join(save_path, phase, subject)

    os.makedirs(poi_path, exist_ok=True)
    os.makedirs(ct_path, exist_ok=True)

    poi_path = os.path.join(poi_path, "poi.json")
    ct_path = os.path.join(ct_path, "ct.nii.gz")


    poi.save(poi_path, verbose=False)   
    ct.save(ct_path, verbose=False)



def prepare_data(
    bids_surgery_info: BIDS_Global_info,
    save_path: str,
    get_files_fn: callable,
    n_workers: int = 8,
):
    master = []

    partial_process_container = partial(
        process_container,
        save_path=save_path,
        get_files_fn=get_files_fn,
    )

    master = pqdm(
        bids_surgery_info.enumerate_subjects(),
        partial_process_container,
        n_jobs=n_workers,
        argument_type="args",
        #exception_behaviour="immediate",
        exception_behaviour="continue"
    )

    
if __name__ == "__main__":

    data_path = "./dataset-folder-gruber"
    save_path = "./gruber_dataset"

    bids_gloabl_info = BIDS_Global_info(
        datasets=[data_path], parents=["rawdata", "derivatives"]
    )


    get_data_files = partial(
        get_files,
        get_poi=get_gruber_poi,
        get_ct_fn=get_ct
    )

    
    prepare_data(
        bids_surgery_info=bids_gloabl_info,
        save_path=save_path,
        get_files_fn=get_data_files,
    )