# Load json data from dir or list of dir
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor


def load_json_data(path: Path):
    path_list = path.glob("*.json")
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(load_json, path_list))
    return results


def load_json(json_file):
    with open(json_file) as f:
        return json.load(f)
