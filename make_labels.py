import json
import re
from pathlib import Path

FOLDER = Path("/gpfs2/scratch/echiu2/cs3540/Converted Spectograms")
OUTPUT = FOLDER / "labels.json"
IMAGE_EXTENSIONS = {".png", ".jpg"}
SPECIES_REGEX = re.compile(r"^(?P<species>.+?)\s+\d+\s+.+$")
TOWHEE = {
    "towhee_spectrogram": "towhee",
}


def species_from_name(file_name):
    """
    grabs species from file name using regex matching
    """
    stem = Path(file_name).stem
    if stem in TOWHEE:
        return TOWHEE[stem]
    match = SPECIES_REGEX.match(stem)
    if match:
        return match.group("species").strip()
    else:
        return None


def collect(folder):
    labels = {}
    unmatched = []

    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        species = species_from_name(path.name)
        if species is None:
            unmatched.append(path.relative_to(folder).as_posix())
        else:
            labels[path.name] = species

    return dict(sorted(labels.items())), unmatched


def main():
    folder = FOLDER

    index, unmatched = collect(folder)

    if unmatched:
        print("unmatched species exist!")
        for file_name in unmatched:
            print(f"{file_name}")

    with OUTPUT.open("w") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
        fh.write("\n")


if __name__ == "__main__":
    main()
