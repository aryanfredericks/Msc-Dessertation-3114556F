import zipfile
import urllib.request
from pathlib import Path

ZIP_URL = "https://raw.githubusercontent.com/JHnlp/BioCreative-V-CDR-Corpus/master/CDR_Data.zip"
OUT_DIR = Path("./datasets/bc5cdr")
ZIP_PATH = OUT_DIR / "CDR_Data.zip"
RAW_DIR = OUT_DIR / "raw"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {ZIP_URL} ...")
    urllib.request.urlretrieve(ZIP_URL, ZIP_PATH)
    print(f"Saved to {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")

    print(f"Extracting to {RAW_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(RAW_DIR)

    # the PubTator files we actually need live at:
    corpus_dir = RAW_DIR / "CDR_Data" / "CDR.Corpus.v010516"
    expected = [
        "CDR_TrainingSet.PubTator.txt",
        "CDR_DevelopmentSet.PubTator.txt",
        "CDR_TestSet.PubTator.txt",
    ]
    print("\nChecking expected files:")
    for name in expected:
        path = corpus_dir / name
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {path}")

    print(f"\nDone. PubTator files are under: {corpus_dir}")


if __name__ == "__main__":
    main()