# strip_exif.py
from PIL import Image
import sys
from pathlib import Path

def strip_exif(input_path, output_path):
    img = Image.open(input_path)
    data = list(img.getdata())

    clean = Image.new(img.mode, img.size)
    clean.putdata(data)
    clean.save(output_path)

if __name__ == "__main__":
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])

    strip_exif(src, dst)
    print("Saved EXIF-clean image:", dst)