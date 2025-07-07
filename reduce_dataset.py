from pathlib import Path

import cv2
from tqdm import tqdm

SOURCE = "./data"

def main() -> None:
    folder_list = [fldr for fldr in Path(SOURCE).iterdir() if "video" in fldr.stem]

    for fldr in folder_list:
        print(fldr.stem)
        Path(f"{SOURCE}/split_{fldr.stem}").mkdir( parents=True, exist_ok=True)
        file_list = sorted(list(Path(fldr).iterdir()))
        counter = 0
        with tqdm(total=len(file_list)) as t:
            for fp in file_list:
                if int(fp.stem) % 5 == 0:
                    img = cv2.imread(fp)
                    cv2.imwrite(f"{SOURCE}/half_{fldr.stem}/{str(counter).zfill(3)}.jpeg", img[::4, ::4])
                    counter += 1
                t.update(1)


if __name__ == "__main__":
    main()
