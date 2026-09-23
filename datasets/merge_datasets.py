import os
import pickle
import argparse

def merge_pickles(dir1, dir2, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))
    common = files1.intersection(files2)

    print(f"Found {len(common)} matching pickle files")

    for fname in sorted(common):
        path1 = os.path.join(dir1, fname)
        path2 = os.path.join(dir2, fname)

        with open(path1, "rb") as f1:
            d1 = pickle.load(f1)

        with open(path2, "rb") as f2:
            d2 = pickle.load(f2)

        merged = {**d1, **d2}

        out_path = os.path.join(out_dir, fname)
        with open(out_path, "wb") as out_f:
            pickle.dump(merged, out_f)

        print(f"Merged: {fname}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge pickle dictionaries with matching filenames from two directories."
    )
    parser.add_argument("dir1", help="First input directory")
    parser.add_argument("dir2", help="Second input directory")
    parser.add_argument("out", help="Output directory for merged pickle files")

    args = parser.parse_args()

    merge_pickles(args.dir1, args.dir2, args.out)


if __name__ == "__main__":
    main()
