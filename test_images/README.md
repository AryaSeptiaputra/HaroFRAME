# test_images/

Drop reference photos here to batch-test the generation pipeline against real
people/model weights (needs GPU -- run this on vast.ai, see `VASTAI.md` at the
repo root; it will not produce real output on the local dev machine).

## Usage

- Supported files: `.jpg`, `.jpeg`, `.png` -- one clear, front-facing photo per
  person is enough (multiple photos of the same person are not auto-grouped;
  each image file is treated as its own subject).
- Optional per-image prompt: add a same-named `.txt` file next to the image,
  e.g. `alex.jpg` + `alex.txt` containing the prompt text for that photo. If no
  `.txt` file exists, the `--prompt` value passed on the command line is used
  for that image instead.
- Run from the repo root:
  ```
  python scripts/test_real_images.py --prompt "a person smiling, gentle breeze" --out outputs/test_real_images
  ```
- Each `<name>.jpg` produces `outputs/test_real_images/<name>.mp4` plus a
  console summary (OK/FAIL per image). A failure on one image does not stop
  the rest of the batch.

## Privacy

This folder's contents are **gitignored** (except this file) -- real photos of
people must never be committed to the repo or pushed to GitHub. Only put
photos here that you're comfortable existing as plain files inside a Docker
container / on the vast.ai instance you're running them on.
