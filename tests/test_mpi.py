#!/usr/bin/env python3
"""Small end-to-end checks for both MPI implementations."""

import os
import shlex
import struct
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MPI_DIR = ROOT / "MPI_parallelization"
IMAGE_WIDTH = 6
IMAGE_HEIGHT = 8
ITERATIONS = 3


def make_pixels():
    return [
        (row * 31 + column * 17 + 9) % 256
        for row in range(IMAGE_HEIGHT)
        for column in range(IMAGE_WIDTH)
    ]


def write_sun_raster(path, pixels):
    header = struct.pack(
        ">8I",
        0x59A66A95,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        8,
        IMAGE_WIDTH * IMAGE_HEIGHT,
        1,
        1,
        768,
    )
    palette = bytes(range(256)) * 3
    path.write_bytes(header + palette + bytes(pixels))


def read_sun_raster(path):
    data = path.read_bytes()
    header = struct.unpack(">8I", data[:32])
    _, width, height, depth, _, raster_type, map_type, map_length = header
    if depth != 8 or raster_type != 1 or map_type != 1:
        raise AssertionError(f"Unexpected Sun Raster header in {path}")

    pixel_start = 32 + map_length
    return width, height, list(data[pixel_start : pixel_start + width * height])


def rounded_division(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    doubled_remainder = 2 * remainder
    if doubled_remainder > denominator or (
        doubled_remainder == denominator and quotient % 2 == 1
    ):
        quotient += 1
    return quotient


def apply_filter(pixels, width, height, filter_id):
    result = pixels.copy()
    for row in range(1, height - 1):
        for column in range(1, width - 1):
            neighbors = [
                pixels[(row + dy) * width + column + dx]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            ]
            center = pixels[row * width + column]
            if filter_id == 0:
                value = rounded_division(sum(neighbors), 9)
            elif filter_id == 1:
                value = rounded_division(sum(neighbors) + 4 * center, 12)
            elif filter_id == 2:
                north, west, east, south = (
                    pixels[(row - 1) * width + column],
                    pixels[row * width + column - 1],
                    pixels[row * width + column + 1],
                    pixels[(row + 1) * width + column],
                )
                value = min(255, 4 * abs(4 * center - north - west - east - south))
            elif filter_id == 3:
                east = pixels[row * width + column + 1]
                south = pixels[(row + 1) * width + column]
                value = min(255, 4 * max(abs(center - east), abs(center - south)))
            else:
                value = sorted(neighbors)[4]
            result[row * width + column] = value
    return result


def run_mpi(executable, image_path, filter_id, output_dir):
    mpi_command = shlex.split(os.environ.get("MPIEXEC", "mpirun"))
    command = mpi_command + ["--oversubscribe", "-np", "2", str(executable)]
    command += [str(image_path), str(filter_id), str(ITERATIONS)]
    environment = os.environ.copy()
    environment["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    environment["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    return subprocess.run(
        command,
        cwd=output_dir,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def main():
    pixels = make_pixels()
    executables = [
        MPI_DIR / "parallel_blocking",
        MPI_DIR / "parallelnon_blocking",
    ]

    with tempfile.TemporaryDirectory(prefix="image-convolution-mpi-") as temp:
        work_dir = Path(temp)
        input_path = work_dir / "input.ras"
        write_sun_raster(input_path, pixels)

        for executable in executables:
            for filter_id in range(5):
                result = run_mpi(executable, input_path, filter_id, work_dir)
                if result.returncode != 0:
                    raise AssertionError(
                        f"{executable.name}, filter {filter_id} failed:\n"
                        f"{result.stdout}\n{result.stderr}"
                    )

                output_path = work_dir / (
                    f"post-convolution_filter{filter_id}_nbIter{ITERATIONS}.ras"
                )
                width, height, actual = read_sun_raster(output_path)
                if len(actual) != width * height:
                    raise AssertionError(
                        f"Unexpected pixel data length in {output_path}: "
                        f"{len(actual)} for {width}x{height}"
                    )
                expected = pixels.copy()
                for _ in range(ITERATIONS):
                    expected = apply_filter(expected, width, height, filter_id)
                if (width, height) != (IMAGE_WIDTH, IMAGE_HEIGHT):
                    raise AssertionError(f"Unexpected output dimensions: {width}x{height}")
                if actual != expected:
                    mismatch = next(
                        i for i, (got, want) in enumerate(zip(actual, expected))
                        if got != want
                    )
                    raise AssertionError(
                        f"{executable.name}, filter {filter_id}: pixel {mismatch} "
                        f"is {actual[mismatch]}, expected {expected[mismatch]}"
                    )

            invalid = subprocess.run(
                shlex.split(os.environ.get("MPIEXEC", "mpirun"))
                + ["--oversubscribe", "-np", "2", str(executable),
                   str(input_path), "9", "1"],
                cwd=work_dir,
                env={
                    **os.environ,
                    "OMPI_ALLOW_RUN_AS_ROOT": "1",
                    "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
                },
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if invalid.returncode == 0:
                raise AssertionError(f"{executable.name} accepted an invalid filter")

    print("Both MPI implementations passed all integration tests.")


if __name__ == "__main__":
    main()
