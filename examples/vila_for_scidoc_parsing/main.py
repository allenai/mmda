import argparse
import contextlib
import json
import os
import re
import urllib.request
from tempfile import NamedTemporaryFile
from typing import Dict, Generator, List, Optional

from layoutparser.elements import Layout, Rectangle, TextBlock
from layoutparser.visualization import draw_box
from PIL.Image import Image
from tqdm import tqdm

from mmda.recipes.core_recipe import CoreRecipe
from mmda.types.annotation import SpanGroup

DEFAULT_DEST_DIR = os.path.expanduser("~/mmda_predictions")


def is_url(url: str) -> bool:
    if os.path.exists(url):
        return False

    # regex to determine if a string is a valid URL
    return re.search(r"^(?:http|ftp)s?://", url) is not None


def get_dir_name(path: str) -> str:
    if is_url(path):
        return path.split("/")[-1].rstrip(".pdf")
    else:
        return os.path.basename(path).rstrip(".pdf")


@contextlib.contextmanager
def download_pdf(url: str) -> Generator[str, None, None]:
    name: Optional[str] = None

    # Create a temporary file
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        # Download the file using urllib
        with urllib.request.urlopen(url) as response:
            # Save the downloaded data to the temporary file
            temp_file.write(response.read())

        # Get the name of the temporary file
        name = temp_file.name

    # return the name of the temporary file
    yield name

    # Delete the temporary file
    os.remove(name)


def draw_tokens(
    image: Image,
    doc_tokens: List[SpanGroup],
    color_map: Optional[Dict[int, str]] = None,
    token_boundary_width: int = 0,
    alpha: float = 0.25,
    **lp_draw_box_kwargs,
):
    """Draw MMDA tokens as rectangles on the an image of a page."""

    w, h = image.size
    layout = [
        TextBlock(
            Rectangle(
                *token.spans[0]
                .box.get_absolute(page_height=h, page_width=w)  # pyright: ignore
                .coordinates
            ),
            type=token.type,
            text=token.text,
        )
        for token in doc_tokens
    ]

    return draw_box(
        image,
        Layout(blocks=layout),
        color_map=color_map,
        box_width=token_boundary_width,
        box_alpha=alpha,
        **lp_draw_box_kwargs,
    )


def draw_blocks(
    image: Image,
    doc_tokens: List[SpanGroup],
    color_map: Optional[Dict[int, str]] = None,
    token_boundary_width: int = 0,
    alpha: float = 0.25,
    **lp_draw_box_kwargs,
):
    w, h = image.size
    layout = [
        TextBlock(
            Rectangle(
                *token.box_group.boxes[0]  # pyright: ignore
                .get_absolute(page_height=h, page_width=w)
                .coordinates
            ),
            type=token.type,
            text=token.text,
        )
        for token in doc_tokens
    ]
    return draw_box(
        image,
        Layout(blocks=layout),
        color_map=color_map,
        box_width=token_boundary_width,
        box_alpha=alpha,
        **lp_draw_box_kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--pdf-path",
        type=str,
        nargs="+",
        required=True,
        help="Path to PDF file(s) to be processed. Can be a URL.",
    )
    parser.add_argument(
        "-d",
        "--destination",
        type=str,
        default=DEFAULT_DEST_DIR,
        help=(
            "Path to directory where to save the results. "
            "A directory will be created for each paper"
        ),
    )
    args = parser.parse_args()

    recipe = CoreRecipe()

    pbar = tqdm(args.pdf_path)
    for pdf_path in pbar:
        pbar.set_description(f"Working on {pdf_path}")

        if is_url(pdf_path):
            with download_pdf(pdf_path) as temp_file:
                doc = recipe.from_path(temp_file)
        else:
            assert os.path.exists(pdf_path), f"PDF file {pdf_path} does not exist."
            doc = recipe.from_path(pdf_path)

        # get location where to save data
        save_folder = os.path.join(args.destination, get_dir_name(pdf_path))
        os.makedirs(save_folder, exist_ok=True)

        # save images with layout and text predictions
        for pid, (page, image) in enumerate(zip(doc.pages, doc.images)):
            new_tokens = []
            for pred in page.vila_span_groups:
                for token in pred.tokens:
                    new_token = token.from_json(token.to_json())
                    new_token.type = pred.type
                    new_token.text = token.text
                    new_tokens.append(new_token)

            # draw sections and equations
            viz = draw_tokens(image, new_tokens, alpha=0.6)
            viz = draw_blocks(viz, page.equations, alpha=0.2)

            viz.save(f"{save_folder}/{pid}.png")

        # Save all text alongside name of section it belongs to
        with open(f"{save_folder}/sections.jsonl", "w") as f:
            for pid, page in enumerate(doc.pages):  # pyright: ignore
                for pred in page.vila_span_groups:
                    data = {
                        "text": str(pred.text),
                        "type": pred.type,
                        "page": int(pid),
                    }
                    f.write(f"{json.dumps(data, sort_keys=True)}\n")

        # Save all equations
        with open(f"{save_folder}/equations.jsonl", "w") as f:
            for pid, page in enumerate(doc.pages):  # pyright: ignore
                for pred in page.equations:
                    data = {
                        "text": str(pred.text),
                        "page": int(pid),
                    }
                    f.write(f"{json.dumps(data, sort_keys=True)}\n")
