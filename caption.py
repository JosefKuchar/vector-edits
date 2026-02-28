import json
import pathlib
import uuid
from typing import Annotated, Any, Iterator, Self, cast

import PIL.Image
import polars as pl
import portalocker
import pydantic
import torch
import tqdm.auto
import transformers
import typer
import vllm
import datasets

import svgai.img
import svgai.svg

app = typer.Typer(
    pretty_exceptions_show_locals=False,
    rich_markup_mode="rich",
)


@app.command()
def main(
    output_path: Annotated[
        pathlib.Path,
        typer.Argument(
            help="Path to the output file",
            writable=True,
            dir_okay=False,
            file_okay=True,
        ),
    ],
    chunk_size: Annotated[int, typer.Option(help="Number of items to process in one chunk")] = 1000,
    max_tokens: Annotated[int, typer.Option(help="Maximum number of tokens in the generated captions")] = 400,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 0.0,
    append_if_exists: Annotated[bool, typer.Option(help="Append to the output file if it exists")] = False,
) -> None:
    if output_path.exists() and not append_if_exists:
        typer.secho(f"Output file {output_path} already exists.", fg=typer.colors.YELLOW)
        typer.confirm("Do you want to append new content to it?", abort=True)

    dataset = datasets.load_dataset("authoranonymous321/VectorEdits", split="validation")

    checkpoint_name = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
    min_img_width = 28
    min_img_height = 28

    lm = vllm.LLM(
        checkpoint_name,
        max_model_len=1536,
        task="generate",
        limit_mm_per_prompt={"image": 2, "video": 0},
        gpu_memory_utilization=0.95,
    )
    processor = transformers.AutoProcessor.from_pretrained(checkpoint_name)
    sampling_params = vllm.SamplingParams(
        max_tokens=max_tokens,
        n=1,
        temperature=temperature,
        logprobs=0,
    )
    offset = 0

    try:
        with tqdm.auto.tqdm(total=len(dataset), desc="Overall progress") as global_pbar:
            while True:
                try:
                    chunk = dataset.select(range(offset, min(offset + chunk_size, len(dataset))))
                    print(chunk)
                    offset += chunk_size
                    global_pbar.update(len(chunk))
                    if len(chunk) == 0:
                        global_pbar.write(
                            typer.style(
                                "Everything done or planned by other workers. Exiting", fg=typer.colors.GREEN
                            )
                        )
                        raise typer.Exit()
                    chunk_output = process_chunk(
                        chunk,
                        lm,
                        sampling_params,
                        processor,
                        512,
                        512,
                        min_img_width,
                        min_img_height,
                    )
                    save_output_chunk(output_path, chunk_output)
                except portalocker.AlreadyLocked:
                    global_pbar.write(typer.style("Cannot acquire lock file", fg=typer.colors.RED))
    finally:
        # graceful exit of vllm
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def read_jsonl_column(path: pathlib.Path, column: str) -> Iterator[Any]:
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            entry = cast(dict, entry)
            yield entry[column]


def save_output_chunk(
    output_path: pathlib.Path,
    chunk_output: pl.DataFrame,
) -> None:
    output_path.touch(exist_ok=True)
    with open(output_path, "a") as f:
        chunk_output.write_ndjson(f)


def make_model_inputs(
    image_1: PIL.Image.Image,
    image_2: PIL.Image.Image,
    processor: transformers.ProcessorMixin,
) -> vllm.TextPrompt:
    prompt = (
        "Create prompt instruction for transforming the first image into the second image. Only output the prompt. Be precise and concise. Use plain language. Do not use any special characters or formatting. Do not include any additional information."
    )

    thread = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
                {"type": "image"}
            ],
        },
    ]

    assert isinstance(processor, transformers.ProcessorMixin)
    tokenized = processor.apply_chat_template(thread, tokenize=False, add_generation_prompt=True)

    return vllm.TextPrompt(
        prompt=tokenized,
        multi_modal_data={
            "image": [image_1, image_2],
        },
    )


class ItemOutput(pydantic.BaseModel):
    item_id_1: str | int
    item_id_2: str | int
    instruction: str
    error: str | None = None

    @classmethod
    def make_error(cls, item_id_1: str | int, item_id_2, error: Exception) -> Self:
        return cls(
            item_id_1=item_id_1,
            item_id_2=item_id_2,
            error=repr(error),
            instruction="",
        )


def process_chunk(
    chunk: pl.DataFrame,
    lm: vllm.LLM,
    sampling_params: vllm.SamplingParams,
    processor: transformers.ProcessorMixin,
    img_width: int,
    img_height: int,
    min_img_width: int,
    min_min_height: int,
) -> pl.DataFrame:
    output_chunk: list[ItemOutput] = []

    chunk_inputs: list[vllm.TextPrompt] = []
    todo_ids: list[Any] = []

    for item in chunk.to_list():
        item_id_1 = item["item_1"]["item_id"]
        item_id_2 = item["item_2"]["item_id"]
        svg_1 = item["item_1"]["item_svg"]
        svg_2 = item["item_2"]["item_svg"]

        try:
            image_1 = svgai.svg.render_fit(svg_1, width=img_width, height=img_height, background="white")
            image_1 = svgai.img.center_pad_image(image_1, min_img_width, min_min_height)
            image_2 = svgai.svg.render_fit(svg_2, width=img_width, height=img_height, background="white")
            image_2 = svgai.img.center_pad_image(image_1, min_img_width, min_min_height)
        except Exception as e:
            output_chunk.append(ItemOutput.make_error(item_id_1, item_id_2, e))
            continue

        item_inputs = make_model_inputs(
            image_1, image_2, processor
        )
        chunk_inputs.append(item_inputs)
        todo_ids.append((item_id_1, item_id_2))

    chunk_outputs = lm.generate(chunk_inputs, sampling_params)
    for (item_id_1, item_id_2), item_output in zip(todo_ids, chunk_outputs):
        assert isinstance(item_output, vllm.RequestOutput)
        captions = [completion.text for completion in item_output.outputs]
        output_chunk.append(
            ItemOutput(
                item_id_1=item_id_1,
                item_id_2=item_id_2,
                instruction=captions[0] if captions else "",
            )
        )

    return pl.DataFrame(output_chunk)


if __name__ == "__main__":
    app()
