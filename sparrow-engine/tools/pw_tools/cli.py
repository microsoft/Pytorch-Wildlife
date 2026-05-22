"""pw-tools: Build-time CLI for PytorchWildlife model management.

Entry point registered in pyproject.toml:
    [project.scripts]
    pw-tools = "pw_tools.cli:app"

Requires: pip install pytorchwildlife[tools]
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="pw-tools",
    help=(
        "PytorchWildlife build-time toolkit. "
        "Convert PyTorch models to ONNX, generate manifests, validate, and upload."
    ),
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

# Sub-command groups
convert_app = typer.Typer(help="Convert PyTorch / TorchScript models to ONNX.", no_args_is_help=True)
manifest_app = typer.Typer(help="Create and inspect TOML model manifests.", no_args_is_help=True)
validate_app = typer.Typer(help="Validate manifests and run inference tests.", no_args_is_help=True)
upload_app = typer.Typer(help="Upload models to HuggingFace Hub.", no_args_is_help=True)

app.add_typer(convert_app, name="convert")
app.add_typer(manifest_app, name="manifest")
app.add_typer(validate_app, name="validate")
app.add_typer(upload_app, name="upload")

# ---------------------------------------------------------------------------
# Lazy-import sub-command modules so missing optional deps produce a clean
# error message pointing at the correct extra, rather than an ImportError
# traceback at CLI startup.
# ---------------------------------------------------------------------------


def _require(package: str, extra: str) -> None:
    """Raise a clean error if *package* is not importable."""
    import importlib

    try:
        importlib.import_module(package)
    except ImportError:
        typer.echo(
            f"[pw-tools] Missing dependency: {package!r}.\n"
            f"Install it with:  pip install pytorchwildlife[{extra}]\n",
            err=True,
        )
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# convert sub-commands
# ---------------------------------------------------------------------------


@convert_app.command("model")
def convert_model(
    source: str = typer.Argument(..., help="Path to .pt / .pth / TorchScript model file."),
    output: str = typer.Option(..., "--output", "-o", help="Destination .onnx path."),
    opset: int = typer.Option(17, "--opset", help="ONNX opset version (recommend 17)."),
    dynamic_batch: bool = typer.Option(False, "--dynamic-batch", help="Mark batch dimension as dynamic."),
    input_shape: str = typer.Option(
        "1,3,640,640",
        "--input-shape",
        help="Comma-separated NCHW shape for example input, e.g. '1,3,640,640'.",
    ),
    input_names: str = typer.Option("images", "--input-names", help="Comma-separated ONNX input names."),
    output_names: str = typer.Option("output0", "--output-names", help="Comma-separated ONNX output names."),
    use_dynamo: bool = typer.Option(
        True,
        "--dynamo/--no-dynamo",
        help="Use torch.onnx.export(dynamo=True) for PyTorch 2.5+ (recommended).",
    ),
    optimize: bool = typer.Option(True, "--optimize/--no-optimize", help="Run ORT graph optimization pass."),
    half: bool = typer.Option(False, "--half", help="Convert to FP16 after export (requires onnxconverter-common)."),
) -> None:
    """Convert a PyTorch or TorchScript model to ONNX.

    Example:
        pw-tools convert model megadetector_v6.pt --output megadetector_v6.onnx --opset 17 --dynamic-batch
    """
    _require("torch", "tools")
    _require("onnx", "tools")
    _require("onnxruntime", "tools")

    from pw_tools.convert import convert_to_onnx

    shape = tuple(int(x) for x in input_shape.split(","))
    i_names = [n.strip() for n in input_names.split(",")]
    o_names = [n.strip() for n in output_names.split(",")]

    convert_to_onnx(
        source_path=source,
        output_path=output,
        opset=opset,
        dynamic_batch=dynamic_batch,
        input_shape=shape,
        input_names=i_names,
        output_names=o_names,
        use_dynamo=use_dynamo,
        optimize=optimize,
        half=half,
    )


# ---------------------------------------------------------------------------
# manifest sub-commands
# ---------------------------------------------------------------------------


@manifest_app.command("create")
def manifest_create(
    onnx_path: str = typer.Argument(..., help="Path to the .onnx model file."),
    name: str = typer.Option(..., "--name", "-n", help="Human-readable model name, e.g. 'megadetector-v6'."),
    task: str = typer.Option(
        ...,
        "--task",
        "-t",
        help="Task type: detection | classification | audio_classification | point_detection",
    ),
    labels: str = typer.Option(None, "--labels", help="Path to labels .txt file (one label per line)."),
    config: str = typer.Option(None, "--config", help="Path to TOML config for preprocessing parameters."),
    output: str = typer.Option(None, "--output", "-o", help="Output manifest .toml path (default: <name>.toml)."),
    version: str = typer.Option("1.0.0", "--version", help="Model version string."),
    license_id: str = typer.Option("MIT", "--license", help="SPDX license identifier."),
    description: str = typer.Option("", "--description", "-d", help="Short one-line description."),
    tags: str = typer.Option("", "--tags", help="Comma-separated tags."),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Prompt for missing fields."),
) -> None:
    """Auto-generate a TOML manifest from an ONNX file.

    Example:
        pw-tools manifest create megadetector_v6.onnx --name megadetector-v6 --task detection --labels labels.txt
    """
    _require("onnx", "tools")
    _require("onnxruntime", "tools")

    from pw_tools.manifest import create_manifest

    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    out = output or f"{name}.toml"

    create_manifest(
        onnx_path=onnx_path,
        name=name,
        task=task,
        labels_path=labels,
        config_path=config,
        output_path=out,
        version=version,
        license_id=license_id,
        description=description,
        tags=tag_list,
        interactive=interactive,
    )


@manifest_app.command("inspect")
def manifest_inspect(
    manifest_path: str = typer.Argument(..., help="Path to a .toml manifest file."),
) -> None:
    """Pretty-print a manifest and show computed SHA256.

    Example:
        pw-tools manifest inspect megadetector-v6.toml
    """
    _require("tomllib", "tools")

    from pw_tools.manifest import inspect_manifest

    inspect_manifest(manifest_path)


# ---------------------------------------------------------------------------
# validate sub-commands
# ---------------------------------------------------------------------------


@validate_app.command("manifest")
def validate_manifest_cmd(
    manifest_path: str = typer.Argument(..., help="Path to a .toml manifest file."),
    test_images: str = typer.Option(None, "--test-images", help="Directory of test images (JPG/PNG)."),
    reference_outputs: str = typer.Option(
        None, "--reference-outputs", help="Directory of .npy reference output arrays."
    ),
    atol: float = typer.Option(1e-3, "--atol", help="Absolute tolerance for output comparison."),
    rtol: float = typer.Option(1e-3, "--rtol", help="Relative tolerance for output comparison."),
) -> None:
    """Validate a manifest: load model, run test images, compare to references.

    Example:
        pw-tools validate manifest megadetector-v6.toml --test-images ./test_data/ --reference-outputs ./refs/
    """
    _require("onnx", "tools")
    _require("onnxruntime", "tools")

    from pw_tools.validate import validate_manifest

    validate_manifest(
        manifest_path=manifest_path,
        test_images_dir=test_images,
        reference_outputs_dir=reference_outputs,
        atol=atol,
        rtol=rtol,
    )


# ---------------------------------------------------------------------------
# upload sub-commands
# ---------------------------------------------------------------------------


@upload_app.command("model")
def upload_model(
    manifest_path: str = typer.Argument(..., help="Path to a .toml manifest file."),
    repo: str = typer.Option(..., "--repo", "-r", help="HuggingFace repo id, e.g. 'pytorchwildlife/megadetector-v6'."),
    token: str = typer.Option(None, "--token", help="HF write token (or set HF_TOKEN env var)."),
    private: bool = typer.Option(False, "--private", help="Create private repository."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate everything but skip the actual upload."),
    commit_message: str = typer.Option("", "--message", "-m", help="Commit message for the upload."),
) -> None:
    """Upload an ONNX model, manifest, labels, and model card to HuggingFace Hub.

    Example:
        pw-tools upload model megadetector-v6.toml --repo pytorchwildlife/megadetector-v6
    """
    _require("huggingface_hub", "tools")

    from pw_tools.upload import upload_to_hub

    upload_to_hub(
        manifest_path=manifest_path,
        repo_id=repo,
        token=token,
        private=private,
        dry_run=dry_run,
        commit_message=commit_message,
    )


if __name__ == "__main__":
    app()
