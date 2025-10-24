import click


from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pharmacophore2mol")
except PackageNotFoundError:
    # Package not installed, mark as development version
    __version__ = "dev"

@click.group()
@click.version_option(version=__version__, prog_name='pharmacophore2mol')
def cli():
    """Pharmacophore2Mol: Command Line Interface (CLI)
    """
    pass

@cli.command()
@click.argument('input_path', type=click.Path(exists=True, dir_okay=True, readable=True))
def evaluate(input_path):
    """
    Evaluate molecules from SDF or XYZ file(s).
    
    Automatically converts XYZ to SDF using OpenBabel if needed.
    
    Examples:
    
        >>> p2m evaluate xyz_dir
        >>> p2m evaluate molecules.sdf

    """
    from pharmacophore2mol.metrics.utils import evaluate_from_file
    
    # Run evaluation
    results = evaluate_from_file(
        input_path=input_path,
    )

    results.print_summary()