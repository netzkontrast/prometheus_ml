# -*- coding: utf-8 -*-

"""Console script for prometheus_ml."""
import sys
import click
import prometheus_ml.cli.server as server


@click.command()
def main(args=None):
    #server.app.run_server(debug=True)
    click.echo("prometheus cli")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
