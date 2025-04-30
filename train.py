from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--loss", choices=[
    ...
    # Add your custom loss here!
])

args = parser.parse_args()

match args.loss:
    case "mc":
        ...
    # Add your custom loss here!