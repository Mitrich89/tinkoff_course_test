import argparse
from train import NgrammModel


def main():
    parser = argparse.ArgumentParser(description='NgrammModel')
    parser.add_argument("--model", help="model filename")
    parser.add_argument("--length", help="length of output generated text")
    parser.add_argument("--n", nargs='?', help="Choose amount of gramms in ngramm-model", default='4')
    parser.add_argument("--prefix", nargs='*', help="beginning of the text", default='')
    args = parser.parse_args()
    with NgrammModel(model_filename=args.model) as model:
        for s in model.generate(int(args.length), int(args.n), args.prefix):
            print(s)


if __name__ == '__main__':
    main()
