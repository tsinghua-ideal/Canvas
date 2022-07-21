import argparse
import json
import os


if __name__ == '__main__':
    # Arguments.
    parser = argparse.ArgumentParser(description='Canvas selector server')
    parser.add_argument('--path', type=str, metavar='PATH', required=True)
    args = parser.parse_args()

    # Collect files to JSONs.
    dirs = [os.path.join(args.path, f) for f in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, f))]

    def filter_legal(path: str):
        files = os.listdir(path)
        if len(files) != 3:
            return False
        a_p, a_s = files[0].split('.')
        b_p, b_s = files[1].split('.')
        c_p, c_s = files[2].split('.')
        if a_p == b_p and b_p == c_p:
            return {a_s, b_s, c_s} == {'py', 'dot', 'json'}

    dirs = list(filter(filter_legal, dirs))

    def to_json(path: str):
        files = os.listdir(path)
        assert len(files) == 3
        a_p, a_s = files[0].split('.')
        b_p, b_s = files[1].split('.')
        c_p, c_s = files[2].split('.')
        assert a_p == b_p and b_p == c_p
        assert {a_s, b_s, c_s} == {'py', 'dot', 'json'}
        prefix = os.path.join(path, a_p)
        with open(f'{prefix}.py', 'r') as py_file:
            with open(f'{prefix}.dot', 'r') as dot_file:
                return json.dumps({'py': py_file.read(), 'dot': dot_file.read()})

    jsons = list(map(to_json, dirs))
    print(f'{len(jsons)} kernels collected')
