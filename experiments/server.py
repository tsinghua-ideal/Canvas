import argparse
import json
import os
import time

from http.server import HTTPServer, BaseHTTPRequestHandler


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        print(self)
        time.sleep(1)

    def do_POST(self):
        print(self)
        time.sleep(1)


if __name__ == '__main__':
    # Arguments.
    parser = argparse.ArgumentParser(description='Canvas selector server')
    parser.add_argument('--path', type=str, metavar='PATH', required=True)
    parser.add_argument('--port', type=int, metavar='PORT', default='80')
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

    # Start HTTP server.
    print(f'Starting listening at localhost:{args.port}')
    server = HTTPServer(('localhost', args.port), Handler)
    server.serve_forever()
