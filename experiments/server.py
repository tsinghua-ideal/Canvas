import argparse
import json
import os
import time

from http.server import HTTPServer, BaseHTTPRequestHandler

pending = {}
training = {}
finished = {}
forbidden = set()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, _, *__):
        pass

    def do_GET(self):
        # Assign an available kernel to the client.
        remote_ip = self.connection.getpeername()[0]
        print(f'Incoming GET request from {remote_ip}: {self.path}')
        if self.path != '/kernel':
            print(' > Bad request')
            return

        # Detect timeout kernels.
        timeout_kernels = []
        for name, pack in training.items():
            if time.time() - training[name]['time'] > 86400 * 10:  # 24 hours: 86400 secs
                timeout_kernels.append(name)
        for name in timeout_kernels:
            assert name not in pending
            assert name in training
            assert name not in finished
            pending[name] = training.pop(name)
            print(f' > Timeout kernel: {name}')

        selected_name, selected_pack = None, None
        for name, pack in sorted(list(pending.items()), key=lambda x: x[0], reverse=True):
            if (remote_ip, name) not in forbidden:
                selected_name, selected_pack = name, pack
                break
        response_json = {'name': '', 'py': '', 'dot': ''}
        if selected_pack is not None:
            print(f' > Response kernel: {selected_name}')
            response_json = {'name': selected_name, 'py': selected_pack['py'], 'dot': selected_pack['dot']}
            training[selected_name] = pending.pop(selected_name)
            training[selected_name]['time'] = time.time()
        else:
            print(f' > No available response')
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response_json).encode())

    def do_POST(self):
        # Response success or failure.
        remote_ip = self.connection.getpeername()[0]
        print(f'Incoming POST request from {remote_ip}: {self.path}')
        if not (self.path.startswith('/success?name=') or self.path.startswith('/failure?name=') or
                self.path.startswith('/test')):
            print(' > Bad request')
            return
        if not self.path.startswith('/test'):
            name = self.path[14:]
            assert len(name) > 0
        else:
            name = None
        if self.path.startswith('/success'):
            if name in pending:
                finished[name] = pending.pop(name)
            if name in training:
                finished[name] = training.pop(name)
            print(f' > Successfully trained: {name}')
        elif self.path.startswith('/failure'):
            assert name not in pending
            assert name in training
            assert name not in finished
            pending[name] = training.pop(name)
            forbidden.add((remote_ip, name))
            print(f' > Failed to train: {name}')
        else:
            print(f' > Testing message ...')
        self.send_response(200)
        self.send_header('Content-type', 'text')
        self.end_headers()


if __name__ == '__main__':
    # Arguments.
    parser = argparse.ArgumentParser(description='Canvas selector server')
    parser.add_argument('--path', type=str, metavar='PATH', required=True)
    parser.add_argument('--host', type=str, metavar='HOST', default='0.0.0.0')
    parser.add_argument('--port', type=int, metavar='PORT', default='8000')
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

    def collect_kernels():
        collects = {}
        for path in dirs:
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
                    collects[path.split('/')[-1]] = {'py': py_file.read(), 'dot': dot_file.read()}
        return collects

    pending = collect_kernels()
    print(f'{len(pending)} kernels collected')

    # Start HTTP server.
    print(f'Starting listening at {args.host}:{args.port}')
    server = HTTPServer((args.host, args.port), Handler)
    server.serve_forever()
