import canvas
import json
import requests


class Handler:
    def __init__(self, args):
        self.addr = args.canvas_selector_address

    def get_kernel(self):
        response = requests.get(f'{self.addr}/kernel')
        j = json.loads(response.text)
        assert 'py' in j and 'dot' in j and 'name' in j
        if not j['name']:
            return None, None
        return j['name'], canvas.KernelPack(torch_code=j['py'], graphviz_code=j['dot'])

    def success(self, name):
        assert name is not None
        requests.post(f'{self.addr}/success?name={name}')

    def failure(self, name):
        if name is not None:
            requests.post(f'{self.addr}/failure?name={name}')
