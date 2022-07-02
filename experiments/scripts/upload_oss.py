import argparse
import os

import oss2


if __name__ == '__main__':
    # Parse arguments.
    parser = argparse.ArgumentParser(description='Upload files to OSS')
    parser.add_argument('--dir', type=str, metavar='DIR', required=True, help='Directory to upload')
    parser.add_argument('--key-id', type=str, default='LTAI5tCx79brCnGXxKGTsAst')
    parser.add_argument('--key-secret', type=str, default='F0IVmA99YzX2x8LWkGrp8WBjVH9qsa')
    parser.add_argument('--end-point', type=str, default='oss-cn-hangzhou.aliyuncs.com')
    parser.add_argument('--bucket', type=str, default='canvas-imagenet')
    parser.add_argument('--prefix', type=str, default='')
    args = parser.parse_args()

    oss_try_times = 10

    def get_oss_bucket():
        return oss2.Bucket(oss2.Auth(args.key_id, args.key_secret), args.end_point, args.bucket, connect_timeout=5)

    length = len(args.dir)
    for root, subdirs, files in os.walk(args.dir):
        assert len(root) >= length
        prefix, _, path = root[0:length], root[length-1], root[length:]
        for file in files:
            success = False
            for i in range(oss_try_times):
                # noinspection PyBroadException
                try:
                    file_path, key_path = os.path.join(root, file), os.path.join(args.prefix, path, file)
                    print(f'Uploading {file_path} to {key_path} ...')
                    get_oss_bucket().put_object_from_file(key_path, file_path)
                    success = True
                    break
                except Exception as ex:
                    print(f'Failed to upload, try {i + 1} time(s)')
                    continue
            assert success
