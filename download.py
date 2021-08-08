import os
import time
import argparse
import requests
from tqdm import tqdm
from joblib import Parallel, delayed
import py7zr

def arglist():

    parser = argparse.ArgumentParser(prog="Google Drive Dataset Downloader",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    dataset = parser.add_argument_group("Required: What Dataset To Download?")
    dataset.add_argument('--CelebA', default=False, action='store_true',
                         help="CelebA Dataset")

    parser.add_argument('-d', '--directory', default=None, type=str,
                        help="Specify the directory to download to")
    parser.add_argument('-n', '--ncpus', default=1, type=int,
                        help="specify the number of CPUs")

    return parser.parse_args()


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_parallel(fnames, drive_path, path, args):
    root_url = "https://docs.google.com/uc?export=download"
    Parallel(n_jobs=args.ncpus, prefer="threads")(
            delayed(download)
            (root_url, name, id, f"{args.directory}/{path}")
            for name, id in tqdm(zip(fnames, drive_path),
                                 total=len(fnames)))


def get_token(response):
    for key, val in response.cookies.items():
        if key.startswith('download_warning'):
            return val
    return None

def save_response(response, fname, dest, chunk_size=32 * 1024):
    size = int(response.headers.get('content-length', 0))
    with open(f"{dest}/{fname}", 'wb') as out:
        for chunk in tqdm(response.iter_content(chunk_size),
                          total=size,
                          leave=False,
                          desc=fname):
            if chunk:
                out.write(chunk)


def download(root_url, fname, drive_id, dest):
    session = requests.Session()
    response = session.get(root_url, params={'id': drive_id}, stream=True)
    token = get_token(response)
    if token:
        params = {'id': drive_id, 'confirm': token}
        response = session.get(root_url, params=params, stream=True)
    save_response(response, fname, dest)


def extract_parallel(path, args):
    files = os.listdir(f"{args.directory}/{path}")
    Parallel(n_jobs=args.ncpus)(
            delayed(extract)(f, f"{args.directory}/{path}")
            for f in tqdm(files))


def extract(zip_file, path):
    print(f"Extracting: {path}/{zip_file}")
    archive = py7zr.SevenZipFile(f"{path}/{zip_file}", mode='r')
    archive.extractall(path=path)
    archive.close()
    os.remove(f"{path}/{zip_file}")


def CelebA(args):
    IMGS = {"align": [f"img_align_celeba_png.7z.{str(i).zfill(3)}" for i in range(1,17)],
            "align_ids": ['0B7EVK8r0v71pSVd0ZjQ3Sks2dzg',
                          '0B7EVK8r0v71pR2NwRnU2cVZ2RTg',
                          '0B7EVK8r0v71peUlHSDVhd0JTamM',
                          '0B7EVK8r0v71pVmYwbmRtV2hZcDA',
                          '0B7EVK8r0v71pVjRlNVB3cDVjaDQ',
                          '0B7EVK8r0v71pa3NIcEgtTXZrM0U',
                          '0B7EVK8r0v71pNE5aQmY5c2ZLOXc',
                          '0B7EVK8r0v71pejhuem9QV2h0MDQ',
                          '0B7EVK8r0v71pZk5QcUlObVltaEE',
                          '0B7EVK8r0v71pLThPNzFETUNMUVE',
                          '0B7EVK8r0v71pZWZ4UGdBbk9UVWs',
                          '0B7EVK8r0v71pSk1zVWN2aHhMZ3c',
                          '0B7EVK8r0v71pNjFfTGYzTWJDdUU',
                          '0B7EVK8r0v71pbFlZaURkY3dhWWM',
                          '0B7EVK8r0v71pczZ0NFNFdFRXSUU',
                          '0B7EVK8r0v71pckZsdFFIYlJoN1k'],
            "imgs": [f"img_celeba.7z.{str(i).zfill(3)}" for i in range(1, 15)],
            "img_ids": ['0B7EVK8r0v71pQy1YUGtHeUM2dUE',
                        '0B7EVK8r0v71peFphOHpxODd5SjQ',
                        '0B7EVK8r0v71pMk5FeXRlOXcxVVU',
                        '0B7EVK8r0v71peXc4WldxZGFUbk0',
                        '0B7EVK8r0v71pMktaV1hjZUJhLWM',
                        '0B7EVK8r0v71pbWFfbGRDOVZxOUU',
                        '0B7EVK8r0v71pQlZrOENSOUhkQ3c',
                        '0B7EVK8r0v71pLVltX2F6dzVwT0E',
                        '0B7EVK8r0v71pVlg5SmtLa1ZiU0k',
                        '0B7EVK8r0v71pa09rcFF4THRmSFU',
                        '0B7EVK8r0v71pNU9BZVBEMF9KN28',
                        '0B7EVK8r0v71pTVd3R2NpQ0FHaGM',
                        '0B7EVK8r0v71paXBad2lfSzlzSlk',
                        '0B7EVK8r0v71pcTFwT1VFZzkzZk0'],
            "anno_names": ['identity_CelebA.txt',
                           'list_bbox_celeba.txt',
                           'list_attr_celeba.txt',
                           'list_lasdmarks_align_celeba.txt',
                           'list_landmarks_celeba.txt'],
            "anno_ids": ['1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS',
                         '0B7EVK8r0v71pbThiMVRxWXZ4dU0',
                         '0B7EVK8r0v71pblRyaVFSWGxPY0U',
                         '0B7EVK8r0v71pd0FJY3Blby1HUTQ',
                         '0B7EVK8r0v71pTzJIdlJWdHczRlU'],
            "eval_names": ['list_eval_partition.txt'],
            "eval_ids": ['0B7EVK8r0v71pY0NSMzRuSXJEVkk'],
            "readme_id": ['0B7EVK8r0v71pOXBhSUdJWU1MYUk'],
            }

    # Make directories and download
    #makedir(f"{args.directory}/Img/img_align_celeba")
    #download_parallel(IMGS['align'],
    #                  IMGS['align_ids'],
    #                  "Img/img_align_celeba",
    #                  args)
    #extract_parallel(f"Img/img_align_celeba",
    #                 args)
    #print(f"Obtained Align Images")

    makedir(f"{args.directory}/Img/img_celeba")
    download_parallel(IMGS['imgs'],
                      IMGS['img_ids'],
                      "Img/img_celeba",
                      args)
    extract_parallel(f"Img/img_celeba",
                     args)
    #makedir(f"{args.directory}/Anno")
    #makedir(f"{args.directory}/Eval")


def main():
    args = arglist()
    makedir(args.directory)
    CelebA(args)


if __name__ == '__main__':
    main()
