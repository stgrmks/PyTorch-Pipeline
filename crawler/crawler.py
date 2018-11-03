_author__ = 'MSteger'
import os

from torchvision import transforms
from google_images_download import google_images_download

def run_crawler(arguments):
    loader = google_images_download.googleimagesdownload()
    return loader.download(arguments)


if __name__ == '__main__':
    crawler_args = {
        'keywords': 'elefant',
        'limit': 5000,
        'size': 'large',
        'print_urls': False,
        'metadata': False,
        'output_directory': '/media/msteger/storage/resources/DreamPhant/downloads/',
        'no_directory': False,
        'chromedriver': '/home/mks/Downloads/chromedriver',
        'related_images': True,
        'extract_metadata': True
    }
    run_crawler(arguments = crawler_args)