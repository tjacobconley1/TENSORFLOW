# Unzip a file
# Tyler Conley

import os
import zipfile

local_zip = 'faces.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/.git/TENSORFLOW/tmp')
zip_ref.close()

local_zip = 'facestest.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/.git/TENSORFLOW/tmp')
zip_ref.close()
