import glob
import os
from google.cloud import storage

storage.blob._DEFAULT_CHUNKSIZE = 5 * 1024 * 1024  # 5 MB
storage.blob._MAX_MULTIPART_SIZE = 100 * 1024 * 1024  # 5 MB

storage_client = storage.Client()
bucket = storage_client.bucket('shield-phishing')

'''Delete files'''

# blobs = bucket.list_blobs(prefix='face_verification/pretrained_models/face_net_nyoki/facenet.onnx')
# for blob in blobs:
#    blob.delete()

'''Create folder'''

# blob = bucket.blob('face_verification/pretrained_models/face_net_nyoki/saved_model/variables/')
# blob.upload_from_string('')

'''Copy blob'''

# source_blob = bucket.blob("mlflow-artifact-store/")
# new_blob = bucket.copy_blob(source_blob, bucket, "prod/" + "mlflow-artifact-store/")

'''Upload file'''

blob1 = bucket.blob('face_verification/pretrained_models/face_net_nyoki/facenet.onnx')
with open('../onnx/facenet.onnx', 'rb') as model:
     blob1.upload_from_file(model)
print('uploaded')

'''Upload folder'''


def upload_local_directory_to_gcs(local_path, gcs_path):
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(local_file, gcs_path + "/" + os.path.basename(local_file))
        else:
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


'''Download file'''

#blob = bucket.blob("face_verification/photos/C03i2p9bk/104365138969033191651/img0.png")
# Download the file to a destination
#blob.download_to_filename("img")
