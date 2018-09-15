from pydrive.auth import GoogleAuth
from apiclient import discovery
import httplib2
from oauth2client.service_account import ServiceAccountCredentials
from pydrive.drive import GoogleDrive
from pathlib import Path



def _uploadToDrive(folder_name, file_name, file_location, parent='1LgT1F3_fZHeoshMGVD3rwh0gj1FaUbgS',  delete=False):
    import os

    credentials = ServiceAccountCredentials.from_json_keyfile_name(str(Path.home())+"/key3.json", scopes='https://www.googleapis.com/auth/drive')

    gauth = GoogleAuth()
    gauth.credentials = credentials

    drive = GoogleDrive(gauth)

    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(parent)}).GetList()

    top_level = None
    for f in file_list:
        if f['title'] == folder_name:
            top_level = f

    if not top_level:
        top_level = drive.CreateFile({'title': folder_name, 'parents':[{'id':parent}], 'mimeType':'application/vnd.google-apps.folder'})
        top_level.Upload()

    parent_id = top_level['id']
    file2 = drive.CreateFile({'title': file_name , 'parents':[{'id':parent_id}]})
    file2.SetContentFile(file_location)
    file2.Upload()
    if delete:
        os.remove(file_location)

def uploadToDrive(folder_name, file_name, file_location, parent='1LgT1F3_fZHeoshMGVD3rwh0gj1FaUbgS',  delete=False):
    try:
        _uploadToDrive(folder_name, file_name, file_location, parent, delete)
    except:
        print("WARNING: upload to drive failed")
