import subprocess
from sys import argv
import os

def from_bucket(filename, path, mongo_db=False):
    # Download mongo database from S3 Bucket
    p1 = subprocess.Popen(['s3cmd', 'sync', 's3://'+filename, path+'/data/'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out1, err1 = p1.communicate()

    if mongo_db:
        f = filename.split('/')[-1]
        db_name = ''
        if filename.endswith('.tar'):
            db_name = f[-1]
            # Remove .tar from end
            db_name = db_name[:-4]
        else:
            print('Mongo DB is not in a .tar file!')
            return False
        # Unzip file
        p2 = subprocess.Popen(['tar', '-xvf', path+'/data/'+f], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out2, err2 = p2.communicate()

        # Save as mongo database
        p3 = subprocess.Popen(['mongorestore', '--db', db_name, './'+db_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out3, err3 = p3.communicate()

        # Remove tar file and folder
        p4 = subprocess.Popen(['rm', path+'/data/'+f], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out4, err4 = p4.communicate()
        p4 = subprocess.Popen(['rm', '-r', f[:-4]], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out4, err4 = p4.communicate()

    return True

if __name__ == '__main__':
    # takes in filename as arg and path to current directory
    # Example filename: dsiprojectdata/rss_feeds_new.tar
    filename, mongo_db = argv[1], argv[2]
    path = '..'
    result = from_bucket(filename, path, mongo_db=mongo_db)
