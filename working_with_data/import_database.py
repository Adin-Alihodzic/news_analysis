import subprocess
from sys import argv

# takes in mongo database name as arg and path to current directory
db_name, path = argv[1], argv[2]

# Download mongo database from S3 Bucket
p1 = subprocess.Popen(['s3cmd', 'sync', 's3://dsiprojectdata/'+db_name+'.tar', path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out1, err1 = p1.communicate()
print('Got database from bucket.')

# Unzip file
p2 = subprocess.Popen(['tar', '-xvf', db_name+'.tar'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out2, err2 = p2.communicate()

# Save as mongo database
p3 = subprocess.Popen(['mongorestore', '--db', db_name, './'+db_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out3, err3 = p3.communicate()
print('Restored database to mongo.')
