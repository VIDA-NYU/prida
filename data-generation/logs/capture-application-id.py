import sys

stdout_filename = sys.argv[1]
application_id = ''
str_application_id_start = 'INFO yarn.Client: Application report for '
str_application_id_end = ' (state: ACCEPTED)'

stdout_file = open(stdout_filename)
line = stdout_file.readline()
while line != '':
    if str_application_id_start in line:
        application_id = line[line.find(str_application_id_start)+len(str_application_id_start):line.find(str_application_id_end)]
        break
    line = stdout_file.readline()
stdout_file.close()

print(application_id)
