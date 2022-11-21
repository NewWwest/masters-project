import os
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def info_log(*args):
    log_message = '\t'.join([str(x) for x in args])
    logging.info(log_message)


def warn_log(*args):
    log_message = '\t'.join([str(x) for x in args])
    logging.warning(log_message)


def get_files_in_from_directory(dir, extension=None, startswith=None):
    files_list = []
    for root, subdirs, files in os.walk(dir):
        for file in files:
            if extension != None and not file.endswith(extension):
                continue

            if startswith != None and not file.startswith(startswith):
                continue
            
            file_path = os.path.join(root, file)
            files_list.append(file_path)
    return files_list


def is_hexadecimal_string(val):
    try:
        test = int(val, 16)
        return True
    except:
        return False