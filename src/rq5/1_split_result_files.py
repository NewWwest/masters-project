import sys
sys.path.insert(0, r'D:\Projects\aaa')

import shutil 
import time
import json


from src.utils.utils import get_files_in_from_directory 



results_location_RollingWindowMiner = 'results/dl/RollingWindowMiner'
results_location_AddedCodeMiner = 'results/dl/AddedCodeMiner'
results_location_AST =  'results/dl/CodeParserMiner_ast'
results_location_Actions =  'results/dl/CodeParserMiner_edit'

new_results_location_RollingWindowMiner = 'results/dl/java2/RollingWindowMiner/'
new_results_location_AddedCodeMiner = 'results/dl/java2/AddedCodeMiner/'
new_results_location_AST =  'results/dl/java2/CodeParserMiner_ast/'
new_results_location_Actions =  'results/dl/java2/CodeParserMiner_edit/'

def filter_files_for_extensions(results_directory, valid_extensions):
    valid_files = []
    files = get_files_in_from_directory(results_directory)
    for filename in files:
        try:
            with open(filename, 'r') as f:
                temp_data = json.load(f)
                temp_data = [x for x in temp_data if x['file_name'].split('.')[-1] in valid_extensions]
                if len(temp_data) > 0:
                    valid_files.append(filename)
        except Exception as e:
            print('Failed to load', filename)
            print(e)

    return valid_files

def main(input_location, result_location):
    valid_extensions = set(['java'])
    valid_files = filter_files_for_extensions(input_location, valid_extensions)
    for f in valid_files:
        shutil.copy(f, result_location)


if __name__ == '__main__':
    start_time = time.time()
    main(results_location_AST, new_results_location_AST)
    main(results_location_Actions, new_results_location_Actions)
    main(results_location_AddedCodeMiner, new_results_location_AddedCodeMiner)
    main(results_location_RollingWindowMiner, new_results_location_RollingWindowMiner)
    print("--- %s seconds ---" % (time.time() - start_time))
    