import pandas as pd
import regex as re
import json

cwe_input = 'cwe_unique.csv'
keywords_inputs = 'vuln_keywords - Copy.txt'
not_covered_output = 'cwe_decloned.csv'
not_covered_tf_output = 'cwe_decloned.json'
keywords_unique_output = 'final_keywords.csv'

def main():
    remove_repetitions()
    produce_tf_file()

def remove_repetitions():
    df = pd.read_csv(cwe_input)
    df2 = pd.read_csv(keywords_inputs)
    regexes = {}
    for _,x in df2.iterrows():
        reeeee = re.compile(r'\b'+x['Name'].lower()+r'\b')
        regexes[x['Name']] = reeeee

    not_covered_cwe = []
    for _,y in df.iterrows():
        to_add = True
        for reee in regexes:
            if regexes[reee].search(y['Name'].lower()):
                to_add = False
                break

        if to_add:
            not_covered_cwe.append(y['Name'])

    not_covered_cwe.sort()
    df_temp=pd.DataFrame(not_covered_cwe)
    df_temp.to_csv(not_covered_output, index=False)

    not_covered_keywords = []
    for _,y in df2.iterrows():
        to_add = True
        for reee in regexes:
            if reee != y['Name'] and regexes[reee].search(y['Name'].lower()):
                to_add = False
                break

        if to_add:
            not_covered_keywords.append(y['Name'].lower())

    not_covered_keywords.sort()
    df_temp=pd.DataFrame(not_covered_keywords)
    df_temp.to_csv(keywords_unique_output, index=False)

    
def produce_tf_file():
    df = pd.read_csv(not_covered_output)
    reeeee = re.compile(r'\b')

    words = {}
    for _,x in df.iterrows():
        stuff = reeeee.split(x['0'])

        for y in stuff:
            if y and y != '' and y != ' ' and y!='\r\n':
                if y not in words:
                    words[y]=0
                words[y]+=1

    sorteds = {k: v for k, v in sorted(words.items(), key=lambda item: -item[1])}

    with open(not_covered_tf_output, 'w') as f:
        json.dump(sorteds, f, indent=2)

if __name__ == '__main__':
    main()