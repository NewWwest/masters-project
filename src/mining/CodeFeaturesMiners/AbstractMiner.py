
separators = ['/','\\','`','*','_','{','}','[',']','(',')','>','#','+','-',',','.','!','$','\'', '\t', '\n', '\r']

class AbstractMiner:
    features = ['basic', 'code_analysis', 'bag_of_words', 'history']
    def __init__(self, feature_classes) -> None:
        self.feature_classes = set(feature_classes)
        self.features = {}


    def _add_feature(self, name, value):
        if value != None:
            if isinstance(value, bool):
                value = int(value)
            self.features[name] = value    
        else:
            self.features[name]=float('nan')

    def _escape_separators(self, text):
        for sep in separators:
            if sep in text:
                text = text.replace(sep, f' {sep} ')

        return text
