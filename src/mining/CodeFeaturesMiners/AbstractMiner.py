# -----------------------------
# Copyright 2022 Software Improvement Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------

from src.mining.CodeFeaturesMiners.constants_and_configs import separators

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
