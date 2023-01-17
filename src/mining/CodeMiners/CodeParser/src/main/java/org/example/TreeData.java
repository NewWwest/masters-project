/*
 * -----------------------------
 * Copyright 2022 Software Improvement Group
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ----------------------------- 
*/
package org.example;

import com.github.gumtreediff.tree.Tree;

public class TreeData {
    public Integer StartByte;
    public Integer EndByte;
    public String Label;
    public TreeData[] Children;

    public TreeData(String title, TreeData child){
        StartByte = child.StartByte;
        EndByte = child.EndByte;
        Label = title;
        Children = new TreeData[]{child};
    }

    public TreeData(TreeData[] children){
        StartByte = -1;
        EndByte = -1;
        Label = "root";
        Children = children;
    }

    public TreeData(Tree t, TreeData[] children){
        StartByte = t.getPos();
        EndByte = t.getEndPos();
        Label = t.toString();
        Children = children;
    }
}
